#!/usr/bin/env python3
"""Wait for the frozen evaluation sweep and render its audited final report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return payload


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--posttrain-orchestration", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--pretrain-step0", type=Path, required=True)
    parser.add_argument("--internal-validation", type=Path, required=True)
    parser.add_argument("--table3-orchestration", type=Path, required=True)
    parser.add_argument("--table3-index", type=Path, required=True)
    parser.add_argument("--aggregate-manifest", type=Path, required=True)
    parser.add_argument("--state-provenance", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument(
        "--expected-complete-stage",
        action="append",
        help="Accepted completed upstream stage; repeat to allow more than one.",
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def archive_partial_outputs(outputs: list[Path], manifest: dict[str, Any]) -> None:
    existing = [path for path in outputs if path.exists()]
    if not existing:
        return
    archive = Path(manifest["identity"]["manifest"]).parent / (
        "interrupted-report-" + utc_now().replace(":", "").replace("+00:00", "Z")
    )
    archive.mkdir(parents=True, exist_ok=False)
    for path in existing:
        shutil.move(str(path), archive / path.name)
    manifest.setdefault("archived_partial_outputs", []).append(
        {
            "archived_at_utc": utc_now(),
            "archive": str(archive),
            "files": [path.name for path in existing],
        }
    )


def main() -> int:
    args = parse_args()
    if args.poll_seconds < 5 or args.poll_seconds > 60:
        raise ValueError("poll interval must be between 5 and 60 seconds")

    python = args.python.resolve(strict=True)
    renderer = args.renderer.resolve(strict=True)
    posttrain = args.posttrain_orchestration.absolute()
    manifest_path = args.manifest.absolute()
    log_path = args.log.absolute()
    outputs = [args.output_md.absolute(), args.output_html.absolute(), args.output_audit.absolute()]
    expected_complete_stages = sorted(
        set(args.expected_complete_stage or ["evaluation_complete_report_pending"])
    )
    input_paths = {
        "training_manifest": args.training_manifest.absolute(),
        "pretrain_step0": args.pretrain_step0.absolute(),
        "internal_validation": args.internal_validation.absolute(),
        "posttrain_orchestration": posttrain,
        "table3_orchestration": args.table3_orchestration.absolute(),
        "table3_index": args.table3_index.absolute(),
        "aggregate_manifest": args.aggregate_manifest.absolute(),
        "state_provenance": args.state_provenance.absolute(),
        "split_manifest": args.split_manifest.absolute(),
        "execution_manifest": args.execution_manifest.absolute(),
    }
    identity = {
        "python": str(python),
        "renderer": {
            "path": str(renderer),
            "sha256": sha256_file(renderer),
        },
        "posttrain_orchestration": str(posttrain),
        "inputs": {name: str(path) for name, path in input_paths.items()},
        "outputs": [str(path) for path in outputs],
        "manifest": str(manifest_path),
        "expected_complete_stages": expected_complete_stages,
    }

    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if manifest.get("identity") != identity:
            raise RuntimeError("existing report-supervisor identity mismatch")
        if manifest.get("status") == "complete":
            for path in outputs:
                path.resolve(strict=True)
            print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
    else:
        manifest = {
            "schema_version": 1,
            "status": "running",
            "stage": "waiting_for_evaluation",
            "started_at_utc": utc_now(),
            "resume_count": 0,
            "identity": identity,
        }

    def update(stage: str, **values: Any) -> None:
        manifest.update(values)
        manifest["status"] = "running"
        manifest["stage"] = stage
        manifest["updated_at_utc"] = utc_now()
        atomic_json_write(manifest_path, manifest)

    try:
        update("waiting_for_evaluation")
        while True:
            if posttrain.is_file():
                evaluation = load_json(posttrain)
                status = evaluation.get("status")
                if status == "failed":
                    raise RuntimeError(
                        "post-training evaluation failed: "
                        + str(evaluation.get("exception", "unknown error"))
                    )
                if status == "complete":
                    if evaluation.get("stage") not in expected_complete_stages:
                        raise RuntimeError("unexpected completed evaluation stage")
                    break
            time.sleep(args.poll_seconds)

        archive_partial_outputs(outputs, manifest)
        update("rendering_audited_report")
        command = [str(python), str(renderer)]
        for name, path in input_paths.items():
            command.extend(["--" + name.replace("_", "-"), str(path)])
        command.extend(
            [
                "--output-md",
                str(outputs[0]),
                "--output-html",
                str(outputs[1]),
                "--output-audit",
                str(outputs[2]),
            ]
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n{utc_now()} report_started command={json.dumps(command)}\n")
            handle.flush()
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)

        audit = load_json(outputs[2].resolve(strict=True))
        if audit.get("status") != "passed":
            raise RuntimeError("generated report audit did not pass")
        markdown_sha = sha256_file(outputs[0].resolve(strict=True))
        html_sha = sha256_file(outputs[1].resolve(strict=True))
        if audit.get("outputs", {}).get("markdown", {}).get("sha256") != markdown_sha:
            raise RuntimeError("markdown output SHA-256 mismatch")
        if audit.get("outputs", {}).get("html", {}).get("sha256") != html_sha:
            raise RuntimeError("HTML output SHA-256 mismatch")

        manifest["status"] = "complete"
        manifest["stage"] = "report_complete"
        manifest["completed_at_utc"] = utc_now()
        manifest["outputs"] = {
            "markdown": {
                "path": str(outputs[0]),
                "sha256": markdown_sha,
                "bytes": outputs[0].stat().st_size,
            },
            "html": {
                "path": str(outputs[1]),
                "sha256": html_sha,
                "bytes": outputs[1].stat().st_size,
            },
            "audit": {
                "path": str(outputs[2]),
                "sha256": sha256_file(outputs[2]),
                "bytes": outputs[2].stat().st_size,
            },
        }
        atomic_json_write(manifest_path, manifest)
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except BaseException as exception:
        manifest["status"] = "failed"
        manifest["stage"] = "failed"
        manifest["failed_at_utc"] = utc_now()
        manifest["exception"] = repr(exception)
        manifest["traceback"] = traceback.format_exc()
        atomic_json_write(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
