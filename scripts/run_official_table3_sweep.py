#!/usr/bin/env python3
"""Run a recoverable, same-code Table 3 baseline and checkpoint sweep."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import torch


CLASSES = (
    ("en", "complete"),
    ("en", "incomplete"),
    ("zh", "complete"),
    ("zh", "incomplete"),
)

CLASS_CONFIG = {
    "en": {
        "dataset": Path("/root/autodl-tmp/dataset/soulx_duplug_eval/extracted/Easy-Turn-Testset-en"),
        "config": Path("/root/SoulX-stage3-dataset/configs/soulx_table3_candidate_en.yaml"),
        "asr": Path("/root/SoulX-stage3-dataset/pretrained_models/SenseVoiceSmall"),
    },
    "zh": {
        "dataset": Path("/root/autodl-tmp/dataset/soulx_duplug_eval/raw/easy_turn_zh_5812651/testset"),
        "config": Path("/root/SoulX-stage3-dataset/configs/soulx_table3_candidate_zh.yaml"),
        "asr": Path("/root/SoulX-stage3-dataset/pretrained_models/paraformer-zh"),
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def output_valid(
    output: Path,
    evaluation_root: Path,
    language: str,
    label: str,
    checkpoint: dict[str, Any] | None,
) -> bool:
    try:
        payload = load_json(output)
        valid = (
            payload.get("status") == "complete"
            and payload.get("language") == language
            and payload.get("label") == label
            and Path(payload["project"]["root"]).resolve() == evaluation_root
            and payload["project"].get("dirty") is False
        )
        continuation = payload.get("continuation_checkpoint")
        if checkpoint is None:
            return valid and continuation is None
        return valid and (
            isinstance(continuation, dict)
            and continuation.get("status") == "accepted"
            and continuation.get("local_step") == checkpoint["local_step"]
            and continuation.get("sha256") == checkpoint["sha256"]
        )
    except Exception:
        return False


def validate_external_baseline(
    baseline_root: Path,
    evaluation_root: Path,
) -> dict[str, Any]:
    """Validate and identify a frozen baseline without modifying it."""

    artifacts = []
    for language, label in CLASSES:
        output = baseline_root / f"{language}-{label}.json"
        if not output_valid(
            output, evaluation_root, language, label, checkpoint=None
        ):
            raise RuntimeError(f"external baseline failed identity check: {output}")
        artifacts.append(
            {
                "path": str(output),
                "sha256": sha256_file(output),
                "bytes": output.stat().st_size,
            }
        )
    gate = baseline_root / "table3-gate.json"
    gate_payload = load_json(gate)
    if gate_payload.get("evaluation_mode") != "baseline":
        raise RuntimeError(f"external baseline gate mode mismatch: {gate}")
    artifacts.append(
        {"path": str(gate), "sha256": sha256_file(gate), "bytes": gate.stat().st_size}
    )
    return {
        "mode": "external_read_only",
        "root": str(baseline_root),
        "artifacts": artifacts,
    }


def archive_partial(root: Path, prefix: str) -> Path | None:
    paths = [
        root / f"{prefix}.json",
        root / f"{prefix}-asr.jsonl",
        root / f"{prefix}-process.log",
        root / f"{prefix}-traces",
    ]
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    archive = root / f"interrupted-retry-{utc_now().replace(':', '').replace('+00:00', 'Z')}"
    archive.mkdir(parents=True, exist_ok=False)
    for path in existing:
        shutil.move(str(path), archive / path.name)
    return archive


def run_class(
    python: Path,
    evaluation_root: Path,
    official_root: Path,
    output_root: Path,
    run_id: str,
    language: str,
    label: str,
    checkpoint: dict[str, Any] | None,
    environment: dict[str, str],
) -> str:
    prefix = f"{language}-{label}"
    output = output_root / f"{prefix}.json"
    if output_valid(output, evaluation_root, language, label, checkpoint):
        print(f"{utc_now()} class_already_complete root={output_root.name} class={prefix}", flush=True)
        return "already_complete"

    archive = archive_partial(output_root, prefix)
    if archive is not None:
        print(f"{utc_now()} partial_archived class={prefix} archive={archive}", flush=True)

    settings = CLASS_CONFIG[language]
    command = [
        str(python),
        str(evaluation_root / "scripts/run_table3_reproduction.py"),
        "--language", language,
        "--label", label,
        "--dataset-root", str(settings["dataset"]),
        "--official-root", str(official_root),
        "--config", str(settings["config"]),
        "--asr-model-dir", str(settings["asr"]),
        "--trace-dir", str(output_root / f"{prefix}-traces"),
        "--asr-cache", str(output_root / f"{prefix}-asr.jsonl"),
        "--output", str(output),
        "--run-id", run_id,
    ]
    if checkpoint is not None:
        command.extend(["--continuation-checkpoint", checkpoint["path"]])
    process_log = output_root / f"{prefix}-process.log"
    print(f"{utc_now()} class_started root={output_root.name} class={prefix}", flush=True)
    with process_log.open("x", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=evaluation_root,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if not output_valid(output, evaluation_root, language, label, checkpoint):
        raise RuntimeError(f"completed class failed identity check: {output}")
    print(f"{utc_now()} class_complete root={output_root.name} class={prefix}", flush=True)
    return "complete"


def run_gate(
    python: Path,
    evaluation_root: Path,
    result_root: Path,
    mode: str,
    environment: dict[str, str],
) -> dict[str, Any]:
    gate = result_root / "table3-gate.json"
    if gate.exists():
        payload = load_json(gate)
        if payload.get("evaluation_mode") == mode:
            return payload
        raise RuntimeError(f"existing gate mode mismatch: {gate}")
    command = [
        str(python),
        str(evaluation_root / "scripts/check_table3_reproduction_gate.py"),
        "--evaluation-mode", mode,
        "--en-complete", str(result_root / "en-complete.json"),
        "--en-incomplete", str(result_root / "en-incomplete.json"),
        "--zh-complete", str(result_root / "zh-complete.json"),
        "--zh-incomplete", str(result_root / "zh-incomplete.json"),
        "--output", str(gate),
    ]
    result = subprocess.run(command, cwd=evaluation_root, env=environment)
    # A baseline evidence gate may return 2 solely because its accuracy differs
    # from the paper target.  The evidence artifact must still be retained.
    allowed = {0, 2} if mode == "baseline" else {0}
    if result.returncode not in allowed or not gate.exists():
        raise RuntimeError(f"Table 3 {mode} gate failed with rc={result.returncode}")
    return load_json(gate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument(
        "--expected-step",
        type=int,
        action="append",
        required=True,
        help="Predeclared checkpoint grid; repeat once per expected local step.",
    )
    parser.add_argument(
        "--baseline-source-root",
        type=Path,
        help="Reuse this frozen, identity-checked baseline read-only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    python = args.python.resolve(strict=True)
    evaluation_root = args.evaluation_root.resolve(strict=True)
    official_root = args.official_root.resolve(strict=True)
    output_root = args.output_root.resolve()
    generated_baseline_root = output_root / "baseline-step000000"
    sweep_root = output_root / "checkpoints"
    manifest_path = output_root / "orchestration_manifest.json"

    commit = git_output(evaluation_root, "rev-parse", "HEAD")
    dirty = git_output(evaluation_root, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise RuntimeError(f"evaluation runtime must be clean: {evaluation_root}")
    if git_output(official_root, "status", "--porcelain"):
        raise RuntimeError(f"official upstream must be clean: {official_root}")

    expected_steps = sorted(args.expected_step)
    if (
        not expected_steps
        or any(step <= 0 for step in expected_steps)
        or len(expected_steps) != len(set(expected_steps))
    ):
        raise RuntimeError("expected steps must be unique positive integers")

    checkpoints = []
    for value in args.checkpoint:
        path = value.resolve(strict=True)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        step = payload.get("local_step")
        if not isinstance(step, int) or step <= 0:
            raise RuntimeError(f"invalid checkpoint local step: {path}")
        checkpoints.append({"local_step": step, "path": str(path), "sha256": sha256_file(path)})
    checkpoints.sort(key=lambda item: item["local_step"])
    if [item["local_step"] for item in checkpoints] != expected_steps:
        raise RuntimeError(
            "checkpoint steps differ from the predeclared expected grid: "
            f"actual={[item['local_step'] for item in checkpoints]} "
            f"expected={expected_steps}"
        )

    if args.baseline_source_root is None:
        baseline_root = generated_baseline_root
        baseline_identity = {
            "mode": "generated_in_output_root",
            "root": str(baseline_root),
        }
    else:
        baseline_root = args.baseline_source_root.resolve(strict=True)
        if baseline_root == generated_baseline_root:
            raise RuntimeError("external baseline must be outside the new output root")
        baseline_identity = validate_external_baseline(
            baseline_root, evaluation_root
        )

    identity = {
        "evaluation_commit": commit,
        "evaluation_root": str(evaluation_root),
        "official_commit": git_output(official_root, "rev-parse", "HEAD"),
        "official_root": str(official_root),
        "expected_steps": expected_steps,
        "checkpoints": checkpoints,
        "baseline": baseline_identity,
    }
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if manifest.get("identity") != identity:
            raise RuntimeError("existing orchestration manifest identity mismatch")
    else:
        output_root.mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema_version": 1,
            "status": "running",
            "started_at_utc": utc_now(),
            "identity": identity,
            "completed_classes": [],
        }
        atomic_json_write(manifest_path, manifest)
    if args.baseline_source_root is None:
        baseline_root.mkdir(parents=True, exist_ok=True)
    sweep_root.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment.update(
        {
            "MODELSCOPE_CACHE": "/root/SoulX-stage3-dataset/pretrained_models/modelscope_cache",
            "HF_HOME": "/root/autodl-tmp/cache/huggingface",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )

    if args.baseline_source_root is None:
        for language, label in CLASSES:
            run_class(
                python, evaluation_root, official_root, baseline_root,
                f"official-current-code-step000000-{commit[:8]}", language, label,
                None, environment,
            )
            marker = f"step000000/{language}/{label}"
            if marker not in manifest["completed_classes"]:
                manifest["completed_classes"].append(marker)
                manifest["updated_at_utc"] = utc_now()
                atomic_json_write(manifest_path, manifest)
        run_gate(python, evaluation_root, baseline_root, "baseline", environment)
    else:
        manifest["baseline_reused_read_only"] = baseline_identity
        manifest["updated_at_utc"] = utc_now()
        atomic_json_write(manifest_path, manifest)

    for checkpoint in checkpoints:
        step = checkpoint["local_step"]
        step_root = sweep_root / f"step{step:06d}"
        step_root.mkdir(parents=True, exist_ok=True)
        run_id = f"official-continual-step{step:06d}-{checkpoint['sha256'][:8]}"
        for language, label in CLASSES:
            run_class(
                python, evaluation_root, official_root, step_root, run_id,
                language, label, checkpoint, environment,
            )
            marker = f"step{step:06d}/{language}/{label}"
            if marker not in manifest["completed_classes"]:
                manifest["completed_classes"].append(marker)
                manifest["updated_at_utc"] = utc_now()
                atomic_json_write(manifest_path, manifest)
        run_gate(python, evaluation_root, step_root, "continuation", environment)
        subprocess.run(
            [
                str(python),
                str(evaluation_root / "scripts/build_continual_table3_index.py"),
                "--baseline-root", str(baseline_root),
                "--sweep-root", str(sweep_root),
                "--output", str(output_root / "sweep_index.json"),
            ],
            cwd=evaluation_root,
            env=environment,
            check=True,
        )
        print(f"{utc_now()} checkpoint_complete step={step}", flush=True)

    manifest["status"] = "complete"
    manifest["completed_at_utc"] = utc_now()
    atomic_json_write(manifest_path, manifest)
    print(f"{utc_now()} official_table3_sweep_complete root={output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
