#!/usr/bin/env python3
"""Run approved frozen shard pipelines sequentially and stop on first failure."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARD_RUNNER = PROJECT_ROOT / "scripts/run_frozen_expansion_shard_full_pipeline.py"
DATA_ROOT = Path("/root/autodl-tmp/dataset/duplexconv")
SUPPORTED_QWEN_ROUTES = (
    "direct-no-proxy-v2",
    "environment-proxy-aware-v2",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def pipeline_manifest_for(contract: dict[str, Any]) -> Path:
    number = int(contract["shard_number"])
    return (
        Path("/root/autodl-tmp/dataset/duplexconv/reports/expansion_v1")
        / f"Edu_{number:04d}/full_pipeline_manifest_v1.json"
    )


def validate_download(contract: dict[str, Any], manifest: dict[str, Any]) -> None:
    if manifest.get("status") not in {"complete", "already_complete"}:
        raise RuntimeError("download manifest is not complete")
    if int(manifest.get("output_bytes", -1)) != int(contract["official_archive_bytes"]):
        raise RuntimeError("download byte count differs from contract")
    if manifest.get("output_sha256") != contract["official_archive_sha256"]:
        raise RuntimeError("download SHA-256 differs from contract")


def candidate_tar_for(shard_number: int) -> Path:
    return (
        DATA_ROOT
        / "work"
        / f"gate_d_edu{shard_number:04d}_v1"
        / "candidate_model_ready_views.tar"
    )


def audit_candidate_tar_cleanup(
    candidate_tar: Path,
    closure: dict[str, Any],
    *,
    required_inputs: list[Path],
) -> dict[str, Any]:
    if closure.get("gate_passed") is not True:
        raise RuntimeError("refusing candidate cleanup without passed Gate D")
    if not candidate_tar.is_file():
        raise RuntimeError(f"candidate tar is absent before cleanup: {candidate_tar}")
    expected_sha = (closure.get("sha256") or {}).get("candidate_tar")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise RuntimeError("Gate D closure has no candidate tar SHA-256")
    actual_sha = sha256_file(candidate_tar)
    if actual_sha != expected_sha:
        raise RuntimeError("candidate tar SHA-256 differs from Gate D closure")
    missing_inputs = [str(path) for path in required_inputs if not path.exists()]
    if missing_inputs:
        raise RuntimeError(f"candidate rebuild inputs are absent: {missing_inputs}")
    return {
        "status": "predelete_audit_passed",
        "candidate_tar": str(candidate_tar),
        "candidate_tar_bytes": candidate_tar.stat().st_size,
        "candidate_tar_sha256": actual_sha,
        "gate_passed": True,
        "required_rebuild_inputs": [str(path) for path in required_inputs],
        "required_rebuild_inputs_present": True,
        "deletion_scope": "exactly one Gate D candidate tar; preserve its directory and all manifests/selections/reports",
        "recoverability": "deterministically rebuildable from preserved model-ready selection and target audio",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", action="append", type=Path, required=True)
    parser.add_argument("--download-manifest", action="append", type=Path, required=True)
    parser.add_argument("--queue-manifest", type=Path, required=True)
    parser.add_argument(
        "--qwen-route",
        choices=SUPPORTED_QWEN_ROUTES,
        default="direct-no-proxy-v2",
    )
    parser.add_argument(
        "--delete-passed-gate-d-candidate-tar",
        action="store_true",
        help="After exact closure/hash/rebuild-input audit, delete only the passed shard's candidate tar.",
    )
    args = parser.parse_args()
    if len(args.contract) != len(args.download_manifest):
        raise ValueError("--contract and --download-manifest counts must match")
    if args.queue_manifest.exists():
        raise FileExistsError(f"refusing to overwrite queue manifest: {args.queue_manifest}")

    entries: list[dict[str, Any]] = []
    seen: set[int] = set()
    for contract_path, download_path in zip(args.contract, args.download_manifest):
        contract_path = contract_path.resolve(strict=True)
        download_path = download_path.resolve(strict=True)
        contract = load_object(contract_path)
        number = int(contract["shard_number"])
        if number in seen:
            raise ValueError(f"duplicate shard in queue: {number}")
        seen.add(number)
        validate_download(contract, load_object(download_path))
        pipeline_manifest = pipeline_manifest_for(contract)
        if pipeline_manifest.exists():
            raise FileExistsError(
                f"refusing shard with an existing pipeline manifest: {pipeline_manifest}"
            )
        entries.append(
            {
                "shard_number": number,
                "shard": f"Edu_{number:04d}",
                "contract": str(contract_path),
                "contract_sha256": sha256_file(contract_path),
                "download_manifest": str(download_path),
                "pipeline_manifest": str(pipeline_manifest),
                "status": "pending",
            }
        )

    state: dict[str, Any] = {
        "schema_version": 1,
        "profile": "duplexconv-frozen-sequential-shard-queue-v1",
        "status": "running",
        "started_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "resource_policy": {
            "qwen_shards_concurrent": 1,
            "gpu_tasks_concurrent": 1,
            "processing_order": "strictly sequential",
            "qwen_route": args.qwen_route,
        },
        "failure_policy": (
            "Stop before starting the next shard on any nonzero runner exit, "
            "non-complete pipeline manifest, or failed final Gate D."
        ),
        "forbidden_actions": [
            "training",
            "checkpoint_evaluation",
            "benchmark_rule_changes",
            "automatic source exclusion or sanitization",
            "automatic retry or output selection",
        ],
        "rolling_cleanup_policy": {
            "enabled": args.delete_passed_gate_d_candidate_tar,
            "scope": "only the exact candidate_model_ready_views.tar of a shard after passed final Gate D and independent SHA/rebuild-input audit",
            "preserve": "raw, target audio, model-ready Parquet, API/ASR/GLM caches, selections, manifests, checksums and reports",
        },
        "entries": entries,
    }
    atomic_json(args.queue_manifest, state)

    for entry in entries:
        entry["status"] = "running"
        entry["started_at_utc"] = utc_now()
        state["current_shard"] = entry["shard"]
        state["updated_at_utc"] = utc_now()
        atomic_json(args.queue_manifest, state)
        command = [
            sys.executable,
            str(SHARD_RUNNER),
            "--contract",
            entry["contract"],
            "--download-manifest",
            entry["download_manifest"],
            "--qwen-route",
            args.qwen_route,
        ]
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if completed.returncode != 0:
            entry["status"] = "failed"
            entry["failed_at_utc"] = utc_now()
            entry["runner_returncode"] = completed.returncode
            state["status"] = "failed_stopped_before_next_shard"
            state["failed_shard"] = entry["shard"]
            state["updated_at_utc"] = utc_now()
            atomic_json(args.queue_manifest, state)
            return completed.returncode

        pipeline = load_object(Path(entry["pipeline_manifest"]))
        result = pipeline.get("result") or {}
        if pipeline.get("status") != "complete" or result.get("gate_d_passed") is not True:
            entry["status"] = "failed_postcondition"
            entry["failed_at_utc"] = utc_now()
            state["status"] = "failed_stopped_before_next_shard"
            state["failed_shard"] = entry["shard"]
            state["updated_at_utc"] = utc_now()
            atomic_json(args.queue_manifest, state)
            return 41
        closure = Path(result["gate_d_closure"])
        if sha256_file(closure) != result["gate_d_closure_sha256"]:
            entry["status"] = "failed_gate_d_closure_hash"
            state["status"] = "failed_stopped_before_next_shard"
            state["failed_shard"] = entry["shard"]
            state["updated_at_utc"] = utc_now()
            atomic_json(args.queue_manifest, state)
            return 42
        if args.delete_passed_gate_d_candidate_tar:
            closure_value = load_object(closure)
            number = int(entry["shard_number"])
            candidate_tar = candidate_tar_for(number)
            required_inputs = [
                DATA_ROOT / "model_ready" / f"edu{number:04d}_stage3_zh_v1",
                DATA_ROOT / "processed" / f"target_audio_edu{number:04d}_v1",
                DATA_ROOT
                / "work"
                / f"gate_d_edu{number:04d}_v1"
                / "selection.jsonl",
                closure,
            ]
            cleanup = audit_candidate_tar_cleanup(
                candidate_tar,
                closure_value,
                required_inputs=required_inputs,
            )
            cleanup["free_bytes_before"] = shutil.disk_usage(DATA_ROOT).free
            cleanup["predelete_audited_at_utc"] = utc_now()
            entry["rolling_cleanup"] = cleanup
            state["updated_at_utc"] = utc_now()
            atomic_json(args.queue_manifest, state)
            candidate_tar.unlink()
            if candidate_tar.exists():
                raise RuntimeError("candidate tar still exists after exact unlink")
            cleanup["status"] = "complete"
            cleanup["deleted_at_utc"] = utc_now()
            cleanup["free_bytes_after"] = shutil.disk_usage(DATA_ROOT).free
            cleanup["free_bytes_increase"] = (
                cleanup["free_bytes_after"] - cleanup["free_bytes_before"]
            )
            cleanup["candidate_tar_absent_after"] = True
            state["updated_at_utc"] = utc_now()
            atomic_json(args.queue_manifest, state)
        entry["status"] = "complete_gate_d_passed"
        entry["completed_at_utc"] = utc_now()
        entry["result"] = result
        state["updated_at_utc"] = utc_now()
        atomic_json(args.queue_manifest, state)

    state.pop("current_shard", None)
    state["status"] = "complete_all_gate_d_passed"
    state["completed_at_utc"] = utc_now()
    state["updated_at_utc"] = utc_now()
    atomic_json(args.queue_manifest, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
