#!/usr/bin/env python3
"""Recover Edu_0015 on the verified direct route, then queue Edu_0016-Edu_0017.

This is orchestration only.  It delegates every dataset stage and every frozen
gate to the existing shard and queue runners, preserves the failed Edu_0015
run, and stops before later shards on any anomaly.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_frozen_expansion_shard_queue as queue_runner  # noqa: E402


PYTHON = Path("/root/autodl-tmp/conda_envs/soulx-duplug-official/bin/python")
DATA_ROOT = Path("/root/autodl-tmp/dataset/duplexconv")
REPORT_ROOT = DATA_ROOT / "reports/expansion_v1"
SHARD_RUNNER = SCRIPTS_DIR / "run_frozen_expansion_shard_full_pipeline.py"
QUEUE_RUNNER = SCRIPTS_DIR / "run_frozen_expansion_shard_queue.py"
ROUTE = "direct-no-proxy-v2"
RUN_VARIANT = "network_retry"

SUPERVISOR_MANIFEST = REPORT_ROOT / "edu0015_0017_network_recovery_supervisor_v6.json"
EDU15_PIPELINE = (
    REPORT_ROOT / "Edu_0015/full_pipeline_manifest_network_retry_v1.json"
)
QUEUE_MANIFEST = REPORT_ROOT / "edu0016_0017_queue_manifest_v6.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_state(state: dict[str, Any]) -> None:
    state["updated_at_utc"] = utc_now()
    queue_runner.atomic_json(SUPERVISOR_MANIFEST, state)


def run(command: list[str]) -> int:
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return completed.returncode


def main() -> int:
    if SUPERVISOR_MANIFEST.exists():
        raise FileExistsError(f"refusing to overwrite {SUPERVISOR_MANIFEST}")
    for path in (EDU15_PIPELINE, QUEUE_MANIFEST):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    state: dict[str, Any] = {
        "schema_version": 1,
        "profile": "edu0015-network-recovery-then-edu0016-0017-queue-v6",
        "status": "running",
        "stage": "edu0015_network_retry",
        "started_at_utc": utc_now(),
        "network_diagnosis": {
            "selected_route": ROUTE,
            "direct_key_preflight_http_status": 200,
            "direct_key_preflight_latency_seconds": 0.769,
            "daily_limit_usd": 10.0,
            "usage_daily_usd_at_preflight": 0.0,
            "reason": "Direct route passed the no-inference-cost key/budget preflight; the prior environment-proxy run failed before inference with connection refused.",
        },
        "approved_scope": [
            "Run a fresh Edu_0015 network_retry variant through the unchanged frozen pipeline.",
            "After exact passed-Gate-D cleanup, run Edu_0016 and Edu_0017 strictly sequentially.",
        ],
        "forbidden_actions": [
            "training",
            "checkpoint evaluation",
            "benchmark rule or threshold changes",
            "state-label model or prompt changes",
            "automatic source exclusion or sanitization",
            "overwriting the failed Edu_0015 evidence",
        ],
        "resource_policy": {
            "qwen_route": ROUTE,
            "qwen_model": "qwen/qwen3-235b-a22b-2507",
            "qwen_workers": 4,
            "daily_api_cap_usd": 10.0,
            "qwen_shards_concurrent": 1,
            "gpu_tasks_concurrent": 1,
        },
        "edu0015_pipeline_manifest": str(EDU15_PIPELINE),
        "edu0016_0017_queue_manifest": str(QUEUE_MANIFEST),
    }
    write_state(state)

    try:
        edu15_command = [
            str(PYTHON),
            str(SHARD_RUNNER),
            "--contract",
            str(PROJECT_ROOT / "configs/duplexconv_edu0015_source_contract.json"),
            "--download-manifest",
            str(REPORT_ROOT / "Edu_0015/download_manifest.json"),
            "--run-variant",
            RUN_VARIANT,
            "--qwen-route",
            ROUTE,
        ]
        state["current_command"] = edu15_command
        write_state(state)
        returncode = run(edu15_command)
        if returncode != 0:
            raise RuntimeError(f"Edu_0015 recovery runner returned {returncode}")

        pipeline = queue_runner.load_object(EDU15_PIPELINE)
        result = pipeline.get("result") or {}
        if pipeline.get("status") != "complete" or result.get("gate_d_passed") is not True:
            raise RuntimeError("Edu_0015 recovery did not complete with passed Gate D")
        closure_path = Path(result["gate_d_closure"])
        if queue_runner.sha256_file(closure_path) != result["gate_d_closure_sha256"]:
            raise RuntimeError("Edu_0015 recovery Gate D closure hash differs")

        state["stage"] = "edu0015_exact_candidate_cleanup"
        state["current_command"] = None
        write_state(state)
        closure = queue_runner.load_object(closure_path)
        candidate_tar = (
            DATA_ROOT
            / "work/gate_d_edu0015_network_retry_v1/candidate_model_ready_views.tar"
        )
        required_inputs = [
            DATA_ROOT / "model_ready/edu0015_network_retry_stage3_zh_v1",
            DATA_ROOT / "processed/target_audio_edu0015_network_retry_v1",
            DATA_ROOT / "work/gate_d_edu0015_network_retry_v1/selection.jsonl",
            closure_path,
        ]
        cleanup = queue_runner.audit_candidate_tar_cleanup(
            candidate_tar, closure, required_inputs=required_inputs
        )
        cleanup["free_bytes_before"] = shutil.disk_usage(DATA_ROOT).free
        cleanup["predelete_audited_at_utc"] = utc_now()
        state["edu0015_rolling_cleanup"] = cleanup
        write_state(state)
        candidate_tar.unlink()
        if candidate_tar.exists():
            raise RuntimeError("Edu_0015 candidate tar remains after exact unlink")
        cleanup["status"] = "complete"
        cleanup["deleted_at_utc"] = utc_now()
        cleanup["free_bytes_after"] = shutil.disk_usage(DATA_ROOT).free
        cleanup["free_bytes_increase"] = (
            cleanup["free_bytes_after"] - cleanup["free_bytes_before"]
        )
        cleanup["candidate_tar_absent_after"] = True
        state["edu0015_result"] = result
        write_state(state)

        state["stage"] = "edu0016_0017_strict_queue"
        queue_command = [
            str(PYTHON),
            str(QUEUE_RUNNER),
            "--contract",
            str(PROJECT_ROOT / "configs/duplexconv_edu0016_source_contract.json"),
            "--download-manifest",
            str(REPORT_ROOT / "Edu_0016/download_manifest.json"),
            "--contract",
            str(PROJECT_ROOT / "configs/duplexconv_edu0017_source_contract.json"),
            "--download-manifest",
            str(REPORT_ROOT / "Edu_0017/download_manifest.json"),
            "--queue-manifest",
            str(QUEUE_MANIFEST),
            "--qwen-route",
            ROUTE,
            "--delete-passed-gate-d-candidate-tar",
        ]
        state["current_command"] = queue_command
        write_state(state)
        returncode = run(queue_command)
        if returncode != 0:
            raise RuntimeError(f"Edu_0016-Edu_0017 queue returned {returncode}")
        queue_state = queue_runner.load_object(QUEUE_MANIFEST)
        if queue_state.get("status") != "complete_all_gate_d_passed":
            raise RuntimeError("Edu_0016-Edu_0017 queue lacks complete status")

        state["status"] = "complete"
        state["stage"] = "all_three_final_gate_d_complete"
        state["current_command"] = None
        state["completed_at_utc"] = utc_now()
        state["queue_manifest_sha256"] = queue_runner.sha256_file(QUEUE_MANIFEST)
        write_state(state)
        return 0
    except BaseException as exc:
        state["status"] = "failed_stopped_before_next_stage"
        state["failed_at_utc"] = utc_now()
        state["error_type"] = type(exc).__name__
        state["error"] = str(exc)
        state["current_command"] = None
        write_state(state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
