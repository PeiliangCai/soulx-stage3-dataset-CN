#!/usr/bin/env python3
"""Run the preregistered endpoint-first Table 3 coarse-to-fine protocol."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

import torch


SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

from run_official_table3_sweep import (  # noqa: E402
    CLASSES,
    atomic_json_write,
    git_output,
    load_json,
    output_valid,
    run_class,
    run_gate,
    sha256_file,
    validate_external_baseline,
)


EXPECTED_ALL_STEPS = [1, 2, 3, 5, 10, 20, 30]
MANDATORY_ORDER = [1, 30, 10, 5, 20]
CONDITIONAL_REFINEMENT_ORDER = [2, 3]
EXPECTED_OFFICIAL_COMMIT = "928b06508ed2de1344208d06fb1f6fb2ebfb1df5"
EXPECTED_EVALUATION_COMMIT = "b17bcf903b9bd896238a4ee9fec495fd75df1401"
PROXY_ENVIRONMENT_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def choose_refinement_steps(index: dict[str, Any]) -> list[int]:
    """Apply the frozen refinement trigger without assuming monotonicity."""

    rows = {row.get("local_step"): row for row in index.get("checkpoints", [])}
    missing = [step for step in MANDATORY_ORDER if step not in rows]
    if missing:
        raise RuntimeError(f"mandatory coarse checkpoints missing from index: {missing}")
    return (
        list(CONDITIONAL_REFINEMENT_ORDER)
        if rows[5].get("obvious_decline_trigger") is True
        else []
    )


def resolve_devices(
    specification: str,
    *,
    environment: dict[str, str] | None = None,
    cuda_device_count: int | None = None,
) -> list[str]:
    """Resolve an explicit CUDA selector list or all currently visible GPUs."""

    value = specification.strip()
    if not value:
        raise ValueError("--devices must not be empty")
    if value.lower() == "auto":
        current_environment = os.environ if environment is None else environment
        visible = current_environment.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible and visible != "-1":
            devices = [item.strip() for item in visible.split(",") if item.strip()]
        else:
            count = torch.cuda.device_count() if cuda_device_count is None else cuda_device_count
            devices = [str(index) for index in range(count)]
    else:
        devices = [item.strip() for item in value.split(",") if item.strip()]

    if not devices:
        raise RuntimeError("no CUDA devices resolved")
    if len(set(devices)) != len(devices):
        raise ValueError(f"duplicate CUDA device selector in {devices}")
    return devices


class ParallelStepFailure(RuntimeError):
    """A step worker failed; no further pending work should be submitted."""

    def __init__(self, failures: list[dict[str, Any]], completed_steps: list[int]):
        self.failures = failures
        self.completed_steps = completed_steps
        super().__init__(
            "parallel checkpoint evaluation failed: "
            + json.dumps(failures, ensure_ascii=False, sort_keys=True)
        )


def execute_step_pool(
    steps: list[int],
    devices: list[str],
    worker,
    *,
    on_event=None,
) -> tuple[dict[int, Any], list[dict[str, Any]]]:
    """Run at most one checkpoint worker per GPU and stop scheduling after failure."""

    if not devices:
        raise ValueError("at least one CUDA device is required")
    if len(set(devices)) != len(devices):
        raise ValueError("CUDA device selectors must be unique")
    pending = iter(steps)
    results: dict[int, Any] = {}
    assignments: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    phase_failed = False

    def emit(event: str, record: dict[str, Any], result=None, error=None) -> None:
        if on_event is not None:
            on_event(event, record, result, error)

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        active = {}

        def submit_next(device: str) -> bool:
            try:
                step = next(pending)
            except StopIteration:
                return False
            record = {
                "local_step": step,
                "device": device,
                "status": "running",
                "started_at_utc": utc_now(),
            }
            assignments.append(record)
            future = executor.submit(worker, step, device)
            active[future] = record
            emit("started", record)
            return True

        for device in devices:
            if not submit_next(device):
                break

        while active:
            completed, _ = wait(active, return_when=FIRST_COMPLETED)
            released_devices = []
            for future in completed:
                record = active.pop(future)
                released_devices.append(record["device"])
                try:
                    result = future.result()
                except BaseException as exception:
                    phase_failed = True
                    record["status"] = "failed"
                    record["completed_at_utc"] = utc_now()
                    record["exception"] = repr(exception)
                    failures.append(
                        {
                            "local_step": record["local_step"],
                            "device": record["device"],
                            "exception": repr(exception),
                        }
                    )
                    emit("failed", record, error=exception)
                else:
                    record["status"] = "complete"
                    record["completed_at_utc"] = utc_now()
                    results[record["local_step"]] = result
                    emit("complete", record, result=result)

            if not phase_failed:
                for device in released_devices:
                    if not submit_next(device):
                        break

    if failures:
        raise ParallelStepFailure(failures, sorted(results))
    return results, assignments


def build_index(
    python: Path,
    evaluation_root: Path,
    baseline_root: Path,
    sweep_root: Path,
    output: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    subprocess.run(
        [
            str(python),
            str(evaluation_root / "scripts/build_continual_table3_index.py"),
            "--baseline-root",
            str(baseline_root),
            "--sweep-root",
            str(sweep_root),
            "--output",
            str(output),
        ],
        cwd=evaluation_root,
        env=environment,
        check=True,
    )
    payload = load_json(output)
    if payload.get("status") != "complete":
        raise RuntimeError("coarse-to-fine sweep index is incomplete")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-source-root", type=Path, required=True)
    parser.add_argument("--internal-validation-index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--expected-step", type=int, action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--devices",
        default="0",
        help=(
            "Comma-separated physical CUDA selectors (for example 0 or 0,1), "
            "or auto for all currently visible GPUs. Each GPU evaluates at most "
            "one checkpoint at a time."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    python = args.python.resolve(strict=True)
    evaluation_root = args.evaluation_root.resolve(strict=True)
    official_root = args.official_root.resolve(strict=True)
    output_root = args.output_root.resolve(strict=True)
    baseline_root = args.baseline_source_root.resolve(strict=True)
    internal_path = args.internal_validation_index.resolve(strict=True)
    manifest_path = args.manifest.absolute()
    sweep_root = output_root / "checkpoints"
    index_path = output_root / "sweep_index.json"
    devices = resolve_devices(args.devices)

    if sorted(args.expected_step) != EXPECTED_ALL_STEPS:
        raise RuntimeError(f"expected grid must remain {EXPECTED_ALL_STEPS}")
    if git_output(evaluation_root, "rev-parse", "HEAD") != EXPECTED_EVALUATION_COMMIT:
        raise RuntimeError("Table 3 evaluation commit drift")
    if git_output(evaluation_root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Table 3 evaluation runtime is dirty")
    if git_output(official_root, "rev-parse", "HEAD") != EXPECTED_OFFICIAL_COMMIT:
        raise RuntimeError("official upstream commit drift")
    if git_output(official_root, "status", "--porcelain"):
        raise RuntimeError("official upstream is dirty")

    internal = load_json(internal_path)
    if (
        internal.get("status") != "complete"
        or internal.get("expected_steps") != EXPECTED_ALL_STEPS
        or internal.get("table3_used_for_selection") is not False
    ):
        raise RuntimeError("frozen internal validation prerequisite failed")

    checkpoints: dict[int, dict[str, Any]] = {}
    for value in args.checkpoint:
        path = value.resolve(strict=True)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        step = payload.get("local_step")
        if step not in EXPECTED_ALL_STEPS or step in checkpoints:
            raise RuntimeError(f"invalid or duplicate compact checkpoint: {path}")
        checkpoints[step] = {
            "local_step": step,
            "path": str(path),
            "sha256": sha256_file(path),
        }
    if sorted(checkpoints) != EXPECTED_ALL_STEPS:
        raise RuntimeError("compact checkpoint grid is incomplete")

    baseline_identity = validate_external_baseline(baseline_root, evaluation_root)
    identity = {
        "profile": "table3-endpoint-first-coarse-to-fine-v1",
        "evaluation_commit": EXPECTED_EVALUATION_COMMIT,
        "evaluation_root": str(evaluation_root),
        "official_commit": EXPECTED_OFFICIAL_COMMIT,
        "official_root": str(official_root),
        "all_checkpoint_steps": EXPECTED_ALL_STEPS,
        "all_checkpoints": [checkpoints[step] for step in EXPECTED_ALL_STEPS],
        "mandatory_evaluation_order": MANDATORY_ORDER,
        "conditional_refinement_order": CONDITIONAL_REFINEMENT_ORDER,
        "refinement_trigger": "evaluate steps 2 and 3 iff step 5 obvious_decline_trigger is true",
        "non_monotonicity_assumed": False,
        "table3_used_for_checkpoint_selection": False,
        "baseline": baseline_identity,
        "internal_validation": {
            "path": str(internal_path),
            "sha256": sha256_file(internal_path),
            "selected_local_step": internal["selection"]["selected_local_step"],
        },
    }
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if manifest.get("identity") != identity:
            raise RuntimeError("existing coarse-to-fine manifest identity mismatch")
        if manifest.get("status") == "complete":
            print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        if manifest.get("status") == "failed":
            manifest.setdefault("failure_history", []).append(
                {
                    key: manifest[key]
                    for key in (
                        "failed_at_utc",
                        "exception",
                        "traceback",
                        "stage",
                        "current_local_step",
                    )
                    if key in manifest
                }
            )
            for key in ("failed_at_utc", "exception", "traceback"):
                manifest.pop(key, None)
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
    else:
        manifest = {
            "schema_version": 1,
            "status": "running",
            "stage": "adopting_step1",
            "started_at_utc": utc_now(),
            "resume_count": 0,
            "identity": identity,
            "completed_classes": [],
            "completed_steps_in_evaluation_order": [],
        }
        atomic_json_write(manifest_path, manifest)

    environment = os.environ.copy()
    removed_proxy_environment_keys = [
        key for key in PROXY_ENVIRONMENT_KEYS if key in environment
    ]
    for key in PROXY_ENVIRONMENT_KEYS:
        environment.pop(key, None)
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

    def update(stage: str, **values: Any) -> None:
        manifest.update(values)
        manifest["status"] = "running"
        manifest["stage"] = stage
        manifest["updated_at_utc"] = utc_now()
        atomic_json_write(manifest_path, manifest)

    scheduler_attempt = {
        "profile": "checkpoint-step-gpu-pool-v1",
        "requested_devices": args.devices,
        "resolved_devices": devices,
        "maximum_parallel_checkpoint_workers": len(devices),
        "one_checkpoint_per_device": True,
        "classes_within_checkpoint": "serial",
        "network_route": "direct-with-proxy-environment-cleared",
        "removed_proxy_environment_keys": removed_proxy_environment_keys,
        "started_at_utc": utc_now(),
        "status": "running",
        "phases": [],
    }
    manifest.setdefault("scheduler_attempts", []).append(scheduler_attempt)
    update(
        "preparing_parallel_mandatory_grid",
        current_local_step=None,
        active_parallel_assignments=[],
        execution_scheduler={
            "profile": scheduler_attempt["profile"],
            "requested_devices": args.devices,
            "resolved_devices": devices,
            "maximum_parallel_checkpoint_workers": len(devices),
            "one_checkpoint_per_device": True,
            "classes_within_checkpoint": "serial",
            "global_manifest_writer": "main-thread-only",
        },
    )

    def run_step_artifacts(step: int, device: str) -> dict[str, Any]:
        checkpoint = checkpoints[step]
        step_root = sweep_root / f"step{step:06d}"
        step_root.mkdir(parents=True, exist_ok=True)
        run_id = f"official-continual-step{step:06d}-{checkpoint['sha256'][:8]}"
        worker_environment = environment.copy()
        worker_environment["CUDA_VISIBLE_DEVICES"] = device
        markers = []
        for language, label in CLASSES:
            run_class(
                python,
                evaluation_root,
                official_root,
                step_root,
                run_id,
                language,
                label,
                checkpoint,
                worker_environment,
            )
            marker = f"step{step:06d}/{language}/{label}"
            markers.append(marker)
        gate = run_gate(
            python,
            evaluation_root,
            step_root,
            "continuation",
            worker_environment,
        )
        if gate.get("evidence_audit_passed") is not True or gate.get("gate_passed") is not True:
            raise RuntimeError(f"step {step} continuation evidence gate failed")
        return {
            "local_step": step,
            "device": device,
            "markers": markers,
            "gate_path": str(step_root / "table3-gate.json"),
        }

    def run_phase(name: str, steps: list[int], stage: str) -> None:
        phase = {
            "name": name,
            "planned_steps_in_logical_order": steps,
            "status": "running",
            "started_at_utc": utc_now(),
            "assignments": [],
            "completed_steps_in_wall_clock_order": [],
        }
        scheduler_attempt["phases"].append(phase)

        def on_event(event: str, record: dict[str, Any], result=None, error=None) -> None:
            if event == "started":
                phase["assignments"].append(record)
            elif event == "complete":
                phase["completed_steps_in_wall_clock_order"].append(
                    record["local_step"]
                )
                for marker in result["markers"]:
                    if marker not in manifest["completed_classes"]:
                        manifest["completed_classes"].append(marker)
            elif event == "failed":
                phase["status"] = "failed"
                phase["failed_step"] = record["local_step"]
                phase["exception"] = repr(error)
            active = [
                {
                    "local_step": assignment["local_step"],
                    "device": assignment["device"],
                    "started_at_utc": assignment["started_at_utc"],
                }
                for assignment in phase["assignments"]
                if assignment["status"] == "running"
            ]
            update(stage, active_parallel_assignments=active)

        execute_step_pool(
            steps,
            devices,
            run_step_artifacts,
            on_event=on_event,
        )
        for step in steps:
            if step not in manifest["completed_steps_in_evaluation_order"]:
                manifest["completed_steps_in_evaluation_order"].append(step)
        phase["status"] = "complete"
        phase["completed_at_utc"] = utc_now()
        update(stage, active_parallel_assignments=[])

    try:
        run_phase(
            "mandatory_coarse_grid",
            MANDATORY_ORDER,
            "running_parallel_mandatory_coarse_grid",
        )

        coarse_index = build_index(
            python,
            evaluation_root,
            baseline_root,
            sweep_root,
            index_path,
            environment,
        )

        refinement = choose_refinement_steps(coarse_index)
        update(
            "coarse_grid_complete",
            refinement_triggered=bool(refinement),
            refinement_steps=refinement,
            step5_obvious_decline_trigger=bool(refinement),
        )
        if refinement:
            run_phase(
                "conditional_early_refinement",
                refinement,
                "running_parallel_conditional_early_refinement",
            )

        final_index = build_index(
            python,
            evaluation_root,
            baseline_root,
            sweep_root,
            index_path,
            environment,
        )
        evaluated_steps = [
            row["local_step"] for row in final_index.get("checkpoints", [])
        ]
        expected_evaluated = sorted(set(MANDATORY_ORDER + refinement))
        if evaluated_steps != expected_evaluated:
            raise RuntimeError(
                f"final evaluated steps mismatch: {evaluated_steps} != {expected_evaluated}"
            )

        manifest["status"] = "complete"
        manifest["stage"] = "coarse_to_fine_table3_complete"
        manifest["completed_at_utc"] = utc_now()
        manifest["evaluated_steps"] = evaluated_steps
        manifest["omitted_steps"] = sorted(set(EXPECTED_ALL_STEPS) - set(evaluated_steps))
        manifest["sweep_index"] = {
            "path": str(index_path),
            "sha256": sha256_file(index_path),
        }
        scheduler_attempt["status"] = "complete"
        scheduler_attempt["completed_at_utc"] = utc_now()
        manifest["active_parallel_assignments"] = []
        atomic_json_write(manifest_path, manifest)
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except BaseException as exception:
        scheduler_attempt["status"] = "failed"
        scheduler_attempt["failed_at_utc"] = utc_now()
        manifest["status"] = "failed"
        manifest["stage"] = "failed"
        manifest["failed_at_utc"] = utc_now()
        manifest["exception"] = repr(exception)
        manifest["traceback"] = traceback.format_exc()
        atomic_json_write(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
