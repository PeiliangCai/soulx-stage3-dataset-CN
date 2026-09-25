#!/usr/bin/env python3
"""Run the frozen B/C/D Table 3 step-5/10 comparison with a GPU pool."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from typing import Any, Callable

import torch


SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

from run_official_table3_coarse_to_fine import resolve_devices  # noqa: E402
from run_official_table3_sweep import (  # noqa: E402
    CLASSES,
    atomic_json_write,
    git_output,
    load_json,
    run_class,
    run_gate,
    sha256_file,
)


EXPECTED_GROUPS = ("B", "C", "D")
EXPECTED_STEPS = (5, 10)
LEGACY_RESUME_CONTROLLER_SHA256 = (
    "a046fa967d340a2b4edb954d4f5c842e5685c1f19670ac01d9b08bfa33133523"
)
PROXY_ENVIRONMENT_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


class BaselineMismatchError(RuntimeError):
    """The fresh baseline differs from the frozen A result."""


class ParallelGroupFailure(RuntimeError):
    """A GPU worker failed; preserve completed peer results for audit."""

    def __init__(self, failures: list[dict[str, Any]], results: dict[str, Any]):
        self.failures = failures
        self.results = results
        super().__init__(
            "parallel Table 3 phase failed: "
            + json.dumps(failures, ensure_ascii=False, sort_keys=True)
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_normalize(value: Any) -> Any:
    """Return the exact value that survives a JSON write/read round trip."""

    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def validate_or_migrate_resume_identity(
    stored_identity: dict[str, Any], current_identity: dict[str, Any]
) -> dict[str, Any] | None:
    """Validate identity, allowing only the known integer-key resume bug migration."""

    stored = json_normalize(stored_identity)
    current = json_normalize(current_identity)
    if stored == current:
        return None

    legacy_view = json_normalize(current)
    controller_path = str(Path(__file__).resolve())
    matching = [
        row
        for row in legacy_view.get("controller_files", [])
        if row.get("path") == controller_path
    ]
    if len(matching) != 1:
        raise RuntimeError("existing orchestration identity mismatch")
    current_sha256 = matching[0].get("sha256")
    matching[0]["sha256"] = LEGACY_RESUME_CONTROLLER_SHA256
    if stored != legacy_view:
        raise RuntimeError("existing orchestration identity mismatch")
    return {
        "reason": "normalize JSON integer step keys after legacy resume comparison bug",
        "controller_path": controller_path,
        "from_sha256": LEGACY_RESUME_CONTROLLER_SHA256,
        "to_sha256": current_sha256,
    }


def execute_group_pool(
    groups: list[str],
    devices: list[str],
    worker: Callable[[str, str], Any],
    *,
    on_event: Callable[[str, dict[str, Any], Any, BaseException | None], None]
    | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Schedule at most one group worker per device and stop after a failure."""

    if not devices:
        raise ValueError("at least one CUDA device is required")
    if len(set(devices)) != len(devices):
        raise ValueError("CUDA device selectors must be unique")
    if len(set(groups)) != len(groups):
        raise ValueError("group names must be unique")

    pending = iter(groups)
    results: dict[str, Any] = {}
    assignments: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    phase_failed = False

    def emit(
        event: str,
        record: dict[str, Any],
        result: Any = None,
        error: BaseException | None = None,
    ) -> None:
        if on_event is not None:
            on_event(event, record, result, error)

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        active: dict[Any, dict[str, Any]] = {}

        def submit_next(device: str) -> bool:
            try:
                group = next(pending)
            except StopIteration:
                return False
            record = {
                "group": group,
                "device": device,
                "status": "running",
                "started_at_utc": utc_now(),
            }
            assignments.append(record)
            future = executor.submit(worker, group, device)
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
                            "group": record["group"],
                            "device": record["device"],
                            "exception_type": type(exception).__name__,
                            "exception": repr(exception),
                        }
                    )
                    emit("failed", record, error=exception)
                else:
                    record["status"] = "complete"
                    record["completed_at_utc"] = utc_now()
                    results[record["group"]] = result
                    emit("complete", record, result=result)

            if not phase_failed:
                for device in released_devices:
                    if not submit_next(device):
                        break

    if failures:
        raise ParallelGroupFailure(failures, results)
    return results, assignments


def validate_baseline(
    baseline_root: Path,
    expected: dict[str, dict[str, int]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    if gate.get("evaluation_mode") != "baseline":
        raise RuntimeError("fresh baseline gate mode mismatch")
    if gate.get("evidence_audit_passed") is not True:
        raise RuntimeError("fresh baseline evidence audit failed")

    observed: dict[str, dict[str, int | float]] = {}
    for language, label in CLASSES:
        key = f"{language}/{label}"
        payload = load_json(baseline_root / f"{language}-{label}.json")
        row = payload["summary"]["by_class"][key]
        actual = {"correct": int(row["correct"]), "total": int(row["total"])}
        if actual != expected.get(key):
            raise BaselineMismatchError(
                f"fresh baseline differs from frozen A baseline for {key}: "
                f"actual={actual} expected={expected.get(key)}"
            )
        observed[key] = {
            **actual,
            "accuracy": float(row["accuracy"]),
        }
    return observed


def validate_config(config: dict[str, Any]) -> None:
    if tuple(config.get("expected_steps", [])) != EXPECTED_STEPS:
        raise RuntimeError(f"expected steps must remain {list(EXPECTED_STEPS)}")
    groups = config.get("groups")
    if not isinstance(groups, dict) or tuple(groups) != EXPECTED_GROUPS:
        raise RuntimeError(f"group order must remain {list(EXPECTED_GROUPS)}")
    expected_waves = [
        ["baseline", "B5", "C5"],
        ["D5", "B10", "C10"],
        ["D10"],
    ]
    if config.get("execution_waves") != expected_waves:
        raise RuntimeError("execution wave order drift")
    if config.get("table3_used_for_training_or_checkpoint_selection") is not False:
        raise RuntimeError("Table 3 selection policy drift")
    if config.get("table2_enabled") is not False or config.get("paid_api_enabled") is not False:
        raise RuntimeError("out-of-scope execution enabled")


def checkpoint_identities(config: dict[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    result: dict[str, dict[int, dict[str, Any]]] = {}
    for group in EXPECTED_GROUPS:
        result[group] = {}
        rows = config["groups"][group].get("checkpoints", {})
        if sorted(int(step) for step in rows) != list(EXPECTED_STEPS):
            raise RuntimeError(f"checkpoint grid drift for group {group}")
        for step in EXPECTED_STEPS:
            declared = rows[str(step)]
            path = Path(declared["path"]).resolve(strict=True)
            digest = sha256_file(path)
            if digest != declared["sha256"]:
                raise RuntimeError(f"checkpoint hash drift for group {group} step {step}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload.get("local_step") != step:
                raise RuntimeError(f"checkpoint local step drift for group {group} step {step}")
            result[group][step] = {
                "path": str(path),
                "sha256": digest,
                "bytes": path.stat().st_size,
                "local_step": step,
            }
    return result


def build_group_index(
    python: Path,
    evaluation_root: Path,
    baseline_root: Path,
    group_root: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    output = group_root / "sweep_index.json"
    subprocess.run(
        [
            str(python),
            str(evaluation_root / "scripts/build_continual_table3_index.py"),
            "--baseline-root",
            str(baseline_root),
            "--sweep-root",
            str(group_root / "checkpoints"),
            "--output",
            str(output),
        ],
        cwd=evaluation_root,
        env=environment,
        check=True,
    )
    payload = load_json(output)
    if payload.get("status") != "complete":
        raise RuntimeError(f"incomplete Table 3 index for {group_root.name}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    python = args.python.resolve(strict=True)
    evaluation_root = args.evaluation_root.resolve(strict=True)
    official_root = args.official_root.resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    output_root = args.output_root.resolve()
    manifest_path = output_root / "orchestration_manifest.json"
    baseline_root = output_root / "baseline-step000000"
    config = load_json(config_path)
    validate_config(config)
    devices = resolve_devices(args.devices)

    evaluation_commit = git_output(evaluation_root, "rev-parse", "HEAD")
    if evaluation_commit != config["expected_evaluation_commit"]:
        raise RuntimeError("Table 3 evaluation commit drift")
    if git_output(evaluation_root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Table 3 evaluation runtime is dirty")
    official_commit = git_output(official_root, "rev-parse", "HEAD")
    if official_commit != config["expected_official_commit"]:
        raise RuntimeError("official upstream commit drift")
    if git_output(official_root, "status", "--porcelain"):
        raise RuntimeError("official upstream is dirty")

    checkpoints = checkpoint_identities(config)
    controller_files = [
        Path(__file__).resolve(),
        (SCRIPT_ROOT / "run_official_table3_sweep.py").resolve(strict=True),
        (SCRIPT_ROOT / "run_official_table3_coarse_to_fine.py").resolve(strict=True),
    ]
    identity = json_normalize({
        "task_id": config["task_id"],
        "protocol": config["protocol"],
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "evaluation_root": str(evaluation_root),
        "evaluation_commit": evaluation_commit,
        "official_root": str(official_root),
        "official_commit": official_commit,
        "controller_files": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in controller_files
        ],
        "fresh_baseline_expected": config["baseline"]["expected"],
        "groups": checkpoints,
        "execution_waves": config["execution_waves"],
        "table3_used_for_training_or_checkpoint_selection": False,
    })

    if manifest_path.exists():
        manifest = load_json(manifest_path)
        identity_migration = validate_or_migrate_resume_identity(
            manifest.get("identity", {}), identity
        )
        if identity_migration is not None:
            identity_migration["migrated_at_utc"] = utc_now()
            manifest.setdefault("identity_migrations", []).append(identity_migration)
            manifest["identity"] = identity
        if manifest.get("status") == "complete":
            print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        if not args.resume:
            raise RuntimeError("incomplete run exists; explicit --resume is required")
        manifest.setdefault("resume_history", []).append(
            {
                "resumed_at_utc": utc_now(),
                "previous_status": manifest.get("status"),
                "previous_stage": manifest.get("stage"),
            }
        )
    else:
        if output_root.exists():
            raise RuntimeError("output root exists without an orchestration manifest")
        output_root.mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema_version": 1,
            "status": "running",
            "stage": "preflight_complete",
            "started_at_utc": utc_now(),
            "identity": identity,
            "completed_classes": [],
            "completed_phases": [],
            "scheduler_attempts": [],
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

    attempt = {
        "profile": "baseline-plus-provisional-bc-three-wave-v1",
        "requested_devices": args.devices,
        "resolved_devices": devices,
        "maximum_parallel_group_workers": len(devices),
        "one_checkpoint_per_device": True,
        "classes_within_checkpoint": "serial",
        "baseline_device": devices[0],
        "removed_proxy_environment_keys": removed_proxy_environment_keys,
        "network_policy": "local-artifacts-offline",
        "started_at_utc": utc_now(),
        "status": "running",
        "phases": [],
    }
    manifest["scheduler_attempts"].append(attempt)

    def run_fresh_baseline(device: str) -> dict[str, Any]:
        baseline_environment = environment.copy()
        baseline_environment["CUDA_VISIBLE_DEVICES"] = device
        baseline_root.mkdir(parents=True, exist_ok=True)
        markers = []
        for language, label in CLASSES:
            run_class(
                python,
                evaluation_root,
                official_root,
                baseline_root,
                f"abcd-fresh-baseline-{evaluation_commit[:8]}",
                language,
                label,
                None,
                baseline_environment,
            )
            markers.append(f"baseline-step000000/{language}/{label}")
        gate = run_gate(
            python, evaluation_root, baseline_root, "baseline", baseline_environment
        )
        observed = validate_baseline(
            baseline_root, config["baseline"]["expected"], gate
        )
        return {
            "task": "baseline",
            "device": device,
            "markers": markers,
            "fresh_baseline": {
                "root": str(baseline_root),
                "observed": observed,
                "gate": {
                    "path": str(baseline_root / "table3-gate.json"),
                    "sha256": sha256_file(baseline_root / "table3-gate.json"),
                    "evidence_audit_passed": gate["evidence_audit_passed"],
                    "paper_accuracy_gate_passed": gate["accuracy_gate_passed"],
                },
            },
        }

    def run_checkpoint(group: str, step: int, device: str) -> dict[str, Any]:
        checkpoint = checkpoints[group][step]
        step_root = output_root / "groups" / group / "checkpoints" / f"step{step:06d}"
        step_root.mkdir(parents=True, exist_ok=True)
        worker_environment = environment.copy()
        worker_environment["CUDA_VISIBLE_DEVICES"] = device
        run_id = f"abcd-{group.lower()}-step{step:06d}-{checkpoint['sha256'][:8]}"
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
            markers.append(f"{group}/step{step:06d}/{language}/{label}")
        gate = run_gate(
            python,
            evaluation_root,
            step_root,
            "continuation",
            worker_environment,
        )
        if gate.get("evidence_audit_passed") is not True or gate.get("gate_passed") is not True:
            raise RuntimeError(f"continuation evidence gate failed for {group} step {step}")
        return {
            "task": f"{group}{step}",
            "group": group,
            "step": step,
            "device": device,
            "markers": markers,
            "gate": {
                "path": str(step_root / "table3-gate.json"),
                "sha256": sha256_file(step_root / "table3-gate.json"),
            },
        }

    def run_wave(name: str, jobs: dict[str, tuple[str, int | None]]) -> dict[str, Any]:
        phase = {
            "name": name,
            "jobs_in_logical_order": list(jobs),
            "status": "running",
            "started_at_utc": utc_now(),
            "assignments": [],
        }
        attempt["phases"].append(phase)
        update(name, current=None, active_assignments=[])

        def worker(task: str, device: str) -> dict[str, Any]:
            group, step = jobs[task]
            if group == "baseline":
                return run_fresh_baseline(device)
            if step is None:
                raise RuntimeError(f"missing checkpoint step for task {task}")
            return run_checkpoint(group, step, device)

        def on_event(
            event: str,
            record: dict[str, Any],
            result: Any,
            error: BaseException | None,
        ) -> None:
            if event == "started":
                phase["assignments"].append(record)
            elif event == "complete":
                for marker in result["markers"]:
                    if marker not in manifest["completed_classes"]:
                        manifest["completed_classes"].append(marker)
            elif event == "failed":
                phase["status"] = "failed"
                phase.setdefault("failures", []).append(
                    {"task": record["group"], "exception": repr(error)}
                )
            active = [
                {
                    "task": row["group"],
                    "device": row["device"],
                    "started_at_utc": row["started_at_utc"],
                }
                for row in phase["assignments"]
                if row["status"] == "running"
            ]
            update(name, active_assignments=active)

        results, _ = execute_group_pool(
            list(jobs), devices, worker, on_event=on_event
        )
        phase["results"] = results
        phase["status"] = "complete"
        phase["completed_at_utc"] = utc_now()
        manifest["completed_phases"].append(name)
        return results

    def refresh_indexes(groups: tuple[str, ...]) -> None:
        indexes = dict(manifest.get("group_indexes", {}))
        for group in groups:
            group_root = output_root / "groups" / group
            payload = build_group_index(
                python,
                evaluation_root,
                baseline_root,
                group_root,
                environment,
            )
            indexes[group] = {
                "path": str(group_root / "sweep_index.json"),
                "sha256": sha256_file(group_root / "sweep_index.json"),
                "evaluated_steps": [row["local_step"] for row in payload["checkpoints"]],
            }
        update("indexes_refreshed", current=None, active_assignments=[], group_indexes=indexes)

    def discard_provisional_step5() -> None:
        records = []
        for group in ("B", "C"):
            path = output_root / "groups" / group / "checkpoints" / "step000005"
            record = {"group": group, "step": 5, "path": str(path), "existed": path.exists()}
            if path.exists():
                expected_parent = (output_root / "groups" / group / "checkpoints").resolve()
                resolved = path.resolve()
                if resolved.parent != expected_parent or resolved.name != "step000005":
                    raise RuntimeError(f"refusing to discard unexpected provisional path: {resolved}")
                record["bytes_before_deletion"] = sum(
                    item.stat().st_size for item in resolved.rglob("*") if item.is_file()
                )
            records.append(record)
        manifest["provisional_discard"] = {
            "reason": "fresh baseline differed from the frozen A baseline",
            "planned_at_utc": utc_now(),
            "records": records,
        }
        atomic_json_write(manifest_path, manifest)
        for record in records:
            path = Path(record["path"])
            if path.exists():
                shutil.rmtree(path)
                record["deleted"] = True
                record["deleted_at_utc"] = utc_now()
            else:
                record["deleted"] = False
        atomic_json_write(manifest_path, manifest)

    try:
        try:
            first_wave = run_wave(
                "parallel_fresh_baseline_plus_provisional_B5_C5",
                {
                    "baseline": ("baseline", None),
                    "B5": ("B", 5),
                    "C5": ("C", 5),
                },
            )
        except ParallelGroupFailure as exception:
            baseline_mismatch = any(
                row["group"] == "baseline"
                and row["exception_type"] == "BaselineMismatchError"
                for row in exception.failures
            )
            if baseline_mismatch:
                discard_provisional_step5()
            raise

        manifest["fresh_baseline"] = first_wave["baseline"]["fresh_baseline"]
        update("fresh_baseline_verified_with_provisional_B5_C5", current=None)
        refresh_indexes(("B", "C"))

        run_wave(
            "parallel_D5_B10_C10",
            {
                "D5": ("D", 5),
                "B10": ("B", 10),
                "C10": ("C", 10),
            },
        )
        refresh_indexes(("B", "C", "D"))

        run_wave("D10", {"D10": ("D", 10)})
        refresh_indexes(("D",))

        for group, index in manifest["group_indexes"].items():
            if index["evaluated_steps"] != list(EXPECTED_STEPS):
                raise RuntimeError(f"final Table 3 grid mismatch for group {group}")

        attempt["status"] = "complete"
        attempt["completed_at_utc"] = utc_now()
        manifest["status"] = "complete"
        manifest["stage"] = "table3_step5_10_complete"
        manifest["current"] = None
        manifest["active_assignments"] = []
        manifest["completed_at_utc"] = utc_now()
        manifest["updated_at_utc"] = manifest["completed_at_utc"]
        atomic_json_write(manifest_path, manifest)
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except BaseException as exception:
        attempt["status"] = "failed"
        attempt["failed_at_utc"] = utc_now()
        manifest["status"] = "failed"
        manifest["stage"] = "failed"
        manifest["current"] = None
        manifest["active_assignments"] = []
        manifest["failed_at_utc"] = utc_now()
        manifest["exception"] = repr(exception)
        manifest["traceback"] = traceback.format_exc()
        manifest["updated_at_utc"] = manifest["failed_at_utc"]
        atomic_json_write(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
