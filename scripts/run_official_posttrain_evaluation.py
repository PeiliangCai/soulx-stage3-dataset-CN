#!/usr/bin/env python3
"""Fail-closed coordinator for post-training internal validation and Table 3."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback
from typing import Any

import torch


EXPECTED_BASE_SHA256 = (
    "b0703dea0b1dbb1cd51e6e7b6514c60907ea4d4b6752cecc3f71cb6445650dbe"
)
EXPECTED_UPSTREAM_COMMIT = "928b06508ed2de1344208d06fb1f6fb2ebfb1df5"
EXPECTED_EVALUATION_COMMIT = "b17bcf903b9bd896238a4ee9fec495fd75df1401"


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


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def validate_result(
    path: Path,
    expected_step: int,
    expected_split_identity: str,
    expected_checkpoint_sha256: str | None,
) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("status") != "complete" or payload.get("local_step") != expected_step:
        raise RuntimeError(f"invalid group-validation result: {path}")
    if payload.get("base_checkpoint", {}).get("sha256") != EXPECTED_BASE_SHA256:
        raise RuntimeError(f"group-validation base identity drift: {path}")
    if (
        payload.get("split_manifest", {}).get("split_identity_sha256")
        != expected_split_identity
    ):
        raise RuntimeError(f"group-validation split identity drift: {path}")
    if payload.get("validation_data", {}).get("row_count") != 2061:
        raise RuntimeError(f"group-validation row-count drift: {path}")
    metrics = payload.get("metrics", {})
    exact = payload.get("exact_token_weighted_metrics", {})
    if (
        not metrics
        or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in metrics.values())
        or exact.get("profile") != "exact-per-row-target-weighted-v1"
        or not math.isfinite(exact.get("objective", float("nan")))
    ):
        raise RuntimeError(f"group-validation metrics are incomplete/non-finite: {path}")
    continuation = payload.get("continuation_checkpoint")
    if expected_step == 0:
        if continuation is not None:
            raise RuntimeError("step-0 validation unexpectedly used an overlay")
    elif (
        not isinstance(continuation, dict)
        or continuation.get("status") != "accepted"
        or continuation.get("local_step") != expected_step
        or continuation.get("sha256") != expected_checkpoint_sha256
    ):
        raise RuntimeError(f"group-validation overlay identity mismatch: {path}")
    return payload


def verify_training(
    training_dir: Path, expected_steps: list[int], expected_split_identity: str
) -> tuple[dict[str, Any], list[Path], dict[str, Any]]:
    manifest_path = training_dir / "run_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("status") != "complete":
        raise RuntimeError("training manifest is not complete")
    if manifest.get("runtime_base_commit") != EXPECTED_UPSTREAM_COMMIT:
        raise RuntimeError("training runtime base commit drift")
    if manifest.get("base_checkpoint", {}).get("sha256") != EXPECTED_BASE_SHA256:
        raise RuntimeError("training base checkpoint identity drift")
    if manifest.get("split", {}).get("split_identity_sha256") != expected_split_identity:
        raise RuntimeError("training split identity drift")
    if manifest.get("final_local_step") != expected_steps[-1]:
        raise RuntimeError("training final step mismatch")
    if manifest.get("checkpoint_steps") != expected_steps:
        raise RuntimeError("training checkpoint grid mismatch")
    updates = manifest.get("updates", [])
    if [row.get("local_step") for row in updates] != list(
        range(1, expected_steps[-1] + 1)
    ):
        raise RuntimeError("training optimizer update sequence is incomplete")
    if any(row.get("microbatches") != 576 or row.get("samples") != 576 for row in updates):
        raise RuntimeError("training effective-batch audit drift")
    if manifest.get("total_samples") != expected_steps[-1] * 576:
        raise RuntimeError("training sample-exposure audit drift")

    checkpoint_paths = []
    checkpoint_audits = []
    for step in expected_steps:
        item = manifest.get("checkpoints", {}).get(str(step))
        if not isinstance(item, dict):
            raise RuntimeError(f"training checkpoint missing at step {step}")
        path = Path(item["path"]).resolve(strict=True)
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"training checkpoint SHA-256 drift at step {step}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        state = payload.get("trainable_state_dict")
        if (
            payload.get("checkpoint_profile") != "soulx-stage3-compact-evaluation-v2"
            or payload.get("local_step") != step
            or payload.get("sample_exposure") != step * 576
            or payload.get("split_identity_sha256") != expected_split_identity
            or payload.get("official_base_checkpoint_sha256") != EXPECTED_BASE_SHA256
            or not isinstance(state, dict)
            or len(state) != manifest.get("trainable_tensor_count")
            or sum(tensor.numel() for tensor in state.values())
            != manifest.get("trainable_parameter_count")
            or any(not torch.isfinite(tensor).all() for tensor in state.values())
        ):
            raise RuntimeError(f"compact checkpoint tensor audit failed at step {step}")
        checkpoint_paths.append(path)
        checkpoint_audits.append(
            {
                "local_step": step,
                "path": str(path),
                "sha256": item["sha256"],
                "bytes": path.stat().st_size,
                "tensor_audit_passed": True,
            }
        )
        del payload, state

    last = (training_dir / "last.ckpt").resolve(strict=True)
    full_checkpoint = {
        "path": str(last),
        "sha256": sha256_file(last),
        "bytes": last.stat().st_size,
    }
    return manifest, checkpoint_paths, {
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
        },
        "compact_checkpoints": checkpoint_audits,
        "full_lightning_checkpoint": full_checkpoint,
    }


def run_group_validation(
    *,
    project_root: Path,
    training_python: Path,
    runtime_root: Path,
    config: Path,
    output: Path,
    log: Path,
    environment: dict[str, str],
    checkpoint: Path | None,
) -> None:
    command = [
        str(training_python),
        str(project_root / "scripts/run_official_group_validation.py"),
        "--runtime-root",
        str(runtime_root),
        "--config",
        str(config),
        "--output",
        str(output),
    ]
    if checkpoint is not None:
        command.extend(["--continuation-checkpoint", str(checkpoint)])
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{utc_now()} attempt_started command={json.dumps(command)}\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--training-python", type=Path, required=True)
    parser.add_argument("--training-runtime", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--evaluation-python", type=Path, required=True)
    parser.add_argument("--evaluation-runtime", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--table3-baseline-root", type=Path, required=True)
    parser.add_argument("--table3-output-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, action="append", required=True)
    parser.add_argument("--expected-split-identity", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve(strict=True)
    training_dir = args.training_dir.resolve(strict=True)
    training_python = args.training_python.resolve(strict=True)
    training_runtime = args.training_runtime.resolve(strict=True)
    training_config = args.training_config.resolve(strict=True)
    evaluation_python = args.evaluation_python.resolve(strict=True)
    evaluation_runtime = args.evaluation_runtime.resolve(strict=True)
    official_root = args.official_root.resolve(strict=True)
    table3_baseline_root = args.table3_baseline_root.resolve(strict=True)
    table3_output_root = args.table3_output_root.absolute()
    output_root = args.output_root.absolute()
    expected_steps = sorted(args.expected_step)
    if expected_steps != [1, 2, 3, 5, 10, 20, 30]:
        raise RuntimeError("this frozen experiment requires steps [1,2,3,5,10,20,30]")
    if args.poll_seconds < 5 or args.poll_seconds > 60:
        raise ValueError("poll interval must be between 5 and 60 seconds")
    if git_output(evaluation_runtime, "rev-parse", "HEAD") != EXPECTED_EVALUATION_COMMIT:
        raise RuntimeError("Table 3 evaluation commit drift")
    if git_output(evaluation_runtime, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Table 3 evaluation runtime is dirty")
    if git_output(official_root, "rev-parse", "HEAD") != EXPECTED_UPSTREAM_COMMIT:
        raise RuntimeError("official upstream commit drift")
    if git_output(official_root, "status", "--porcelain"):
        raise RuntimeError("official upstream is dirty")

    manifest_path = output_root / "orchestration_manifest.json"
    identity = {
        "project_root": str(project_root),
        "training_dir": str(training_dir),
        "training_python": str(training_python),
        "training_runtime": str(training_runtime),
        "training_config": {
            "path": str(training_config),
            "sha256": sha256_file(training_config),
        },
        "evaluation_python": str(evaluation_python),
        "evaluation_runtime": str(evaluation_runtime),
        "evaluation_commit": EXPECTED_EVALUATION_COMMIT,
        "official_root": str(official_root),
        "official_commit": EXPECTED_UPSTREAM_COMMIT,
        "table3_baseline_root": str(table3_baseline_root),
        "table3_output_root": str(table3_output_root),
        "expected_steps": expected_steps,
        "expected_split_identity": args.expected_split_identity,
    }
    if manifest_path.exists():
        orchestration = load_json(manifest_path)
        if orchestration.get("identity") != identity:
            raise RuntimeError("existing post-training orchestration identity mismatch")
        orchestration["resume_count"] = int(orchestration.get("resume_count", 0)) + 1
    else:
        if output_root.exists():
            raise FileExistsError(f"output root exists without manifest: {output_root}")
        output_root.mkdir(parents=True)
        orchestration = {
            "schema_version": 1,
            "status": "running",
            "stage": "waiting_for_training",
            "started_at_utc": utc_now(),
            "resume_count": 0,
            "identity": identity,
            "group_validation_results": {},
        }

    def update(stage: str, **values: Any) -> None:
        orchestration.update(values)
        orchestration["status"] = "running"
        orchestration["stage"] = stage
        orchestration["updated_at_utc"] = utc_now()
        atomic_json_write(manifest_path, orchestration)

    try:
        update("waiting_for_training")
        training_manifest_path = training_dir / "run_manifest.json"
        while True:
            if training_manifest_path.is_file():
                current = load_json(training_manifest_path)
                status = current.get("status")
                if status == "failed":
                    raise RuntimeError("formal training reported failure")
                if status == "complete":
                    break
            time.sleep(args.poll_seconds)

        # The audit callback marks fit completion close to Lightning's native
        # checkpoint finalization.  Wait for a stable full checkpoint before
        # hashing rather than racing the writer.
        last_checkpoint = training_dir / "last.ckpt"
        previous_size = None
        stable_observations = 0
        for _ in range(max(2, 600 // args.poll_seconds)):
            if last_checkpoint.is_file():
                current_size = last_checkpoint.stat().st_size
                if current_size > 0 and current_size == previous_size:
                    stable_observations += 1
                else:
                    stable_observations = 0
                previous_size = current_size
                if stable_observations >= 1:
                    break
            time.sleep(args.poll_seconds)
        else:
            raise RuntimeError("full Lightning last.ckpt did not become stable within 10 minutes")

        update("auditing_training")
        training, checkpoints, training_audit = verify_training(
            training_dir, expected_steps, args.expected_split_identity
        )
        update("running_exact_group_validation", training_audit=training_audit)

        group_root = output_root / "group_validation"
        group_root.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        cache_root = "/root/autodl-tmp/cache/huggingface_continual_edu0001_0045"
        environment.update(
            {
                "HF_HOME": cache_root,
                "HF_DATASETS_CACHE": f"{cache_root}/datasets",
                "XDG_CACHE_HOME": f"{cache_root}/xdg",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "WANDB_MODE": "offline",
                "PYTHONDONTWRITEBYTECODE": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        results: list[Path] = []
        for step, checkpoint in [(0, None), *zip(expected_steps, checkpoints)]:
            output = group_root / f"step{step:06d}.json"
            log = group_root / f"step{step:06d}.log"
            checkpoint_sha = (
                training["checkpoints"][str(step)]["sha256"] if step else None
            )
            if not output.exists():
                run_group_validation(
                    project_root=project_root,
                    training_python=training_python,
                    runtime_root=training_runtime,
                    config=training_config,
                    output=output,
                    log=log,
                    environment=environment,
                    checkpoint=checkpoint,
                )
            validate_result(
                output,
                step,
                args.expected_split_identity,
                checkpoint_sha,
            )
            orchestration["group_validation_results"][str(step)] = {
                "path": str(output),
                "sha256": sha256_file(output),
                "status": "complete",
            }
            update("running_exact_group_validation")
            results.append(output)

        validation_index = group_root / "index.json"
        if not validation_index.exists():
            command = [
                str(training_python),
                str(project_root / "scripts/build_official_group_validation_index.py"),
                "--baseline",
                str(results[0]),
                "--output",
                str(validation_index),
            ]
            for step, result in zip(expected_steps, results[1:]):
                command.extend(["--expected-step", str(step)])
                command.extend(["--checkpoint-result", str(result)])
            subprocess.run(command, cwd=project_root, env=environment, check=True)
        validation_payload = load_json(validation_index)
        if (
            validation_payload.get("status") != "complete"
            or validation_payload.get("expected_steps") != expected_steps
            or validation_payload.get("table3_used_for_selection") is not False
        ):
            raise RuntimeError("internal validation index failed its frozen audit")
        update(
            "running_table3",
            group_validation_index={
                "path": str(validation_index),
                "sha256": sha256_file(validation_index),
                "selected_local_step": validation_payload["selection"][
                    "selected_local_step"
                ],
            },
        )

        command = [
            str(training_python),
            str(project_root / "scripts/run_official_table3_sweep.py"),
            "--python",
            str(evaluation_python),
            "--evaluation-root",
            str(evaluation_runtime),
            "--official-root",
            str(official_root),
            "--output-root",
            str(table3_output_root),
            "--baseline-source-root",
            str(table3_baseline_root),
        ]
        for step, checkpoint in zip(expected_steps, checkpoints):
            command.extend(["--expected-step", str(step)])
            command.extend(["--checkpoint", str(checkpoint)])
        table3_log = output_root / "table3_orchestration.log"
        with table3_log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n{utc_now()} attempt_started command={json.dumps(command)}\n")
            handle.flush()
            subprocess.run(
                command,
                cwd=project_root,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        table3_manifest = table3_output_root / "orchestration_manifest.json"
        table3_index = table3_output_root / "sweep_index.json"
        if load_json(table3_manifest).get("status") != "complete":
            raise RuntimeError("Table 3 orchestration did not complete")
        table3_payload = load_json(table3_index)
        if [row["local_step"] for row in table3_payload.get("checkpoints", [])] != expected_steps:
            raise RuntimeError("Table 3 sweep index step grid mismatch")

        orchestration["status"] = "complete"
        orchestration["stage"] = "evaluation_complete_report_pending"
        orchestration["completed_at_utc"] = utc_now()
        orchestration["table3"] = {
            "manifest": {
                "path": str(table3_manifest),
                "sha256": sha256_file(table3_manifest),
            },
            "index": {"path": str(table3_index), "sha256": sha256_file(table3_index)},
            "log": str(table3_log),
        }
        atomic_json_write(manifest_path, orchestration)
        print(json.dumps(orchestration, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except BaseException as exception:
        orchestration["status"] = "failed"
        orchestration["failed_at_utc"] = utc_now()
        orchestration["exception"] = repr(exception)
        orchestration["traceback"] = traceback.format_exc()
        atomic_json_write(manifest_path, orchestration)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
