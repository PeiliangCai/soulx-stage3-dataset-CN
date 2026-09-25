#!/usr/bin/env python3
"""Fail-closed B/C/D queue around the patched official Lightning entrypoint."""

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
import traceback
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from duplexconv_stage3.official_ablation_queue import (
    EXPECTED_BASE_SHA256,
    EXPECTED_CHECKPOINT_STEPS,
    EXPECTED_EVALUATION_STEPS,
    EXPECTED_UPSTREAM_COMMIT,
    audit_variant_config,
    load_json,
    load_yaml,
    sha256_file,
    validate_group_validation_result,
    validate_training_manifest,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def runtime_identity(runtime_root: Path) -> dict[str, Any]:
    diff = subprocess.check_output(
        ["git", "-C", str(runtime_root), "diff", "--no-ext-diff"], text=True
    )
    return {
        "commit": git_output(runtime_root, "rev-parse", "HEAD"),
        "dirty_lines": git_output(
            runtime_root, "status", "--porcelain", "--untracked-files=all"
        ).splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
    }


def parse_experiment(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("experiment must use NAME=/path/config.yaml") from error
    if name not in {"B", "C", "D"}:
        raise argparse.ArgumentTypeError("experiment name must be B, C, or D")
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--reference-training-manifest", type=Path, required=True)
    parser.add_argument("--experiment", action="append", type=parse_experiment, required=True)
    parser.add_argument("--orchestration-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--tmp-root", type=Path, required=True)
    parser.add_argument("--checkpoint-link-root", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--evaluation-step", type=int, action="append", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def run_logged(command: list[str], log_path: Path, environment: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{utc_now()} command={json.dumps(command)}\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def verify_runtime_against_reference(
    runtime_root: Path, reference_manifest: dict[str, Any]
) -> dict[str, Any]:
    current = runtime_identity(runtime_root)
    reference = reference_manifest["runtime"]
    if current["commit"] != EXPECTED_UPSTREAM_COMMIT or current["commit"] != reference["commit"]:
        raise RuntimeError("training runtime commit drift")
    if current["tracked_diff_sha256"] != reference["tracked_diff_sha256"]:
        raise RuntimeError("training runtime tracked patch drift")
    if current["dirty_lines"] != reference["dirty_lines"]:
        raise RuntimeError("training runtime dirty-file identity drift")
    files = []
    for item in reference["audited_files"]:
        path = runtime_root / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"training runtime audited-file drift: {item['path']}")
        files.append({"path": item["path"], "sha256": item["sha256"]})
    return {**current, "audited_files": files}


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    project_root = args.project_root.resolve(strict=True)
    runtime_root = args.runtime_root.resolve(strict=True)
    training_python = args.python.resolve(strict=True)
    reference_config_path = args.reference_config.resolve(strict=True)
    reference_manifest_path = args.reference_training_manifest.resolve(strict=True)
    if project_root != PROJECT_ROOT:
        raise RuntimeError("project-root does not match this script's repository")
    if sorted(args.evaluation_step) != EXPECTED_EVALUATION_STEPS:
        raise RuntimeError(
            f"evaluation steps must be exactly {EXPECTED_EVALUATION_STEPS} for this approved run"
        )
    experiments = dict(args.experiment)
    if list(experiments) != ["B", "C", "D"]:
        raise RuntimeError("experiments must be supplied exactly once in B, C, D order")
    if len(experiments) != len(args.experiment):
        raise RuntimeError("duplicate experiment name")

    orchestration_root = args.orchestration_root.absolute()
    evaluation_root = args.evaluation_root.absolute()
    cache_root = args.cache_root.absolute()
    tmp_root = args.tmp_root.absolute()
    manifest_path = orchestration_root / "orchestration_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"orchestration manifest already exists: {manifest_path}")
    if cache_root.exists():
        raise FileExistsError(f"fresh queue cache root already exists: {cache_root}")

    reference_manifest = load_json(reference_manifest_path)
    if reference_manifest.get("status") != "complete":
        raise RuntimeError("A reference training is not complete")
    if reference_manifest.get("base_checkpoint", {}).get("sha256") != EXPECTED_BASE_SHA256:
        raise RuntimeError("A reference base checkpoint drift")
    runtime = verify_runtime_against_reference(runtime_root, reference_manifest)
    reference_config = load_yaml(reference_config_path)

    group_audits = {}
    for name, raw_path in experiments.items():
        config_path = raw_path.resolve(strict=True)
        config = load_yaml(config_path)
        audit = audit_variant_config(name, reference_config, config)
        training_dir = Path(config["train_config"]["continual_audit_dir"]).absolute()
        if training_dir != Path(config["train_config"]["default_root_dir"]).absolute():
            raise RuntimeError(f"{name} audit/default output directories differ")
        if training_dir.exists():
            raise FileExistsError(f"{name} training output already exists: {training_dir}")
        group_evaluation = evaluation_root / name
        if group_evaluation.exists():
            raise FileExistsError(f"{name} evaluation output already exists: {group_evaluation}")
        checkpoint_link = args.checkpoint_link_root.absolute() / training_dir.name
        if checkpoint_link.exists() or checkpoint_link.is_symlink():
            raise FileExistsError(f"{name} checkpoint link already exists: {checkpoint_link}")
        split_manifest_path = Path(config["dataset_config"]["split_manifest_path"])
        split_manifest = load_json(split_manifest_path.resolve(strict=True))
        for split_name in ("train", "validation"):
            data_path = Path(config["dataset_config"][f"{split_name}_data_path"])
            artifact = split_manifest["artifacts"][split_name]
            if data_path.resolve(strict=True) != Path(artifact["path"]).resolve(strict=True):
                raise RuntimeError(f"{name} {split_name} path differs from manifest")
            if sha256_file(data_path) != artifact["sha256"]:
                raise RuntimeError(f"{name} {split_name} hash drift")
        if split_manifest.get("source_leakage_count") != 0:
            raise RuntimeError(f"{name} split reports source leakage")
        group_audits[name] = {
            **audit,
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "training_dir": str(training_dir),
            "evaluation_dir": str(group_evaluation),
            "checkpoint_link": str(checkpoint_link),
            "split_manifest": {
                "path": str(split_manifest_path.resolve()),
                "sha256": sha256_file(split_manifest_path),
                "split_identity_sha256": split_manifest["split_identity_sha256"],
                "train_rows": split_manifest["train"]["row_count"],
                "validation_rows": split_manifest["validation"]["row_count"],
                "source_leakage_count": split_manifest["source_leakage_count"],
            },
        }

    free_bytes = shutil.disk_usage(orchestration_root.parent).free
    if free_bytes < 30 * 1024**3:
        raise RuntimeError("data disk has less than the frozen 30 GiB free-space gate")
    return {
        "project_root": str(project_root),
        "runtime_root": str(runtime_root),
        "training_python": str(training_python),
        "reference_config": {
            "path": str(reference_config_path),
            "sha256": sha256_file(reference_config_path),
        },
        "reference_training_manifest": {
            "path": str(reference_manifest_path),
            "sha256": sha256_file(reference_manifest_path),
        },
        "runtime": runtime,
        "saved_checkpoint_steps": EXPECTED_CHECKPOINT_STEPS,
        "external_internal_validation_steps": [0] + EXPECTED_EVALUATION_STEPS,
        "table3_enabled": False,
        "table2_enabled": False,
        "device": args.device,
        "free_bytes_at_preflight": free_bytes,
        "cache_root": str(cache_root),
        "tmp_root": str(tmp_root),
        "groups": group_audits,
    }


def environment_for(args: argparse.Namespace, group: str) -> dict[str, str]:
    cache_root = args.cache_root.absolute()
    tmp_dir = args.tmp_root.absolute() / group
    for path in (
        cache_root / "home",
        cache_root / "datasets",
        cache_root / "xdg",
        tmp_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": args.device,
            "WANDB_MODE": "offline",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HOME": str(cache_root / "home"),
            "HF_DATASETS_CACHE": str(cache_root / "datasets"),
            "XDG_CACHE_HOME": str(cache_root / "xdg"),
            "TMPDIR": str(tmp_dir),
            "TEMP": str(tmp_dir),
            "TMP": str(tmp_dir),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return environment


def validate_training_outputs(
    training_dir: Path,
    expected_split_identity: str,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    manifest_path = training_dir / "run_manifest.json"
    manifest = load_json(manifest_path)
    summary = validate_training_manifest(manifest, expected_split_identity)
    checkpoints = {}
    import torch

    for step in EXPECTED_CHECKPOINT_STEPS:
        item = manifest.get("checkpoints", {}).get(str(step))
        if not isinstance(item, dict):
            raise RuntimeError(f"compact checkpoint missing at step {step}")
        path = Path(item["path"]).resolve(strict=True)
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"compact checkpoint hash drift at step {step}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        state = payload.get("trainable_state_dict")
        if (
            payload.get("checkpoint_profile") != "soulx-stage3-compact-evaluation-v2"
            or payload.get("local_step") != step
            or payload.get("sample_exposure") != step * 576
            or payload.get("split_identity_sha256") != expected_split_identity
            or not isinstance(state, dict)
            or len(state) != manifest.get("trainable_tensor_count")
            or any(not torch.isfinite(tensor).all() for tensor in state.values())
        ):
            raise RuntimeError(f"compact checkpoint tensor audit failed at step {step}")
        checkpoints[step] = {
            "path": str(path),
            "sha256": item["sha256"],
            "bytes": path.stat().st_size,
            "tensor_audit_passed": True,
        }
        del payload, state
    last = (training_dir / "last.ckpt").resolve(strict=True)
    summary.update(
        {
            "manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": sha256_file(manifest_path),
            },
            "last_checkpoint": {
                "path": str(last),
                "sha256": sha256_file(last),
                "bytes": last.stat().st_size,
            },
            "compact_checkpoints": [checkpoints[step] for step in EXPECTED_CHECKPOINT_STEPS],
        }
    )
    return summary, checkpoints


def main() -> int:
    args = parse_args()
    audit = preflight(args)
    if args.preflight_only:
        print(json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2))
        return 0

    orchestration_root = args.orchestration_root.absolute()
    manifest_path = orchestration_root / "orchestration_manifest.json"
    orchestration_root.mkdir(parents=True, exist_ok=True)
    args.cache_root.absolute().mkdir(parents=True, exist_ok=False)
    args.tmp_root.absolute().mkdir(parents=True, exist_ok=True)
    args.evaluation_root.absolute().mkdir(parents=True, exist_ok=False)
    args.checkpoint_link_root.absolute().mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "task_id": "duplexconv_edu0001_0045_abcd_official_training_v1",
        "status": "running",
        "started_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "current_group": None,
        "current_stage": "preflight_complete",
        "approved_scope": (
            "Train B/C/D serially for 30 official-Lightning optimizer steps; "
            "externally validate only steps 0, 5 and 10; no Table 2/Table 3"
        ),
        "preflight": audit,
        "groups": {
            name: {"status": "pending", **audit["groups"][name]}
            for name in ("B", "C", "D")
        },
    }

    def write_manifest() -> None:
        manifest["updated_at_utc"] = utc_now()
        atomic_json_write(manifest_path, manifest)

    write_manifest()
    try:
        experiments = dict(args.experiment)
        for name in ("B", "C", "D"):
            group = manifest["groups"][name]
            config_path = experiments[name].resolve(strict=True)
            training_dir = Path(group["training_dir"])
            evaluation_dir = Path(group["evaluation_dir"])
            environment = environment_for(args, name)
            manifest["current_group"] = name

            group["status"] = "running"
            group["started_at_utc"] = utc_now()
            group["current_stage"] = "step0_validation"
            manifest["current_stage"] = f"{name}:step0_validation"
            write_manifest()
            step0_path = evaluation_dir / "group_validation/step000000.json"
            run_logged(
                [
                    audit["training_python"],
                    str(PROJECT_ROOT / "scripts/run_official_group_validation.py"),
                    "--runtime-root",
                    audit["runtime_root"],
                    "--config",
                    str(config_path),
                    "--output",
                    str(step0_path),
                ],
                evaluation_dir / "group_validation/step000000.log",
                environment,
            )
            step0 = load_json(step0_path)
            group["validation"] = {
                "0": validate_group_validation_result(
                    step0,
                    0,
                    group["split_manifest"]["split_identity_sha256"],
                    None,
                )
            }
            group["current_stage"] = "training"
            manifest["current_stage"] = f"{name}:training"
            write_manifest()

            training_dir.mkdir(parents=True, exist_ok=False)
            run_logged(
                [
                    audit["training_python"],
                    str(Path(audit["runtime_root"]) / "finetune.py"),
                    "--config_path",
                    str(config_path),
                ],
                training_dir / "training.log",
                environment,
            )
            training_summary, checkpoints = validate_training_outputs(
                training_dir, group["split_manifest"]["split_identity_sha256"]
            )
            group["training"] = training_summary

            for step in EXPECTED_EVALUATION_STEPS:
                group["current_stage"] = f"step{step}_validation"
                manifest["current_stage"] = f"{name}:step{step}_validation"
                write_manifest()
                output = evaluation_dir / f"group_validation/step{step:06d}.json"
                run_logged(
                    [
                        audit["training_python"],
                        str(PROJECT_ROOT / "scripts/run_official_group_validation.py"),
                        "--runtime-root",
                        audit["runtime_root"],
                        "--config",
                        str(config_path),
                        "--output",
                        str(output),
                        "--continuation-checkpoint",
                        checkpoints[step]["path"],
                    ],
                    evaluation_dir / f"group_validation/step{step:06d}.log",
                    environment,
                )
                result = load_json(output)
                group["validation"][str(step)] = validate_group_validation_result(
                    result,
                    step,
                    group["split_manifest"]["split_identity_sha256"],
                    checkpoints[step]["sha256"],
                )
                write_manifest()

            group["current_stage"] = "partial_validation_index"
            manifest["current_stage"] = f"{name}:partial_validation_index"
            write_manifest()
            index_path = evaluation_dir / "group_validation/index_step000005_000010.json"
            index_log = evaluation_dir / "group_validation/index_step000005_000010.log"
            command = [
                audit["training_python"],
                str(PROJECT_ROOT / "scripts/build_official_group_validation_index.py"),
                "--baseline",
                str(step0_path),
            ]
            for step in EXPECTED_EVALUATION_STEPS:
                command.extend(
                    [
                        "--checkpoint-result",
                        str(evaluation_dir / f"group_validation/step{step:06d}.json"),
                        "--expected-step",
                        str(step),
                    ]
                )
            command.extend(["--output", str(index_path)])
            run_logged(command, index_log, environment)
            index = load_json(index_path)
            if (
                index.get("status") != "complete"
                or index.get("expected_steps") != EXPECTED_EVALUATION_STEPS
                or index.get("table3_used_for_selection") is not False
            ):
                raise RuntimeError(f"{name} partial internal-validation index failed")
            group["partial_validation_index"] = {
                "path": str(index_path),
                "sha256": sha256_file(index_path),
                "evaluated_steps": EXPECTED_EVALUATION_STEPS,
                "selection_is_partial": True,
                "selection": index["selection"],
            }

            checkpoint_link = Path(group["checkpoint_link"])
            checkpoint_link.symlink_to(training_dir, target_is_directory=True)
            group["status"] = "complete"
            group["current_stage"] = "complete"
            group["completed_at_utc"] = utc_now()
            manifest["current_stage"] = f"{name}:complete"
            write_manifest()

        manifest["status"] = "complete"
        manifest["current_group"] = None
        manifest["current_stage"] = "complete"
        manifest["completed_at_utc"] = utc_now()
        write_manifest()
        return 0
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["exception"] = repr(error)
        manifest["traceback"] = traceback.format_exc()
        if manifest["current_group"] is not None:
            manifest["groups"][manifest["current_group"]]["status"] = "failed"
        write_manifest()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
