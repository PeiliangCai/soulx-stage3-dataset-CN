"""Pure audit helpers for the official B/C/D Stage 3 training queue."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml


EXPECTED_BASE_SHA256 = (
    "b0703dea0b1dbb1cd51e6e7b6514c60907ea4d4b6752cecc3f71cb6445650dbe"
)
EXPECTED_UPSTREAM_COMMIT = "928b06508ed2de1344208d06fb1f6fb2ebfb1df5"
EXPECTED_CHECKPOINT_STEPS = [1, 2, 3, 5, 10, 20, 30]
EXPECTED_EVALUATION_STEPS = [5, 10]
EXPECTED_EFFECTIVE_BATCH = 576
EXPECTED_TOTAL_STEPS = 30
EXPECTED_VALIDATION_ROWS = 2061

IDENTITY_KEYS = {
    "train_config.continual_run_id",
    "train_config.continual_audit_dir",
    "train_config.default_root_dir",
    "train_config.wandb_run_name",
    "train_config.wandb_save_dir",
}
DATA_KEYS = {
    "dataset_config.train_data_path",
    "dataset_config.validation_data_path",
    "dataset_config.split_manifest_path",
}
WEIGHT_KEYS = {
    "train_config.user_complete_loss_rate",
    "train_config.user_incomplete_loss_rate",
}
ALLOWED_DIFF_KEYS = IDENTITY_KEYS | DATA_KEYS | WEIGHT_KEYS
EXPECTED_DIFF_KEYS = {
    "B": IDENTITY_KEYS | WEIGHT_KEYS,
    "C": IDENTITY_KEYS | DATA_KEYS,
    "D": IDENTITY_KEYS | DATA_KEYS | WEIGHT_KEYS,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return payload


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"YAML root is not a mapping: {path}")
    return payload


def flatten_mapping(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, dict):
            result.update(flatten_mapping(child, path))
        else:
            result[path] = child
    return result


def audit_variant_config(
    name: str, reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Require the exact predeclared A/B/C/D configuration differences."""

    if name not in EXPECTED_DIFF_KEYS:
        raise ValueError(f"unknown ablation group: {name}")
    baseline = flatten_mapping(reference)
    variant = flatten_mapping(candidate)
    if set(baseline) != set(variant):
        raise RuntimeError(f"{name} config key set differs from A")
    differences = {
        key: {"A": baseline[key], name: variant[key]}
        for key in baseline
        if baseline[key] != variant[key]
    }
    actual_keys = set(differences)
    if actual_keys - ALLOWED_DIFF_KEYS:
        raise RuntimeError(
            f"{name} has unapproved config differences: "
            f"{sorted(actual_keys - ALLOWED_DIFF_KEYS)}"
        )
    if actual_keys != EXPECTED_DIFF_KEYS[name]:
        raise RuntimeError(
            f"{name} config difference set mismatch: "
            f"actual={sorted(actual_keys)} expected={sorted(EXPECTED_DIFF_KEYS[name])}"
        )

    if name in {"B", "D"}:
        if not (
            variant["train_config.user_complete_loss_rate"]
            == variant["train_config.user_incomplete_loss_rate"]
            == 0.185
        ):
            raise RuntimeError(f"{name} equal-loss values drifted")
    if name == "C":
        if not (
            variant["train_config.user_complete_loss_rate"] == 0.24
            and variant["train_config.user_incomplete_loss_rate"] == 0.13
        ):
            raise RuntimeError("C original loss values drifted")
    if name in {"C", "D"}:
        for key in DATA_KEYS:
            if "balanced_ci_v1" not in str(variant[key]):
                raise RuntimeError(f"{name} does not reference balanced data for {key}")

    required = {
        "dataset_config.batch_size": 1,
        "train_config.accumulate_grad_batches": EXPECTED_EFFECTIVE_BATCH,
        "train_config.total_steps": EXPECTED_TOTAL_STEPS,
        "train_config.seed": 42,
        "train_config.origin_step_estimate": 1800,
        "train_config.continual_checkpoint_steps": EXPECTED_CHECKPOINT_STEPS,
    }
    for key, expected in required.items():
        if variant.get(key) != expected:
            raise RuntimeError(
                f"{name} frozen config value drifted: {key}={variant.get(key)!r}"
            )
    return {
        "group": name,
        "difference_keys": sorted(actual_keys),
        "differences": differences,
        "config_identity_sha256": canonical_sha256(candidate),
    }


def validate_training_manifest(
    manifest: dict[str, Any], expected_split_identity: str
) -> dict[str, Any]:
    """Validate the audit callback's completed official-Lightning record."""

    if manifest.get("status") != "complete":
        raise RuntimeError("training manifest is not complete")
    if manifest.get("runtime_base_commit") != EXPECTED_UPSTREAM_COMMIT:
        raise RuntimeError("training upstream commit drift")
    if manifest.get("base_checkpoint", {}).get("sha256") != EXPECTED_BASE_SHA256:
        raise RuntimeError("training base checkpoint drift")
    if manifest.get("split", {}).get("split_identity_sha256") != expected_split_identity:
        raise RuntimeError("training split identity drift")
    if manifest.get("final_local_step") != EXPECTED_TOTAL_STEPS:
        raise RuntimeError("training final step mismatch")
    if manifest.get("checkpoint_steps") != EXPECTED_CHECKPOINT_STEPS:
        raise RuntimeError("training checkpoint grid mismatch")
    updates = manifest.get("updates", [])
    if [item.get("local_step") for item in updates] != list(
        range(1, EXPECTED_TOTAL_STEPS + 1)
    ):
        raise RuntimeError("optimizer update sequence is incomplete")
    if any(
        item.get("microbatches") != EXPECTED_EFFECTIVE_BATCH
        or item.get("samples") != EXPECTED_EFFECTIVE_BATCH
        for item in updates
    ):
        raise RuntimeError("effective batch audit drift")
    if manifest.get("total_samples") != EXPECTED_TOTAL_STEPS * EXPECTED_EFFECTIVE_BATCH:
        raise RuntimeError("sample exposure mismatch")
    if not math.isfinite(float(manifest.get("cuda_peak_memory_bytes", float("nan")))):
        raise RuntimeError("training peak CUDA memory is missing")
    return {
        "final_local_step": EXPECTED_TOTAL_STEPS,
        "optimizer_update_count": len(updates),
        "sample_exposure": manifest["total_samples"],
        "split_identity_sha256": expected_split_identity,
        "cuda_peak_memory_bytes": manifest["cuda_peak_memory_bytes"],
    }


def validate_group_validation_result(
    result: dict[str, Any],
    expected_step: int,
    expected_split_identity: str,
    expected_checkpoint_sha256: str | None,
) -> dict[str, Any]:
    """Validate one step-0 or continuation internal-validation artifact."""

    if result.get("status") != "complete" or result.get("local_step") != expected_step:
        raise RuntimeError("group-validation status/step mismatch")
    if result.get("base_checkpoint", {}).get("sha256") != EXPECTED_BASE_SHA256:
        raise RuntimeError("group-validation base checkpoint drift")
    if (
        result.get("split_manifest", {}).get("split_identity_sha256")
        != expected_split_identity
    ):
        raise RuntimeError("group-validation split identity drift")
    if result.get("validation_data", {}).get("row_count") != EXPECTED_VALIDATION_ROWS:
        raise RuntimeError("group-validation row count drift")
    if result.get("optimizer_created") is not False:
        raise RuntimeError("group validation created an optimizer")
    if result.get("training_updates_performed") != 0:
        raise RuntimeError("group validation performed training")
    exact = result.get("exact_token_weighted_metrics", {})
    values = [exact.get("objective"), exact.get("accuracy")]
    for head in exact.get("heads", {}).values():
        values.extend(
            [head.get("token_weighted_cross_entropy"), head.get("token_weighted_accuracy")]
        )
    if (
        exact.get("profile") != "exact-per-row-target-weighted-v1"
        or not values
        or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values)
    ):
        raise RuntimeError("group-validation metrics are incomplete/non-finite")
    continuation = result.get("continuation_checkpoint")
    if expected_step == 0:
        if continuation is not None:
            raise RuntimeError("step 0 unexpectedly loaded a continuation checkpoint")
    elif (
        not isinstance(continuation, dict)
        or continuation.get("status") != "accepted"
        or continuation.get("local_step") != expected_step
        or continuation.get("sha256") != expected_checkpoint_sha256
    ):
        raise RuntimeError("group-validation checkpoint identity mismatch")
    return {
        "local_step": expected_step,
        "objective": exact["objective"],
        "accuracy": exact["accuracy"],
        "heads": exact["heads"],
    }
