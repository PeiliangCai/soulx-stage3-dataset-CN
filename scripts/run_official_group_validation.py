#!/usr/bin/env python3
"""Run a full frozen-split validation without entering the training loop."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from duplexconv_stage3.table3_reproduction import apply_continuation_checkpoint


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


def summarize_token_weighted_validation(model: Any) -> dict[str, Any]:
    """Recover exact target-weighted loss/accuracy from official per-row tensors."""

    heads: dict[str, Any] = {}
    total_loss_sum = 0.0
    total_correct_sum = 0.0
    total_targets = 0.0
    for head in model.LOSS_HEADS:
        loss_sum = 0.0
        correct_sum = 0.0
        valid_targets = 0.0
        for loss, accuracy, count in zip(
            model.val_losses[head.name],
            model.val_accs[head.name],
            model.val_valid_targets[head.name],
        ):
            count_value = float(count.detach().cpu())
            if count_value <= 0:
                continue
            loss_value = float(loss.detach().cpu())
            accuracy_value = float(accuracy.detach().cpu())
            if not np.isfinite(loss_value) or not np.isfinite(accuracy_value):
                raise RuntimeError(
                    f"non-finite validation value with targets for head {head.name}"
                )
            loss_sum += loss_value * count_value
            correct_sum += accuracy_value * count_value
            valid_targets += count_value
        if valid_targets <= 0:
            raise RuntimeError(f"validation head has no valid targets: {head.name}")
        heads[head.name] = {
            "valid_targets": int(valid_targets),
            "token_weighted_cross_entropy": loss_sum / valid_targets,
            "token_weighted_accuracy": correct_sum / valid_targets,
        }
        total_loss_sum += loss_sum
        total_correct_sum += correct_sum
        total_targets += valid_targets
    return {
        "profile": "exact-per-row-target-weighted-v1",
        "objective": total_loss_sum / total_targets,
        "accuracy": total_correct_sum / total_targets,
        "valid_targets": int(total_targets),
        "heads": heads,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--continuation-checkpoint",
        type=Path,
        help="Compact official-Lightning trainable overlay; omission evaluates step 0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_root = args.runtime_root.resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite validation artifact: {output_path}")

    sys.path.insert(0, str(runtime_root))
    from config.config import RunConfig
    from models.state_prediction_data import State_Prediction_DataModule
    from models.state_prediction_model import State_Prediction_Model

    config = OmegaConf.merge(RunConfig(), OmegaConf.load(config_path))
    seed = int(config.train_config.seed)
    pl.seed_everything(seed, workers=True)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    started_at = utc_now()
    model = State_Prediction_Model(config)
    continuation_checkpoint = None
    if args.continuation_checkpoint is not None:
        continuation_checkpoint = apply_continuation_checkpoint(
            model, args.continuation_checkpoint
        )
    data = State_Prediction_DataModule(config)
    # The patched upstream DataModule creates the frozen validation dataset in
    # its fit setup.  Pass the resulting official val_dataloader explicitly so
    # Lightning does not reinterpret this audit as a training run.
    data.setup(stage="fit")
    validation_rows = len(data.asr_val)
    if (
        continuation_checkpoint is not None
        and continuation_checkpoint["split_identity_sha256"]
        != data.split_audit["split_identity_sha256"]
    ):
        raise RuntimeError("continuation checkpoint frozen-split identity mismatch")

    trainer = pl.Trainer(
        accelerator=config.train_config.accelerator,
        devices=config.train_config.num_gpu_per_node,
        num_nodes=config.train_config.num_node,
        precision=config.train_config.precision,
        strategy=config.train_config.strategy,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        sync_batchnorm=config.train_config.sync_batchnorm,
    )
    results = trainer.validate(model=model, dataloaders=data.val_dataloader(), verbose=False)
    if len(results) != 1:
        raise RuntimeError(f"expected one validation result, received {len(results)}")
    metrics = {
        name: float(value)
        for name, value in results[0].items()
        if name.startswith("val_")
    }
    if not metrics or any(not np.isfinite(value) for value in metrics.values()):
        raise RuntimeError("validation metrics are empty or non-finite")
    exact_token_weighted_metrics = summarize_token_weighted_validation(model)
    if not np.isfinite(exact_token_weighted_metrics["objective"]):
        raise RuntimeError("exact token-weighted validation objective is non-finite")

    base_checkpoint = Path(config.model_config.init_ckpt_path).resolve(strict=True)
    validation_data = Path(config.dataset_config.validation_data_path).resolve(strict=True)
    split_manifest = Path(config.dataset_config.split_manifest_path).resolve(strict=True)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "evaluation_profile": "official-lightning-group-validation-only-v1",
        "local_step": (
            continuation_checkpoint["local_step"]
            if continuation_checkpoint is not None
            else 0
        ),
        "estimated_total_optimizer_step": (
            continuation_checkpoint["estimated_total_optimizer_step"]
            if continuation_checkpoint is not None
            else int(config.train_config.origin_step_estimate)
        ),
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "training_updates_performed": 0,
        "backward_calls_performed": 0,
        "optimizer_created": False,
        "runtime_root": str(runtime_root),
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "base_checkpoint": {
            "path": str(base_checkpoint),
            "sha256": sha256_file(base_checkpoint),
            "bytes": base_checkpoint.stat().st_size,
        },
        "continuation_checkpoint": continuation_checkpoint,
        "validation_data": {
            "path": str(validation_data),
            "sha256": sha256_file(validation_data),
            "row_count": validation_rows,
        },
        "split_manifest": {
            "path": str(split_manifest),
            "sha256": sha256_file(split_manifest),
            "split_identity_sha256": data.split_audit["split_identity_sha256"],
            "source_leakage_count": data.split_audit["source_leakage_count"],
        },
        "trainer": {
            "entrypoint": "pytorch_lightning.Trainer.validate",
            "precision": str(trainer.precision),
            "devices": trainer.num_devices,
            "world_size": trainer.world_size,
        },
        "metrics": metrics,
        "exact_token_weighted_metrics": exact_token_weighted_metrics,
        "cuda_peak_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    atomic_json_write(output_path, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
