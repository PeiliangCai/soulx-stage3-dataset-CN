"""Audit-only callback for weight continuation through Lightning Trainer.fit().

This module records what the upstream Trainer does.  It does not implement
forward, backward, gradient accumulation, optimizer stepping, AMP, or the LR
scheduler.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf


OFFICIAL_BASE_CHECKPOINT_SHA256 = (
    "b0703dea0b1dbb1cd51e6e7b6514c60907ea4d4b6752cecc3f71cb6445650dbe"
)
UPSTREAM_BASE_COMMIT = "928b06508ed2de1344208d06fb1f6fb2ebfb1df5"


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


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def package_versions() -> dict[str, str | None]:
    result = {}
    for name in (
        "torch",
        "torchaudio",
        "transformers",
        "pytorch-lightning",
        "peft",
        "datasets",
        "numpy",
        "omegaconf",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def git_identity(root: Path) -> dict[str, Any]:
    def output(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()

    audited_files = (
        "config/config.py",
        "finetune.py",
        "models/_train_heads.py",
        "models/state_prediction_data.py",
        "models/state_prediction_model.py",
        "utils/continual_audit.py",
    )
    diff = subprocess.check_output(
        ["git", "-C", str(root), "diff", "--no-ext-diff"], text=True
    )
    return {
        "root": str(root),
        "commit": output("rev-parse", "HEAD"),
        "dirty_lines": output(
            "status", "--porcelain", "--untracked-files=all"
        ).splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "audited_files": [
            {
                "path": relative,
                "sha256": sha256_file(root / relative),
                "bytes": (root / relative).stat().st_size,
            }
            for relative in audited_files
        ],
    }


class OfficialContinualAuditCallback(pl.Callback):
    """Observe official Lightning updates and save compact eval overlays."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        train = config.train_config
        self.run_id = str(train.continual_run_id)
        self.audit_dir = Path(train.continual_audit_dir).absolute()
        self.checkpoint_steps = sorted(
            set(int(step) for step in train.continual_checkpoint_steps)
        )
        self.origin_step = int(train.origin_step_estimate)
        self.last_global_step = 0
        self.microbatches_since_update = 0
        self.samples_since_update = 0
        self.total_microbatches = 0
        self.total_samples = 0
        self.loss_sum_since_update = 0.0
        self.loss_count_since_update = 0
        self.optimizer_lr_for_update = None
        self.manifest: dict[str, Any] = {}

    @property
    def state_key(self) -> str:
        return f"OfficialContinualAuditCallback:{self.run_id}"

    def state_dict(self) -> dict[str, Any]:
        return {
            "last_global_step": self.last_global_step,
            "microbatches_since_update": self.microbatches_since_update,
            "samples_since_update": self.samples_since_update,
            "total_microbatches": self.total_microbatches,
            "total_samples": self.total_samples,
            "loss_sum_since_update": self.loss_sum_since_update,
            "loss_count_since_update": self.loss_count_since_update,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for key, value in state_dict.items():
            setattr(self, key, value)

    def _write_manifest(self) -> None:
        self.manifest["updated_at_utc"] = utc_now()
        atomic_json_write(self.audit_dir / "run_manifest.json", self.manifest)

    def _compact_checkpoint(self, trainer, pl_module, step: int) -> dict[str, Any]:
        state = {
            name: parameter.detach().cpu().clone()
            for name, parameter in pl_module.named_parameters()
            if parameter.requires_grad
        }
        payload = {
            "schema_version": 3,
            "checkpoint_profile": "soulx-stage3-compact-evaluation-v2",
            "training_flow": "official-pytorch-lightning-trainer-fit-v1",
            "runtime_base_commit": UPSTREAM_BASE_COMMIT,
            "official_base_checkpoint_sha256": OFFICIAL_BASE_CHECKPOINT_SHA256,
            "local_step": step,
            "origin_step_estimate": self.origin_step,
            "estimated_total_optimizer_step": self.origin_step + step,
            "peak_learning_rate": float(self.config.train_config.learning_rate),
            "split_identity_sha256": trainer.datamodule.split_audit[
                "split_identity_sha256"
            ],
            "sample_exposure": self.total_samples,
            "trainable_state_dict": state,
        }
        path = self.audit_dir / f"checkpoints/evaluation/step{step:06d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        torch.save(payload, temporary)
        temporary.replace(path)
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "profile": payload["checkpoint_profile"],
        }

    def on_fit_start(self, trainer, pl_module) -> None:
        if self.audit_dir.exists() and (self.audit_dir / "run_manifest.json").exists():
            raise FileExistsError(f"audit run already exists: {self.audit_dir}")
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        runtime_root = Path(__file__).resolve().parents[1]
        base_checkpoint = Path(
            self.config.model_config.init_ckpt_path
        ).resolve(strict=True)
        base_sha256 = sha256_file(base_checkpoint)
        if base_sha256 != OFFICIAL_BASE_CHECKPOINT_SHA256:
            raise RuntimeError("official base checkpoint identity drift")
        trainable = {
            name: parameter.numel()
            for name, parameter in pl_module.named_parameters()
            if parameter.requires_grad
        }
        optimizer = trainer.optimizers[0]
        scheduler = trainer.lr_scheduler_configs[0].scheduler
        self.last_global_step = int(trainer.global_step)
        self.manifest = {
            "schema_version": 1,
            "status": "running",
            "run_id": self.run_id,
            "started_at_utc": utc_now(),
            "training_flow": {
                "entrypoint": "finetune.py::train",
                "trainer": "pytorch_lightning.Trainer.fit",
                "training_step": "State_Prediction_Model.training_step",
                "optimizer_factory": "State_Prediction_Model.configure_optimizers",
                "manual_optimization": False,
            },
            "runtime": git_identity(runtime_root),
            "runtime_base_commit": UPSTREAM_BASE_COMMIT,
            "base_checkpoint": {
                "path": str(base_checkpoint),
                "sha256": base_sha256,
                "bytes": base_checkpoint.stat().st_size,
                "contains_optimizer_scheduler_scaler_global_step": False,
            },
            "origin_step_estimate": self.origin_step,
            "origin_step_estimate_confidence": "low",
            "origin_step_estimate_basis": "public Stage 3 config total_steps=1800",
            "exact_optimizer_resume": bool(self.config.train_config.ckpt_path),
            "optimizer_state_initialization": (
                "Lightning checkpoint resume"
                if self.config.train_config.ckpt_path
                else "new official AdamW; released weight file has no optimizer state"
            ),
            "config": OmegaConf.to_container(self.config, resolve=True),
            "split": trainer.datamodule.split_audit,
            "environment": package_versions(),
            "trainer": {
                "devices": trainer.num_devices,
                "num_nodes": trainer.num_nodes,
                "world_size": trainer.world_size,
                "precision": str(trainer.precision),
                "accumulate_grad_batches": trainer.accumulate_grad_batches,
                "max_steps": trainer.max_steps,
                "val_check_interval": trainer.val_check_interval,
            },
            "optimizer": {
                "class": optimizer.__class__.__module__
                + "."
                + optimizer.__class__.__name__,
                "betas": list(optimizer.param_groups[0]["betas"]),
                "eps": optimizer.param_groups[0]["eps"],
                "weight_decay": optimizer.param_groups[0]["weight_decay"],
                "initial_lr": optimizer.param_groups[0]["initial_lr"],
                "lr_at_fit_start": optimizer.param_groups[0]["lr"],
            },
            "scheduler": {
                "class": scheduler.__class__.__module__
                + "."
                + scheduler.__class__.__name__,
                "last_epoch_at_fit_start": scheduler.last_epoch,
                "state_dict_at_fit_start": scheduler.state_dict(),
            },
            "checkpoint_steps": self.checkpoint_steps,
            "checkpoints": {},
            "updates": [],
            "validations": [],
            "total_parameter_count": sum(p.numel() for p in pl_module.parameters()),
            "trainable_parameter_count": sum(trainable.values()),
            "trainable_tensor_count": len(trainable),
        }
        self._write_manifest()

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        self.optimizer_lr_for_update = float(optimizer.param_groups[0]["lr"])

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ) -> None:
        batch_size = int(batch["input_ids"].shape[0])
        self.microbatches_since_update += 1
        self.samples_since_update += batch_size * int(trainer.world_size)
        self.total_microbatches += 1
        self.total_samples += batch_size * int(trainer.world_size)
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if torch.is_tensor(loss) and torch.isfinite(loss.detach()).all():
            self.loss_sum_since_update += float(loss.detach().cpu())
            self.loss_count_since_update += 1

        current_step = int(trainer.global_step)
        if current_step == self.last_global_step:
            return
        if current_step != self.last_global_step + 1:
            raise RuntimeError("unexpected Lightning global_step jump")
        record = {
            "local_step": current_step,
            "estimated_total_optimizer_step": self.origin_step + current_step,
            "microbatches": self.microbatches_since_update,
            "samples": self.samples_since_update,
            "cumulative_microbatches": self.total_microbatches,
            "cumulative_samples": self.total_samples,
            "mean_training_step_loss": (
                self.loss_sum_since_update / self.loss_count_since_update
                if self.loss_count_since_update
                else None
            ),
            "optimizer_lr_used": self.optimizer_lr_for_update,
            "recorded_at_utc": utc_now(),
        }
        append_jsonl(self.audit_dir / "optimizer_updates.jsonl", record)
        self.manifest["updates"].append(record)
        if current_step in self.checkpoint_steps:
            checkpoint = self._compact_checkpoint(trainer, pl_module, current_step)
            self.manifest["checkpoints"][str(current_step)] = checkpoint
        self.last_global_step = current_step
        self.microbatches_since_update = 0
        self.samples_since_update = 0
        self.loss_sum_since_update = 0.0
        self.loss_count_since_update = 0
        self.optimizer_lr_for_update = None
        self._write_manifest()

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        metrics = {}
        for name, value in trainer.callback_metrics.items():
            if name.startswith("val_") and torch.is_tensor(value):
                metrics[name] = float(value.detach().cpu())
        record = {
            "local_step": int(trainer.global_step),
            "estimated_total_optimizer_step": self.origin_step
            + int(trainer.global_step),
            "metrics": metrics,
            "recorded_at_utc": utc_now(),
        }
        append_jsonl(self.audit_dir / "validation_metrics.jsonl", record)
        self.manifest["validations"].append(record)
        self._write_manifest()

    def on_exception(self, trainer, pl_module, exception) -> None:
        if self.manifest:
            self.manifest["status"] = "failed"
            self.manifest["exception"] = repr(exception)
            self._write_manifest()

    def on_fit_end(self, trainer, pl_module) -> None:
        self.manifest["status"] = "complete"
        self.manifest["completed_at_utc"] = utc_now()
        self.manifest["final_local_step"] = int(trainer.global_step)
        self.manifest["final_estimated_total_optimizer_step"] = (
            self.origin_step + int(trainer.global_step)
        )
        self.manifest["total_microbatches"] = self.total_microbatches
        self.manifest["total_samples"] = self.total_samples
        self.manifest["cuda_peak_memory_bytes"] = torch.cuda.max_memory_allocated()
        self._write_manifest()
