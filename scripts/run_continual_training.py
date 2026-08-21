#!/usr/bin/env python3
"""Run auditable SoulX Stage 3 continuation with compact checkpoints."""

from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from duplexconv_stage3.continual_training import (
    atomic_json_write,
    continuation_learning_rate,
    sha256_file,
    utc_now,
)


OFFICIAL_CHECKPOINT_SHA256 = (
    "b0703dea0b1dbb1cd51e6e7b6514c60907ea4d4b6752cecc3f71cb6445650dbe"
)
RUNTIME_BASE_COMMIT = "928b06508ed2de1344208d06fb1f6fb2ebfb1df5"
DEFAULT_CHECKPOINT_STEPS = (0, 5, 10, 20, 30, 45, 60, 90, 120, 180, 240, 300)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_steps(value: str) -> list[int]:
    result = sorted(set(int(item) for item in value.split(",") if item.strip()))
    if not result or result[0] != 0 or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("checkpoint steps must start at zero")
    return result


def package_versions() -> dict[str, str | None]:
    names = (
        "torch",
        "torchaudio",
        "transformers",
        "pytorch-lightning",
        "peft",
        "datasets",
        "numpy",
        "omegaconf",
    )
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def git_identity(root: Path) -> dict[str, Any]:
    def output(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    dirty = output("status", "--porcelain", "--untracked-files=all").splitlines()
    return {
        "root": str(root),
        "commit": output("rev-parse", "HEAD"),
        "tree": output("rev-parse", "HEAD^{tree}"),
        "dirty_lines": dirty,
    }


def runtime_identity(runtime_dir: Path) -> dict[str, Any]:
    files = []
    for relative in (
        "models/state_prediction_model.py",
        "models/state_prediction_data.py",
        "models/_train_heads.py",
        "config/config.py",
    ):
        path = runtime_dir / relative
        files.append({"path": relative, "sha256": sha256_file(path)})
    return {
        "root": str(runtime_dir),
        "base_commit": RUNTIME_BASE_COMMIT,
        "files": files,
    }


def move_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def load_frozen_datasets(config, split_manifest: dict[str, Any], dataset_class):
    from datasets import load_dataset

    # Construct once through the official class to preserve its tokenizer,
    # token masks, __getitem__, and collator.  Only its random row split is
    # replaced by the already-frozen source-conversation Parquet files.
    train_dataset = dataset_class(config, "train")
    train_dataset.data_list = load_dataset(
        "parquet",
        data_files=split_manifest["artifacts"]["train"]["path"],
        split="train",
    )
    validation_dataset = copy.copy(train_dataset)
    validation_dataset.data_list = load_dataset(
        "parquet",
        data_files=split_manifest["artifacts"]["validation"]["path"],
        split="train",
    )
    expected_train = split_manifest["train"]["row_count"]
    expected_validation = split_manifest["validation"]["row_count"]
    if len(train_dataset) != expected_train or len(validation_dataset) != expected_validation:
        raise RuntimeError("frozen split row count mismatch")
    train_ids = set(train_dataset.data_list["index"])
    validation_ids = set(validation_dataset.data_list["index"])
    if train_ids & validation_ids:
        raise RuntimeError("frozen split row leakage")
    return train_dataset, validation_dataset


def valid_count(labels: torch.Tensor) -> int:
    return int(torch.count_nonzero(labels[:, 1:] != -100))


def validation_metrics(model, dataset, device: str) -> dict[str, Any]:
    started = time.monotonic()
    was_training = model.training
    model.eval()
    heads = model.LOSS_HEADS
    loss_sums = Counter()
    correct = Counter()
    targets = Counter()
    row_total_sum = 0.0
    finite_rows = 0
    with torch.no_grad():
        for index in range(len(dataset)):
            batch = move_batch(dataset.collator([dataset[index]]), device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(
                    (batch["input_ids"], batch["audio_masks"], batch["label_text_ids"])
                )
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                total_loss, losses, _ = model._compute_heads(
                    logits, predictions, batch, heads
                )
            row_value = float(total_loss.detach().float().cpu())
            if not math.isfinite(row_value):
                raise RuntimeError(f"non-finite validation loss at row {index}")
            row_total_sum += row_value
            finite_rows += 1
            for head in heads:
                labels = batch[head.label_key][:, 1:]
                mask = labels != -100
                count = int(mask.sum())
                if count:
                    head_loss = float(losses[head.name].detach().float().cpu())
                    if not math.isfinite(head_loss):
                        raise RuntimeError(
                            f"non-finite validation {head.name} loss at row {index}"
                        )
                    loss_sums[head.name] += head_loss * count
                    correct[head.name] += int(
                        (predictions.detach()[:, :-1][mask] == labels[mask]).sum()
                    )
                    targets[head.name] += count
            del outputs, logits, predictions, batch
    if was_training:
        model.train()
    head_metrics = {}
    for head in heads:
        count = targets[head.name]
        head_metrics[head.name] = {
            "loss": loss_sums[head.name] / count if count else None,
            "accuracy": correct[head.name] / count if count else None,
            "correct": correct[head.name],
            "valid_targets": count,
        }
    weighted_token_loss = sum(
        float(getattr(model.train_config, head.weight_attr))
        * head_metrics[head.name]["loss"]
        for head in heads
        if head_metrics[head.name]["loss"] is not None
    )
    state_names = ("idle", "nonidle", "user_complete", "user_incomplete", "user_backchannel")
    state_macro_accuracy = sum(head_metrics[name]["accuracy"] for name in state_names) / len(state_names)
    return {
        "row_count": finite_rows,
        "official_row_mean_weighted_loss": row_total_sum / finite_rows,
        "token_weighted_objective": weighted_token_loss,
        "state_macro_accuracy": state_macro_accuracy,
        "heads": head_metrics,
        "runtime_seconds": time.monotonic() - started,
    }


class IndexStream:
    def __init__(self, size: int, seed: int, state: dict[str, int] | None = None):
        self.size = size
        self.seed = seed
        self.epoch = int((state or {}).get("epoch", 0))
        self.cursor = int((state or {}).get("cursor", 0))
        self.consumed = int((state or {}).get("consumed", 0))
        self.order = self._order(self.epoch)
        if not 0 <= self.cursor <= self.size:
            raise ValueError("invalid stream cursor")

    def _order(self, epoch: int) -> list[int]:
        order = list(range(self.size))
        random.Random(self.seed + epoch).shuffle(order)
        return order

    def next(self) -> int:
        if self.cursor == self.size:
            self.epoch += 1
            self.cursor = 0
            self.order = self._order(self.epoch)
        result = self.order[self.cursor]
        self.cursor += 1
        self.consumed += 1
        return result

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "cursor": self.cursor, "consumed": self.consumed}


def cpu_trainable_state(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def save_checkpoint(
    path: Path,
    profile: str,
    model,
    local_step: int,
    peak_lr: float,
    origin_step_estimate: int,
    split_identity: str,
    stream: IndexStream,
    optimizer=None,
    scaler=None,
) -> dict[str, Any]:
    payload = {
        "schema_version": 2,
        "checkpoint_profile": profile,
        "runtime_base_commit": RUNTIME_BASE_COMMIT,
        "official_base_checkpoint_sha256": OFFICIAL_CHECKPOINT_SHA256,
        "local_step": local_step,
        "origin_step_estimate": origin_step_estimate,
        "estimated_total_optimizer_step": origin_step_estimate + local_step,
        "peak_learning_rate": peak_lr,
        "split_identity_sha256": split_identity,
        "stream_state": stream.state_dict(),
        "trainable_state_dict": cpu_trainable_state(model),
    }
    if optimizer is not None and scaler is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
        payload["grad_scaler_state_dict"] = scaler.state_dict()
        payload["rng_state"] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "profile": profile,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--purpose", choices=("calibration", "formal"), required=True)
    parser.add_argument("--peak-lr", type=float, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument(
        "--checkpoint-steps",
        type=parse_steps,
        default=list(DEFAULT_CHECKPOINT_STEPS),
    )
    parser.add_argument("--origin-step-estimate", type=int, default=1800)
    parser.add_argument("--official-global-effective-batch", type=int, default=576)
    parser.add_argument("--rewarm-steps", type=int, default=5)
    parser.add_argument("--accumulate-grad-batches", type=int, default=72)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_steps <= 0 or args.checkpoint_steps[-1] > args.max_steps:
        raise ValueError("checkpoint grid exceeds max steps")
    if args.peak_lr <= 0 or args.accumulate_grad_batches <= 0:
        raise ValueError("learning rate and accumulation must be positive")

    runtime_dir = args.runtime_dir.resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    split_path = args.split_manifest.resolve(strict=True)
    run_dir = args.run_dir.absolute()
    manifest_path = run_dir / "run_manifest.json"
    step_log = run_dir / "training_steps.jsonl"
    validation_log = run_dir / "validation_metrics.jsonl"
    latest_checkpoint = run_dir / "checkpoints/latest_resume.pt"
    if args.resume:
        if not manifest_path.is_file() or not latest_checkpoint.is_file():
            raise FileNotFoundError("resume manifest/latest checkpoint is missing")
    elif run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    else:
        (run_dir / "checkpoints/evaluation").mkdir(parents=True)

    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    for artifact in split_manifest["artifacts"].values():
        path = Path(artifact["path"]).resolve(strict=True)
        if sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"split artifact drift: {path}")

    sys.path.insert(0, str(runtime_dir))
    from config.config import RunConfig
    from models.state_prediction_data import State_Prediction_Dataset
    from models.state_prediction_model import State_Prediction_Model

    config = OmegaConf.merge(RunConfig(), OmegaConf.load(config_path))
    config.train_config.learning_rate = args.peak_lr
    config.train_config.total_steps = args.max_steps
    config.train_config.accumulate_grad_batches = args.accumulate_grad_batches
    seed = int(config.train_config.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    train_dataset, validation_dataset = load_frozen_datasets(
        config, split_manifest, State_Prediction_Dataset
    )
    initialization_log = run_dir / "initialization.log"
    log_mode = "a" if args.resume else "x"
    with initialization_log.open(log_mode, encoding="utf-8") as handle:
        with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            model = State_Prediction_Model(config).train().to("cuda")
    if sha256_file(Path(config.model_config.init_ckpt_path)) != OFFICIAL_CHECKPOINT_SHA256:
        raise RuntimeError("official base checkpoint identity drift")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.peak_lr,
        weight_decay=float(config.train_config.weight_decay),
        betas=tuple(config.train_config.betas),
        eps=float(config.train_config.eps),
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=16384.0, enabled=True)

    local_step = 0
    stream_state = None
    if args.resume:
        resume_payload = torch.load(latest_checkpoint, map_location="cpu", weights_only=False)
        if resume_payload["split_identity_sha256"] != split_manifest["split_identity_sha256"]:
            raise RuntimeError("resume split identity mismatch")
        named = dict(model.named_parameters())
        if set(resume_payload["trainable_state_dict"]) != {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }:
            raise RuntimeError("resume trainable key set mismatch")
        with torch.no_grad():
            for name, value in resume_payload["trainable_state_dict"].items():
                named[name].copy_(value.to(named[name].device))
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        scaler.load_state_dict(resume_payload["grad_scaler_state_dict"])
        rng_state = resume_payload["rng_state"]
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch_cpu"])
        torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
        local_step = int(resume_payload["local_step"])
        stream_state = resume_payload["stream_state"]
    stream = IndexStream(len(train_dataset), seed, stream_state)

    base_checkpoint = Path(config.model_config.init_ckpt_path).resolve(strict=True)
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    effective_batch = int(config.dataset_config.batch_size) * args.accumulate_grad_batches
    if args.resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["run_id"] != args.run_id or manifest["peak_learning_rate"] != args.peak_lr:
            raise RuntimeError("resume arguments do not match manifest")
        if sha256_file(latest_checkpoint) != manifest["latest_resume"]["sha256"]:
            raise RuntimeError("latest resume checkpoint hash mismatch")
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
        manifest["status"] = "running"
        manifest["resumed_at_utc"] = utc_now()
    else:
        manifest = {
            "schema_version": 2,
            "status": "running",
            "run_id": args.run_id,
            "purpose": args.purpose,
            "started_at_utc": utc_now(),
            "project": git_identity(PROJECT_ROOT),
            "runtime": runtime_identity(runtime_dir),
            "config": {
                "path": str(config_path),
                "sha256": sha256_file(config_path),
                "resolved": OmegaConf.to_container(config, resolve=True),
            },
            "split": {
                "manifest_path": str(split_path),
                "manifest_sha256": sha256_file(split_path),
                "identity_sha256": split_manifest["split_identity_sha256"],
                "train": split_manifest["train"],
                "validation": split_manifest["validation"],
            },
            "base_checkpoint": {
                "path": str(base_checkpoint),
                "sha256": OFFICIAL_CHECKPOINT_SHA256,
                "bytes": base_checkpoint.stat().st_size,
                "published_step_metadata_present": False,
            },
            "origin_step_estimate": args.origin_step_estimate,
            "origin_step_estimate_confidence": "low",
            "origin_step_estimate_basis": "public training-code stage3 total_steps=1800",
            "exact_optimizer_resume": False,
            "optimizer_state_initialization": "new AdamW; official release has no optimizer state",
            "resume_semantics": "latest grid point restores trainable tensors, AdamW, GradScaler, Python/NumPy/Torch RNG and deterministic data cursor",
            "peak_learning_rate": args.peak_lr,
            "lr_schedule": {
                "profile": "short-rewarm-plus-origin-offset-inverse-sqrt-v1",
                "rewarm_steps": args.rewarm_steps,
            },
            "precision": "fp16-autocast-with-gradscaler",
            "initial_grad_scale": 16384.0,
            "batch_size_per_microbatch": int(config.dataset_config.batch_size),
            "gradient_accumulation": args.accumulate_grad_batches,
            "local_effective_batch": effective_batch,
            "official_reference_global_effective_batch": args.official_global_effective_batch,
            "official_sample_equivalent_step_per_local_step": effective_batch / args.official_global_effective_batch,
            "checkpoint_steps": args.checkpoint_steps,
            "max_steps": args.max_steps,
            "seed": seed,
            "trainable_parameter_count": trainable_count,
            "total_parameter_count": sum(p.numel() for p in model.parameters()),
            "environment": {
                "python": platform.python_version(),
                "executable": sys.executable,
                "packages": package_versions(),
                "gpu": torch.cuda.get_device_name(0),
                "torch_cuda": torch.version.cuda,
            },
            "step_records": 0,
            "validation_records": 0,
            "checkpoints": {},
            "latest_resume": None,
            "best_validation_resume": None,
            "amp_overflows": [],
            "resume_count": 0,
        }
    atomic_json_write(manifest_path, manifest)

    checkpoint_steps = set(args.checkpoint_steps)
    if local_step == 0:
        baseline_validation = validation_metrics(model, validation_dataset, "cuda")
        baseline_record = {
            "local_step": 0,
            "estimated_total_optimizer_step": args.origin_step_estimate,
            "official_sample_equivalent_continuation_step": 0.0,
            "learning_rate": 0.0,
            "metrics": baseline_validation,
            "recorded_at_utc": utc_now(),
        }
        append_jsonl(validation_log, baseline_record)
        manifest["validation_records"] = 1
        manifest["baseline_validation"] = baseline_record
        atomic_json_write(manifest_path, manifest)

    started = time.monotonic()
    while local_step < args.max_steps:
        pending_step = local_step + 1
        lr = continuation_learning_rate(
            args.peak_lr,
            args.origin_step_estimate,
            pending_step,
            args.rewarm_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        objective_sum = 0.0
        per_head_loss_sums = Counter()
        per_head_correct = Counter()
        per_head_targets = Counter()
        micro_indexes = []
        step_started = time.monotonic()
        for _ in range(args.accumulate_grad_batches):
            dataset_index = stream.next()
            sample = train_dataset[dataset_index]
            micro_indexes.append(sample["index"])
            batch = move_batch(train_dataset.collator([sample]), "cuda")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(
                    (batch["input_ids"], batch["audio_masks"], batch["label_text_ids"])
                )
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                loss, losses, _ = model._compute_heads(
                    logits, predictions, batch, model.LOSS_HEADS
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss before step {pending_step}")
            scaler.scale(loss / args.accumulate_grad_batches).backward()
            objective_sum += float(loss.detach().float().cpu())
            for head in model.LOSS_HEADS:
                labels = batch[head.label_key][:, 1:]
                mask = labels != -100
                count = int(mask.sum())
                if count:
                    head_loss = float(losses[head.name].detach().float().cpu())
                    per_head_loss_sums[head.name] += head_loss * count
                    per_head_targets[head.name] += count
                    per_head_correct[head.name] += int(
                        (predictions.detach()[:, :-1][mask] == labels[mask]).sum()
                    )
            del outputs, logits, predictions, batch
        scaler.unscale_(optimizer)
        grad_squares = []
        nonfinite_gradient_names = []
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    nonfinite_gradient_names.append(name)
                else:
                    grad_squares.append(torch.sum(parameter.grad.detach().float() ** 2))
        grad_norm = float(torch.sqrt(torch.stack(grad_squares).sum()).cpu()) if grad_squares else 0.0
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        if nonfinite_gradient_names:
            overflow = {
                "pending_step": pending_step,
                "consumed_microbatches": stream.consumed,
                "grad_scaler_before": scale_before,
                "grad_scaler_after": scale_after,
                "nonfinite_gradient_name_sample": nonfinite_gradient_names[:10],
                "recorded_at_utc": utc_now(),
            }
            manifest["amp_overflows"].append(overflow)
            atomic_json_write(manifest_path, manifest)
            continue

        local_step = pending_step
        heads = {}
        for head in model.LOSS_HEADS:
            count = per_head_targets[head.name]
            heads[head.name] = {
                "loss": per_head_loss_sums[head.name] / count if count else None,
                "accuracy": per_head_correct[head.name] / count if count else None,
                "correct": per_head_correct[head.name],
                "valid_targets": count,
            }
        step_record = {
            "local_step": local_step,
            "estimated_total_optimizer_step": args.origin_step_estimate + local_step,
            "official_sample_equivalent_continuation_step": stream.consumed / args.official_global_effective_batch,
            "learning_rate": lr,
            "optimizer_objective_mean": objective_sum / args.accumulate_grad_batches,
            "heads": heads,
            "gradient_l2_norm": grad_norm,
            "grad_scaler_before": scale_before,
            "grad_scaler_after": scale_after,
            "microbatch_count": args.accumulate_grad_batches,
            "cumulative_microbatches": stream.consumed,
            "epoch_equivalent": stream.consumed / len(train_dataset),
            "source_row_index_first": micro_indexes[0],
            "source_row_index_last": micro_indexes[-1],
            "runtime_seconds": time.monotonic() - step_started,
            "recorded_at_utc": utc_now(),
        }
        append_jsonl(step_log, step_record)
        manifest["step_records"] = int(manifest.get("step_records", 0)) + 1
        manifest["last_step"] = step_record
        manifest["cuda_peak_memory_bytes"] = torch.cuda.max_memory_allocated()

        if local_step in checkpoint_steps:
            validation = validation_metrics(model, validation_dataset, "cuda")
            validation_record = {
                "local_step": local_step,
                "estimated_total_optimizer_step": args.origin_step_estimate + local_step,
                "official_sample_equivalent_continuation_step": stream.consumed / args.official_global_effective_batch,
                "learning_rate": lr,
                "metrics": validation,
                "recorded_at_utc": utc_now(),
            }
            append_jsonl(validation_log, validation_record)
            manifest["validation_records"] = int(manifest.get("validation_records", 0)) + 1
            snapshot_path = run_dir / f"checkpoints/evaluation/step{local_step:06d}.pt"
            snapshot = save_checkpoint(
                snapshot_path,
                "soulx-stage3-compact-evaluation-v2",
                model,
                local_step,
                args.peak_lr,
                args.origin_step_estimate,
                split_manifest["split_identity_sha256"],
                stream,
            )
            latest = save_checkpoint(
                latest_checkpoint,
                "soulx-stage3-compact-trainable-resume-v2",
                model,
                local_step,
                args.peak_lr,
                args.origin_step_estimate,
                split_manifest["split_identity_sha256"],
                stream,
                optimizer,
                scaler,
            )
            current_objective = validation["token_weighted_objective"]
            previous_best = manifest.get("best_validation_resume")
            if previous_best is None or current_objective < previous_best["validation_objective"]:
                best = save_checkpoint(
                    run_dir / "checkpoints/best_validation_resume.pt",
                    "soulx-stage3-compact-trainable-resume-v2",
                    model,
                    local_step,
                    args.peak_lr,
                    args.origin_step_estimate,
                    split_manifest["split_identity_sha256"],
                    stream,
                    optimizer,
                    scaler,
                )
                manifest["best_validation_resume"] = {
                    **best,
                    "local_step": local_step,
                    "validation_objective": current_objective,
                }
            manifest["checkpoints"][str(local_step)] = {
                "evaluation": snapshot,
                "validation": validation_record,
            }
            manifest["latest_resume"] = {**latest, "local_step": local_step}
        manifest["updated_at_utc"] = utc_now()
        manifest["wall_time_seconds"] = time.monotonic() - started
        atomic_json_write(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "step": local_step,
                    "lr": lr,
                    "loss": step_record["optimizer_objective_mean"],
                    "epoch_equivalent": step_record["epoch_equivalent"],
                    "validation": local_step in checkpoint_steps,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    manifest["status"] = "complete"
    manifest["completed_at_utc"] = utc_now()
    manifest["wall_time_seconds"] = time.monotonic() - started
    atomic_json_write(manifest_path, manifest)
    print(json.dumps({"status": "complete", "run_id": args.run_id, "step": local_step}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
