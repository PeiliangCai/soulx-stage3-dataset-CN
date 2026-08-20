#!/usr/bin/env python3
"""Run five real patched Stage 3 optimizer steps and verify compact reload."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_examples(dataset, count: int = 5) -> list[int]:
    state_tokens = [
        "<|user_complete|>",
        "<|user_incomplete|>",
        "<|user_backchannel|>",
        "<|user_nonidle|>",
        "<|user_idle|>",
    ]
    candidates = []
    for index, row in enumerate(dataset.data_list):
        sequence = row["sequence"]
        length = len(dataset.tokenizer.encode(sequence))
        if 100 <= length <= 600:
            states = {token for token in state_tokens if token in sequence}
            if states - {"<|user_idle|>"}:
                candidates.append((length, index, states))
    chosen = []
    covered = set()
    for _ in range(count):
        options = [item for item in candidates if item[1] not in chosen]
        options.sort(key=lambda item: (-len(item[2] - covered), item[0], item[1]))
        if not options:
            raise ValueError("not enough short real examples for five-step preflight")
        _, index, states = options[0]
        chosen.append(index)
        covered.update(states)
    if covered != set(state_tokens):
        raise ValueError(f"five-step sample does not cover all states: {sorted(covered)}")
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    runtime_dir = args.runtime_dir.resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    report_path = args.report.absolute()
    checkpoint_path = args.checkpoint.absolute()
    if report_path.exists() or checkpoint_path.exists():
        raise FileExistsError("report/checkpoint output already exists")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(runtime_dir))
    from config.config import RunConfig
    from models.state_prediction_data import State_Prediction_Dataset
    from models.state_prediction_model import State_Prediction_Model

    config = OmegaConf.merge(RunConfig(), OmegaConf.load(config_path))
    seed = int(config.train_config.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    started = time.monotonic()
    dataset = State_Prediction_Dataset(config, "train")
    chosen = choose_examples(dataset)
    model = State_Prediction_Model(config).train().to("cuda")
    optimizer_config = model.configure_optimizers()
    optimizer = optimizer_config["optimizer"]
    scheduler = optimizer_config["lr_scheduler"]["scheduler"]
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    tracked_name = next(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("audio_projector")
    )
    tracked_parameter = dict(model.named_parameters())[tracked_name]
    tracked_before = tracked_parameter.detach().float().cpu().clone()
    steps = []
    overflow_attempts = []
    attempt = 0
    while len(steps) < 5:
        attempt += 1
        if attempt > 30:
            raise ValueError("AMP failed to complete five optimizer steps in 30 attempts")
        dataset_index = chosen[len(steps)]
        sample = dataset[dataset_index]
        batch = dataset.collator([sample])
        batch = {
            key: value.to("cuda") if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(
                (batch["input_ids"], batch["audio_masks"], batch["label_text_ids"])
            )
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            loss, losses, accuracies = model._compute_heads(
                logits, predictions, batch, model.LOSS_HEADS
            )
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite total loss at attempt {attempt}")
        if not all(torch.isfinite(value) for value in losses.values()):
            raise ValueError(f"non-finite head loss at attempt {attempt}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nonfinite_gradient_names = [
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        ]
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()
        if nonfinite_gradient_names:
            if scale_after >= scale_before:
                raise ValueError("GradScaler did not back off after non-finite gradients")
            overflow_attempts.append(
                {
                    "attempt": attempt,
                    "pending_step": len(steps) + 1,
                    "index": sample["index"],
                    "grad_scaler_before": float(scale_before),
                    "grad_scaler_after": float(scale_after),
                    "nonfinite_gradient_name_sample": nonfinite_gradient_names[:5],
                }
            )
            continue
        scheduler.step()
        valid_targets = {
            head.name: int(torch.count_nonzero(batch[head.label_key][:, 1:] != -100))
            for head in model.LOSS_HEADS
        }
        steps.append(
            {
                "step": len(steps) + 1,
                "attempt": attempt,
                "index": sample["index"],
                "token_length": int(len(sample["input_id"])),
                "total_loss": float(loss.detach().float().cpu()),
                "head_losses": {
                    name: float(value.detach().float().cpu()) for name, value in losses.items()
                },
                "head_accuracies": {
                    name: (
                        float(value.detach().float().cpu()) if torch.isfinite(value) else None
                    )
                    for name, value in accuracies.items()
                },
                "valid_targets": valid_targets,
                "grad_scaler_before": float(scale_before),
                "grad_scaler_after": float(scale_after),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )

    tracked_after = tracked_parameter.detach().float().cpu().clone()
    parameter_delta_l2 = float(torch.linalg.vector_norm(tracked_after - tracked_before))
    if parameter_delta_l2 == 0.0:
        raise ValueError("tracked trainable projector parameter did not change")

    trainable_state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    checkpoint = {
        "schema_version": 1,
        "checkpoint_profile": "soulx-stage3-compact-trainable-resume-v1",
        "base_runtime_commit": "928b06508ed2de1344208d06fb1f6fb2ebfb1df5",
        "step": 5,
        "trainable_state_dict": trainable_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "grad_scaler_state_dict": scaler.state_dict(),
        "config": OmegaConf.to_container(config, resolve=True),
    }
    torch.save(checkpoint, checkpoint_path)
    checkpoint_size = checkpoint_path.stat().st_size
    checkpoint_sha256 = sha256_file(checkpoint_path)

    with torch.no_grad():
        tracked_parameter.add_(1.0)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    named_parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in loaded["trainable_state_dict"].items():
            named_parameters[name].copy_(value.to(named_parameters[name].device))
    if not torch.equal(tracked_parameter.detach().cpu(), tracked_after):
        raise ValueError("compact trainable checkpoint reload mismatch")
    optimizer.load_state_dict(loaded["optimizer_state_dict"])
    scheduler.load_state_dict(loaded["scheduler_state_dict"])
    scaler.load_state_dict(loaded["grad_scaler_state_dict"])

    report = {
        "schema_version": 1,
        "status": "passed",
        "runtime_dir": str(runtime_dir),
        "runtime_base_commit": "928b06508ed2de1344208d06fb1f6fb2ebfb1df5",
        "config_path": str(config_path),
        "dataset_train_length": len(dataset),
        "steps": steps,
        "amp_overflow_attempt_count": len(overflow_attempts),
        "amp_overflow_attempts": overflow_attempts,
        "all_five_states_covered": True,
        "tracked_parameter": tracked_name,
        "tracked_parameter_delta_l2": parameter_delta_l2,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "total_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "cuda_peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "wall_time_seconds": round(time.monotonic() - started, 6),
        "checkpoint": {
            "path": str(checkpoint_path),
            "profile": checkpoint["checkpoint_profile"],
            "bytes": checkpoint_size,
            "sha256": checkpoint_sha256,
            "reload_passed": True,
            "includes_optimizer_scheduler_scaler": True,
        },
        "head_valid_target_totals": dict(
            Counter(
                {
                    head.name: sum(item["valid_targets"][head.name] for item in steps)
                    for head in model.LOSS_HEADS
                }
            )
        ),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
