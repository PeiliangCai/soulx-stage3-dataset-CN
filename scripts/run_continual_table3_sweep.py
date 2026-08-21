#!/usr/bin/env python3
"""Sequentially run the four frozen Table 3 classes for each checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from duplexconv_stage3.continual_evaluation import CLASS_KEYS, build_sweep_index
from duplexconv_stage3.continual_training import sha256_file


CLASS_CONFIG = {
    "en": {
        "dataset": "/root/autodl-tmp/dataset/soulx_duplug_eval/extracted/Easy-Turn-Testset-en",
        "config": "/root/SoulX-stage3-dataset/configs/soulx_table3_candidate_en.yaml",
        "asr": "/root/SoulX-stage3-dataset/pretrained_models/SenseVoiceSmall",
    },
    "zh": {
        "dataset": "/root/autodl-tmp/dataset/soulx_duplug_eval/raw/easy_turn_zh_5812651/testset",
        "config": "/root/SoulX-stage3-dataset/configs/soulx_table3_candidate_zh.yaml",
        "asr": "/root/SoulX-stage3-dataset/pretrained_models/paraformer-zh",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--official-root",
        type=Path,
        default=PROJECT_ROOT / "third_party/SoulX-Duplug-upstream",
    )
    args = parser.parse_args()
    python = args.python.resolve(strict=True)
    official_root = args.official_root.resolve(strict=True)
    output_root = args.output_root.absolute()
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "sweep_index.json"
    for checkpoint_value in args.checkpoint:
        checkpoint = checkpoint_value.resolve(strict=True)
        import torch

        metadata = torch.load(checkpoint, map_location="cpu", weights_only=True)
        step = int(metadata["local_step"])
        checkpoint_hash = sha256_file(checkpoint)
        step_root = output_root / f"step{step:06d}"
        step_root.mkdir()
        run_id = f"continual-step{step:06d}-{checkpoint_hash[:8]}"
        for language, label in CLASS_KEYS:
            output = step_root / f"{language}-{label}.json"
            trace_dir = step_root / f"{language}-{label}-traces"
            cache = step_root / f"{language}-{label}-asr.jsonl"
            process_log = step_root / f"{language}-{label}-process.log"
            settings = CLASS_CONFIG[language]
            command = [
                str(python),
                str(PROJECT_ROOT / "scripts/run_table3_reproduction.py"),
                "--language", language,
                "--label", label,
                "--dataset-root", settings["dataset"],
                "--official-root", str(official_root),
                "--config", settings["config"],
                "--asr-model-dir", settings["asr"],
                "--trace-dir", str(trace_dir),
                "--asr-cache", str(cache),
                "--output", str(output),
                "--run-id", run_id,
                "--continuation-checkpoint", str(checkpoint),
            ]
            with process_log.open("x", encoding="utf-8") as handle:
                subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=os.environ.copy(),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        gate_path = step_root / "table3-gate.json"
        gate_command = [
            str(python),
            str(PROJECT_ROOT / "scripts/check_table3_reproduction_gate.py"),
            "--evaluation-mode", "continuation",
            "--en-complete", str(step_root / "en-complete.json"),
            "--en-incomplete", str(step_root / "en-incomplete.json"),
            "--zh-complete", str(step_root / "zh-complete.json"),
            "--zh-incomplete", str(step_root / "zh-incomplete.json"),
            "--output", str(gate_path),
        ]
        subprocess.run(gate_command, cwd=PROJECT_ROOT, check=True)
        index = build_sweep_index(args.baseline_root, output_root, index_path)
        print(
            json.dumps(
                {
                    "completed_step": step,
                    "checkpoint_sha256": checkpoint_hash,
                    "evaluated_checkpoint_count": len(index["checkpoints"]),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
