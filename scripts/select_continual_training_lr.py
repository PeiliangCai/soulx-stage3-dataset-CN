#!/usr/bin/env python3
"""Freeze the formal continuation LR from two validation-only calibrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from duplexconv_stage3.continual_training import (
    atomic_json_write,
    select_lr_candidate,
    sha256_file,
    utc_now,
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.candidate) != 2:
        raise ValueError("pass exactly two --candidate run directories")
    if args.output.exists():
        raise FileExistsError(f"selection output exists: {args.output}")
    candidates = []
    split_identities = set()
    baseline_identities = set()
    for directory_value in args.candidate:
        directory = directory_value.resolve(strict=True)
        manifest_path = directory / "run_manifest.json"
        validation_path = directory / "validation_metrics.jsonl"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["status"] != "complete" or manifest["purpose"] != "calibration":
            raise RuntimeError(f"candidate is not a complete calibration: {directory}")
        rows = load_jsonl(validation_path)
        baseline = next(item for item in rows if item["local_step"] == 0)["metrics"]
        final_record = max(rows, key=lambda item: item["local_step"])
        if final_record["local_step"] != manifest["max_steps"]:
            raise RuntimeError(f"candidate lacks final validation: {directory}")
        split_identities.add(manifest["split"]["identity_sha256"])
        baseline_for_identity = dict(baseline)
        baseline_for_identity.pop("runtime_seconds", None)
        baseline_identities.add(
            json.dumps(
                baseline_for_identity,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        candidates.append(
            {
                "run_id": manifest["run_id"],
                "run_dir": str(directory),
                "manifest_sha256": sha256_file(manifest_path),
                "validation_log_sha256": sha256_file(validation_path),
                "peak_lr": manifest["peak_learning_rate"],
                "final_local_step": final_record["local_step"],
                "amp_overflow_count": len(manifest["amp_overflows"]),
                "baseline": baseline,
                "final": final_record["metrics"],
                "checkpoints": [
                    {"local_step": item["local_step"], "metrics": item["metrics"]}
                    for item in rows
                ],
            }
        )
    if len(split_identities) != 1:
        raise RuntimeError("calibration split identities differ")
    if len(baseline_identities) != 1:
        raise RuntimeError("calibration baseline validation metrics differ")
    result = select_lr_candidate(candidates)
    result.update(
        {
            "schema_version": 1,
            "status": "frozen",
            "created_at_utc": utc_now(),
            "selection_protocol": "validation-only-guard-safe-horizon-v2",
            "split_identity_sha256": next(iter(split_identities)),
            "collapse_guard": "ineligible if any state-head accuracy drops by >5pp",
            "tie_break": "if final objectives differ by <=1%, select lower LR",
            "all_ineligible_fallback": "select longest guard-safe horizon, then lower LR",
        }
    )
    atomic_json_write(args.output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
