#!/usr/bin/env python3
"""Audit and select checkpoints using only frozen internal validation results."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


STATE_HEADS = (
    "idle",
    "nonidle",
    "user_complete",
    "user_incomplete",
    "user_backchannel",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "complete":
        raise RuntimeError(f"group validation is not complete: {path}")
    exact = payload.get("exact_token_weighted_metrics")
    if exact is None or exact.get("profile") != "exact-per-row-target-weighted-v1":
        raise RuntimeError(f"exact token-weighted metrics are missing: {path}")
    scalar_values = [exact.get("objective"), exact.get("accuracy")]
    for head in exact.get("heads", {}).values():
        scalar_values.extend(
            [head.get("token_weighted_cross_entropy"), head.get("token_weighted_accuracy")]
        )
    if not scalar_values or any(
        not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in scalar_values
    ):
        raise RuntimeError(f"non-finite exact validation metric: {path}")
    return payload


def select_checkpoint(
    baseline: dict[str, Any], candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    baseline_heads = baseline["exact_token_weighted_metrics"]["heads"]
    rows = []
    for payload in candidates:
        exact = payload["exact_token_weighted_metrics"]
        drops = {
            name: baseline_heads[name]["token_weighted_accuracy"]
            - exact["heads"][name]["token_weighted_accuracy"]
            for name in STATE_HEADS
        }
        eligible = all(value <= 0.05 for value in drops.values())
        rows.append(
            {
                "local_step": int(payload["local_step"]),
                "estimated_total_optimizer_step": int(
                    payload["estimated_total_optimizer_step"]
                ),
                "token_weighted_objective": float(exact["objective"]),
                "token_weighted_accuracy": float(exact["accuracy"]),
                "state_accuracy_drops_vs_step0": drops,
                "guard_passed": eligible,
                "eligible": eligible,
            }
        )
    eligible_rows = [row for row in rows if row["eligible"]]
    if not eligible_rows:
        return {
            "selected_local_step": 0,
            "selected_estimated_total_optimizer_step": int(
                baseline["estimated_total_optimizer_step"]
            ),
            "reason": "no continuation checkpoint passed the predeclared 5pp state-head guard",
            "one_percent_earlier_tie_break_used": False,
            "candidates": rows,
        }
    minimum = min(row["token_weighted_objective"] for row in eligible_rows)
    tie_pool = [
        row
        for row in eligible_rows
        if row["token_weighted_objective"] <= minimum * 1.01
    ]
    selected = min(tie_pool, key=lambda row: row["local_step"])
    return {
        "selected_local_step": selected["local_step"],
        "selected_estimated_total_optimizer_step": selected[
            "estimated_total_optimizer_step"
        ],
        "reason": (
            "earliest eligible checkpoint within 1% of the lowest exact "
            "token-weighted validation objective"
        ),
        "one_percent_earlier_tie_break_used": (
            selected["token_weighted_objective"] > minimum
            or len(tie_pool) > 1
        ),
        "minimum_eligible_objective": minimum,
        "candidates": rows,
    }


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite validation index: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--checkpoint-result", type=Path, action="append", required=True)
    parser.add_argument("--expected-step", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected_steps = sorted(args.expected_step)
    if (
        not expected_steps
        or any(step <= 0 for step in expected_steps)
        or len(expected_steps) != len(set(expected_steps))
    ):
        raise ValueError("expected steps must be unique positive integers")
    baseline_path = args.baseline.resolve(strict=True)
    baseline = load_result(baseline_path)
    if baseline.get("local_step") != 0:
        raise RuntimeError("group-validation baseline is not step 0")
    result_pairs = [
        (path.resolve(strict=True), load_result(path.resolve(strict=True)))
        for path in args.checkpoint_result
    ]
    result_pairs.sort(key=lambda pair: pair[1]["local_step"])
    result_paths = [pair[0] for pair in result_pairs]
    candidates = [pair[1] for pair in result_pairs]
    actual_steps = [int(payload["local_step"]) for payload in candidates]
    if actual_steps != expected_steps:
        raise RuntimeError(
            f"group-validation step grid mismatch: actual={actual_steps} expected={expected_steps}"
        )

    split_identity = baseline["split_manifest"]["split_identity_sha256"]
    base_sha = baseline["base_checkpoint"]["sha256"]
    for payload in candidates:
        continuation = payload.get("continuation_checkpoint")
        if not isinstance(continuation, dict) or continuation.get("status") != "accepted":
            raise RuntimeError("continuation checkpoint audit is absent or rejected")
        if continuation.get("local_step") != payload["local_step"]:
            raise RuntimeError("continuation checkpoint/result step mismatch")
        if payload["split_manifest"]["split_identity_sha256"] != split_identity:
            raise RuntimeError("group-validation split identity drift")
        if payload["base_checkpoint"]["sha256"] != base_sha:
            raise RuntimeError("group-validation official base identity drift")
        if payload["validation_data"]["row_count"] != baseline["validation_data"]["row_count"]:
            raise RuntimeError("group-validation row-count drift")

    selection = select_checkpoint(baseline, candidates)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "profile": "official-group-validation-exact-token-weighted-sweep-v1",
        "generated_at_utc": utc_now(),
        "selection_source": "frozen_internal_validation_only",
        "table3_used_for_selection": False,
        "selection_rule": {
            "nonfinite": "reject",
            "state_head_guard": "no exact token-weighted state-head accuracy may drop by more than 5 percentage points versus step 0",
            "objective": "lowest exact per-row target-weighted validation cross-entropy over all seven heads",
            "tie_break": "if eligible objectives are within 1% of the minimum, select the earlier local step",
            "fallback": "retain official step 0 if no continuation checkpoint passes the guard",
        },
        "expected_steps": expected_steps,
        "split_identity_sha256": split_identity,
        "official_base_checkpoint_sha256": base_sha,
        "baseline": {
            "path": str(baseline_path),
            "sha256": sha256_file(baseline_path),
            "metrics": baseline["metrics"],
            "exact_token_weighted_metrics": baseline[
                "exact_token_weighted_metrics"
            ],
        },
        "checkpoints": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "local_step": result["local_step"],
                "estimated_total_optimizer_step": result[
                    "estimated_total_optimizer_step"
                ],
                "continuation_checkpoint": result["continuation_checkpoint"],
                "metrics": result["metrics"],
                "exact_token_weighted_metrics": result[
                    "exact_token_weighted_metrics"
                ],
            }
            for path, result in zip(result_paths, candidates)
        ],
        "selection": selection,
    }
    atomic_json_write(args.output.absolute(), payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
