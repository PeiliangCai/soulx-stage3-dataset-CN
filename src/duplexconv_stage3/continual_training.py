"""Reproducible data splitting and bookkeeping for Stage 3 continuation."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Any, Iterable


STATE_TOKENS = (
    "user_idle",
    "user_nonidle",
    "user_backchannel",
    "user_complete",
    "user_incomplete",
)
STATE_PATTERN = re.compile(
    r"<\|(" + "|".join(re.escape(value) for value in STATE_TOKENS) + r")\|>"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def continuation_learning_rate(
    peak_lr: float,
    origin_step_estimate: int,
    local_step: int,
    rewarm_steps: int,
) -> float:
    """Offset inverse-square-root decay with a short optimizer-state rewarm."""

    if peak_lr <= 0 or origin_step_estimate <= 0 or local_step <= 0:
        raise ValueError("peak_lr, origin_step_estimate and local_step must be positive")
    if rewarm_steps <= 0:
        raise ValueError("rewarm_steps must be positive")
    ramp = min(1.0, local_step / rewarm_steps)
    offset_decay = math.sqrt(
        origin_step_estimate / (origin_step_estimate + local_step)
    )
    return peak_lr * ramp * offset_decay


def select_lr_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Select LR from validation only using a rule frozen before Table 3."""

    if len(candidates) != 2:
        raise ValueError("exactly two LR candidates are required")
    evaluated = []
    for item in candidates:
        baseline = item["baseline"]
        final = item["final"]
        state_names = (
            "idle",
            "nonidle",
            "user_complete",
            "user_incomplete",
            "user_backchannel",
        )
        checkpoints = item.get(
            "checkpoints",
            [{"local_step": item.get("final_local_step", 0), "metrics": final}],
        )
        guard_trajectory = []
        for checkpoint in checkpoints:
            metrics = checkpoint["metrics"]
            drops_at_step = {
                name: baseline["heads"][name]["accuracy"]
                - metrics["heads"][name]["accuracy"]
                for name in state_names
            }
            guard_trajectory.append(
                {
                    "local_step": checkpoint["local_step"],
                    "state_accuracy_drops": drops_at_step,
                    "guard_passed": all(value <= 0.05 for value in drops_at_step.values()),
                }
            )
        drops = guard_trajectory[-1]["state_accuracy_drops"]
        nonfinite = not math.isfinite(final["token_weighted_objective"])
        collapse = any(value > 0.05 for value in drops.values())
        evaluated.append(
            {
                **item,
                "state_accuracy_drops": drops,
                "guard_trajectory": guard_trajectory,
                "last_guard_safe_step": max(
                    checkpoint["local_step"]
                    for checkpoint in guard_trajectory
                    if checkpoint["guard_passed"]
                ),
                "eligible": not nonfinite and not collapse,
                "ineligibility_reasons": [
                    reason
                    for condition, reason in (
                        (nonfinite, "non-finite final validation objective"),
                        (collapse, "one or more state-head accuracies dropped by >5pp"),
                    )
                    if condition
                ],
            }
        )
    eligible = [item for item in evaluated if item["eligible"]]
    eligibility_fallback_used = not eligible
    if eligible:
        pool = sorted(
            eligible,
            key=lambda item: (
                item["final"]["token_weighted_objective"],
                item["peak_lr"],
            ),
        )
        selected = pool[0]
        reason = "lowest final token-weighted validation objective among guard-passing candidates"
    else:
        pool = sorted(
            evaluated,
            key=lambda item: (-item["last_guard_safe_step"], item["peak_lr"]),
        )
        selected = pool[0]
        reason = "all candidates failed the final guard; selected longest guard-safe horizon, then lower LR"
    if eligible and len(pool) > 1:
        low, high = sorted(pool[:2], key=lambda item: item["peak_lr"])
        denominator = min(
            low["final"]["token_weighted_objective"],
            high["final"]["token_weighted_objective"],
        )
        relative_gap = abs(
            low["final"]["token_weighted_objective"]
            - high["final"]["token_weighted_objective"]
        ) / denominator
        if relative_gap <= 0.01:
            selected = low
            reason = "final objectives within 1%; conservative lower-LR tie-break"
    return {
        "selected_run_id": selected["run_id"],
        "selected_peak_lr": selected["peak_lr"],
        "selection_reason": reason,
        "eligibility_fallback_used": eligibility_fallback_used,
        "benchmark_used_for_selection": False,
        "candidates": evaluated,
    }


def atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def deterministic_group_split(
    source_ids: Iterable[str], validation_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Split complete source conversations with one predeclared RNG draw."""

    unique = sorted(set(source_ids))
    if not unique:
        raise ValueError("source_ids is empty")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    validation_count = round(len(unique) * validation_fraction)
    validation_count = max(1, min(len(unique) - 1, validation_count))
    shuffled = list(unique)
    random.Random(seed).shuffle(shuffled)
    validation = sorted(shuffled[:validation_count])
    validation_set = set(validation)
    train = sorted(item for item in unique if item not in validation_set)
    return train, validation


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    state_counts: Counter[str] = Counter()
    source_ids = set()
    view_ids = set()
    ntrack_rows: Counter[str] = Counter()
    qwen_event_ids = set()
    chunk_count = 0
    for row in rows:
        metadata = row["metadata"]
        source_ids.add(metadata["source_id"])
        view_ids.add(metadata["view_id"])
        ntrack_rows[str(metadata["source_ntrack"])] += 1
        qwen_event_ids.update(metadata["qwen_event_ids"])
        chunk_count += int(metadata["chunk_count"])
        state_counts.update(STATE_PATTERN.findall(row["sequence"]))
    return {
        "source_conversation_count": len(source_ids),
        "target_view_count": len(view_ids),
        "row_count": len(rows),
        "chunk_count": chunk_count,
        "duration_hours_from_160ms_chunks": chunk_count * 0.16 / 3600,
        "rows_by_source_ntrack": dict(sorted(ntrack_rows.items())),
        "state_token_counts": {
            token: state_counts[token] for token in STATE_TOKENS
        },
        "unique_qwen_labeled_event_count": len(qwen_event_ids),
        "row_index_identity_sha256": canonical_sha256(
            sorted(row["index"] for row in rows)
        ),
    }


def load_window_metadata(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            index = row.get("index")
            if not isinstance(index, str) or index in result:
                raise ValueError(f"invalid/duplicate window index at line {line_number}")
            result[index] = row
    if not result:
        raise ValueError("window metadata is empty")
    return result


def build_split_rows(
    indexes: list[str],
    sequences: list[str],
    metadata_by_index: dict[str, dict[str, Any]],
    validation_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if len(indexes) != len(sequences) or len(set(indexes)) != len(indexes):
        raise ValueError("dataset indexes/sequences are invalid")
    if set(indexes) != set(metadata_by_index):
        raise ValueError("Parquet and window metadata index sets differ")
    source_ids = [metadata_by_index[index]["source_id"] for index in indexes]
    train_sources, validation_sources = deterministic_group_split(
        source_ids, validation_fraction, seed
    )
    train_set = set(train_sources)
    validation_set = set(validation_sources)
    if train_set & validation_set or train_set | validation_set != set(source_ids):
        raise AssertionError("invalid source-conversation partition")
    train_rows = []
    validation_rows = []
    for index, sequence in zip(indexes, sequences):
        item = {
            "index": index,
            "sequence": sequence,
            "metadata": metadata_by_index[index],
        }
        target = train_rows if item["metadata"]["source_id"] in train_set else validation_rows
        target.append(item)
    manifest = {
        "schema_version": 1,
        "profile": "source-conversation-group-aware-random-v1",
        "seed": seed,
        "validation_fraction_requested": validation_fraction,
        "source_order_before_rng": "lexicographic",
        "rng": "python.random.Random(seed).shuffle",
        "train_source_ids": train_sources,
        "validation_source_ids": validation_sources,
        "source_leakage_count": len(train_set & validation_set),
        "train": summarize_rows(train_rows),
        "validation": summarize_rows(validation_rows),
    }
    manifest["split_identity_sha256"] = canonical_sha256(
        {
            "profile": manifest["profile"],
            "seed": seed,
            "train_source_ids": train_sources,
            "validation_source_ids": validation_sources,
            "train_rows": manifest["train"]["row_index_identity_sha256"],
            "validation_rows": manifest["validation"]["row_index_identity_sha256"],
        }
    )
    return train_rows, validation_rows, manifest
