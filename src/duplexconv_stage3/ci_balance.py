"""Deterministic Complete/Incomplete active-row balancing for Stage 3 data."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from typing import Any, Iterable

from duplexconv_stage3.continual_training import (
    STATE_PATTERN,
    STATE_TOKENS,
    canonical_sha256,
)


BALANCE_PROFILE = "ci-active-row-downsample-v1"
CATEGORY_BOTH = "complete_and_incomplete"
CATEGORY_COMPLETE_ONLY = "complete_only"
CATEGORY_INCOMPLETE_ONLY = "incomplete_only"
CATEGORY_NEITHER = "neither_complete_nor_incomplete"
CATEGORIES = (
    CATEGORY_BOTH,
    CATEGORY_COMPLETE_ONLY,
    CATEGORY_INCOMPLETE_ONLY,
    CATEGORY_NEITHER,
)


def sequence_state_counts(sequence: str) -> Counter[str]:
    """Count the five Stage 3 state tokens in one serialized sequence."""

    return Counter(STATE_PATTERN.findall(sequence))


def ci_category(state_counts: Counter[str]) -> str:
    """Classify a row by whether Complete and Incomplete are active."""

    has_complete = state_counts["user_complete"] > 0
    has_incomplete = state_counts["user_incomplete"] > 0
    if has_complete and has_incomplete:
        return CATEGORY_BOTH
    if has_complete:
        return CATEGORY_COMPLETE_ONLY
    if has_incomplete:
        return CATEGORY_INCOMPLETE_ONLY
    return CATEGORY_NEITHER


def largest_remainder_allocation(
    counts: dict[tuple[str, str, str], int], target: int
) -> tuple[dict[tuple[str, str, str], int], dict[tuple[str, str, str], int]]:
    """Allocate an exact integer target proportionally with stable tie breaks."""

    if not counts or any(value < 0 for value in counts.values()):
        raise ValueError("stratum counts must be non-empty and non-negative")
    total = sum(counts.values())
    if not 0 <= target <= total:
        raise ValueError("allocation target must be between zero and total")

    allocation: dict[tuple[str, str, str], int] = {}
    remainders: dict[tuple[str, str, str], int] = {}
    for key, count in counts.items():
        allocation[key], remainders[key] = divmod(target * count, total)

    remaining = target - sum(allocation.values())
    ranked = sorted(counts, key=lambda key: (-remainders[key], key))
    for key in ranked[:remaining]:
        allocation[key] += 1

    if sum(allocation.values()) != target:
        raise AssertionError("largest-remainder allocation did not close")
    if any(allocation[key] > counts[key] for key in counts):
        raise AssertionError("largest-remainder allocation exceeds a stratum")
    return allocation, remainders


def stable_seeded_rank(seed: int, index: str) -> str:
    """Return a cross-version deterministic ranking key for one row index."""

    return hashlib.sha256(f"{seed}\0{index}".encode("utf-8")).hexdigest()


def _stratum(metadata: dict[str, Any], has_backchannel: bool) -> tuple[str, str, str]:
    try:
        shard = str(metadata["source_shard"])
        ntrack = str(metadata["source_ntrack"])
    except KeyError as error:
        raise ValueError(f"metadata lacks balancing field: {error.args[0]}") from error
    return shard, ntrack, "backchannel" if has_backchannel else "no_backchannel"


def _row_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    category_counts: Counter[str] = Counter()
    state_counts: Counter[str] = Counter()
    source_ids: set[str] = set()
    chunk_count = 0
    row_indexes = []
    for row in rows:
        category_counts[row["category"]] += 1
        state_counts.update(row["state_counts"])
        source_ids.add(str(row["metadata"]["source_id"]))
        chunk_count += int(row["metadata"]["chunk_count"])
        row_indexes.append(row["index"])
    complete_active = category_counts[CATEGORY_BOTH] + category_counts[CATEGORY_COMPLETE_ONLY]
    incomplete_active = category_counts[CATEGORY_BOTH] + category_counts[CATEGORY_INCOMPLETE_ONLY]
    return {
        "row_count": len(row_indexes),
        "row_index_identity_sha256": canonical_sha256(sorted(row_indexes)),
        "source_conversation_count": len(source_ids),
        "chunk_count": chunk_count,
        "duration_hours_from_160ms_chunks": chunk_count * 0.16 / 3600,
        "category_row_counts": {
            category: category_counts[category] for category in CATEGORIES
        },
        "complete_active_row_count": complete_active,
        "incomplete_active_row_count": incomplete_active,
        "state_token_counts": {token: state_counts[token] for token in STATE_TOKENS},
    }


def select_ci_balanced_rows(
    rows: list[dict[str, str]],
    metadata_by_index: dict[str, dict[str, Any]],
    seed: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Downsample Complete-only rows to the Incomplete-only active-row count.

    Rows containing both states, only Incomplete, or neither state are retained.
    Complete-only rows are selected proportionally by source shard, source channel
    count, and Backchannel presence. The returned row order matches the input.
    """

    indexes = [row.get("index") for row in rows]
    if not rows or any(not isinstance(index, str) for index in indexes):
        raise ValueError("rows must contain non-empty string indexes")
    if len(set(indexes)) != len(indexes):
        raise ValueError("row indexes must be unique")
    if set(indexes) != set(metadata_by_index):
        raise ValueError("row and metadata index sets differ")

    enriched = []
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        counts = sequence_state_counts(row["sequence"])
        category = ci_category(counts)
        item = {
            **row,
            "metadata": metadata_by_index[row["index"]],
            "state_counts": counts,
            "category": category,
        }
        enriched.append(item)
        by_category[category].append(item)

    complete_only = by_category[CATEGORY_COMPLETE_ONLY]
    incomplete_only = by_category[CATEGORY_INCOMPLETE_ONLY]
    target = len(incomplete_only)
    if target == 0:
        raise ValueError("cannot balance without Incomplete-only rows")
    if len(complete_only) < target:
        raise ValueError(
            "profile only downsamples Complete-only rows, but Complete-only is smaller"
        )

    rows_by_stratum: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in complete_only:
        key = _stratum(
            item["metadata"], item["state_counts"]["user_backchannel"] > 0
        )
        rows_by_stratum[key].append(item)
    available = {key: len(value) for key, value in rows_by_stratum.items()}
    allocation, remainders = largest_remainder_allocation(available, target)

    selected_complete_only_indexes: set[str] = set()
    strata_audit = []
    for key in sorted(rows_by_stratum):
        ranked = sorted(
            rows_by_stratum[key],
            key=lambda item: (stable_seeded_rank(seed, item["index"]), item["index"]),
        )
        chosen = ranked[: allocation[key]]
        selected_complete_only_indexes.update(item["index"] for item in chosen)
        strata_audit.append(
            {
                "source_shard": key[0],
                "source_ntrack": key[1],
                "backchannel_presence": key[2],
                "available_complete_only_rows": available[key],
                "selected_complete_only_rows": allocation[key],
                "largest_remainder_numerator_remainder": remainders[key],
                "selected_index_identity_sha256": canonical_sha256(
                    sorted(item["index"] for item in chosen)
                ),
            }
        )

    selected_enriched = [
        item
        for item in enriched
        if item["category"] != CATEGORY_COMPLETE_ONLY
        or item["index"] in selected_complete_only_indexes
    ]
    output_rows = [
        {"index": item["index"], "sequence": item["sequence"]}
        for item in selected_enriched
    ]
    input_summary = _row_summary(enriched)
    output_summary = _row_summary(selected_enriched)
    if output_summary["complete_active_row_count"] != output_summary[
        "incomplete_active_row_count"
    ]:
        raise AssertionError("Complete/Incomplete active-row balance did not close")
    if len(output_rows) != len({row["index"] for row in output_rows}):
        raise AssertionError("balanced output contains duplicate rows")

    audit = {
        "profile": BALANCE_PROFILE,
        "seed": seed,
        "selection_rank": "sha256(str(seed) + NUL + row_index), ascending",
        "allocation": (
            "largest remainder proportional to Complete-only stratum size; "
            "ties use lexicographic stratum order"
        ),
        "stratum_fields": [
            "source_shard",
            "source_ntrack",
            "backchannel_presence",
        ],
        "input": input_summary,
        "output": output_summary,
        "selected_complete_only_row_count": len(selected_complete_only_indexes),
        "selected_complete_only_index_identity_sha256": canonical_sha256(
            sorted(selected_complete_only_indexes)
        ),
        "removed_complete_only_row_count": len(complete_only)
        - len(selected_complete_only_indexes),
        "strata": strata_audit,
    }
    return output_rows, audit
