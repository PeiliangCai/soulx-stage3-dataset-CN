#!/usr/bin/env python3
"""Finalize v2 leakage calibration without using candidate training audio.

The first v2 calibration treated the expected overlap between ``clean_input``
and ``input`` inside one benchmark sample as a cross-sample collision.  This
script preserves that raw result, audits every reported short-control overlap
against the benchmark manifest, and derives the numeric threshold from the
already frozen benchmark-only positive and negative controls.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from duplexconv_stage3 import audio_leakage as v1  # noqa: E402
from duplexconv_stage3 import audio_leakage_v2 as v2  # noqa: E402


CALIBRATION_PROFILE = "duplexconv-benchmark-audio-leakage-calibration-v2.1"
NEGATIVE_MARGIN = 0.02
THRESHOLD_QUANTUM = 0.005
REQUIRED_POSITIVE_P01_MARGIN = 0.01


def _all_window_hashes(record: dict[str, Any]) -> set[str]:
    return {
        digest
        for values in record["content_window_sha256"].values()
        for digest in values
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def finalize(raw_path: Path, records_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite calibration: {output_path}")
    raw = _load_json(raw_path)
    records = v2.load_records(records_path)
    records_sha256 = v1.sha256_file(records_path)
    if raw.get("status") != "complete":
        raise RuntimeError("raw calibration is not complete")
    if raw.get("candidate_training_shard_used_for_calibration") is not False:
        raise RuntimeError("raw calibration used candidate training data")
    if raw.get("records_sha256") != records_sha256:
        raise RuntimeError("raw calibration and benchmark records differ")
    if raw.get("implementation_sha256") != v1.sha256_file(Path(v2.__file__)):
        raise RuntimeError("v2 implementation changed after raw calibration")
    if raw.get("v1_dependency_sha256") != v1.sha256_file(Path(v1.__file__)):
        raise RuntimeError("v1 dependency changed after raw calibration")
    if raw.get("content_parameters") != v2.content_parameters():
        raise RuntimeError("content parameters changed after raw calibration")

    record_by_id = {record["record_id"]: record for record in records}
    content_to_records: dict[str, set[str]] = defaultdict(set)
    window_to_records: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record["content_pcm16_sha256"] is not None:
            content_to_records[record["content_pcm16_sha256"]].add(
                record["record_id"]
            )
        for digest in _all_window_hashes(record):
            window_to_records[digest].add(record["record_id"])

    audited_short_controls: list[dict[str, Any]] = []
    cross_sample_collisions = 0
    associated_component_overlaps = 0
    for control in raw["short_padded_controls"]:
        origin = record_by_id[control["origin_record_id"]]
        hit_ids = set(
            content_to_records.get(origin["content_pcm16_sha256"], ())
        )
        for digest in _all_window_hashes(origin):
            hit_ids.update(window_to_records.get(digest, ()))
        hit_ids.discard(origin["record_id"])
        if len(hit_ids) != control["nonorigin_exact_or_window_collision_count"]:
            raise RuntimeError(
                "cannot reproduce raw short-control collision count for "
                f"{origin['record_id']}"
            )
        associated_ids = sorted(
            record_id
            for record_id in hit_ids
            if record_by_id[record_id]["sample_key"] == origin["sample_key"]
        )
        cross_sample_ids = sorted(hit_ids - set(associated_ids))
        associated_component_overlaps += len(associated_ids)
        cross_sample_collisions += len(cross_sample_ids)
        audited_short_controls.append(
            {
                **control,
                "origin_sample_key": origin["sample_key"],
                "associated_component_overlap_record_ids": associated_ids,
                "cross_sample_collision_record_ids": cross_sample_ids,
                "cross_sample_collision_count": len(cross_sample_ids),
            }
        )

    positive = raw["positive_similarity"]
    negative = raw["negative_similarity"]
    negative_maximum = float(negative["maximum"])
    recommended = (
        math.ceil(
            (negative_maximum + NEGATIVE_MARGIN) / THRESHOLD_QUANTUM - 1e-12
        )
        * THRESHOLD_QUANTUM
    )
    positive_p01_margin = float(positive["p01"]) - recommended
    short_near_candidates = sum(
        int(control["nonorigin_near_candidate_count"])
        for control in audited_short_controls
    )
    silence_ok = raw["silence_control"] == {
        "content_samples": 0,
        "content_window_count": 0,
        "chromaprint_frame_count": 0,
    }
    calibration_passed = (
        raw["positive_retrieval_recall"] == 1.0
        and int(negative["count"]) > 0
        and positive_p01_margin >= REQUIRED_POSITIVE_P01_MARGIN
        and raw["short_padded_origin_detection_recall"] == 1.0
        and cross_sample_collisions == 0
        and short_near_candidates == 0
        and silence_ok
    )
    result = {
        "schema_version": 2,
        "representation_profile": raw["profile"],
        "calibration_profile": CALIBRATION_PROFILE,
        "status": "complete",
        "completed_at_utc": v1.utc_now(),
        "raw_calibration_path": str(raw_path.resolve()),
        "raw_calibration_sha256": v1.sha256_file(raw_path),
        "records_path": str(records_path.resolve()),
        "records_sha256": records_sha256,
        "candidate_training_shard_used_for_calibration": False,
        "logical_sample_identity": "benchmark record sample_key",
        "same_sample_component_overlap_policy": (
            "expected association; only cross-sample overlap is a collision"
        ),
        "positive_query_count": raw["positive_query_count"],
        "positive_retrieval_recall": raw["positive_retrieval_recall"],
        "positive_similarity": positive,
        "negative_similarity": negative,
        "short_padded_control_count": raw["short_padded_control_count"],
        "short_padded_origin_detection_recall": raw[
            "short_padded_origin_detection_recall"
        ],
        "short_padded_associated_component_overlap_count": (
            associated_component_overlaps
        ),
        "short_padded_cross_sample_collision_count": cross_sample_collisions,
        "short_padded_nonorigin_near_candidate_count": short_near_candidates,
        "silence_control": raw["silence_control"],
        "threshold_derivation": (
            "ceil((max_negative + 0.02) / 0.005) * 0.005; no inherited "
            "v1 floor because v2 uses a different compact-content representation"
        ),
        "negative_margin": NEGATIVE_MARGIN,
        "threshold_quantum": THRESHOLD_QUANTUM,
        "recommended_near_duplicate_similarity": recommended,
        "minimum_aligned_frames": raw["minimum_aligned_frames"],
        "positive_p01_minus_recommended_margin": positive_p01_margin,
        "required_positive_p01_margin": REQUIRED_POSITIVE_P01_MARGIN,
        "calibration_passed": calibration_passed,
        "content_parameters": raw["content_parameters"],
        "implementation_path": raw["implementation_path"],
        "implementation_sha256": raw["implementation_sha256"],
        "v1_dependency_path": raw["v1_dependency_path"],
        "v1_dependency_sha256": raw["v1_dependency_sha256"],
        "calibration_protocol_path": str(Path(__file__).resolve()),
        "calibration_protocol_sha256": v1.sha256_file(Path(__file__)),
        "short_padded_controls": audited_short_controls,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    v1.atomic_write_text(output_path, json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-calibration", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(args.raw_calibration, args.records, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
