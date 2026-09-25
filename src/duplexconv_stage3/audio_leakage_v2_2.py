"""Separately versioned v2.2 benchmark-audio leakage decision rule.

The v2 compact-content representation remains an immutable dependency.  V2.2
changes only the Chromaprint near-match decision: an alignment must satisfy the
frozen bit-similarity, aligned-frame, and independently calibrated LSH-vote
requirements.  Exact PCM/content/window evidence is unchanged.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import io
import json
import math
from pathlib import Path
import tarfile
from typing import Any, Sequence

import numpy as np
import soundfile as sf

from . import audio_leakage as v1
from . import audio_leakage_v2 as v2


PROFILE = "duplexconv-benchmark-audio-leakage-v2.2"
CALIBRATION_PROFILE = "duplexconv-benchmark-audio-leakage-calibration-v2.2"
SPLIT_SEED = 20260822
CALIBRATION_FRACTION = 0.70


def next_power_of_two_strictly_above(value: int) -> int:
    if value < 0:
        raise ValueError("value must be non-negative")
    return 1 << int(value).bit_length()


def split_sample_keys(
    sample_keys: Sequence[str],
    *,
    seed: int = SPLIT_SEED,
    calibration_fraction: float = CALIBRATION_FRACTION,
) -> tuple[set[str], set[str]]:
    unique = sorted(set(sample_keys))
    if len(unique) < 2:
        raise ValueError("at least two unique sample keys are required")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration fraction must be between zero and one")
    ordered = sorted(
        unique,
        key=lambda key: (v1.sha256_bytes(f"{seed}\0{key}".encode("utf-8")), key),
    )
    cut = min(len(ordered) - 1, max(1, math.floor(len(ordered) * calibration_fraction)))
    return set(ordered[:cut]), set(ordered[cut:])


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


def _control_passes(
    match: v1.Match | None,
    *,
    similarity_threshold: float,
    minimum_aligned_frames: int,
    minimum_lsh_votes: int,
) -> bool:
    return bool(
        match is not None
        and match.similarity >= similarity_threshold
        and match.aligned_frames >= minimum_aligned_frames
        and match.votes >= minimum_lsh_votes
    )


def _split_name(sample_key: str, calibration_keys: set[str]) -> str:
    return "calibration" if sample_key in calibration_keys else "validation"


def calibrate_benchmark(
    records_path: Path,
    zip_root: Path,
    base_config_path: Path,
    base_calibration_path: Path,
    output_path: Path,
    *,
    split_seed: int = SPLIT_SEED,
    calibration_fraction: float = CALIBRATION_FRACTION,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite calibration: {output_path}")
    progress_path = output_path.parent / "calibration_run_manifest.json"
    if progress_path.exists():
        raise FileExistsError(f"refusing to overwrite progress: {progress_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = v2.load_records(records_path)
    base_config = v2._load_frozen_config(base_config_path)
    base_calibration = _load_json(base_calibration_path)
    if not base_calibration.get("calibration_passed"):
        raise RuntimeError("base v2.1 calibration did not pass")
    if (
        base_config.get("calibration_result_sha256")
        != v1.sha256_file(base_calibration_path)
    ):
        raise RuntimeError("base v2 config and calibration differ")
    if base_config["benchmark_records_sha256"] != v1.sha256_file(records_path):
        raise RuntimeError("base v2 config and benchmark records differ")

    similarity_threshold = float(
        base_config["near_duplicate_similarity_threshold"]
    )
    minimum_aligned_frames = int(base_config["minimum_aligned_frames"])
    identity_counts = Counter(
        record["content_pcm16_sha256"]
        for record in records
        if record["content_pcm16_sha256"] is not None
    )
    positive_eligible = [
        record
        for record in records
        if Path(record["member"]).name == "input.wav"
        and record["chromaprint_frame_count"]
        >= max(96, minimum_aligned_frames + 32)
        and identity_counts[record["content_pcm16_sha256"]] == 1
    ]
    negative_eligible_indices = [
        index
        for index, record in enumerate(records)
        if Path(record["member"]).name in {"input.wav", "clean_input.wav"}
        and record["chromaprint_frame_count"] >= minimum_aligned_frames
    ]
    relevant_keys = [record["sample_key"] for record in positive_eligible]
    relevant_keys.extend(records[index]["sample_key"] for index in negative_eligible_indices)
    calibration_keys, validation_keys = split_sample_keys(
        relevant_keys,
        seed=split_seed,
        calibration_fraction=calibration_fraction,
    )
    if calibration_keys & validation_keys:
        raise AssertionError("sample-key split overlaps")

    started = v1.utc_now()
    progress: dict[str, Any] = {
        "schema_version": 1,
        "profile": CALIBRATION_PROFILE,
        "status": "running",
        "started_at_utc": started,
        "updated_at_utc": started,
        "candidate_training_shard_used_for_calibration": False,
        "phase": "positive_controls",
        "positive_source_count_completed": 0,
        "positive_source_count_total": len(positive_eligible),
        "implementation_path": str(Path(__file__).resolve()),
        "implementation_sha256": v1.sha256_file(Path(__file__)),
        "v2_dependency_sha256": v1.sha256_file(Path(v2.__file__)),
        "v1_dependency_sha256": v1.sha256_file(Path(v1.__file__)),
    }
    v1.atomic_write_text(progress_path, json.dumps(progress, indent=2) + "\n")

    index = v1.FingerprintIndex(records)
    record_index_by_id = {
        record["record_id"]: index_value
        for index_value, record in enumerate(records)
    }
    chromaprint = v1.Chromaprint()
    positive_controls: list[dict[str, Any]] = []
    positive_eligible.sort(key=lambda item: (item["sample_key"], item["record_id"]))
    for completed, record in enumerate(positive_eligible, 1):
        audio, sample_rate = v1._load_zip_audio(zip_root, record)
        origin_index = record_index_by_id[record["record_id"]]
        for transform, transformed, transformed_rate in v2._positive_transforms(
            audio, int(sample_rate)
        ):
            query = v2._content_record(transformed, transformed_rate, chromaprint)
            origin = next(
                (
                    match
                    for match in index.query(
                        query["chromaprint"],
                        min_aligned_frames=minimum_aligned_frames,
                        max_results=80,
                    )
                    if match.record_index == origin_index
                ),
                None,
            )
            positive_controls.append(
                {
                    "split": _split_name(record["sample_key"], calibration_keys),
                    "sample_key": record["sample_key"],
                    "origin_record_id": record["record_id"],
                    "transform": transform,
                    "origin_retrieved": origin is not None,
                    "similarity": origin.similarity if origin else None,
                    "aligned_frames": origin.aligned_frames if origin else 0,
                    "lsh_votes": origin.votes if origin else 0,
                }
            )
        if completed % 10 == 0 or completed == len(positive_eligible):
            progress.update(
                {
                    "updated_at_utc": v1.utc_now(),
                    "positive_source_count_completed": completed,
                }
            )
            v1.atomic_write_text(progress_path, json.dumps(progress, indent=2) + "\n")

    window_sets = [_all_window_hashes(record) for record in records]
    negative_pairs: dict[tuple[int, int], dict[str, Any]] = {}
    progress.update(
        {
            "phase": "negative_controls",
            "negative_record_count_completed": 0,
            "negative_record_count_total": len(negative_eligible_indices),
        }
    )
    v1.atomic_write_text(progress_path, json.dumps(progress, indent=2) + "\n")
    for completed, left_index in enumerate(negative_eligible_indices, 1):
        left = records[left_index]
        left_split = _split_name(left["sample_key"], calibration_keys)
        for match in index.query(
            index.fingerprints[left_index],
            min_aligned_frames=minimum_aligned_frames,
            max_results=80,
        ):
            right_index = match.record_index
            right = records[right_index]
            if right_index == left_index or right["zip"] != left["zip"]:
                continue
            if right["sample_key"] == left["sample_key"]:
                continue
            right_split = _split_name(right["sample_key"], calibration_keys)
            if right_split != left_split:
                continue
            if right["content_pcm16_sha256"] == left["content_pcm16_sha256"]:
                continue
            if window_sets[left_index] & window_sets[right_index]:
                continue
            pair_key = tuple(sorted((left_index, right_index)))
            existing = negative_pairs.get(pair_key)
            if existing is None or match.similarity > existing["similarity"]:
                negative_pairs[pair_key] = {
                    "split": left_split,
                    "left_sample_key": left["sample_key"],
                    "right_sample_key": right["sample_key"],
                    "left_record_id": left["record_id"],
                    "right_record_id": right["record_id"],
                    "similarity": match.similarity,
                    "aligned_frames": match.aligned_frames,
                    "lsh_votes": match.votes,
                }
        if completed % 20 == 0 or completed == len(negative_eligible_indices):
            progress.update(
                {
                    "updated_at_utc": v1.utc_now(),
                    "negative_record_count_completed": completed,
                    "negative_pair_count": len(negative_pairs),
                }
            )
            v1.atomic_write_text(progress_path, json.dumps(progress, indent=2) + "\n")

    negative_controls = sorted(
        negative_pairs.values(),
        key=lambda item: (
            item["split"],
            item["left_record_id"],
            item["right_record_id"],
        ),
    )
    calibration_negatives = [
        item for item in negative_controls if item["split"] == "calibration"
    ]
    validation_negatives = [
        item for item in negative_controls if item["split"] == "validation"
    ]
    calibration_positives = [
        item for item in positive_controls if item["split"] == "calibration"
    ]
    validation_positives = [
        item for item in positive_controls if item["split"] == "validation"
    ]
    if not all(
        (
            calibration_negatives,
            validation_negatives,
            calibration_positives,
            validation_positives,
        )
    ):
        raise RuntimeError("calibration/validation controls must all be non-empty")
    calibration_negative_max_votes = max(
        int(item["lsh_votes"]) for item in calibration_negatives
    )
    minimum_lsh_votes = next_power_of_two_strictly_above(
        calibration_negative_max_votes
    )

    def positive_passes(item: dict[str, Any]) -> bool:
        if not item["origin_retrieved"]:
            return False
        return (
            float(item["similarity"]) >= similarity_threshold
            and int(item["aligned_frames"]) >= minimum_aligned_frames
            and int(item["lsh_votes"]) >= minimum_lsh_votes
        )

    def negative_gate_hit(item: dict[str, Any]) -> bool:
        return (
            float(item["similarity"]) >= similarity_threshold
            and int(item["aligned_frames"]) >= minimum_aligned_frames
            and int(item["lsh_votes"]) >= minimum_lsh_votes
        )

    calibration_positive_failures = sum(
        not positive_passes(item) for item in calibration_positives
    )
    validation_positive_failures = sum(
        not positive_passes(item) for item in validation_positives
    )
    calibration_negative_gate_hits = sum(
        negative_gate_hit(item) for item in calibration_negatives
    )
    validation_negative_gate_hits = sum(
        negative_gate_hit(item) for item in validation_negatives
    )
    validation_negative_max_votes = max(
        int(item["lsh_votes"]) for item in validation_negatives
    )
    split_audit = {
        "seed": split_seed,
        "calibration_fraction": calibration_fraction,
        "algorithm": (
            "sort unique sample_key by SHA256(seed + NUL + sample_key); "
            "first floor(N*fraction) keys calibrate, remaining keys validate"
        ),
        "calibration_sample_key_count": len(calibration_keys),
        "validation_sample_key_count": len(validation_keys),
        "overlap_count": len(calibration_keys & validation_keys),
        "calibration_sample_keys_sha256": v1.sha256_bytes(
            v1.canonical_json(sorted(calibration_keys)).encode("utf-8")
        ),
        "validation_sample_keys_sha256": v1.sha256_bytes(
            v1.canonical_json(sorted(validation_keys)).encode("utf-8")
        ),
    }
    calibration_passed = (
        split_audit["overlap_count"] == 0
        and calibration_positive_failures == 0
        and validation_positive_failures == 0
        and calibration_negative_gate_hits == 0
        and validation_negative_gate_hits == 0
        and validation_negative_max_votes < minimum_lsh_votes
        and base_calibration["short_padded_origin_detection_recall"] == 1.0
        and base_calibration["short_padded_cross_sample_collision_count"] == 0
        and base_calibration["short_padded_nonorigin_near_candidate_count"] == 0
        and base_calibration["silence_control"]
        == {
            "content_samples": 0,
            "content_window_count": 0,
            "chromaprint_frame_count": 0,
        }
    )
    result = {
        "schema_version": 1,
        "profile": CALIBRATION_PROFILE,
        "representation_profile": v2.PROFILE,
        "status": "complete",
        "started_at_utc": started,
        "completed_at_utc": v1.utc_now(),
        "candidate_training_shard_used_for_calibration": False,
        "records_path": str(records_path.resolve()),
        "records_sha256": v1.sha256_file(records_path),
        "zip_root": str(zip_root.resolve()),
        "base_v2_config_path": str(base_config_path.resolve()),
        "base_v2_config_sha256": v1.sha256_file(base_config_path),
        "base_v2_1_calibration_path": str(base_calibration_path.resolve()),
        "base_v2_1_calibration_sha256": v1.sha256_file(base_calibration_path),
        "split_audit": split_audit,
        "similarity_threshold_source": "unchanged frozen v2 threshold",
        "near_duplicate_similarity_threshold": similarity_threshold,
        "minimum_aligned_frames": minimum_aligned_frames,
        "minimum_lsh_votes_derivation": (
            "strictly next power of two above calibration negative maximum"
        ),
        "calibration_negative_maximum_lsh_votes": calibration_negative_max_votes,
        "minimum_lsh_votes": minimum_lsh_votes,
        "validation_negative_maximum_lsh_votes": validation_negative_max_votes,
        "positive_control_summary": {
            "calibration_count": len(calibration_positives),
            "validation_count": len(validation_positives),
            "calibration_failure_count": calibration_positive_failures,
            "validation_failure_count": validation_positive_failures,
            "minimum_similarity": min(
                float(item["similarity"])
                for item in positive_controls
                if item["similarity"] is not None
            ),
            "minimum_lsh_votes": min(
                int(item["lsh_votes"]) for item in positive_controls
            ),
        },
        "negative_control_summary": {
            "calibration_count": len(calibration_negatives),
            "validation_count": len(validation_negatives),
            "calibration_gate_hit_count": calibration_negative_gate_hits,
            "validation_gate_hit_count": validation_negative_gate_hits,
            "maximum_similarity": max(
                float(item["similarity"]) for item in negative_controls
            ),
            "maximum_lsh_votes": max(
                int(item["lsh_votes"]) for item in negative_controls
            ),
        },
        "short_and_silence_controls_source": "frozen v2.1 calibration",
        "short_padded_origin_detection_recall": base_calibration[
            "short_padded_origin_detection_recall"
        ],
        "short_padded_cross_sample_collision_count": base_calibration[
            "short_padded_cross_sample_collision_count"
        ],
        "short_padded_nonorigin_near_candidate_count": base_calibration[
            "short_padded_nonorigin_near_candidate_count"
        ],
        "silence_control": base_calibration["silence_control"],
        "calibration_passed": calibration_passed,
        "content_parameters": v2.content_parameters(),
        "implementation_path": str(Path(__file__).resolve()),
        "implementation_sha256": v1.sha256_file(Path(__file__)),
        "v2_dependency_path": str(Path(v2.__file__).resolve()),
        "v2_dependency_sha256": v1.sha256_file(Path(v2.__file__)),
        "v1_dependency_path": str(Path(v1.__file__).resolve()),
        "v1_dependency_sha256": v1.sha256_file(Path(v1.__file__)),
        "positive_controls": positive_controls,
        "negative_controls": negative_controls,
    }
    v1.atomic_write_text(output_path, json.dumps(result, indent=2) + "\n")
    progress = {
        **progress,
        "status": "complete",
        "phase": "complete",
        "updated_at_utc": result["completed_at_utc"],
        "completed_at_utc": result["completed_at_utc"],
        "calibration_result": str(output_path.resolve()),
        "calibration_result_sha256": v1.sha256_file(output_path),
        "calibration_passed": calibration_passed,
        "minimum_lsh_votes": minimum_lsh_votes,
    }
    v1.atomic_write_text(progress_path, json.dumps(progress, indent=2) + "\n")
    return result


def _load_frozen_config(path: Path) -> dict[str, Any]:
    config = _load_json(path)
    required = {
        "profile",
        "benchmark_records_sha256",
        "implementation_sha256",
        "v2_dependency_sha256",
        "v1_dependency_sha256",
        "base_v2_config_path",
        "base_v2_config_sha256",
        "near_duplicate_similarity_threshold",
        "minimum_aligned_frames",
        "minimum_lsh_votes",
        "content_parameters",
    }
    if not required.issubset(config):
        raise ValueError(f"frozen config missing keys: {sorted(required - set(config))}")
    if config["profile"] != PROFILE:
        raise ValueError("frozen config profile mismatch")
    if config["implementation_sha256"] != v1.sha256_file(Path(__file__)):
        raise ValueError("v2.2 implementation does not match frozen config")
    if config["v2_dependency_sha256"] != v1.sha256_file(Path(v2.__file__)):
        raise ValueError("v2 dependency does not match frozen config")
    if config["v1_dependency_sha256"] != v1.sha256_file(Path(v1.__file__)):
        raise ValueError("v1 dependency does not match frozen config")
    base_path = Path(config["base_v2_config_path"])
    if config["base_v2_config_sha256"] != v1.sha256_file(base_path):
        raise ValueError("base v2 frozen config does not match v2.2 config")
    if config["content_parameters"] != v2.content_parameters():
        raise ValueError("content parameters do not match frozen config")
    if int(config["minimum_lsh_votes"]) <= 0:
        raise ValueError("minimum LSH votes must be positive")
    return config


def score_candidate_tar(
    candidate_tar: Path,
    benchmark_records_path: Path,
    frozen_config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = _load_frozen_config(frozen_config_path)
    if v1.sha256_file(benchmark_records_path) != config["benchmark_records_sha256"]:
        raise RuntimeError("benchmark records do not match frozen config")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    records = v2.load_records(benchmark_records_path)
    index = v1.FingerprintIndex(records)
    exact: dict[str, list[int]] = defaultdict(list)
    normalized: dict[str, list[int]] = defaultdict(list)
    content_exact: dict[str, list[int]] = defaultdict(list)
    windows: dict[str, list[int]] = defaultdict(list)
    for index_value, record in enumerate(records):
        exact[record["pcm16k_mono_sha256"]].append(index_value)
        normalized[record["pcm16k_mono_peak_normalized_sha256"]].append(index_value)
        if record["content_pcm16_sha256"] is not None:
            content_exact[record["content_pcm16_sha256"]].append(index_value)
        for digest in _all_window_hashes(record):
            windows[digest].append(index_value)

    output_dir.mkdir(parents=True)
    candidates_path = output_dir / "candidate_identity_manifest.jsonl"
    matches_path = output_dir / "candidate_matches.jsonl"
    candidates_partial = candidates_path.with_suffix(candidates_path.suffix + ".partial")
    matches_partial = matches_path.with_suffix(matches_path.suffix + ".partial")
    chromaprint = v1.Chromaprint()
    counts: Counter[str] = Counter()
    quarantine: set[str] = set()
    with tarfile.open(candidate_tar, "r:*") as archive, candidates_partial.open(
        "w", encoding="utf-8"
    ) as candidates_handle, matches_partial.open("w", encoding="utf-8") as matches_handle:
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith(".wav"):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"cannot read tar member: {member.name}")
            audio, sample_rate = sf.read(
                io.BytesIO(extracted.read()), dtype="float32", always_2d=True
            )
            views: list[tuple[int | str, np.ndarray]] = [
                (channel, audio[:, channel : channel + 1])
                for channel in range(audio.shape[1])
            ]
            views.append(("mix", audio.mean(axis=1, keepdims=True)))
            for channel, view_audio in views:
                candidate = v2._candidate_record(
                    source_member=member.name,
                    channel=channel,
                    audio=view_audio,
                    sample_rate=int(sample_rate),
                    chromaprint=chromaprint,
                )
                candidate_id = v1.sha256_bytes(
                    f"{member.name}\0{channel}".encode("utf-8")
                )[:24]
                public = {
                    key: value for key, value in candidate.items() if key != "chromaprint"
                }
                public["candidate_id"] = candidate_id
                public["chromaprint_raw_u32_base64"] = v1.encode_fingerprint(
                    candidate["chromaprint"]
                )
                public["chromaprint_frame_count"] = int(candidate["chromaprint"].size)
                candidates_handle.write(v1.canonical_json(public) + "\n")
                counts["candidate_views"] += 1

                exact_indices = set(exact.get(candidate["pcm16k_mono_sha256"], ()))
                normalized_indices = set(
                    normalized.get(candidate["pcm16k_mono_peak_normalized_sha256"], ())
                )
                content_indices = (
                    set(content_exact.get(candidate["content_pcm16_sha256"], ()))
                    if candidate["content_pcm16_sha256"] is not None
                    else set()
                )
                window_indices: set[int] = set()
                for digest in _all_window_hashes(candidate):
                    window_indices.update(windows.get(digest, ()))
                near = index.query(
                    candidate["chromaprint"],
                    min_aligned_frames=int(config["minimum_aligned_frames"]),
                    max_results=5,
                )
                near_hits = [
                    match
                    for match in near
                    if _control_passes(
                        match,
                        similarity_threshold=float(
                            config["near_duplicate_similarity_threshold"]
                        ),
                        minimum_aligned_frames=int(config["minimum_aligned_frames"]),
                        minimum_lsh_votes=int(config["minimum_lsh_votes"]),
                    )
                ]
                if exact_indices:
                    counts["exact_audio_matches"] += 1
                if normalized_indices:
                    counts["normalized_audio_matches"] += 1
                if content_indices:
                    counts["content_exact_matches"] += 1
                if window_indices:
                    counts["content_window_matches"] += 1
                if near_hits:
                    counts["near_duplicate_matches"] += 1
                if (
                    exact_indices
                    or normalized_indices
                    or content_indices
                    or window_indices
                    or near_hits
                ):
                    quarantine.add(member.name)
                result = {
                    "candidate_id": candidate_id,
                    "source_member": member.name,
                    "channel": channel,
                    "exact_benchmark_record_ids": [
                        records[index_value]["record_id"]
                        for index_value in sorted(exact_indices)
                    ],
                    "normalized_benchmark_record_ids": [
                        records[index_value]["record_id"]
                        for index_value in sorted(normalized_indices)
                    ],
                    "content_exact_benchmark_record_ids": [
                        records[index_value]["record_id"]
                        for index_value in sorted(content_indices)
                    ],
                    "content_window_benchmark_record_ids": [
                        records[index_value]["record_id"]
                        for index_value in sorted(window_indices)
                    ],
                    "near_candidates": [
                        {
                            "benchmark_record_id": records[match.record_index][
                                "record_id"
                            ],
                            "benchmark_zip": records[match.record_index]["zip"],
                            "benchmark_member": records[match.record_index]["member"],
                            "similarity": match.similarity,
                            "aligned_frames": match.aligned_frames,
                            "offset": match.offset,
                            "votes": match.votes,
                            "passes_similarity": match.similarity
                            >= float(config["near_duplicate_similarity_threshold"]),
                            "passes_lsh_votes": match.votes
                            >= int(config["minimum_lsh_votes"]),
                            "is_gate_hit": match in near_hits,
                        }
                        for match in near
                    ],
                }
                matches_handle.write(v1.canonical_json(result) + "\n")
    candidates_partial.replace(candidates_path)
    matches_partial.replace(matches_path)
    manifest = {
        "schema_version": 1,
        "profile": PROFILE,
        "completed_at_utc": v1.utc_now(),
        "candidate_tar": str(candidate_tar.resolve()),
        "candidate_tar_bytes": candidate_tar.stat().st_size,
        "candidate_tar_sha256": v1.sha256_file(candidate_tar),
        "benchmark_records": str(benchmark_records_path.resolve()),
        "benchmark_records_sha256": v1.sha256_file(benchmark_records_path),
        "frozen_config": str(frozen_config_path.resolve()),
        "frozen_config_sha256": v1.sha256_file(frozen_config_path),
        "near_duplicate_similarity_threshold": float(
            config["near_duplicate_similarity_threshold"]
        ),
        "minimum_aligned_frames": int(config["minimum_aligned_frames"]),
        "minimum_lsh_votes": int(config["minimum_lsh_votes"]),
        "counts": dict(counts),
        "quarantined_source_member_count": len(quarantine),
        "quarantined_source_members": sorted(quarantine),
        "candidate_identity_manifest": str(candidates_path.resolve()),
        "candidate_identity_manifest_sha256": v1.sha256_file(candidates_path),
        "candidate_matches": str(matches_path.resolve()),
        "candidate_matches_sha256": v1.sha256_file(matches_path),
        "gate_passed": not quarantine,
    }
    v1.atomic_write_text(
        output_dir / "run_manifest.json", json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--records", type=Path, required=True)
    calibrate.add_argument("--zip-root", type=Path, required=True)
    calibrate.add_argument("--base-config", type=Path, required=True)
    calibrate.add_argument("--base-calibration", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    calibrate.add_argument(
        "--calibration-fraction", type=float, default=CALIBRATION_FRACTION
    )
    score = subparsers.add_parser("score-tar")
    score.add_argument("--candidate-tar", type=Path, required=True)
    score.add_argument("--benchmark-records", type=Path, required=True)
    score.add_argument("--frozen-config", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "calibrate":
        result = calibrate_benchmark(
            args.records,
            args.zip_root,
            args.base_config,
            args.base_calibration,
            args.output,
            split_seed=args.split_seed,
            calibration_fraction=args.calibration_fraction,
        )
    elif args.command == "score-tar":
        result = score_candidate_tar(
            args.candidate_tar,
            args.benchmark_records,
            args.frozen_config,
            args.output_dir,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
