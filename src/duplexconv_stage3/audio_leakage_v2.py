"""Low-information-aware benchmark audio leakage gate.

Version 1 is intentionally left untouched as immutable evidence.  This module
builds a separate benchmark manifest after removing inactive 20 ms frames from
the content representation.  Pure silence cannot create exact-window or
Chromaprint evidence, while short active utterances remain detectable through
gain-invariant content hashes and one-second windows.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import io
import json
import math
from pathlib import Path
import random
import tarfile
from typing import Any, Iterable, Iterator, Sequence
import zipfile

import numpy as np
import soundfile as sf

from . import audio_leakage as v1


PROFILE = "duplexconv-benchmark-audio-leakage-v2"
CANONICAL_SAMPLE_RATE = v1.CANONICAL_SAMPLE_RATE
ACTIVITY_FRAME_MS = 20
ACTIVITY_FRAME_SAMPLES = CANONICAL_SAMPLE_RATE * ACTIVITY_FRAME_MS // 1000
ACTIVITY_RMS_AFTER_PEAK_NORMALIZATION = 0.01
ACTIVITY_EXPANSION_FRAMES = 0
CONTENT_WINDOWS = ((1.0, 0.25), (4.0, 1.0))
CHROMAPRINT_ALGORITHM = v1.CHROMAPRINT_ALGORITHM
LSH_BITS = v1.LSH_BITS
LSH_SHIFTS = v1.LSH_SHIFTS
LSH_MAX_POSTINGS = v1.LSH_MAX_POSTINGS
DEFAULT_MIN_ALIGNED_FRAMES = 64


def content_parameters() -> dict[str, Any]:
    return {
        "canonical_sample_rate": CANONICAL_SAMPLE_RATE,
        "activity_frame_ms": ACTIVITY_FRAME_MS,
        "activity_rms_after_peak_normalization": (
            ACTIVITY_RMS_AFTER_PEAK_NORMALIZATION
        ),
        "activity_expansion_frames": ACTIVITY_EXPANSION_FRAMES,
        "content_windows": [
            {"seconds": seconds, "hop_seconds": hop}
            for seconds, hop in CONTENT_WINDOWS
        ],
        "chromaprint_algorithm": CHROMAPRINT_ALGORITHM,
        "lsh_bits": LSH_BITS,
        "lsh_shifts": list(LSH_SHIFTS),
        "lsh_max_postings": LSH_MAX_POSTINGS,
    }


def compact_active_pcm16(canonical_pcm: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    value = np.asarray(canonical_pcm, dtype=np.int16)
    if value.ndim != 1:
        raise ValueError("canonical PCM must be mono")
    if value.size == 0:
        return value.astype("<i2", copy=True), {
            "input_samples": 0,
            "active_frame_count": 0,
            "total_frame_count": 0,
            "content_samples": 0,
            "content_seconds": 0.0,
        }

    normalized = v1.peak_normalized_pcm16(value.astype(np.float32) / 32768.0)
    frame_count = math.ceil(normalized.size / ACTIVITY_FRAME_SAMPLES)
    padded = np.pad(
        normalized.astype(np.float32) / 32768.0,
        (0, frame_count * ACTIVITY_FRAME_SAMPLES - normalized.size),
    )
    frames = padded.reshape(frame_count, ACTIVITY_FRAME_SAMPLES)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    active = rms >= ACTIVITY_RMS_AFTER_PEAK_NORMALIZATION
    if ACTIVITY_EXPANSION_FRAMES:
        expanded = active.copy()
        for distance in range(1, ACTIVITY_EXPANSION_FRAMES + 1):
            expanded[distance:] |= active[:-distance]
            expanded[:-distance] |= active[distance:]
        active = expanded

    pieces = []
    for frame_index in np.flatnonzero(active):
        start = int(frame_index) * ACTIVITY_FRAME_SAMPLES
        stop = min(normalized.size, start + ACTIVITY_FRAME_SAMPLES)
        if stop > start:
            pieces.append(normalized[start:stop])
    compact = (
        np.concatenate(pieces).astype("<i2", copy=False)
        if pieces
        else np.empty(0, dtype="<i2")
    )
    return compact, {
        "input_samples": int(value.size),
        "active_frame_count": int(active.sum()),
        "total_frame_count": int(frame_count),
        "content_samples": int(compact.size),
        "content_seconds": compact.size / CANONICAL_SAMPLE_RATE,
    }


def content_window_hashes(content_pcm: np.ndarray) -> dict[str, list[str]]:
    value = np.asarray(content_pcm, dtype=np.int16)
    result: dict[str, list[str]] = {}
    if value.size == 0:
        return {
            f"{seconds:g}s_hop_{hop:g}s": [] for seconds, hop in CONTENT_WINDOWS
        }
    for seconds, hop_seconds in CONTENT_WINDOWS:
        key = f"{seconds:g}s_hop_{hop_seconds:g}s"
        window = int(seconds * CANONICAL_SAMPLE_RATE)
        hop = int(hop_seconds * CANONICAL_SAMPLE_RATE)
        if value.size < window:
            result[key] = []
            continue
        result[key] = [
            v1.sha256_bytes(value[start : start + window].tobytes())
            for start in range(0, value.size - window + 1, hop)
        ]
    return result


def _content_record(
    audio: np.ndarray, sample_rate: int, chromaprint: v1.Chromaprint
) -> dict[str, Any]:
    canonical = v1.canonical_pcm16_mono(audio, sample_rate)
    normalized_whole = v1.peak_normalized_pcm16(
        canonical.astype(np.float32) / 32768.0
    )
    content, activity = compact_active_pcm16(canonical)
    fingerprint = chromaprint.fingerprint(content, CANONICAL_SAMPLE_RATE)
    return {
        "pcm16k_mono_sha256": v1.sha256_bytes(canonical.tobytes()),
        "pcm16k_mono_peak_normalized_sha256": v1.sha256_bytes(
            normalized_whole.tobytes()
        ),
        "content_pcm16_sha256": (
            v1.sha256_bytes(content.tobytes()) if content.size else None
        ),
        "content_window_sha256": content_window_hashes(content),
        "activity": activity,
        "chromaprint": fingerprint,
    }


def fingerprint_record(
    *,
    audio: np.ndarray,
    sample_rate: int,
    raw_bytes: bytes,
    zip_relative: str,
    member: str,
    chromaprint: v1.Chromaprint,
) -> dict[str, Any]:
    content = _content_record(audio, sample_rate, chromaprint)
    channels = 1 if audio.ndim == 1 else int(audio.shape[1])
    frames = int(audio.shape[0])
    return {
        "record_id": v1._record_id(zip_relative, member),
        "zip": zip_relative,
        "member": member,
        "sample_key": v1._sample_key(member),
        "numeric_sample_id": v1._numeric_sample_id(member),
        "audio_file_sha256": v1.sha256_bytes(raw_bytes),
        "audio_file_bytes": len(raw_bytes),
        "sample_rate": sample_rate,
        "channels": channels,
        "frames": frames,
        "duration_seconds": frames / sample_rate,
        **{key: value for key, value in content.items() if key != "chromaprint"},
        "chromaprint_raw_u32_base64": v1.encode_fingerprint(
            content["chromaprint"]
        ),
        "chromaprint_frame_count": int(content["chromaprint"].size),
    }


def load_records(path: Path) -> list[dict[str, Any]]:
    return v1.load_records(path)


def _all_window_hashes(record: dict[str, Any]) -> set[str]:
    return {
        digest
        for values in record["content_window_sha256"].values()
        for digest in values
    }


def build_benchmark_manifest(zip_root: Path, output_dir: Path) -> dict[str, Any]:
    checksums_path = zip_root / "checksums.sha256"
    expected = v1._parse_checksums(checksums_path)
    actual_zips = sorted([*zip_root.glob("v1.0/*.zip"), *zip_root.glob("v1.5/*.zip")])
    actual_relative = [str(path.relative_to(zip_root)) for path in actual_zips]
    if actual_relative != sorted(expected):
        raise RuntimeError("benchmark ZIP list does not match checksums.sha256")
    for path in actual_zips:
        relative = str(path.relative_to(zip_root))
        if v1.sha256_file(path) != expected[relative]:
            raise RuntimeError(f"benchmark ZIP hash mismatch: {relative}")

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    build_manifest_path = output_dir / "benchmark_build_manifest.json"
    records_path = output_dir / "benchmark_identity_manifest.jsonl"
    partial = records_path.with_suffix(records_path.suffix + ".partial")
    chromaprint = v1.Chromaprint()
    count = 0
    duration = 0.0
    content_duration = 0.0
    fingerprints = 0
    empty_content = 0
    content_hashes: set[str] = set()
    windows: set[str] = set()
    started = v1.utc_now()
    progress: dict[str, Any] = {
        "schema_version": 2,
        "profile": PROFILE,
        "status": "running",
        "started_at_utc": started,
        "updated_at_utc": started,
        "zip_root": str(zip_root.resolve()),
        "zip_count": len(actual_zips),
        "audio_file_count_completed": 0,
        "implementation_path": str(Path(__file__).resolve()),
        "implementation_sha256": v1.sha256_file(Path(__file__)),
        "v1_dependency_sha256": v1.sha256_file(Path(v1.__file__)),
    }
    v1.atomic_write_text(build_manifest_path, json.dumps(progress, indent=2) + "\n")
    with partial.open("w", encoding="utf-8") as target:
        for zip_path in actual_zips:
            relative = str(zip_path.relative_to(zip_root))
            with zipfile.ZipFile(zip_path) as archive:
                for member in v1._iter_real_wavs(archive):
                    raw = archive.read(member)
                    audio, sample_rate = sf.read(
                        io.BytesIO(raw), dtype="float32", always_2d=True
                    )
                    record = fingerprint_record(
                        audio=audio,
                        sample_rate=int(sample_rate),
                        raw_bytes=raw,
                        zip_relative=relative,
                        member=member,
                        chromaprint=chromaprint,
                    )
                    target.write(v1.canonical_json(record) + "\n")
                    count += 1
                    duration += record["duration_seconds"]
                    content_duration += record["activity"]["content_seconds"]
                    fingerprints += record["chromaprint_frame_count"]
                    if record["content_pcm16_sha256"] is None:
                        empty_content += 1
                    else:
                        content_hashes.add(record["content_pcm16_sha256"])
                    windows.update(_all_window_hashes(record))
                    if count % 25 == 0:
                        progress.update(
                            {
                                "updated_at_utc": v1.utc_now(),
                                "audio_file_count_completed": count,
                                "current_zip": relative,
                                "current_member": member,
                            }
                        )
                        v1.atomic_write_text(
                            build_manifest_path, json.dumps(progress, indent=2) + "\n"
                        )
    partial.replace(records_path)
    manifest = {
        "schema_version": 2,
        "profile": PROFILE,
        "status": "complete",
        "started_at_utc": started,
        "completed_at_utc": v1.utc_now(),
        "zip_root": str(zip_root.resolve()),
        "zip_count": len(actual_zips),
        "zip_total_bytes": sum(path.stat().st_size for path in actual_zips),
        "zip_checksums_sha256": v1.sha256_file(checksums_path),
        "records_path": str(records_path.resolve()),
        "records_sha256": v1.sha256_file(records_path),
        "audio_file_count": count,
        "audio_duration_hours_with_repeated_components": duration / 3600.0,
        "content_duration_hours_with_repeated_components": content_duration / 3600.0,
        "empty_content_record_count": empty_content,
        "unique_content_identity_count": len(content_hashes),
        "unique_content_window_count": len(windows),
        "chromaprint_frame_count": fingerprints,
        "content_parameters": content_parameters(),
        "implementation_path": str(Path(__file__).resolve()),
        "implementation_sha256": v1.sha256_file(Path(__file__)),
        "v1_dependency_path": str(Path(v1.__file__).resolve()),
        "v1_dependency_sha256": v1.sha256_file(Path(v1.__file__)),
        "permanent_extracted_copy": False,
    }
    v1.atomic_write_text(build_manifest_path, json.dumps(manifest, indent=2) + "\n")
    return manifest


def _positive_transforms(
    audio: np.ndarray, sample_rate: int
) -> Iterator[tuple[str, np.ndarray, int]]:
    yield "gain_0.35", audio * 0.35, sample_rate
    yield "resample_12000", v1.resample_audio(audio, sample_rate, 12_000), 12_000
    padding_left = np.zeros((6 * sample_rate, audio.shape[1]), dtype=np.float32)
    padding_right = np.zeros((8 * sample_rate, audio.shape[1]), dtype=np.float32)
    yield "zero_pad_6s_8s", np.concatenate((padding_left, audio, padding_right)), sample_rate
    start = int(1.5 * sample_rate)
    end = int(1.0 * sample_rate)
    if audio.shape[0] > start + end + int(10.0 * sample_rate):
        yield "crop_1.5s_1.0s", audio[start : audio.shape[0] - end], sample_rate


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate percentile of empty sequence")
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def calibrate_benchmark(
    records_path: Path,
    zip_root: Path,
    output_path: Path,
    *,
    sample_count: int = 48,
    short_sample_count: int = 48,
    seed: int = 20260822,
    min_aligned_frames: int = DEFAULT_MIN_ALIGNED_FRAMES,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite calibration: {output_path}")
    records = load_records(records_path)
    index = v1.FingerprintIndex(records)
    identity_counts = Counter(
        record["content_pcm16_sha256"]
        for record in records
        if record["content_pcm16_sha256"] is not None
    )
    eligible = [
        record
        for record in records
        if Path(record["member"]).name == "input.wav"
        and record["chromaprint_frame_count"] >= max(96, min_aligned_frames + 32)
        and identity_counts[record["content_pcm16_sha256"]] == 1
    ]
    eligible.sort(key=lambda item: (item["content_pcm16_sha256"], item["record_id"]))
    rng = random.Random(seed)
    selected = rng.sample(eligible, min(sample_count, len(eligible)))
    chromaprint = v1.Chromaprint()
    positives: list[dict[str, Any]] = []
    for record in selected:
        audio, sample_rate = v1._load_zip_audio(zip_root, record)
        for transform, transformed, transformed_rate in _positive_transforms(
            audio, int(sample_rate)
        ):
            query = _content_record(transformed, transformed_rate, chromaprint)
            matches = index.query(
                query["chromaprint"],
                min_aligned_frames=min_aligned_frames,
                max_results=60,
            )
            origin = next(
                (
                    match
                    for match in matches
                    if records[match.record_index]["record_id"] == record["record_id"]
                ),
                None,
            )
            positives.append(
                {
                    "origin_record_id": record["record_id"],
                    "transform": transform,
                    "origin_retrieved": origin is not None,
                    "origin_similarity": origin.similarity if origin else None,
                    "origin_aligned_frames": origin.aligned_frames if origin else 0,
                    "query_content_seconds": query["activity"]["content_seconds"],
                }
            )

    negative_pairs: list[dict[str, Any]] = []
    negative_indices = [
        record_index
        for record_index, record in enumerate(records)
        if record["zip"] == "v1.0/candor_pause_handling.zip"
        and Path(record["member"]).name == "input.wav"
        and record["chromaprint_frame_count"] >= min_aligned_frames
    ]
    for record_index in negative_indices:
        origin = records[record_index]
        for match in index.query(
            index.fingerprints[record_index],
            min_aligned_frames=min_aligned_frames,
            max_results=80,
        ):
            candidate = records[match.record_index]
            if candidate["zip"] != origin["zip"]:
                continue
            if Path(candidate["member"]).name != "input.wav":
                continue
            if candidate.get("numeric_sample_id") == origin.get("numeric_sample_id"):
                continue
            if candidate["content_pcm16_sha256"] == origin["content_pcm16_sha256"]:
                continue
            negative_pairs.append(
                {
                    "left_record_id": origin["record_id"],
                    "right_record_id": candidate["record_id"],
                    "similarity": match.similarity,
                    "aligned_frames": match.aligned_frames,
                }
            )

    short_candidates = [
        record
        for record in records
        if record["zip"].startswith("v1.5/")
        and Path(record["member"]).name == "clean_input.wav"
        and 0 < record["activity"]["content_seconds"] <= 4.0
        and identity_counts[record["content_pcm16_sha256"]] == 1
    ]
    short_candidates.sort(
        key=lambda item: (item["content_pcm16_sha256"], item["record_id"])
    )
    selected_short = rng.sample(
        short_candidates, min(short_sample_count, len(short_candidates))
    )
    content_identity_to_records: dict[str, set[str]] = defaultdict(set)
    content_window_to_records: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record["content_pcm16_sha256"] is not None:
            content_identity_to_records[record["content_pcm16_sha256"]].add(
                record["record_id"]
            )
        for digest in _all_window_hashes(record):
            content_window_to_records[digest].add(record["record_id"])
    short_controls: list[dict[str, Any]] = []
    for record in selected_short:
        audio, sample_rate = v1._load_zip_audio(zip_root, record)
        left = np.zeros((7 * sample_rate, audio.shape[1]), dtype=np.float32)
        right = np.zeros((11 * sample_rate, audio.shape[1]), dtype=np.float32)
        padded = np.concatenate((left, audio, right))
        query = _content_record(padded, int(sample_rate), chromaprint)
        exact_ids = set(
            content_identity_to_records.get(query["content_pcm16_sha256"], ())
        )
        window_ids: set[str] = set()
        for digest in _all_window_hashes(query):
            window_ids.update(content_window_to_records.get(digest, ()))
        origin_detected = record["record_id"] in exact_ids or record["record_id"] in window_ids
        nonorigin = (exact_ids | window_ids) - {record["record_id"]}
        near = index.query(
            query["chromaprint"],
            min_aligned_frames=min_aligned_frames,
            max_results=80,
        )
        nonorigin_near = [
            match
            for match in near
            if records[match.record_index]["record_id"] != record["record_id"]
        ]
        short_controls.append(
            {
                "origin_record_id": record["record_id"],
                "origin_detected_by_content_exact_or_window": origin_detected,
                "nonorigin_exact_or_window_collision_count": len(nonorigin),
                "nonorigin_near_candidate_count": len(nonorigin_near),
                "maximum_nonorigin_near_similarity": max(
                    (item.similarity for item in nonorigin_near), default=None
                ),
                "query_chromaprint_frames": int(query["chromaprint"].size),
                "query_content_seconds": query["activity"]["content_seconds"],
            }
        )

    silence = _content_record(
        np.zeros((30 * CANONICAL_SAMPLE_RATE, 1), dtype=np.float32),
        CANONICAL_SAMPLE_RATE,
        chromaprint,
    )
    silence_control = {
        "content_samples": silence["activity"]["content_samples"],
        "content_window_count": len(_all_window_hashes(silence)),
        "chromaprint_frame_count": int(silence["chromaprint"].size),
    }
    positive_scores = [
        item["origin_similarity"]
        for item in positives
        if item["origin_similarity"] is not None
    ]
    negative_scores = [item["similarity"] for item in negative_pairs]
    negative_scores.extend(
        item["maximum_nonorigin_near_similarity"]
        for item in short_controls
        if item["maximum_nonorigin_near_similarity"] is not None
    )
    retrieval_recall = len(positive_scores) / len(positives) if positives else 0.0
    negative_max = max(negative_scores, default=0.0)
    recommended = math.ceil(max(0.80, negative_max + 0.02) * 200.0) / 200.0
    positive_p01 = _percentile(positive_scores, 0.01) if positive_scores else 0.0
    margin = positive_p01 - recommended
    short_origin_recall = (
        sum(item["origin_detected_by_content_exact_or_window"] for item in short_controls)
        / len(short_controls)
        if short_controls
        else 0.0
    )
    short_collision_count = sum(
        item["nonorigin_exact_or_window_collision_count"] for item in short_controls
    )
    calibration_passed = (
        retrieval_recall == 1.0
        and bool(negative_scores)
        and margin >= 0.01
        and short_origin_recall == 1.0
        and short_collision_count == 0
        and silence_control
        == {
            "content_samples": 0,
            "content_window_count": 0,
            "chromaprint_frame_count": 0,
        }
    )
    result = {
        "schema_version": 2,
        "profile": PROFILE,
        "status": "complete",
        "completed_at_utc": v1.utc_now(),
        "records_path": str(records_path.resolve()),
        "records_sha256": v1.sha256_file(records_path),
        "zip_root": str(zip_root.resolve()),
        "seed": seed,
        "candidate_training_shard_used_for_calibration": False,
        "long_positive_source_count": len(selected),
        "positive_query_count": len(positives),
        "positive_retrieval_recall": retrieval_recall,
        "positive_similarity": {
            "minimum": min(positive_scores, default=None),
            "p01": positive_p01,
            "median": _percentile(positive_scores, 0.5) if positive_scores else None,
        },
        "negative_similarity": {
            "count": len(negative_scores),
            "maximum": negative_max,
            "p99": _percentile(negative_scores, 0.99) if negative_scores else None,
            "median": _percentile(negative_scores, 0.5) if negative_scores else None,
        },
        "short_padded_control_count": len(short_controls),
        "short_padded_origin_detection_recall": short_origin_recall,
        "short_padded_nonorigin_exact_or_window_collision_count": short_collision_count,
        "silence_control": silence_control,
        "threshold_derivation": "ceil(max(0.80, max_negative+0.02) * 200) / 200",
        "recommended_near_duplicate_similarity": recommended,
        "minimum_aligned_frames": min_aligned_frames,
        "positive_p01_minus_recommended_margin": margin,
        "calibration_passed": calibration_passed,
        "content_parameters": content_parameters(),
        "implementation_path": str(Path(__file__).resolve()),
        "implementation_sha256": v1.sha256_file(Path(__file__)),
        "v1_dependency_path": str(Path(v1.__file__).resolve()),
        "v1_dependency_sha256": v1.sha256_file(Path(v1.__file__)),
        "positive_controls": positives,
        "negative_controls": negative_pairs,
        "short_padded_controls": short_controls,
    }
    v1.atomic_write_text(output_path, json.dumps(result, indent=2) + "\n")
    return result


def _load_frozen_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "profile",
        "benchmark_records_sha256",
        "implementation_sha256",
        "v1_dependency_sha256",
        "near_duplicate_similarity_threshold",
        "minimum_aligned_frames",
        "content_parameters",
    }
    if not required.issubset(config):
        raise ValueError(f"frozen config missing keys: {sorted(required - set(config))}")
    if config["profile"] != PROFILE:
        raise ValueError("frozen config profile mismatch")
    if config["implementation_sha256"] != v1.sha256_file(Path(__file__)):
        raise ValueError("v2 implementation does not match frozen config")
    if config["v1_dependency_sha256"] != v1.sha256_file(Path(v1.__file__)):
        raise ValueError("v1 dependency does not match frozen config")
    if config["content_parameters"] != content_parameters():
        raise ValueError("content parameters do not match frozen config")
    return config


def _candidate_record(
    *,
    source_member: str,
    channel: int | str,
    audio: np.ndarray,
    sample_rate: int,
    chromaprint: v1.Chromaprint,
) -> dict[str, Any]:
    record = _content_record(audio, sample_rate, chromaprint)
    return {
        "source_member": source_member,
        "channel": channel,
        "sample_rate": sample_rate,
        "frames": int(audio.shape[0]),
        "duration_seconds": audio.shape[0] / sample_rate,
        **record,
    }


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
    records = load_records(benchmark_records_path)
    index = v1.FingerprintIndex(records)
    exact = defaultdict(list)
    normalized = defaultdict(list)
    content_exact = defaultdict(list)
    windows = defaultdict(list)
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
    counts = Counter()
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
                candidate = _candidate_record(
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
                content_indices = set(
                    content_exact.get(candidate["content_pcm16_sha256"], ())
                ) if candidate["content_pcm16_sha256"] is not None else set()
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
                    if match.similarity
                    >= float(config["near_duplicate_similarity_threshold"])
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
                if exact_indices or normalized_indices or content_indices or window_indices or near_hits:
                    quarantine.add(member.name)
                result = {
                    "candidate_id": candidate_id,
                    "source_member": member.name,
                    "channel": channel,
                    "exact_benchmark_record_ids": [
                        records[i]["record_id"] for i in sorted(exact_indices)
                    ],
                    "normalized_benchmark_record_ids": [
                        records[i]["record_id"] for i in sorted(normalized_indices)
                    ],
                    "content_exact_benchmark_record_ids": [
                        records[i]["record_id"] for i in sorted(content_indices)
                    ],
                    "content_window_benchmark_record_ids": [
                        records[i]["record_id"] for i in sorted(window_indices)
                    ],
                    "near_candidates": [
                        {
                            "benchmark_record_id": records[match.record_index]["record_id"],
                            "benchmark_zip": records[match.record_index]["zip"],
                            "benchmark_member": records[match.record_index]["member"],
                            "similarity": match.similarity,
                            "aligned_frames": match.aligned_frames,
                            "offset": match.offset,
                            "votes": match.votes,
                            "is_gate_hit": match in near_hits,
                        }
                        for match in near
                    ],
                }
                matches_handle.write(v1.canonical_json(result) + "\n")
    candidates_partial.replace(candidates_path)
    matches_partial.replace(matches_path)
    manifest = {
        "schema_version": 2,
        "profile": PROFILE,
        "completed_at_utc": v1.utc_now(),
        "candidate_tar": str(candidate_tar.resolve()),
        "candidate_tar_bytes": candidate_tar.stat().st_size,
        "candidate_tar_sha256": v1.sha256_file(candidate_tar),
        "benchmark_records": str(benchmark_records_path.resolve()),
        "benchmark_records_sha256": v1.sha256_file(benchmark_records_path),
        "frozen_config": str(frozen_config_path.resolve()),
        "frozen_config_sha256": v1.sha256_file(frozen_config_path),
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
    build = subparsers.add_parser("build-benchmark")
    build.add_argument("--zip-root", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--records", type=Path, required=True)
    calibrate.add_argument("--zip-root", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--sample-count", type=int, default=48)
    calibrate.add_argument("--short-sample-count", type=int, default=48)
    calibrate.add_argument("--seed", type=int, default=20260822)
    calibrate.add_argument(
        "--minimum-aligned-frames", type=int, default=DEFAULT_MIN_ALIGNED_FRAMES
    )
    score = subparsers.add_parser("score-tar")
    score.add_argument("--candidate-tar", type=Path, required=True)
    score.add_argument("--benchmark-records", type=Path, required=True)
    score.add_argument("--frozen-config", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-benchmark":
        result = build_benchmark_manifest(args.zip_root, args.output_dir)
    elif args.command == "calibrate":
        result = calibrate_benchmark(
            args.records,
            args.zip_root,
            args.output,
            sample_count=args.sample_count,
            short_sample_count=args.short_sample_count,
            seed=args.seed,
            min_aligned_frames=args.minimum_aligned_frames,
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
