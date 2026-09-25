"""Auditable audio leakage fingerprints for DuplexConv expansion.

The benchmark is read directly from its ZIP files.  No permanent extracted
copy is needed.  Near-duplicate retrieval uses the system Chromaprint library;
the decision threshold is deliberately supplied by a frozen configuration and
is never learned from a candidate training shard.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import random
import tarfile
from typing import Any, Iterable, Iterator, Sequence
import zipfile

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf


PROFILE = "duplexconv-benchmark-audio-leakage-v1"
CANONICAL_SAMPLE_RATE = 16_000
WINDOW_SECONDS = 4.0
WINDOW_HOP_SECONDS = 1.0
CHROMAPRINT_ALGORITHM = 1
LSH_BITS = 12
LSH_SHIFTS = (0, 5, 10, 15, 20)
LSH_MAX_POSTINGS = 512
DEFAULT_MIN_ALIGNED_FRAMES = 24
BIT_COUNTS = np.asarray([int(i).bit_count() for i in range(256)], dtype=np.uint8)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(text, encoding="utf-8")
    partial.replace(path)


def _as_float_mono(audio: np.ndarray) -> np.ndarray:
    value = np.asarray(audio, dtype=np.float32)
    if value.ndim == 1:
        return value
    if value.ndim != 2 or value.shape[1] < 1:
        raise ValueError(f"invalid audio shape: {value.shape}")
    return value.mean(axis=1, dtype=np.float32)


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    value = np.asarray(audio, dtype=np.float32)
    if source_rate == target_rate:
        return value.copy()
    ratio = Fraction(target_rate, source_rate)
    return resample_poly(value, ratio.numerator, ratio.denominator, axis=0).astype(
        np.float32, copy=False
    )


def float_to_pcm16(audio: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 32767.0 / 32768.0)
    return np.rint(clipped * 32768.0).astype("<i2")


def canonical_pcm16_mono(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    mono = _as_float_mono(audio)
    return float_to_pcm16(resample_audio(mono, sample_rate, CANONICAL_SAMPLE_RATE))


def peak_normalized_pcm16(audio: np.ndarray) -> np.ndarray:
    value = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(value))) if value.size else 0.0
    if peak > 0:
        value = value * (0.95 / peak)
    return float_to_pcm16(value)


def normalized_window_hashes(canonical_pcm: np.ndarray) -> list[str]:
    audio = np.asarray(canonical_pcm, dtype=np.int16).astype(np.float32) / 32768.0
    window = int(WINDOW_SECONDS * CANONICAL_SAMPLE_RATE)
    hop = int(WINDOW_HOP_SECONDS * CANONICAL_SAMPLE_RATE)
    if audio.size == 0:
        return []
    if audio.size < window:
        return [sha256_bytes(peak_normalized_pcm16(audio).tobytes())]
    starts = range(0, audio.size - window + 1, hop)
    return [
        sha256_bytes(peak_normalized_pcm16(audio[start : start + window]).tobytes())
        for start in starts
    ]


class Chromaprint:
    """Minimal ctypes binding to libchromaprint's raw fingerprint API."""

    def __init__(self, library: str = "libchromaprint.so.1") -> None:
        self.library_name = library
        self.lib = ctypes.CDLL(library)
        self.lib.chromaprint_new.argtypes = [ctypes.c_int]
        self.lib.chromaprint_new.restype = ctypes.c_void_p
        self.lib.chromaprint_start.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.lib.chromaprint_start.restype = ctypes.c_int
        self.lib.chromaprint_feed.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int16),
            ctypes.c_int,
        ]
        self.lib.chromaprint_feed.restype = ctypes.c_int
        self.lib.chromaprint_finish.argtypes = [ctypes.c_void_p]
        self.lib.chromaprint_finish.restype = ctypes.c_int
        self.lib.chromaprint_get_raw_fingerprint.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.chromaprint_get_raw_fingerprint.restype = ctypes.c_int
        self.lib.chromaprint_dealloc.argtypes = [ctypes.c_void_p]
        self.lib.chromaprint_free.argtypes = [ctypes.c_void_p]

    def fingerprint(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        value = np.asarray(audio)
        if value.ndim == 1:
            value = value[:, None]
        if value.ndim != 2 or value.shape[1] < 1:
            raise ValueError(f"invalid Chromaprint audio shape: {value.shape}")
        pcm = value if value.dtype == np.int16 else float_to_pcm16(value)
        pcm = np.ascontiguousarray(pcm, dtype=np.int16)
        context = self.lib.chromaprint_new(CHROMAPRINT_ALGORITHM)
        if not context:
            raise RuntimeError("chromaprint_new failed")
        pointer = ctypes.POINTER(ctypes.c_uint32)()
        size = ctypes.c_int()
        try:
            if not self.lib.chromaprint_start(context, sample_rate, pcm.shape[1]):
                raise RuntimeError("chromaprint_start failed")
            if not self.lib.chromaprint_feed(
                context,
                pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                int(pcm.size),
            ):
                raise RuntimeError("chromaprint_feed failed")
            if not self.lib.chromaprint_finish(context):
                raise RuntimeError("chromaprint_finish failed")
            if not self.lib.chromaprint_get_raw_fingerprint(
                context, ctypes.byref(pointer), ctypes.byref(size)
            ):
                raise RuntimeError("chromaprint_get_raw_fingerprint failed")
            if size.value == 0:
                return np.empty(0, dtype="<u4")
            return np.ctypeslib.as_array(pointer, shape=(size.value,)).astype(
                "<u4", copy=True
            )
        finally:
            if pointer:
                self.lib.chromaprint_dealloc(pointer)
            self.lib.chromaprint_free(context)


def encode_fingerprint(value: np.ndarray) -> str:
    return base64.b64encode(np.asarray(value, dtype="<u4").tobytes()).decode("ascii")


def decode_fingerprint(value: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(value), dtype="<u4").copy()


def _record_id(zip_relative: str, member: str) -> str:
    return sha256_bytes(f"{zip_relative}\0{member}".encode("utf-8"))[:24]


def _sample_key(member: str) -> str:
    parts = [part for part in Path(member).parts if part not in ("__MACOSX",)]
    return "/".join(parts[:2]) if len(parts) >= 2 else member


def _numeric_sample_id(member: str) -> str | None:
    parts = Path(member).parts
    if len(parts) >= 2 and parts[1].isdigit():
        return parts[1]
    return None


def fingerprint_record(
    *,
    audio: np.ndarray,
    sample_rate: int,
    raw_bytes: bytes,
    zip_relative: str,
    member: str,
    chromaprint: Chromaprint,
) -> dict[str, Any]:
    canonical = canonical_pcm16_mono(audio, sample_rate)
    normalized = peak_normalized_pcm16(canonical.astype(np.float32) / 32768.0)
    fingerprint = chromaprint.fingerprint(audio, sample_rate)
    channels = 1 if audio.ndim == 1 else int(audio.shape[1])
    frames = int(audio.shape[0])
    return {
        "record_id": _record_id(zip_relative, member),
        "zip": zip_relative,
        "member": member,
        "sample_key": _sample_key(member),
        "numeric_sample_id": _numeric_sample_id(member),
        "audio_file_sha256": sha256_bytes(raw_bytes),
        "audio_file_bytes": len(raw_bytes),
        "sample_rate": sample_rate,
        "channels": channels,
        "frames": frames,
        "duration_seconds": frames / sample_rate,
        "pcm16k_mono_sha256": sha256_bytes(canonical.tobytes()),
        "pcm16k_mono_peak_normalized_sha256": sha256_bytes(normalized.tobytes()),
        "normalized_window_sha256": normalized_window_hashes(canonical),
        "chromaprint_raw_u32_base64": encode_fingerprint(fingerprint),
        "chromaprint_frame_count": int(fingerprint.size),
    }


def _parse_checksums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        result[relative.strip()] = digest
    return result


def _iter_real_wavs(archive: zipfile.ZipFile) -> Iterator[str]:
    for name in archive.namelist():
        path = Path(name)
        if not name.lower().endswith(".wav"):
            continue
        if "__MACOSX" in path.parts or path.name.startswith("._"):
            continue
        yield name


def build_benchmark_manifest(zip_root: Path, output_dir: Path) -> dict[str, Any]:
    checksums_path = zip_root / "checksums.sha256"
    expected = _parse_checksums(checksums_path)
    actual_zips = sorted([*zip_root.glob("v1.0/*.zip"), *zip_root.glob("v1.5/*.zip")])
    actual_relative = [str(path.relative_to(zip_root)) for path in actual_zips]
    if actual_relative != sorted(expected):
        raise RuntimeError("benchmark ZIP list does not match checksums.sha256")
    for path in actual_zips:
        relative = str(path.relative_to(zip_root))
        if sha256_file(path) != expected[relative]:
            raise RuntimeError(f"benchmark ZIP hash mismatch: {relative}")

    output_dir.mkdir(parents=True, exist_ok=True)
    build_manifest_path = output_dir / "benchmark_build_manifest.json"
    records_path = output_dir / "benchmark_identity_manifest.jsonl"
    partial = records_path.with_suffix(records_path.suffix + ".partial")
    chromaprint = Chromaprint()
    count = 0
    duration = 0.0
    fingerprints = 0
    exact_hashes: set[str] = set()
    windows: set[str] = set()
    started = utc_now()
    progress: dict[str, Any] = {
        "schema_version": 1,
        "profile": PROFILE,
        "status": "running",
        "started_at_utc": started,
        "updated_at_utc": started,
        "zip_root": str(zip_root.resolve()),
        "zip_count": len(actual_zips),
        "zip_total_bytes": sum(path.stat().st_size for path in actual_zips),
        "audio_file_count_completed": 0,
        "current_zip": None,
        "current_member": None,
        "records_partial_path": str(partial.resolve()),
        "implementation_path": str(Path(__file__).resolve()),
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    atomic_write_text(build_manifest_path, json.dumps(progress, indent=2) + "\n")
    with partial.open("w", encoding="utf-8") as target:
        for zip_path in actual_zips:
            relative = str(zip_path.relative_to(zip_root))
            with zipfile.ZipFile(zip_path) as archive:
                for member in _iter_real_wavs(archive):
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
                    target.write(canonical_json(record) + "\n")
                    count += 1
                    duration += record["duration_seconds"]
                    fingerprints += record["chromaprint_frame_count"]
                    exact_hashes.add(record["pcm16k_mono_sha256"])
                    windows.update(record["normalized_window_sha256"])
                    if count % 25 == 0:
                        progress.update(
                            {
                                "updated_at_utc": utc_now(),
                                "audio_file_count_completed": count,
                                "current_zip": relative,
                                "current_member": member,
                                "audio_duration_hours_completed": duration / 3600.0,
                            }
                        )
                        atomic_write_text(
                            build_manifest_path, json.dumps(progress, indent=2) + "\n"
                        )
    partial.replace(records_path)

    manifest = {
        "schema_version": 1,
        "profile": PROFILE,
        "status": "complete",
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "zip_root": str(zip_root.resolve()),
        "zip_count": len(actual_zips),
        "zip_total_bytes": sum(path.stat().st_size for path in actual_zips),
        "zip_checksums_file": str(checksums_path.resolve()),
        "zip_checksums_sha256": sha256_file(checksums_path),
        "records_path": str(records_path.resolve()),
        "records_sha256": sha256_file(records_path),
        "audio_file_count": count,
        "audio_duration_hours_with_repeated_components": duration / 3600.0,
        "unique_pcm16k_mono_identity_count": len(exact_hashes),
        "unique_normalized_window_count": len(windows),
        "chromaprint_frame_count": fingerprints,
        "canonical_sample_rate": CANONICAL_SAMPLE_RATE,
        "window_seconds": WINDOW_SECONDS,
        "window_hop_seconds": WINDOW_HOP_SECONDS,
        "chromaprint": {
            "library": chromaprint.library_name,
            "algorithm": CHROMAPRINT_ALGORITHM,
            "raw_element_type": "little-endian uint32",
        },
        "permanent_extracted_copy": False,
        "implementation_path": str(Path(__file__).resolve()),
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    atomic_write_text(build_manifest_path, json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            records.append(record)
    return records


def bit_agreement(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.size == 0:
        raise ValueError("fingerprints must have the same non-zero shape")
    xor = np.bitwise_xor(left.astype("<u4", copy=False), right.astype("<u4", copy=False))
    differing = int(BIT_COUNTS[xor.view(np.uint8)].sum())
    return 1.0 - differing / (32.0 * left.size)


@dataclass(frozen=True)
class Match:
    record_index: int
    offset: int
    aligned_frames: int
    similarity: float
    votes: int


class FingerprintIndex:
    def __init__(self, records: Sequence[dict[str, Any]]) -> None:
        self.records = list(records)
        self.fingerprints = [
            decode_fingerprint(record["chromaprint_raw_u32_base64"])
            for record in self.records
        ]
        postings: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        mask = (1 << LSH_BITS) - 1
        for record_index, fingerprint in enumerate(self.fingerprints):
            for position, value in enumerate(fingerprint.tolist()):
                for shift in LSH_SHIFTS:
                    postings[(shift, (value >> shift) & mask)].append(
                        (record_index, position)
                    )
        self.postings = {
            key: value
            for key, value in postings.items()
            if len(value) <= LSH_MAX_POSTINGS
        }

    def query(
        self,
        fingerprint: np.ndarray,
        *,
        min_aligned_frames: int = DEFAULT_MIN_ALIGNED_FRAMES,
        max_results: int = 20,
        max_alignments: int = 4000,
    ) -> list[Match]:
        query = np.asarray(fingerprint, dtype="<u4")
        if query.size < min_aligned_frames:
            return []
        votes: Counter[tuple[int, int]] = Counter()
        mask = (1 << LSH_BITS) - 1
        for query_position, value in enumerate(query.tolist()):
            for shift in LSH_SHIFTS:
                for record_index, benchmark_position in self.postings.get(
                    (shift, (value >> shift) & mask), ()
                ):
                    votes[(record_index, benchmark_position - query_position)] += 1

        matches: list[Match] = []
        for (record_index, offset), vote_count in votes.most_common(max_alignments):
            benchmark = self.fingerprints[record_index]
            query_start = max(0, -offset)
            benchmark_start = max(0, offset)
            aligned = min(
                query.size - query_start, benchmark.size - benchmark_start
            )
            if aligned < min_aligned_frames:
                continue
            score = bit_agreement(
                query[query_start : query_start + aligned],
                benchmark[benchmark_start : benchmark_start + aligned],
            )
            matches.append(
                Match(record_index, offset, int(aligned), score, vote_count)
            )
        matches.sort(key=lambda item: (-item.similarity, -item.aligned_frames, -item.votes))
        best_per_record: list[Match] = []
        seen: set[int] = set()
        for match in matches:
            if match.record_index in seen:
                continue
            seen.add(match.record_index)
            best_per_record.append(match)
            if len(best_per_record) == max_results:
                break
        return best_per_record


def _load_zip_audio(zip_root: Path, record: dict[str, Any]) -> tuple[np.ndarray, int]:
    with zipfile.ZipFile(zip_root / record["zip"]) as archive:
        raw = archive.read(record["member"])
    return sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)


def _positive_transforms(
    audio: np.ndarray, sample_rate: int, seed: int
) -> Iterator[tuple[str, np.ndarray, int]]:
    yield "gain_0.35", audio * 0.35, sample_rate
    yield "resample_12000", resample_audio(audio, sample_rate, 12_000), 12_000
    mono_or_channels = np.asarray(audio, dtype=np.float32)
    rng = np.random.default_rng(seed)
    signal_power = float(np.mean(np.square(mono_or_channels)))
    if signal_power > 0:
        noise_power = signal_power / (10.0 ** (30.0 / 10.0))
        noise = rng.normal(0.0, math.sqrt(noise_power), mono_or_channels.shape).astype(
            np.float32
        )
        yield "noise_30db", np.clip(mono_or_channels + noise, -1.0, 1.0), sample_rate
    start = int(1.5 * sample_rate)
    end = int(1.0 * sample_rate)
    if audio.shape[0] > start + end + int(6.0 * sample_rate):
        yield "crop_1.5s_1.0s", audio[start : audio.shape[0] - end], sample_rate


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate percentile of empty sequence")
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def calibrate_benchmark(
    records_path: Path,
    zip_root: Path,
    output_path: Path,
    *,
    sample_count: int = 48,
    seed: int = 20260822,
    min_aligned_frames: int = DEFAULT_MIN_ALIGNED_FRAMES,
) -> dict[str, Any]:
    atomic_write_text(
        output_path,
        json.dumps(
            {
                "schema_version": 1,
                "profile": PROFILE,
                "status": "running",
                "started_at_utc": utc_now(),
                "records_path": str(records_path.resolve()),
                "zip_root": str(zip_root.resolve()),
                "seed": seed,
                "requested_sample_count": sample_count,
                "implementation_path": str(Path(__file__).resolve()),
                "implementation_sha256": sha256_file(Path(__file__)),
            },
            indent=2,
        )
        + "\n",
    )
    records = load_records(records_path)
    index = FingerprintIndex(records)
    eligible = [
        record
        for record in records
        if Path(record["member"]).name == "input.wav"
        and record["chromaprint_frame_count"] >= max(96, min_aligned_frames + 32)
    ]
    eligible.sort(key=lambda item: (item["pcm16k_mono_sha256"], item["record_id"]))
    unique: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for record in eligible:
        if record["pcm16k_mono_sha256"] in seen_hashes:
            continue
        seen_hashes.add(record["pcm16k_mono_sha256"])
        unique.append(record)
    rng = random.Random(seed)
    selected = rng.sample(unique, min(sample_count, len(unique)))
    chromaprint = Chromaprint()
    positives: list[dict[str, Any]] = []
    diagnostic_controls: list[dict[str, Any]] = []
    for sample_index, record in enumerate(selected):
        audio, sample_rate = _load_zip_audio(zip_root, record)
        for transform, transformed, transformed_rate in _positive_transforms(
            audio, int(sample_rate), seed + sample_index
        ):
            query_fp = chromaprint.fingerprint(transformed, transformed_rate)
            matches = index.query(
                query_fp,
                min_aligned_frames=min_aligned_frames,
                max_results=40,
            )
            origin = [
                match
                for match in matches
                if records[match.record_index]["record_id"] == record["record_id"]
            ]
            best_origin = origin[0] if origin else None
            control = {
                "origin_record_id": record["record_id"],
                "origin_zip": record["zip"],
                "origin_member": record["member"],
                "transform": transform,
                "query_chromaprint_frames": int(query_fp.size),
                "origin_retrieved": best_origin is not None,
                "origin_similarity": best_origin.similarity if best_origin else None,
                "origin_aligned_frames": best_origin.aligned_frames if best_origin else 0,
            }
            if transform == "noise_30db":
                diagnostic_controls.append(control)
            else:
                positives.append(control)

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
        origin_windows = set(origin["normalized_window_sha256"])
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
            if candidate["pcm16k_mono_sha256"] == origin["pcm16k_mono_sha256"]:
                continue
            if origin_windows.intersection(candidate["normalized_window_sha256"]):
                continue
            negative_pairs.append(
                {
                    "left_record_id": origin["record_id"],
                    "left_member": origin["member"],
                    "right_record_id": candidate["record_id"],
                    "right_member": candidate["member"],
                    "similarity": match.similarity,
                    "aligned_frames": match.aligned_frames,
                }
            )

    positive_scores = [
        item["origin_similarity"]
        for item in positives
        if item["origin_similarity"] is not None
    ]
    retrieval_recall = len(positive_scores) / len(positives) if positives else 0.0
    negative_scores = [item["similarity"] for item in negative_pairs]
    negative_max = max(negative_scores, default=0.0)
    recommended = math.ceil(max(0.80, negative_max + 0.02) * 200.0) / 200.0
    positive_p01 = percentile(positive_scores, 0.01) if positive_scores else 0.0
    margin = positive_p01 - recommended
    calibration_passed = (
        retrieval_recall == 1.0 and bool(negative_scores) and margin >= 0.01
    )
    result = {
        "schema_version": 1,
        "profile": PROFILE,
        "status": "complete",
        "completed_at_utc": utc_now(),
        "records_path": str(records_path.resolve()),
        "records_sha256": sha256_file(records_path),
        "zip_root": str(zip_root.resolve()),
        "seed": seed,
        "selection": (
            "deterministic random sample of unique input.wav identities with "
            ">=max(96, minimum_aligned_frames+32) Chromaprint frames"
        ),
        "selected_source_count": len(selected),
        "positive_query_count": len(positives),
        "positive_retrieval_recall": retrieval_recall,
        "positive_similarity": {
            "minimum": min(positive_scores, default=None),
            "p01": positive_p01,
            "p05": percentile(positive_scores, 0.05) if positive_scores else None,
            "median": percentile(positive_scores, 0.5) if positive_scores else None,
        },
        "negative_control_definition": (
            "distinct numeric IDs among natural candor_pause_handling/input.wav "
            "records, excluding identical canonical PCM and exact normalized windows"
        ),
        "negative_similarity": {
            "count": len(negative_scores),
            "maximum": negative_max,
            "p99": percentile(negative_scores, 0.99) if negative_scores else None,
            "median": percentile(negative_scores, 0.5) if negative_scores else None,
        },
        "threshold_derivation": "ceil(max(0.80, max_negative+0.02) * 200) / 200",
        "recommended_near_duplicate_similarity": recommended,
        "minimum_aligned_frames": min_aligned_frames,
        "positive_p01_minus_recommended_margin": margin,
        "calibration_passed": calibration_passed,
        "positive_controls": positives,
        "diagnostic_noise_controls_not_used_for_threshold": diagnostic_controls,
        "negative_controls": negative_pairs,
        "implementation_path": str(Path(__file__).resolve()),
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    atomic_write_text(output_path, json.dumps(result, indent=2) + "\n")
    return result


def _load_frozen_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "profile",
        "benchmark_records_sha256",
        "implementation_sha256",
        "near_duplicate_similarity_threshold",
        "minimum_aligned_frames",
        "fingerprint_parameters",
    }
    if not required.issubset(config):
        raise ValueError(f"frozen config missing keys: {sorted(required - set(config))}")
    if config["profile"] != PROFILE:
        raise ValueError("frozen config profile mismatch")
    if config["implementation_sha256"] != sha256_file(Path(__file__)):
        raise ValueError("audio leakage implementation does not match frozen config")
    expected_parameters = {
        "canonical_sample_rate": CANONICAL_SAMPLE_RATE,
        "window_seconds": WINDOW_SECONDS,
        "window_hop_seconds": WINDOW_HOP_SECONDS,
        "chromaprint_algorithm": CHROMAPRINT_ALGORITHM,
        "lsh_bits": LSH_BITS,
        "lsh_shifts": list(LSH_SHIFTS),
        "lsh_max_postings": LSH_MAX_POSTINGS,
    }
    if config["fingerprint_parameters"] != expected_parameters:
        raise ValueError("audio leakage parameters do not match frozen config")
    return config


def _candidate_record(
    *,
    source_member: str,
    channel: int | str,
    audio: np.ndarray,
    sample_rate: int,
    chromaprint: Chromaprint,
) -> dict[str, Any]:
    canonical = canonical_pcm16_mono(audio, sample_rate)
    normalized = peak_normalized_pcm16(canonical.astype(np.float32) / 32768.0)
    fingerprint = chromaprint.fingerprint(audio, sample_rate)
    return {
        "source_member": source_member,
        "channel": channel,
        "sample_rate": sample_rate,
        "frames": int(audio.shape[0]),
        "duration_seconds": audio.shape[0] / sample_rate,
        "pcm16k_mono_sha256": sha256_bytes(canonical.tobytes()),
        "pcm16k_mono_peak_normalized_sha256": sha256_bytes(normalized.tobytes()),
        "normalized_window_sha256": normalized_window_hashes(canonical),
        "chromaprint": fingerprint,
    }


def score_candidate_tar(
    candidate_tar: Path,
    benchmark_records_path: Path,
    frozen_config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = _load_frozen_config(frozen_config_path)
    if sha256_file(benchmark_records_path) != config["benchmark_records_sha256"]:
        raise RuntimeError("benchmark records do not match frozen config")
    records = load_records(benchmark_records_path)
    index = FingerprintIndex(records)
    exact = defaultdict(list)
    normalized = defaultdict(list)
    windows = defaultdict(list)
    for index_value, record in enumerate(records):
        exact[record["pcm16k_mono_sha256"]].append(index_value)
        normalized[record["pcm16k_mono_peak_normalized_sha256"]].append(index_value)
        for digest in record["normalized_window_sha256"]:
            windows[digest].append(index_value)

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / "candidate_identity_manifest.jsonl"
    matches_path = output_dir / "candidate_matches.jsonl"
    candidates_partial = candidates_path.with_suffix(candidates_path.suffix + ".partial")
    matches_partial = matches_path.with_suffix(matches_path.suffix + ".partial")
    chromaprint = Chromaprint()
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
            raw = extracted.read()
            audio, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
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
                candidate_id = sha256_bytes(
                    f"{member.name}\0{channel}".encode("utf-8")
                )[:24]
                candidate_public = {
                    key: value for key, value in candidate.items() if key != "chromaprint"
                }
                candidate_public["candidate_id"] = candidate_id
                candidate_public["chromaprint_frame_count"] = int(
                    candidate["chromaprint"].size
                )
                candidates_handle.write(canonical_json(candidate_public) + "\n")
                counts["candidate_views"] += 1

                exact_indices = set(exact.get(candidate["pcm16k_mono_sha256"], ()))
                normalized_indices = set(
                    normalized.get(candidate["pcm16k_mono_peak_normalized_sha256"], ())
                )
                window_indices: set[int] = set()
                for digest in candidate["normalized_window_sha256"]:
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
                if window_indices:
                    counts["window_matches"] += 1
                if near_hits:
                    counts["near_duplicate_matches"] += 1
                if exact_indices or normalized_indices or window_indices or near_hits:
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
                    "window_benchmark_record_ids": [
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
                matches_handle.write(canonical_json(result) + "\n")
    candidates_partial.replace(candidates_path)
    matches_partial.replace(matches_path)
    gate_passed = not quarantine
    manifest = {
        "schema_version": 1,
        "profile": PROFILE,
        "completed_at_utc": utc_now(),
        "candidate_tar": str(candidate_tar.resolve()),
        "candidate_tar_bytes": candidate_tar.stat().st_size,
        "candidate_tar_sha256": sha256_file(candidate_tar),
        "benchmark_records": str(benchmark_records_path.resolve()),
        "benchmark_records_sha256": sha256_file(benchmark_records_path),
        "frozen_config": str(frozen_config_path.resolve()),
        "frozen_config_sha256": sha256_file(frozen_config_path),
        "counts": dict(counts),
        "quarantined_source_member_count": len(quarantine),
        "quarantined_source_members": sorted(quarantine),
        "candidate_identity_manifest": str(candidates_path.resolve()),
        "candidate_identity_manifest_sha256": sha256_file(candidates_path),
        "candidate_matches": str(matches_path.resolve()),
        "candidate_matches_sha256": sha256_file(matches_path),
        "gate_passed": gate_passed,
    }
    atomic_write_text(output_dir / "run_manifest.json", json.dumps(manifest, indent=2) + "\n")
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
    calibrate.add_argument("--seed", type=int, default=20260822)
    calibrate.add_argument("--minimum-aligned-frames", type=int, default=DEFAULT_MIN_ALIGNED_FRAMES)
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
