#!/usr/bin/env python3
"""Independently diagnose one frozen audio-leakage gate hit.

This tool is evidence-only: it reads the original candidate tar, the benchmark
ZIP, and the preserved gate manifests.  It never changes gate thresholds,
candidate data, or the mechanically authoritative gate result.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile
from typing import Any, Iterable
import zipfile

import numpy as np
from scipy import signal
from scipy.signal import resample_poly
import soundfile as sf


PROFILE = "duplexconv-frozen-gate-v2.2-failure-readonly-analysis-v2"
ANALYSIS_MODE = (
    "read_only_no_threshold_change_no_retry_no_exclusion_no_sanitization"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            yield value


def decode_fingerprint(value: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(value), dtype="<u4").copy()


def read_candidate_audio(
    tar_path: Path, source_member: str
) -> tuple[np.ndarray, int, dict[str, Any]]:
    with tarfile.open(tar_path, "r:*") as archive:
        member = archive.getmember(source_member)
        extracted = archive.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"cannot read tar member: {source_member}")
        raw = extracted.read()
    audio, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    return audio, int(sample_rate), {
        "tar_member_bytes": int(member.size),
        "audio_file_sha256": hashlib.sha256(raw).hexdigest(),
    }


def read_benchmark_audio(
    zip_root: Path, record: dict[str, Any]
) -> tuple[np.ndarray, int, dict[str, Any]]:
    zip_path = zip_root / record["zip"]
    with zipfile.ZipFile(zip_path) as archive:
        raw = archive.read(record["member"])
    if hashlib.sha256(raw).hexdigest() != record["audio_file_sha256"]:
        raise RuntimeError("benchmark member hash differs from identity manifest")
    audio, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    return audio, int(sample_rate), {
        "zip_path": str(zip_path.resolve()),
        "zip_sha256": sha256_file(zip_path),
        "audio_file_sha256": hashlib.sha256(raw).hexdigest(),
    }


def resample_mono(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    value = np.asarray(audio, dtype=np.float64)
    if value.ndim == 2:
        value = value.mean(axis=1)
    common = math.gcd(source_rate, target_rate)
    if source_rate != target_rate:
        value = resample_poly(
            value, target_rate // common, source_rate // common
        ).astype(np.float64, copy=False)
    return value


def rms_dbfs(audio: np.ndarray) -> float | None:
    value = np.asarray(audio, dtype=np.float64)
    if not value.size:
        return None
    rms = float(np.sqrt(np.mean(np.square(value))))
    return 20.0 * math.log10(max(rms, np.finfo(np.float64).tiny))


def peak_dbfs(audio: np.ndarray) -> float | None:
    value = np.asarray(audio, dtype=np.float64)
    if not value.size:
        return None
    peak = float(np.max(np.abs(value)))
    return 20.0 * math.log10(max(peak, np.finfo(np.float64).tiny))


def frame_energy_diagnostic(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    value = np.asarray(audio, dtype=np.float64)
    frame_samples = max(1, round(sample_rate * 0.020))
    frame_count = math.ceil(value.size / frame_samples)
    padded = np.pad(value, (0, frame_count * frame_samples - value.size))
    frames = padded.reshape(frame_count, frame_samples)
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
    display_floor_dbfs = -120.0
    frame_dbfs = np.full(frame_rms.shape, display_floor_dbfs, dtype=np.float64)
    nonzero = frame_rms > 0
    frame_dbfs[nonzero] = 20.0 * np.log10(frame_rms[nonzero])
    return {
        "frame_ms": 20,
        "frame_count": int(frame_count),
        "display_floor_dbfs_for_exact_zero_frames": display_floor_dbfs,
        "fraction_exact_zero_frames": float(np.mean(~nonzero)),
        "frame_rms_dbfs_percentiles": {
            str(q): float(np.percentile(frame_dbfs, q)) for q in (5, 25, 50, 75, 95)
        },
        "fraction_frames_below_dbfs": {
            str(threshold): float(np.mean(frame_dbfs < threshold))
            for threshold in (-60, -50, -40)
        },
    }


def spectrum_diagnostic(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    value = np.asarray(audio, dtype=np.float64)
    if not value.size:
        return {}
    frequencies, psd = signal.welch(
        value,
        fs=sample_rate,
        window="hann",
        nperseg=min(4096, value.size),
        noverlap=min(2048, max(0, value.size // 2)),
        scaling="spectrum",
    )
    psd = np.maximum(psd, np.finfo(np.float64).tiny)
    normalized = psd / psd.sum()
    centroid = float(np.sum(frequencies * normalized))
    bandwidth = float(np.sqrt(np.sum(np.square(frequencies - centroid) * normalized)))
    cumulative = np.cumsum(normalized)
    rolloff = float(frequencies[min(np.searchsorted(cumulative, 0.95), frequencies.size - 1)])
    geometric = float(np.exp(np.mean(np.log(psd))))
    arithmetic = float(np.mean(psd))
    return {
        "welch_bin_count": int(psd.size),
        "spectral_centroid_hz": centroid,
        "spectral_bandwidth_hz": bandwidth,
        "spectral_rolloff_95_hz": rolloff,
        "spectral_flatness": geometric / arithmetic,
    }


def maximum_lagged_pearson(
    candidate: np.ndarray,
    benchmark: np.ndarray,
    sample_rate: int,
    minimum_overlap_seconds: list[float],
) -> list[dict[str, Any]]:
    """Compute normalized Pearson correlation at every sample lag via FFT."""

    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(benchmark, dtype=np.float64)
    cross = signal.correlate(left, right, mode="full", method="fft")
    lags = signal.correlation_lags(left.size, right.size, mode="full")
    left_start = np.maximum(lags, 0)
    right_start = np.maximum(-lags, 0)
    overlap = np.minimum(left.size - left_start, right.size - right_start)
    left_stop = left_start + overlap
    right_stop = right_start + overlap

    def prefix(value: np.ndarray) -> np.ndarray:
        return np.concatenate(([0.0], np.cumsum(value, dtype=np.float64)))

    left_sum_prefix = prefix(left)
    right_sum_prefix = prefix(right)
    left_square_prefix = prefix(np.square(left))
    right_square_prefix = prefix(np.square(right))
    left_sum = left_sum_prefix[left_stop] - left_sum_prefix[left_start]
    right_sum = right_sum_prefix[right_stop] - right_sum_prefix[right_start]
    left_square_sum = left_square_prefix[left_stop] - left_square_prefix[left_start]
    right_square_sum = right_square_prefix[right_stop] - right_square_prefix[right_start]
    count = overlap.astype(np.float64)
    numerator = cross - left_sum * right_sum / count
    left_variance_sum = np.maximum(left_square_sum - np.square(left_sum) / count, 0.0)
    right_variance_sum = np.maximum(
        right_square_sum - np.square(right_sum) / count, 0.0
    )
    denominator = np.sqrt(left_variance_sum * right_variance_sum)
    correlation = np.full(cross.shape, np.nan, dtype=np.float64)
    valid = denominator > np.finfo(np.float64).tiny
    correlation[valid] = numerator[valid] / denominator[valid]

    result: list[dict[str, Any]] = []
    for seconds in minimum_overlap_seconds:
        samples = min(round(seconds * sample_rate), left.size, right.size)
        eligible = (overlap >= samples) & np.isfinite(correlation)
        if not np.any(eligible):
            continue
        eligible_indices = np.flatnonzero(eligible)
        index = int(eligible_indices[np.argmax(np.abs(correlation[eligible]))])
        result.append(
            {
                "minimum_overlap_seconds": samples / sample_rate,
                "maximum_absolute_pearson": float(abs(correlation[index])),
                "signed_pearson": float(correlation[index]),
                "lag_samples_candidate_relative_to_benchmark": int(lags[index]),
                "lag_seconds_candidate_relative_to_benchmark": float(
                    lags[index] / sample_rate
                ),
                "actual_overlap_seconds": float(overlap[index] / sample_rate),
            }
        )
    return result


def hamming_bits(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype="<u4")
    return np.unpackbits(raw.view(np.uint8)).reshape(raw.size, 32).sum(axis=1)


def fingerprint_information(fingerprint: np.ndarray) -> dict[str, Any]:
    value = np.asarray(fingerprint, dtype="<u4")
    transitions = hamming_bits(np.bitwise_xor(value[1:], value[:-1]))
    bits = np.unpackbits(value.view(np.uint8)).reshape(value.size, 32)
    probabilities = bits.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -(
            probabilities * np.log2(probabilities)
            + (1.0 - probabilities) * np.log2(1.0 - probabilities)
        )
    entropy = np.nan_to_num(entropy)
    return {
        "frame_count": int(value.size),
        "unique_frame_count": int(np.unique(value).size),
        "unique_frame_fraction": float(np.unique(value).size / value.size),
        "transition_hamming_mean": float(np.mean(transitions)) if transitions.size else None,
        "transition_hamming_median": float(np.median(transitions)) if transitions.size else None,
        "zero_transition_fraction": float(np.mean(transitions == 0)) if transitions.size else None,
        "mean_per_bit_binary_entropy": float(np.mean(entropy)),
        "constant_bit_fraction": float(np.mean((probabilities == 0) | (probabilities == 1))),
    }


def aligned_fingerprint_information(
    candidate: np.ndarray, benchmark: np.ndarray, offset: int
) -> dict[str, Any]:
    candidate_start = max(0, -offset)
    benchmark_start = max(0, offset)
    aligned = min(
        candidate.size - candidate_start, benchmark.size - benchmark_start
    )
    candidate_part = candidate[candidate_start : candidate_start + aligned]
    benchmark_part = benchmark[benchmark_start : benchmark_start + aligned]
    differing_bits = hamming_bits(np.bitwise_xor(candidate_part, benchmark_part))
    return {
        "candidate_start_frame": int(candidate_start),
        "benchmark_start_frame": int(benchmark_start),
        "aligned_frame_count": int(aligned),
        "mean_differing_bits_per_32_bit_frame": float(np.mean(differing_bits)),
        "median_differing_bits_per_32_bit_frame": float(np.median(differing_bits)),
        "minimum_differing_bits": int(np.min(differing_bits)),
        "maximum_differing_bits": int(np.max(differing_bits)),
        "bit_agreement_recomputed": float(1.0 - np.sum(differing_bits) / (32.0 * aligned)),
        "candidate_aligned_segment": fingerprint_information(candidate_part),
        "benchmark_aligned_segment": fingerprint_information(benchmark_part),
    }


def recurrent_collisions(
    expansion_root: Path, benchmark_record_id: str
) -> dict[str, Any]:
    hits: set[tuple[str, str]] = set()
    evidence_files: list[str] = []
    for path in sorted(expansion_root.glob("Edu_*/leakage_gate_v2_2/candidate_matches.jsonl")):
        file_has_hit = False
        for row in load_jsonl(path):
            if any(
                item.get("benchmark_record_id") == benchmark_record_id
                and item.get("is_gate_hit") is True
                for item in row.get("near_candidates", [])
            ):
                hits.add((path.parents[1].name, Path(row["source_member"]).stem))
                file_has_hit = True
        if file_has_hit:
            evidence_files.append(str(path.resolve()))
    ordered = sorted(hits)
    return {
        "same_benchmark_record_distinct_gate_hit_source_count_in_preserved_original_source_gate_results": len(ordered),
        "shard_and_source_ids": [
            {"shard": shard, "source_id": source} for shard, source in ordered
        ],
        "evidence_file_count": len(evidence_files),
        "evidence_files": evidence_files,
        "interpretation": (
            "Repeated hits from distinct DuplexConv source conversations against one "
            "benchmark record are evidence for a recurrent fingerprint collision; they "
            "do not override the frozen gate result."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--benchmark-zip-root", type=Path, required=True)
    parser.add_argument("--expansion-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_json.exists() or args.output_md.exists():
        raise FileExistsError("refusing to overwrite an existing diagnostic report")

    gate_manifest_path = args.gate_dir / "run_manifest.json"
    matches_path = args.gate_dir / "candidate_matches.jsonl"
    identities_path = args.gate_dir / "candidate_identity_manifest.jsonl"
    gate = json.loads(gate_manifest_path.read_text(encoding="utf-8"))
    if gate.get("gate_passed") is not False:
        raise RuntimeError("diagnostic requires a preserved failed gate run")

    gate_hits: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in load_jsonl(matches_path):
        for near in row.get("near_candidates", []):
            if near.get("is_gate_hit") is True:
                gate_hits.append((row, near))
    if len(gate_hits) != 1:
        raise RuntimeError(f"expected exactly one gate hit, found {len(gate_hits)}")
    match_row, hit = gate_hits[0]

    candidate_record = next(
        row
        for row in load_jsonl(identities_path)
        if row.get("candidate_id") == match_row["candidate_id"]
    )
    benchmark_records_path = Path(gate["benchmark_records"])
    benchmark_record = next(
        row
        for row in load_jsonl(benchmark_records_path)
        if row.get("record_id") == hit["benchmark_record_id"]
    )

    candidate_audio, candidate_rate, candidate_raw = read_candidate_audio(
        Path(gate["candidate_tar"]), match_row["source_member"]
    )
    channel = int(match_row["channel"])
    candidate_channel = candidate_audio[:, channel]
    benchmark_audio, benchmark_rate, benchmark_raw = read_benchmark_audio(
        args.benchmark_zip_root, benchmark_record
    )
    candidate_16k = resample_mono(candidate_channel, candidate_rate, 16_000)
    benchmark_16k = resample_mono(benchmark_audio, benchmark_rate, 16_000)

    full_overlap = min(candidate_16k.size, benchmark_16k.size) / 16_000
    overlaps = [value for value in (1.0, 4.0, 10.0, 15.0, 30.0, 60.0, 80.0) if value < full_overlap]
    overlaps.append(full_overlap)
    correlations = maximum_lagged_pearson(
        candidate_16k, benchmark_16k, 16_000, overlaps
    )

    candidate_fp = decode_fingerprint(candidate_record["chromaprint_raw_u32_base64"])
    benchmark_fp = decode_fingerprint(
        benchmark_record["chromaprint_raw_u32_base64"]
    )
    aligned = aligned_fingerprint_information(candidate_fp, benchmark_fp, int(hit["offset"]))
    if aligned["aligned_frame_count"] != int(hit["aligned_frames"]):
        raise RuntimeError("recomputed aligned fingerprint frame count differs from gate")
    if not math.isclose(
        aligned["bit_agreement_recomputed"], float(hit["similarity"]), abs_tol=1e-15
    ):
        raise RuntimeError("recomputed fingerprint similarity differs from gate")

    literal_reuse = any(
        item["maximum_absolute_pearson"] >= 0.95 and item["minimum_overlap_seconds"] >= 4.0
        for item in correlations
    )
    exact_count = len(match_row["exact_benchmark_record_ids"])
    normalized_count = len(match_row["normalized_benchmark_record_ids"])
    content_count = len(match_row["content_exact_benchmark_record_ids"])
    window_count = len(match_row["content_window_benchmark_record_ids"])
    recurrent = recurrent_collisions(args.expansion_root, hit["benchmark_record_id"])

    report: dict[str, Any] = {
        "schema_version": 2,
        "profile": PROFILE,
        "completed_at_utc": utc_now(),
        "analysis_mode": ANALYSIS_MODE,
        "source": {
            "source_id": Path(match_row["source_member"]).stem,
            "source_member": match_row["source_member"],
            "candidate_id": match_row["candidate_id"],
            "candidate_channel": channel,
            "duration_seconds": candidate_audio.shape[0] / candidate_rate,
            "sample_rate": candidate_rate,
            "channel_count": int(candidate_audio.shape[1]),
            "tar_member_bytes": candidate_raw["tar_member_bytes"],
            "audio_file_sha256": candidate_raw["audio_file_sha256"],
            "source_structurally_usable": bool(candidate_audio.shape[1] == 2),
        },
        "benchmark": {
            "record_id": benchmark_record["record_id"],
            "sample_key": benchmark_record["sample_key"],
            "member": benchmark_record["member"],
            "zip": benchmark_record["zip"],
            "duration_seconds": benchmark_audio.shape[0] / benchmark_rate,
            "sample_rate": benchmark_rate,
            "channel_count": int(benchmark_audio.shape[1]),
            **benchmark_raw,
        },
        "frozen_gate": {
            "profile": gate["profile"],
            "near_duplicate_similarity_threshold": gate["near_duplicate_similarity_threshold"],
            "minimum_aligned_frames": gate["minimum_aligned_frames"],
            "minimum_lsh_votes": gate["minimum_lsh_votes"],
            "similarity": hit["similarity"],
            "aligned_frames": hit["aligned_frames"],
            "lsh_votes": hit["votes"],
            "offset": hit["offset"],
            "exact_audio_match_count": exact_count,
            "normalized_audio_match_count": normalized_count,
            "content_exact_match_count": content_count,
            "window_match_count": window_count,
            "near_duplicate_gate_hit_count": 1,
            "gate_passed": False,
            "gate_result_is_authoritative": True,
        },
        "independent_waveform_diagnostic": {
            "method": (
                "48 kHz source channel resampled to 16 kHz, compared with benchmark "
                "mono using normalized Pearson correlation over every sample lag"
            ),
            "candidate_rms_dbfs": rms_dbfs(candidate_16k),
            "candidate_peak_dbfs": peak_dbfs(candidate_16k),
            "benchmark_rms_dbfs": rms_dbfs(benchmark_16k),
            "benchmark_peak_dbfs": peak_dbfs(benchmark_16k),
            "candidate_frame_energy": frame_energy_diagnostic(candidate_16k, 16_000),
            "benchmark_frame_energy": frame_energy_diagnostic(benchmark_16k, 16_000),
            "lagged_pearson": correlations,
            "literal_waveform_reuse_confirmed": literal_reuse,
        },
        "spectrum_diagnostic": {
            "method": "Welch power spectrum over each complete waveform; descriptive only",
            "candidate": spectrum_diagnostic(candidate_16k, 16_000),
            "benchmark": spectrum_diagnostic(benchmark_16k, 16_000),
            "used_as_gate_override": False,
        },
        "fingerprint_information_diagnostic": {
            "candidate_complete": fingerprint_information(candidate_fp),
            "benchmark_complete": fingerprint_information(benchmark_fp),
            "frozen_alignment": aligned,
            "candidate_active_content_seconds": candidate_record["activity"]["content_seconds"],
            "candidate_active_frame_count_20ms": candidate_record["activity"]["active_frame_count"],
            "candidate_total_frame_count_20ms": candidate_record["activity"]["total_frame_count"],
        },
        "recurrent_benchmark_collision_diagnostic": recurrent,
        "provenance": {
            "candidate_tar": str(Path(gate["candidate_tar"]).resolve()),
            "candidate_tar_sha256": gate["candidate_tar_sha256"],
            "gate_manifest": str(gate_manifest_path.resolve()),
            "gate_manifest_sha256": sha256_file(gate_manifest_path),
            "candidate_matches_sha256": sha256_file(matches_path),
            "candidate_identity_manifest_sha256": sha256_file(identities_path),
            "benchmark_identity_manifest": str(benchmark_records_path.resolve()),
            "benchmark_identity_manifest_sha256": sha256_file(benchmark_records_path),
            "diagnostic_implementation": str(Path(__file__).resolve()),
            "diagnostic_implementation_sha256": sha256_file(Path(__file__)),
        },
    }

    max_full = correlations[-1]["maximum_absolute_pearson"]
    max_one = correlations[0]["maximum_absolute_pearson"]
    recurrent_count = recurrent[
        "same_benchmark_record_distinct_gate_hit_source_count_in_preserved_original_source_gate_results"
    ]
    if not literal_reuse and not any((exact_count, normalized_count, content_count, window_count)):
        report["conclusion"] = (
            "Literal waveform reuse was not confirmed. Zero exact/content/window matches, "
            f"low lagged waveform correlation and {recurrent_count} distinct source "
            "collisions against the same benchmark record support a recurrent "
            "Chromaprint near-duplicate false-positive explanation. This independent "
            "diagnostic does not override the mechanically valid frozen gate failure."
        )
    else:
        report["conclusion"] = (
            "The independent evidence is not sufficient to classify this as a recurrent "
            "Chromaprint false positive. The frozen gate failure remains authoritative."
        )

    markdown = f"""# Edu_0017 冻结 Gate B/C 命中只读诊断

状态：原始冻结门禁仍为失败；本诊断未修改阈值、未排除样本、未重试流水线，也未启动付费 Qwen 或 GPU 工作。

唯一命中为 `{match_row['source_member']}` 声道 {channel} 对 `{benchmark_record['member']}`：相似度 {hit['similarity']:.10f}、对齐 {hit['aligned_frames']} 帧、LSH {hit['votes']} 票；冻结阈值分别为 {gate['near_duplicate_similarity_threshold']}、{gate['minimum_aligned_frames']} 帧、{gate['minimum_lsh_votes']} 票。因此机械门禁结果保持失败，诊断无权覆盖。

独立波形核验未确认字面音频复用：精确、归一化、活跃内容精确和内容窗口匹配均为 0；完整 {full_overlap:.3f} 秒候选重叠的最大绝对 Pearson 相关系数为 {max_full:.8f}，即使把最小重叠放宽到 1 秒，最大值也只有 {max_one:.8f}。候选 RMS 为 {rms_dbfs(candidate_16k):.2f} dBFS，benchmark 为 {rms_dbfs(benchmark_16k):.2f} dBFS。报告同时保存了 20 ms 能量分布和 Welch 频谱描述值，但这些描述值不用于覆盖门禁。

候选在去除非活跃帧后只有 {candidate_record['activity']['content_seconds']:.1f} 秒活跃内容，生成 {candidate_fp.size} 个 Chromaprint 帧，冻结门禁以 offset={hit['offset']} 将这 {hit['aligned_frames']} 帧全部对齐到 benchmark。重算位一致率为 {aligned['bit_agreement_recomputed']:.10f}，与门禁输出一致，说明诊断没有改写命中计算。保存的原始来源门禁结果中，同一个 benchmark 记录已对 {recurrent_count} 个不同来源产生 Gate 命中。

结论：{report['conclusion']}
"""
    atomic_write_text(args.output_json, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    atomic_write_text(args.output_md, markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
