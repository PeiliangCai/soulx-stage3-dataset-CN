#!/usr/bin/env python3
"""Independently close a final model-ready benchmark leakage Gate D run."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def benchmark_identifier_strings(records: list[dict]) -> set[str]:
    strings: set[str] = set()
    for record in records:
        for field in ("record_id", "sample_key", "member", "zip", "numeric_sample_id"):
            value = record.get(field)
            if value is None:
                continue
            text = str(value)
            strings.add(text)
            if field in {"member", "zip"}:
                path = PurePosixPath(text)
                strings.add(path.name)
                strings.add(path.stem)
    return strings


def flattened_windows(record: dict) -> set[str]:
    result: set[str] = set()
    for values in record.get("content_window_sha256", {}).values():
        result.update(values)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ready-dir", type=Path, required=True)
    parser.add_argument("--model-ready-validation", type=Path, required=True)
    parser.add_argument("--target-audio-manifest", type=Path, required=True)
    parser.add_argument("--selection-list", type=Path, required=True)
    parser.add_argument("--candidate-tar", type=Path, required=True)
    parser.add_argument("--gate-output-dir", type=Path, required=True)
    parser.add_argument("--benchmark-records", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--expected-view-count", type=int, required=True)
    parser.add_argument("--expected-source-count", type=int, required=True)
    parser.add_argument("--excluded-source-id")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    windows_path = args.model_ready_dir / "metadata/windows.jsonl"
    stats_path = args.model_ready_dir / "stats.json"
    identity_path = args.gate_output_dir / "candidate_identity_manifest.jsonl"
    matches_path = args.gate_output_dir / "candidate_matches.jsonl"
    run_path = args.gate_output_dir / "run_manifest.json"

    windows = read_jsonl(windows_path)
    audio_manifest = read_jsonl(args.target_audio_manifest)
    identities = read_jsonl(identity_path)
    matches = read_jsonl(matches_path)
    benchmark = read_jsonl(args.benchmark_records)
    run = json.loads(run_path.read_text(encoding="utf-8"))

    view_ids = {record["view_id"] for record in windows}
    source_ids = {record["source_id"] for record in windows}
    audio_by_view = {record["view_id"]: record["wav_relative_path"] for record in audio_manifest}
    expected_members = {audio_by_view[view_id] for view_id in view_ids}

    selection_lines = [
        line.strip()
        for line in args.selection_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selection_members = set(selection_lines)
    with tarfile.open(args.candidate_tar, "r") as archive:
        tar_lines = [member.name for member in archive.getmembers() if member.isfile()]
    tar_members = set(tar_lines)

    scored_member_counts = Counter(record["source_member"] for record in matches)
    identity_member_counts = Counter(record["source_member"] for record in identities)

    benchmark_strings = benchmark_identifier_strings(benchmark)
    source_collisions = sorted(source_ids & benchmark_strings)

    benchmark_exact = {record["pcm16k_mono_sha256"] for record in benchmark}
    benchmark_normalized = {
        record["pcm16k_mono_peak_normalized_sha256"] for record in benchmark
    }
    benchmark_content = {record["content_pcm16_sha256"] for record in benchmark}
    benchmark_windows: set[str] = set()
    for record in benchmark:
        benchmark_windows.update(flattened_windows(record))

    exact_count = sum(record["pcm16k_mono_sha256"] in benchmark_exact for record in identities)
    normalized_count = sum(
        record["pcm16k_mono_peak_normalized_sha256"] in benchmark_normalized
        for record in identities
    )
    content_count = sum(record["content_pcm16_sha256"] in benchmark_content for record in identities)
    window_count = sum(bool(flattened_windows(record) & benchmark_windows) for record in identities)
    near_count = sum(
        bool(candidate.get("is_gate_hit"))
        for record in matches
        for candidate in record.get("near_candidates", [])
    )

    excluded_source_absent = True
    if args.excluded_source_id:
        marker = args.excluded_source_id
        excluded_source_absent = (
            marker not in source_ids
            and all(marker not in member for member in selection_lines)
            and all(marker not in member for member in tar_lines)
            and all(marker not in record["source_member"] for record in identities)
            and all(marker not in record["source_member"] for record in matches)
        )

    view_partition_closed = (
        len(view_ids) == args.expected_view_count
        and len(selection_lines) == args.expected_view_count
        and len(selection_members) == args.expected_view_count
        and len(tar_lines) == args.expected_view_count
        and len(tar_members) == args.expected_view_count
        and expected_members == selection_members == tar_members
    )
    source_partition_closed = len(source_ids) == args.expected_source_count
    scorer_partition_closed = (
        len(matches) == 2 * args.expected_view_count
        and len(identities) == 2 * args.expected_view_count
        and set(scored_member_counts) == expected_members
        and set(identity_member_counts) == expected_members
        and all(count == 2 for count in scored_member_counts.values())
        and all(count == 2 for count in identity_member_counts.values())
    )

    gate_passed = all(
        (
            view_partition_closed,
            source_partition_closed,
            scorer_partition_closed,
            not source_collisions,
            exact_count == 0,
            normalized_count == 0,
            content_count == 0,
            window_count == 0,
            near_count == 0,
            run.get("gate_passed") is True,
            run.get("quarantined_source_member_count") == 0,
            excluded_source_absent,
        )
    )

    result = {
        "schema_version": 1,
        "profile": "duplexconv-model-ready-benchmark-leakage-gate-d-v1",
        "completed_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "selection_definition": "All unique view_id values contributing at least one row in model-ready metadata/windows.jsonl",
        "model_ready_view_count": len(view_ids),
        "model_ready_source_id_count": len(source_ids),
        "scored_source_member_count": len(scored_member_counts),
        "candidate_record_count": len(matches),
        "identity_record_count": len(identities),
        "all_source_members_scored_exactly_twice": scorer_partition_closed,
        "mono_scorer_note": (
            "The unchanged frozen scorer emits channel 0 and an identical mono mix for each mono WAV. "
            f"Every one of the {len(view_ids)} final views was therefore conservatively scored twice."
        ),
        "selection_missing_count": len(expected_members - selection_members),
        "selection_extra_count": len(selection_members - expected_members),
        "selection_tar_partition_closed": view_partition_closed,
        "excluded_source_id": args.excluded_source_id,
        "sanitized_excluded_source_absent": excluded_source_absent,
        "source_identifier_comparison": {
            "benchmark_record_count": len(benchmark),
            "compared_benchmark_fields": [
                "record_id",
                "sample_key",
                "member",
                "zip",
                "numeric_sample_id",
                "path basename",
                "path stem",
            ],
            "source_leakage_count": len(source_collisions),
            "exact_source_identifier_collision_count": len(source_collisions),
            "collisions": source_collisions,
        },
        "audio_comparison": {
            "exact_audio_match_count": exact_count,
            "normalized_audio_match_count": normalized_count,
            "content_exact_match_count": content_count,
            "window_match_count": window_count,
            "near_duplicate_gate_hit_count": near_count,
            "near_duplicate_similarity_threshold": run["near_duplicate_similarity_threshold"],
            "minimum_aligned_frames": run["minimum_aligned_frames"],
            "minimum_lsh_votes": run["minimum_lsh_votes"],
        },
        "gate_passed": gate_passed,
        "sha256": {
            "model_ready_windows": sha256(windows_path),
            "model_ready_stats": sha256(stats_path),
            "model_ready_validation": sha256(args.model_ready_validation),
            "target_audio_manifest": sha256(args.target_audio_manifest),
            "candidate_selection_list": sha256(args.selection_list),
            "candidate_tar": sha256(args.candidate_tar),
            "candidate_identity_manifest": sha256(identity_path),
            "candidate_matches": sha256(matches_path),
            "scorer_run_manifest": sha256(run_path),
            "benchmark_identity_manifest": sha256(args.benchmark_records),
            "frozen_config": sha256(args.frozen_config),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
