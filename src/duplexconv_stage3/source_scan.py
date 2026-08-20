"""Scan official DuplexConv archives and build target-vs-rest source views.

This module deliberately reads only the immutable official archives.  It does
not consume any legacy extracted, processed, reviewed, or model-ready data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from typing import Any, Iterable, Sequence
import wave


CHUNK_MS = 160
SOURCE_VIEW_PROFILE = "target-vs-rest-v1"
DATASET_VERSION = "duplexconv_edu0018_stage3_zh_v1"
SUPPORTED_TRACK_COUNTS = (2, 3)
STATE_MAP = {
    "<|complete|>": "complete",
    "<|incomplete|>": "incomplete",
    "<|backchannel|>": "backchannel",
    "<|wait|>": "wait",
}
EXPECTED = {
    "source_count": 500,
    "source_ntrack": {"2": 495, "3": 5},
    "target_view_count": 1005,
    "event_count": 8505,
    "state_distribution": {
        "complete": 4570,
        "incomplete": 1527,
        "backchannel": 798,
        "wait": 11,
        "missing": 1599,
    },
}


def stable_event_id(source_id: str, channel: int, ordinal: int) -> str:
    return f"{source_id}/ch{channel:02d}/event{ordinal:04d}"


def stable_view_id(source_id: str, target_channel: int) -> str:
    return f"{source_id}/target-ch{target_channel:02d}"


def normalize_state(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or value not in STATE_MAP:
        raise ValueError(f"unsupported official state: {value!r}")
    return STATE_MAP[value]


def interval_chunk_bounds(
    start_ms: float, end_ms: float, chunk_count: int
) -> tuple[int, int]:
    """Return the half-open chunk range touched by a half-open time interval."""
    if chunk_count < 0:
        raise ValueError("chunk_count must be non-negative")
    if not math.isfinite(start_ms) or not math.isfinite(end_ms):
        raise ValueError("interval must be finite")
    if start_ms < 0 or end_ms < start_ms:
        raise ValueError(f"invalid interval: {start_ms}, {end_ms}")
    if end_ms == start_ms or chunk_count == 0:
        start = min(chunk_count, max(0, math.floor(start_ms / CHUNK_MS)))
        return start, start
    start = max(0, math.floor(start_ms / CHUNK_MS))
    stop = min(chunk_count, math.ceil(end_ms / CHUNK_MS))
    return min(start, stop), stop


def clip_terminal_end(
    end_ms: float, duration_ms: float, tolerance_ms: float = CHUNK_MS
) -> tuple[float, float]:
    """Clip a small metadata-only tail overhang and return (effective, overhang).

    DuplexConv timestamps are quantized and a few official terminal timestamps
    exceed the exact WAV frame duration by less than one Stage 3 chunk.  Only
    this terminal case is repairable; a larger overhang remains structural.
    """
    if end_ms <= duration_ms:
        return end_ms, 0.0
    overhang_ms = end_ms - duration_ms
    if overhang_ms <= tolerance_ms:
        return duration_ms, overhang_ms
    raise ValueError(
        f"terminal overhang {overhang_ms:.6f} ms exceeds {tolerance_ms:.6f} ms"
    )


def activity_from_intervals(
    intervals_ms: Iterable[tuple[float, float]], chunk_count: int
) -> list[bool]:
    activity = [False] * chunk_count
    for start_ms, end_ms in intervals_ms:
        start, stop = interval_chunk_bounds(start_ms, end_ms, chunk_count)
        for index in range(start, stop):
            activity[index] = True
    return activity


def target_vs_rest_activity(
    channel_activity: Sequence[Sequence[bool]], target_channel: int
) -> dict[str, list[Any]]:
    if not channel_activity:
        raise ValueError("channel_activity cannot be empty")
    if target_channel < 0 or target_channel >= len(channel_activity):
        raise ValueError("target channel is out of range")
    chunk_count = len(channel_activity[0])
    if any(len(item) != chunk_count for item in channel_activity):
        raise ValueError("channel activity lengths do not match")

    target = list(channel_activity[target_channel])
    other_count = [0] * chunk_count
    for channel, activity in enumerate(channel_activity):
        if channel == target_channel:
            continue
        for index, active in enumerate(activity):
            other_count[index] += int(bool(active))
    other = [count > 0 for count in other_count]
    overlap = [left and right for left, right in zip(target, other)]
    return {
        "target_active_by_chunk": target,
        "other_active_by_chunk": other,
        "other_active_count_by_chunk": other_count,
        "overlap_by_chunk": overlap,
    }


def _basename_stem(name: str, suffix: str) -> str:
    base = PurePosixPath(name).name
    if not base.endswith(suffix):
        raise ValueError(f"unexpected archive member suffix: {name}")
    return base[: -len(suffix)]


def _member_map(
    archive: tarfile.TarFile, suffix: str
) -> tuple[dict[str, tarfile.TarInfo], list[str]]:
    result: dict[str, tarfile.TarInfo] = {}
    duplicates: list[str] = []
    for member in archive.getmembers():
        if not member.isfile() or not member.name.endswith(suffix):
            continue
        source_id = _basename_stem(member.name, suffix)
        if source_id in result:
            duplicates.append(source_id)
        else:
            result[source_id] = member
    return result, sorted(set(duplicates))


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} is not numeric: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} is not finite: {value!r}")
    return result


def _extract_events(
    source_id: str,
    metadata: dict[str, Any],
    duration_ms: float,
) -> tuple[
    list[dict[str, Any]],
    list[list[tuple[float, float]]],
    list[str],
    list[str],
]:
    errors: list[str] = []
    anomalies: list[str] = []
    all_events: list[dict[str, Any]] = []
    asr = metadata.get("asr")
    if not isinstance(asr, list):
        return [], [], ["metadata_asr_not_list"], []

    channel_intervals: list[list[tuple[float, float]]] = [
        [] for _ in range(len(asr))
    ]
    for channel, events in enumerate(asr):
        if not isinstance(events, list):
            errors.append(f"channel_{channel}:events_not_list")
            continue
        previous_start = -1.0
        for ordinal, event in enumerate(events):
            event_id = stable_event_id(source_id, channel, ordinal)
            if not isinstance(event, dict):
                errors.append(f"{event_id}:event_not_object")
                continue
            try:
                start_ms = _finite_number(event.get("startInMs"), "startInMs")
                end_ms = _finite_number(event.get("endInMs"), "endInMs")
            except ValueError as exc:
                errors.append(f"{event_id}:{exc}")
                continue
            if start_ms < previous_start:
                errors.append(f"{event_id}:event_start_not_monotonic")
            previous_start = start_ms
            event_repairs: list[dict[str, Any]] = []
            effective_end_ms = end_ms
            if start_ms < 0 or end_ms < start_ms or start_ms > duration_ms:
                errors.append(f"{event_id}:event_interval_out_of_bounds")
            else:
                try:
                    effective_end_ms, event_overhang_ms = clip_terminal_end(
                        end_ms, duration_ms
                    )
                    if event_overhang_ms > 1.0:
                        anomaly = (
                            f"{event_id}:event_terminal_overhang_ms:"
                            f"{event_overhang_ms:.6f}"
                        )
                        anomalies.append(anomaly)
                        event_repairs.append(
                            {
                                "type": "clip_terminal_event_end",
                                "original_end_ms": end_ms,
                                "effective_end_ms": effective_end_ms,
                                "overhang_ms": event_overhang_ms,
                            }
                        )
                except ValueError:
                    errors.append(f"{event_id}:event_interval_out_of_bounds")

            try:
                official_state = normalize_state(event.get("state"))
            except ValueError as exc:
                official_state = None
                errors.append(f"{event_id}:{exc}")

            labels = event.get("labels") if isinstance(event.get("labels"), dict) else {}
            text = labels.get("txt", "")
            if not isinstance(text, str):
                errors.append(f"{event_id}:labels_txt_not_string")
                text = str(text)

            speaker = event.get("speaker") if isinstance(event.get("speaker"), dict) else {}
            segments = speaker.get("segments", [])
            if segments is None:
                segments = []
            if not isinstance(segments, list):
                errors.append(f"{event_id}:speaker_segments_not_list")
                segments = []

            valid_segments_ms: list[list[float]] = []
            for segment_index, segment in enumerate(segments):
                if not isinstance(segment, dict):
                    errors.append(
                        f"{event_id}:segment_{segment_index}:segment_not_object"
                    )
                    continue
                try:
                    segment_start = 1000.0 * _finite_number(
                        segment.get("startSec"), "startSec"
                    )
                    segment_end = 1000.0 * _finite_number(
                        segment.get("endSec"), "endSec"
                    )
                except ValueError as exc:
                    errors.append(f"{event_id}:segment_{segment_index}:{exc}")
                    continue
                if segment_start < -1.0 or segment_end < segment_start or segment_start > duration_ms:
                    errors.append(
                        f"{event_id}:segment_{segment_index}:interval_out_of_bounds"
                    )
                    continue
                segment_start = max(0.0, segment_start)
                try:
                    effective_segment_end, segment_overhang_ms = clip_terminal_end(
                        segment_end, duration_ms
                    )
                except ValueError:
                    errors.append(
                        f"{event_id}:segment_{segment_index}:interval_out_of_bounds"
                    )
                    continue
                if segment_overhang_ms > 1.0:
                    anomaly = (
                        f"{event_id}:segment_{segment_index}_terminal_overhang_ms:"
                        f"{segment_overhang_ms:.6f}"
                    )
                    anomalies.append(anomaly)
                    event_repairs.append(
                        {
                            "type": "clip_terminal_speaker_segment_end",
                            "segment_index": segment_index,
                            "original_end_ms": segment_end,
                            "effective_end_ms": effective_segment_end,
                            "overhang_ms": segment_overhang_ms,
                        }
                    )
                segment_end = effective_segment_end
                if segment_end > segment_start:
                    channel_intervals[channel].append((segment_start, segment_end))
                    valid_segments_ms.append([segment_start, segment_end])

            envelope_valid = (
                start_ms >= 0
                and effective_end_ms > start_ms
                and effective_end_ms <= duration_ms
            )
            all_events.append(
                {
                    "event_id": event_id,
                    "source_id": source_id,
                    "channel": channel,
                    "ordinal": ordinal,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "effective_end_ms": effective_end_ms,
                    "lid": event.get("LID"),
                    "text": text,
                    "official_state_raw": event.get("state"),
                    "official_state": official_state,
                    "speaker_segments_ms": valid_segments_ms,
                    "activity_source": (
                        "speaker_segments" if valid_segments_ms else "none"
                    ),
                    "activity_fallback_candidate": bool(
                        not valid_segments_ms and envelope_valid and text.strip()
                    ),
                    "event_envelope_ms": [start_ms, end_ms],
                    "effective_event_envelope_ms": [start_ms, effective_end_ms],
                    "deterministic_repairs": event_repairs,
                }
            )
    return all_events, channel_intervals, errors, anomalies


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(
                json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            handle.write("\n")


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _validate_expected(summary: dict[str, Any]) -> None:
    checks = {
        "source_count": summary["sources"]["total"] == EXPECTED["source_count"],
        "source_ntrack": summary["sources"]["by_ntrack"]
        == EXPECTED["source_ntrack"],
        "target_view_count": summary["views"]["upper_bound"]
        == EXPECTED["target_view_count"],
        "event_count": summary["events"]["total"] == EXPECTED["event_count"],
        "state_distribution": summary["events"]["state_distribution"]
        == EXPECTED["state_distribution"],
        "view_closure": summary["views"]["upper_bound"]
        == summary["views"]["structurally_usable"]
        + summary["views"]["source_quarantined"],
    }
    summary["expected_checks"] = checks
    summary["gate_3_source_contract_passed"] = all(checks.values())
    if not summary["gate_3_source_contract_passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"source contract checks failed: {failed}")


def scan_archives(
    audio_archive: Path,
    metadata_archive: Path,
    output_dir: Path,
    *,
    compute_input_hashes: bool = True,
) -> dict[str, Any]:
    audio_archive = audio_archive.resolve(strict=True)
    metadata_archive = metadata_archive.resolve(strict=True)
    output_dir = output_dir.absolute()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"temporary directory already exists: {temporary}")
    temporary.mkdir()

    source_inventory: list[dict[str, Any]] = []
    event_records: list[dict[str, Any]] = []
    view_records: list[dict[str, Any]] = []
    quarantine_records: list[dict[str, Any]] = []
    anomaly_records: list[dict[str, Any]] = []
    by_ntrack: dict[str, int] = {}
    state_distribution = {
        "complete": 0,
        "incomplete": 0,
        "backchannel": 0,
        "wait": 0,
        "missing": 0,
    }
    view_upper_bound = 0
    duration_seconds_by_ntrack: dict[str, float] = {}

    try:
        with tarfile.open(audio_archive, "r:*") as audio_tar, tarfile.open(
            metadata_archive, "r:*"
        ) as metadata_tar:
            wav_members, duplicate_wav = _member_map(audio_tar, ".wav")
            json_members, duplicate_json = _member_map(metadata_tar, ".json")
            audio_ids = set(wav_members)
            metadata_ids = set(json_members)
            selected_ids = sorted(audio_ids)

            archive_errors: list[str] = []
            if duplicate_wav:
                archive_errors.append(f"duplicate_wav_ids:{duplicate_wav}")
            if duplicate_json:
                archive_errors.append(f"duplicate_json_ids:{duplicate_json}")
            if audio_ids - metadata_ids:
                archive_errors.append(
                    f"audio_without_metadata:{sorted(audio_ids - metadata_ids)}"
                )
            if archive_errors:
                raise RuntimeError("; ".join(archive_errors))

            for source_id in selected_ids:
                errors: list[str] = []
                wav_member = wav_members[source_id]
                metadata_member = json_members[source_id]
                metadata_file = metadata_tar.extractfile(metadata_member)
                if metadata_file is None:
                    raise RuntimeError(f"cannot read metadata member: {metadata_member.name}")
                metadata = json.load(metadata_file)
                if not isinstance(metadata, dict):
                    raise RuntimeError(f"metadata is not an object: {source_id}")

                wav_file = audio_tar.extractfile(wav_member)
                if wav_file is None:
                    raise RuntimeError(f"cannot read WAV member: {wav_member.name}")
                with wave.open(wav_file, "rb") as wav_reader:
                    wav_channels = wav_reader.getnchannels()
                    wav_sample_rate = wav_reader.getframerate()
                    wav_sample_width = wav_reader.getsampwidth()
                    wav_frame_count = wav_reader.getnframes()
                wav_duration_seconds = wav_frame_count / wav_sample_rate
                duration_ms = 1000.0 * wav_duration_seconds
                chunk_frames = wav_sample_rate * CHUNK_MS // 1000
                chunk_count = math.ceil(wav_frame_count / chunk_frames)

                ntrack = metadata.get("nTrack")
                if not isinstance(ntrack, int):
                    errors.append("metadata_ntrack_not_integer")
                elif ntrack not in SUPPORTED_TRACK_COUNTS:
                    errors.append(f"unsupported_ntrack:{ntrack}")
                if ntrack != wav_channels:
                    errors.append(
                        f"metadata_wav_channel_mismatch:{ntrack}!={wav_channels}"
                    )
                if metadata.get("fs") != wav_sample_rate:
                    errors.append(
                        f"metadata_wav_sample_rate_mismatch:{metadata.get('fs')}!={wav_sample_rate}"
                    )
                if wav_sample_width != 2:
                    errors.append(f"wav_sample_width_not_s16:{wav_sample_width}")
                asr = metadata.get("asr")
                if not isinstance(asr, list) or len(asr) != ntrack:
                    errors.append(
                        f"metadata_asr_track_mismatch:{len(asr) if isinstance(asr, list) else 'not-list'}!={ntrack}"
                    )
                try:
                    metadata_duration = _finite_number(
                        metadata.get("timeLenInSec"), "timeLenInSec"
                    )
                    if abs(metadata_duration - wav_duration_seconds) > max(
                        0.001, 1.0 / wav_sample_rate
                    ):
                        errors.append(
                            "metadata_wav_duration_mismatch:"
                            f"{metadata_duration}!={wav_duration_seconds}"
                        )
                except ValueError as exc:
                    errors.append(str(exc))

                events, channel_intervals, event_errors, event_anomalies = _extract_events(
                    source_id, metadata, duration_ms
                )
                errors.extend(event_errors)
                structurally_usable = not errors
                event_ids_by_channel: list[list[str]] = [
                    [] for _ in range(wav_channels)
                ]
                for event in events:
                    event["source_structurally_usable"] = structurally_usable
                    event_records.append(event)
                    if 0 <= event["channel"] < wav_channels:
                        event_ids_by_channel[event["channel"]].append(event["event_id"])
                    state_distribution[event["official_state"] or "missing"] += 1

                effective_ntrack = (
                    ntrack if isinstance(ntrack, int) and ntrack > 0 else wav_channels
                )
                view_upper_bound += effective_ntrack
                key = str(effective_ntrack)
                by_ntrack[key] = by_ntrack.get(key, 0) + 1
                duration_seconds_by_ntrack[key] = (
                    duration_seconds_by_ntrack.get(key, 0.0)
                    + wav_duration_seconds * effective_ntrack
                )
                source_record = {
                    "source_id": source_id,
                    "audio_member": wav_member.name,
                    "metadata_member": metadata_member.name,
                    "ntrack": ntrack,
                    "wav_channels": wav_channels,
                    "sample_rate": wav_sample_rate,
                    "sample_width_bytes": wav_sample_width,
                    "frame_count": wav_frame_count,
                    "duration_seconds": wav_duration_seconds,
                    "chunk_count": chunk_count,
                    "channel_event_counts": [
                        len(items) for items in event_ids_by_channel
                    ],
                    "channel_speaker_segment_counts": [
                        len(items) for items in channel_intervals
                    ],
                    "structurally_usable": structurally_usable,
                    "errors": sorted(set(errors)),
                    "anomalies": sorted(set(event_anomalies)),
                }
                source_inventory.append(source_record)
                if event_anomalies:
                    anomaly_records.append(
                        {
                            "source_id": source_id,
                            "anomalies": sorted(set(event_anomalies)),
                        }
                    )

                if not structurally_usable:
                    quarantine_records.append(
                        {
                            "source_id": source_id,
                            "view_count": effective_ntrack,
                            "errors": sorted(set(errors)),
                        }
                    )
                    continue

                channel_activity = [
                    activity_from_intervals(intervals, chunk_count)
                    for intervals in channel_intervals
                ]
                for target_channel in range(ntrack):
                    relation = target_vs_rest_activity(
                        channel_activity, target_channel
                    )
                    if any(len(values) != chunk_count for values in relation.values()):
                        raise RuntimeError(
                            f"activity length closure failed for {source_id} ch{target_channel}"
                        )
                    view_records.append(
                        {
                            "view_id": stable_view_id(source_id, target_channel),
                            "source_id": source_id,
                            "source_ntrack": ntrack,
                            "conversation_domain": (
                                "multi_party_supplemental"
                                if ntrack == 3
                                else "two_party"
                            ),
                            "source_view_profile": SOURCE_VIEW_PROFILE,
                            "target_channel": target_channel,
                            "reference_channels": [
                                channel
                                for channel in range(ntrack)
                                if channel != target_channel
                            ],
                            "target_event_ids": event_ids_by_channel[target_channel],
                            "duration_seconds": wav_duration_seconds,
                            "source_sample_rate": wav_sample_rate,
                            "source_frame_count": wav_frame_count,
                            "chunk_ms": CHUNK_MS,
                            "chunk_count": chunk_count,
                            **relation,
                        }
                    )

        source_quarantined_views = sum(
            record["view_count"] for record in quarantine_records
        )
        summary: dict[str, Any] = {
            "schema_version": 1,
            "dataset_version": DATASET_VERSION,
            "source_view_profile": SOURCE_VIEW_PROFILE,
            "chunk_ms": CHUNK_MS,
            "inputs": {
                "audio_archive": str(audio_archive),
                "audio_archive_bytes": audio_archive.stat().st_size,
                "metadata_archive": str(metadata_archive),
                "metadata_archive_bytes": metadata_archive.stat().st_size,
            },
            "sources": {
                "total": len(source_inventory),
                "structurally_usable": sum(
                    item["structurally_usable"] for item in source_inventory
                ),
                "quarantined": len(quarantine_records),
                "with_deterministic_tail_repairs": len(anomaly_records),
                "by_ntrack": dict(sorted(by_ntrack.items())),
            },
            "views": {
                "upper_bound": view_upper_bound,
                "structurally_usable": len(view_records),
                "source_quarantined": source_quarantined_views,
                "by_ntrack": {
                    key: sum(
                        1 for item in view_records if str(item["source_ntrack"]) == key
                    )
                    for key in sorted(by_ntrack)
                },
                "duration_seconds_by_ntrack": {
                    key: round(value, 6)
                    for key, value in sorted(duration_seconds_by_ntrack.items())
                },
                "total_duration_seconds": round(
                    sum(duration_seconds_by_ntrack.values()), 6
                ),
                "total_duration_hours": round(
                    sum(duration_seconds_by_ntrack.values()) / 3600.0, 6
                ),
            },
            "events": {
                "total": len(event_records),
                "state_distribution": state_distribution,
                "activity_from_speaker_segments": sum(
                    event["activity_source"] == "speaker_segments"
                    for event in event_records
                ),
                "activity_fallback_candidates": sum(
                    event["activity_fallback_candidate"] for event in event_records
                ),
            },
        }
        if compute_input_hashes:
            summary["inputs"]["audio_archive_sha256"] = sha256_file(audio_archive)
            summary["inputs"]["metadata_archive_sha256"] = sha256_file(
                metadata_archive
            )
        _validate_expected(summary)

        _write_jsonl(temporary / "source_inventory.jsonl", source_inventory)
        _write_jsonl(temporary / "events.jsonl", event_records)
        _write_jsonl(temporary / "target_views.jsonl", view_records)
        _write_jsonl(temporary / "source_quarantine.jsonl", quarantine_records)
        _write_jsonl(temporary / "source_anomalies.jsonl", anomaly_records)
        _write_json(temporary / "summary.json", summary)
        checksum_lines = []
        for path in sorted(temporary.iterdir(), key=lambda item: item.name):
            if path.is_file():
                checksum_lines.append(f"{sha256_file(path)}  {path.name}\n")
        (temporary / "checksums.sha256").write_text(
            "".join(checksum_lines), encoding="utf-8"
        )
        temporary.rename(output_dir)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan official DuplexConv archives into target-vs-rest views."
    )
    parser.add_argument("--audio-archive", required=True, type=Path)
    parser.add_argument("--metadata-archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--skip-input-hashes",
        action="store_true",
        help="Skip expensive archive hashing (not allowed for the formal scan).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = scan_archives(
        args.audio_archive,
        args.metadata_archive,
        args.output_dir,
        compute_input_hashes=not args.skip_input_hashes,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
