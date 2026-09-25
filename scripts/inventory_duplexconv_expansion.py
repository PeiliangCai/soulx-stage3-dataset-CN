#!/usr/bin/env python3
"""Inventory all Edu metadata into inferred 500-source audio shards."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tarfile
from typing import Any


SOURCE_RE = re.compile(r"Edu--(\d+)\.(json|wav)")
SHARD_SIZE = 500
ANCHOR_SHARD_NUMBER = 18
SAMPLE_WIDTH_BYTES = 2
TARGET_SAMPLE_RATE = 16_000
EDU0018_QWEN_ACCEPTED_COST_USD = 0.2187791
EDU0018_QWEN_REQUEST_SOURCES = 404


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def normalize_state(value: Any) -> str:
    if value is None:
        return "missing"
    text = str(value).strip().lower()
    if not text:
        return "missing"
    if text.startswith("<|") and text.endswith("|>"):
        text = text[2:-2]
    return text


def source_number(name: str, extension: str) -> int:
    match = SOURCE_RE.fullmatch(Path(name).name)
    if not match or match.group(2) != extension:
        raise ValueError(f"unexpected Edu member name: {name}")
    return int(match.group(1))


def summarize(records: list[dict[str, Any]], shard_number: int) -> dict[str, Any]:
    state_counts: Counter[str] = Counter()
    lid_counts: Counter[str] = Counter()
    track_counts: Counter[str] = Counter()
    event_count = 0
    sources_with_missing = 0
    source_seconds = 0.0
    target_seconds = 0.0
    raw_pcm_bytes = 0
    target_pcm_bytes = 0
    for record in records:
        duration = float(record["timeLenInSec"])
        tracks = int(record["nTrack"])
        sample_rate = int(record["fs"])
        channels = record["asr"]
        if len(channels) != tracks:
            raise ValueError(f"asr/nTrack mismatch: {record['_source_id']}")
        track_counts[str(tracks)] += 1
        source_seconds += duration
        target_seconds += duration * tracks
        raw_pcm_bytes += round(duration * sample_rate * tracks * SAMPLE_WIDTH_BYTES)
        target_pcm_bytes += round(duration * TARGET_SAMPLE_RATE * tracks * SAMPLE_WIDTH_BYTES)
        missing_here = False
        for channel in channels:
            for event in channel:
                event_count += 1
                state = normalize_state(event.get("state"))
                state_counts[state] += 1
                missing_here = missing_here or state == "missing"
                lid_counts[str(event.get("LID") or "missing").lower()] += 1
        sources_with_missing += int(missing_here)
    missing_events = state_counts["missing"]
    request_cost = sources_with_missing * (
        EDU0018_QWEN_ACCEPTED_COST_USD / EDU0018_QWEN_REQUEST_SOURCES
    )
    known_events = event_count - missing_events
    return {
        "shard_number": shard_number,
        "inferred_archive_name": f"Edu_{shard_number:04d}.tar",
        "membership_status": (
            "verified_exact_against_local_audio_archive"
            if shard_number == ANCHOR_SHARD_NUMBER
            else "inferred_from_sorted_metadata_chunks_pending_official_manifest"
        ),
        "source_count": len(records),
        "first_source_id": records[0]["_source_id"],
        "last_source_id": records[-1]["_source_id"],
        "track_count_distribution": dict(sorted(track_counts.items())),
        "source_clock_hours": source_seconds / 3600,
        "target_view_hours": target_seconds / 3600,
        "event_count": event_count,
        "state_counts": dict(sorted(state_counts.items())),
        "missing_state_events": missing_events,
        "sources_with_missing_state": sources_with_missing,
        "known_state_events": known_events,
        "wait_events": state_counts["wait"],
        "incomplete_events": state_counts["incomplete"],
        "backchannel_events": state_counts["backchannel"],
        "lid_counts": dict(sorted(lid_counts.items())),
        "estimated_raw_pcm_bytes": raw_pcm_bytes,
        "estimated_target_16k_pcm_bytes": target_pcm_bytes,
        "estimated_raw_plus_target_bytes": raw_pcm_bytes + target_pcm_bytes,
        "qwen_cost_estimate_usd_from_edu0018_request_rate": request_cost,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-archive", type=Path, required=True)
    parser.add_argument("--anchor-audio-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata = args.metadata_archive.resolve(strict=True)
    anchor = args.anchor_audio_archive.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite inventory: {output}")

    started = utc_now()
    records = []
    with tarfile.open(metadata, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        members.sort(key=lambda member: source_number(member.name, "json"))
        for member in members:
            source_id = f"Edu--{source_number(member.name, 'json'):06d}"
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError(f"cannot read metadata member: {member.name}")
            value = json.load(handle)
            if not isinstance(value, dict):
                raise ValueError(f"metadata root is not an object: {member.name}")
            value["_source_id"] = source_id
            records.append(value)

    with tarfile.open(anchor, "r") as archive:
        anchor_ids = sorted(
            f"Edu--{source_number(member.name, 'wav'):06d}"
            for member in archive.getmembers()
            if member.isfile()
        )
    anchor_slice = records[
        (ANCHOR_SHARD_NUMBER - 1) * SHARD_SIZE : ANCHOR_SHARD_NUMBER * SHARD_SIZE
    ]
    inferred_anchor_ids = [item["_source_id"] for item in anchor_slice]
    if anchor_ids != inferred_anchor_ids:
        raise RuntimeError("sorted metadata shard rule does not reproduce Edu_0018")

    shards = []
    for offset in range(0, len(records), SHARD_SIZE):
        shards.append(summarize(records[offset : offset + SHARD_SIZE], offset // SHARD_SIZE + 1))
    totals = summarize(records, 0)
    totals.pop("inferred_archive_name")
    totals.pop("membership_status")
    totals.pop("shard_number")
    payload = {
        "schema_version": 1,
        "status": "complete",
        "profile": "duplexconv-edu-metadata-expansion-inventory-v1",
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "python": sys.version,
        },
        "inputs": {
            "metadata_archive": {
                "path": str(metadata),
                "bytes": metadata.stat().st_size,
                "sha256": sha256_file(metadata),
            },
            "anchor_audio_archive": {
                "path": str(anchor),
                "bytes": anchor.stat().st_size,
                "sha256": sha256_file(anchor),
                "shard_number": ANCHOR_SHARD_NUMBER,
                "member_count": len(anchor_ids),
            },
        },
        "membership_inference": {
            "rule": "sort all Edu metadata by numeric source ID and group consecutive records into chunks of 500",
            "anchor_verification": "Edu_0018 exact 500/500 member match",
            "limitation": "all non-anchor shard names and member ranges remain inferred until checked against the official archive manifest",
        },
        "cost_estimate_basis": {
            "edu0018_accepted_qwen_cost_usd": EDU0018_QWEN_ACCEPTED_COST_USD,
            "edu0018_grouped_source_requests": EDU0018_QWEN_REQUEST_SOURCES,
            "warning": "linear planning estimate only; actual token lengths and missing-state distribution can change cost",
        },
        "shard_count": len(shards),
        "shards": shards,
        "totals": totals,
        "selection_used_benchmark_results": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "run_manifest.json", payload)
    atomic_write(
        output / "shards.jsonl",
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in shards),
    )
    lines = [
        "# DuplexConv Edu expansion metadata inventory",
        "",
        f"- Metadata sources: {totals['source_count']:,}",
        f"- Inferred shards: {len(shards)} (500 sources each except the final shard)",
        f"- Source clock hours: {totals['source_clock_hours']:.3f}",
        f"- Target-view hours: {totals['target_view_hours']:.3f}",
        f"- Events: {totals['event_count']:,}",
        f"- Missing-state events: {totals['missing_state_events']:,}",
        f"- Sources with missing state: {totals['sources_with_missing_state']:,}",
        f"- Estimated raw PCM: {totals['estimated_raw_pcm_bytes'] / 1e9:.3f} GB",
        f"- Estimated target-view 16 kHz PCM: {totals['estimated_target_16k_pcm_bytes'] / 1e9:.3f} GB",
        f"- Linear Qwen estimate: ${totals['qwen_cost_estimate_usd_from_edu0018_request_rate']:.3f}",
        "",
        "Only Edu_0018 membership is verified against a local official audio tar. All other shard memberships must be checked against the official file manifest before download.",
        "",
    ]
    atomic_write(output / "inventory_summary.md", "\n".join(lines))
    print(json.dumps({"status": "complete", "output": str(output), "shards": len(shards), "totals": totals}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
