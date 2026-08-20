"""Build auditable per-chunk Stage 3 state timelines."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import shutil
from typing import Any, Sequence

from .source_scan import interval_chunk_bounds, sha256_file
from .state_labeling import canonical_json, read_jsonl


TIMELINE_PROFILE = "duplexconv-stage3-state-timeline-v1"
STATE_TOKEN_BY_EVENT_STATE = {
    "complete": "user_nonidle",
    "incomplete": "user_nonidle",
    "backchannel": "user_backchannel",
}
TERMINAL_TOKEN_BY_EVENT_STATE = {
    "complete": "user_complete",
    "incomplete": "user_incomplete",
}


def _event_activity_chunks(
    event: dict[str, Any],
    *,
    chunk_count: int,
    fallback_evidence_ids: set[str],
) -> tuple[list[int], str, str]:
    segments = event.get("speaker_segments_ms") or []
    if segments:
        chunks = set()
        for start_ms, end_ms in segments:
            start, stop = interval_chunk_bounds(start_ms, end_ms, chunk_count)
            chunks.update(range(start, stop))
        return sorted(chunks), "speaker_segments", "applied"
    if event.get("activity_fallback_candidate"):
        if event["event_id"] not in fallback_evidence_ids:
            return [], "event_envelope_fallback", "missing_paraformer_evidence"
        start_ms, end_ms = event["effective_event_envelope_ms"]
        start, stop = interval_chunk_bounds(start_ms, end_ms, chunk_count)
        return list(range(start, stop)), "event_envelope_fallback", "applied"
    return [], "none", "non_acoustic_placeholder"


def _contiguous_ranges(chunks: Sequence[int]) -> list[list[int]]:
    if not chunks:
        return []
    ranges = []
    start = previous = chunks[0]
    for chunk in chunks[1:]:
        if chunk != previous + 1:
            ranges.append([start, previous + 1])
            start = chunk
        previous = chunk
    ranges.append([start, previous + 1])
    return ranges


def build_view_timeline(
    *,
    view: dict[str, Any],
    events: Sequence[dict[str, Any]],
    asr: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    view_id = view["view_id"]
    original_chunk_count = view["chunk_count"]
    if asr["view_id"] != view_id or asr["chunk_count"] != original_chunk_count:
        raise ValueError(f"ASR/view contract mismatch for {view_id}")
    if len(asr["chunk_asr_targets"]) != original_chunk_count:
        raise ValueError(f"ASR chunk target count mismatch for {view_id}")

    fallback_evidence_ids = set(asr["fallback_event_ids_with_token_evidence"])
    assignments = []
    resolved_activity: dict[str, list[int]] = {}
    needs_terminal_padding = False
    for event in sorted(events, key=lambda item: item["event_id"]):
        if event["final_state"] not in STATE_TOKEN_BY_EVENT_STATE:
            raise ValueError(f"invalid final state for {event['event_id']}")
        chunks, activity_source, status = _event_activity_chunks(
            event,
            chunk_count=original_chunk_count,
            fallback_evidence_ids=fallback_evidence_ids,
        )
        resolved_activity[event["event_id"]] = chunks
        decision_chunk = None
        if status == "applied" and event["final_state"] in TERMINAL_TOKEN_BY_EVENT_STATE:
            decision_chunk = chunks[-1] + 1
            needs_terminal_padding |= decision_chunk >= original_chunk_count
        assignments.append(
            {
                "event_id": event["event_id"],
                "final_state": event["final_state"],
                "state_label_source": event["state_label_source"],
                "activity_source": activity_source,
                "activity_chunk_ranges": _contiguous_ranges(chunks),
                "decision_chunk": decision_chunk,
                "status": status,
            }
        )

    effective_chunk_count = original_chunk_count + int(needs_terminal_padding)
    claims: dict[int, list[dict[str, str]]] = defaultdict(list)
    assignment_by_id = {item["event_id"]: item for item in assignments}
    for event in events:
        assignment = assignment_by_id[event["event_id"]]
        if assignment["status"] != "applied":
            continue
        activity_state = STATE_TOKEN_BY_EVENT_STATE[event["final_state"]]
        for chunk in resolved_activity[event["event_id"]]:
            claims[chunk].append(
                {"event_id": event["event_id"], "kind": "activity", "state": activity_state}
            )
        if assignment["decision_chunk"] is not None:
            claims[assignment["decision_chunk"]].append(
                {
                    "event_id": event["event_id"],
                    "kind": "terminal",
                    "state": TERMINAL_TOKEN_BY_EVENT_STATE[event["final_state"]],
                }
            )

    quarantines = []
    forced_quarantine_chunks: set[int] = set()
    for assignment in assignments:
        if assignment["status"] == "missing_paraformer_evidence":
            event = next(item for item in events if item["event_id"] == assignment["event_id"])
            start, stop = interval_chunk_bounds(
                *event["effective_event_envelope_ms"], original_chunk_count
            )
            quarantines.append(
                {
                    "view_id": view_id,
                    "kind": "fallback_event_without_paraformer_evidence",
                    "event_ids": [assignment["event_id"]],
                    "chunk_range": [start, stop],
                }
            )
            forced_quarantine_chunks.update(range(start, stop))

    chunk_states: list[str | None] = []
    conflict_chunks = []
    for chunk in range(effective_chunk_count):
        chunk_claims = claims.get(chunk, [])
        distinct_states = sorted({item["state"] for item in chunk_claims})
        if len(distinct_states) > 1:
            chunk_states.append(None)
            conflict_chunks.append(chunk)
            for item in chunk_claims:
                assignment_by_id[item["event_id"]]["status"] = "state_conflict"
        elif distinct_states:
            chunk_states.append(distinct_states[0])
        else:
            chunk_states.append("user_idle")
    for start, stop in _contiguous_ranges(conflict_chunks):
        event_ids = sorted(
            {
                item["event_id"]
                for chunk in range(start, stop)
                for item in claims.get(chunk, [])
            }
        )
        quarantines.append(
            {
                "view_id": view_id,
                "kind": "state_claim_conflict",
                "event_ids": event_ids,
                "chunk_range": [start, stop],
            }
        )

    primary_activity = list(view["target_active_by_chunk"])
    covered_primary = {
        chunk
        for event_id, chunks in resolved_activity.items()
        if assignment_by_id[event_id]["status"] in {"applied", "state_conflict"}
        for chunk in chunks
    }
    uncovered_primary = [
        chunk for chunk, active in enumerate(primary_activity) if active and chunk not in covered_primary
    ]
    for start, stop in _contiguous_ranges(uncovered_primary):
        quarantines.append(
            {
                "view_id": view_id,
                "kind": "target_activity_without_state_event",
                "event_ids": [],
                "chunk_range": [start, stop],
            }
        )
        for chunk in range(start, stop):
            chunk_states[chunk] = None

    for chunk in forced_quarantine_chunks:
        chunk_states[chunk] = None

    for token_index in asr["asr_token_outside_target_activity"]:
        token = asr["tokens"][token_index]
        chunk = token["emit_chunk"]
        chunk_states[chunk] = None
        quarantines.append(
            {
                "view_id": view_id,
                "kind": "asr_token_outside_target_activity",
                "event_ids": [],
                "token_indices": [token_index],
                "chunk_range": [chunk, chunk + 1],
            }
        )

    padding = effective_chunk_count - original_chunk_count
    return (
        {
            "timeline_profile": TIMELINE_PROFILE,
            "view_id": view_id,
            "source_id": view["source_id"],
            "source_ntrack": view["source_ntrack"],
            "target_channel": view["target_channel"],
            "reference_channels": view["reference_channels"],
            "original_chunk_count": original_chunk_count,
            "effective_chunk_count": effective_chunk_count,
            "terminal_silence_padding_chunks": padding,
            "chunk_asr_targets": list(asr["chunk_asr_targets"]) + [""] * padding,
            "chunk_states": chunk_states,
            "target_active_by_chunk": primary_activity + [False] * padding,
            "other_active_by_chunk": list(view["other_active_by_chunk"]) + [False] * padding,
            "other_active_count_by_chunk": list(view["other_active_count_by_chunk"]) + [0] * padding,
            "overlap_by_chunk": list(view["overlap_by_chunk"]) + [False] * padding,
            "event_assignments": assignments,
            "asr_cache_signature": asr["cache_signature"],
            "asr_token_count": len(asr["tokens"]),
            "quarantine_count": len(quarantines),
        },
        quarantines,
    )


def build_timelines(
    *, scan_dir: Path, state_dir: Path, asr_dir: Path, output_dir: Path
) -> dict[str, Any]:
    scan_dir = scan_dir.resolve(strict=True)
    state_dir = state_dir.resolve(strict=True)
    asr_dir = asr_dir.resolve(strict=True)
    output_dir = output_dir.absolute()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}"
    temporary.mkdir()
    try:
        views = {item["view_id"]: item for item in read_jsonl(scan_dir / "target_views.jsonl")}
        events_by_view: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in read_jsonl(state_dir / "events_with_final_state.jsonl"):
            view_id = f"{event['source_id']}/target-ch{event['channel']:02d}"
            events_by_view[view_id].append(event)
        asr_results = {item["view_id"]: item for item in read_jsonl(asr_dir / "asr_results.jsonl")}
        asr_quarantine = list(read_jsonl(asr_dir / "asr_quarantine.jsonl"))
        if asr_quarantine:
            raise ValueError("ASR quarantine is non-empty; resolve it before timeline construction")
        if set(views) != set(asr_results):
            raise ValueError("target view and ASR result ID sets differ")

        timelines = []
        quarantines = []
        for view_id in sorted(views):
            timeline, view_quarantines = build_view_timeline(
                view=views[view_id],
                events=events_by_view[view_id],
                asr=asr_results[view_id],
            )
            timelines.append(timeline)
            quarantines.extend(view_quarantines)
        with (temporary / "timelines.jsonl").open("w", encoding="utf-8") as handle:
            for item in timelines:
                handle.write(canonical_json(item) + "\n")
        with (temporary / "timeline_quarantine.jsonl").open("w", encoding="utf-8") as handle:
            for item in quarantines:
                handle.write(canonical_json(item) + "\n")

        assignment_status_counts = Counter(
            assignment["status"]
            for timeline in timelines
            for assignment in timeline["event_assignments"]
        )
        state_counts = Counter(
            state for timeline in timelines for state in timeline["chunk_states"]
        )
        summary = {
            "schema_version": 1,
            "timeline_profile": TIMELINE_PROFILE,
            "view_count": len(timelines),
            "event_count": sum(len(item["event_assignments"]) for item in timelines),
            "original_chunk_count": sum(item["original_chunk_count"] for item in timelines),
            "effective_chunk_count": sum(item["effective_chunk_count"] for item in timelines),
            "terminal_silence_padding_chunk_count": sum(
                item["terminal_silence_padding_chunks"] for item in timelines
            ),
            "assignment_status_counts": dict(sorted(assignment_status_counts.items())),
            "chunk_state_counts": {
                "quarantined": state_counts.pop(None, 0),
                **dict(sorted(state_counts.items())),
            },
            "quarantine_record_count": len(quarantines),
            "views_with_quarantine": len({item["view_id"] for item in quarantines}),
            "quarantine_kind_counts": dict(
                sorted(Counter(item["kind"] for item in quarantines).items())
            ),
        }
        (temporary / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        checksum_lines = []
        for path in sorted(temporary.iterdir(), key=lambda item: item.name):
            if path.is_file():
                checksum_lines.append(f"{sha256_file(path)}  {path.name}\n")
        (temporary / "checksums.sha256").write_text("".join(checksum_lines), encoding="utf-8")
        temporary.rename(output_dir)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Stage 3 state timelines.")
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--asr-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_timelines(
        scan_dir=args.scan_dir,
        state_dir=args.state_dir,
        asr_dir=args.asr_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
