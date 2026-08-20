"""Export audited Stage 3 timelines as official two-column Parquet data."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Sequence

from .source_scan import sha256_file
from .state_labeling import canonical_json, read_jsonl


DATASET_VERSION = "duplexconv_edu0018_stage3_zh_v1"
SEQUENCE_PROFILE = "soulx-stage3-chunk-groups-v1"
PREFIX = "<|task_duplex_predict|><|punctuation_off|>"
EOS = "<|end_of_sentence|>"
MAX_TOKEN_LENGTH = 1500
VALID_STATES = {
    "user_idle",
    "user_nonidle",
    "user_complete",
    "user_incomplete",
    "user_backchannel",
}


def contiguous_usable_spans(states: Sequence[str | None]) -> list[tuple[int, int]]:
    spans = []
    start = None
    for index, state in enumerate(list(states) + [None]):
        if state is not None and start is None:
            start = index
        elif state is None and start is not None:
            spans.append((start, index))
            start = None
    return spans


def make_chunk_group(audio_tokens: Sequence[int], text: str, state: str) -> str:
    if len(audio_tokens) != 2:
        raise ValueError("each Stage 3 chunk requires exactly two audio tokens")
    if state not in VALID_STATES:
        raise ValueError(f"invalid Stage 3 state: {state}")
    if not isinstance(text, str) or "<|" in text or "|>" in text:
        raise ValueError("ASR text contains an invalid control-token fragment")
    return (
        f"<|audio_{audio_tokens[0]}|><|audio_{audio_tokens[1]}|>"
        f"{text}{EOS}<|{state}|>"
    )


def greedy_windows(
    *, tokenizer: Any, groups: Sequence[str], span_start: int, max_token_length: int
) -> tuple[list[tuple[int, int, str, int]], list[int]]:
    prefix_length = len(tokenizer.encode(PREFIX, add_special_tokens=False))
    windows = []
    oversized = []
    start = 0
    length = prefix_length
    for index, group in enumerate(groups):
        group_length = len(tokenizer.encode(group, add_special_tokens=False))
        if prefix_length + group_length > max_token_length:
            if index > start:
                sequence = PREFIX + "".join(groups[start:index])
                actual = len(tokenizer.encode(sequence))
                if actual > max_token_length:
                    raise ValueError("tokenizer boundary assumption exceeded max length")
                windows.append((span_start + start, span_start + index, sequence, actual))
            oversized.append(span_start + index)
            start = index + 1
            length = prefix_length
            continue
        if index > start and length + group_length > max_token_length:
            sequence = PREFIX + "".join(groups[start:index])
            actual = len(tokenizer.encode(sequence))
            if actual > max_token_length:
                raise ValueError("tokenizer boundary assumption exceeded max length")
            windows.append((span_start + start, span_start + index, sequence, actual))
            start = index
            length = prefix_length
        length += group_length
    if start < len(groups):
        sequence = PREFIX + "".join(groups[start:])
        actual = len(tokenizer.encode(sequence))
        if actual > max_token_length:
            raise ValueError("tokenizer boundary assumption exceeded max length")
        windows.append((span_start + start, span_start + len(groups), sequence, actual))
    return windows, oversized


def _event_ids_for_window(assignments: Sequence[dict[str, Any]], start: int, stop: int) -> list[str]:
    event_ids = []
    for assignment in assignments:
        intersects = any(
            range_start < stop and range_stop > start
            for range_start, range_stop in assignment["activity_chunk_ranges"]
        )
        decision = assignment["decision_chunk"]
        if intersects or (decision is not None and start <= decision < stop):
            event_ids.append(assignment["event_id"])
    return sorted(event_ids)


def export_model_ready(
    *,
    timeline_dir: Path,
    glm_dir: Path,
    tokenizer_dir: Path,
    output_dir: Path,
    upstream_commit: str,
    max_token_length: int = MAX_TOKEN_LENGTH,
) -> dict[str, Any]:
    timeline_dir = timeline_dir.resolve(strict=True)
    glm_dir = glm_dir.resolve(strict=True)
    tokenizer_dir = tokenizer_dir.resolve(strict=True)
    output_dir = output_dir.absolute()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}"
    (temporary / "data").mkdir(parents=True)
    (temporary / "metadata").mkdir()
    (temporary / "quarantine").mkdir()
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        expected_token_ids = {
            "task_duplex_predict": 151670,
            "punctuation_off": 151672,
            "end_of_sentence": 151674,
            "user_complete": 151676,
            "user_backchannel": 151677,
            "user_incomplete": 151678,
            "user_idle": 151680,
            "user_nonidle": 151681,
            "audio_0": 151700,
            "audio_51865": 203565,
        }
        for name, expected in expected_token_ids.items():
            token = f"<|{name}|>"
            actual = tokenizer.encode(token, add_special_tokens=False)
            if actual != [expected]:
                raise ValueError(f"tokenizer contract mismatch for {token}: {actual}")

        timelines = list(read_jsonl(timeline_dir / "timelines.jsonl"))
        glm_by_view = {item["view_id"]: item for item in read_jsonl(glm_dir / "audio_tokens.jsonl")}
        glm_quarantine = list(read_jsonl(glm_dir / "glm_quarantine.jsonl"))
        if glm_quarantine:
            raise ValueError("GLM quarantine is non-empty; resolve it before export")
        if {item["view_id"] for item in timelines} != set(glm_by_view):
            raise ValueError("timeline and GLM view ID sets differ")

        rows = []
        metadata_rows = []
        quarantine_rows = []
        exported_chunks: set[tuple[str, int]] = set()
        for timeline in timelines:
            view_id = timeline["view_id"]
            glm = glm_by_view[view_id]
            chunk_count = timeline["effective_chunk_count"]
            if glm["effective_chunk_count"] != chunk_count:
                raise ValueError(f"GLM/timeline chunk mismatch for {view_id}")
            if not (
                len(timeline["chunk_states"])
                == len(timeline["chunk_asr_targets"])
                == len(glm["chunk_audio_tokens"])
                == chunk_count
            ):
                raise ValueError(f"timeline arrays do not close for {view_id}")

            for chunk, state in enumerate(timeline["chunk_states"]):
                if state is None:
                    quarantine_rows.append(
                        {"view_id": view_id, "chunk_range": [chunk, chunk + 1], "reason": "timeline_quarantine"}
                    )
            for span_start, span_stop in contiguous_usable_spans(timeline["chunk_states"]):
                groups = [
                    make_chunk_group(
                        glm["chunk_audio_tokens"][chunk],
                        timeline["chunk_asr_targets"][chunk],
                        timeline["chunk_states"][chunk],
                    )
                    for chunk in range(span_start, span_stop)
                ]
                windows, oversized = greedy_windows(
                    tokenizer=tokenizer,
                    groups=groups,
                    span_start=span_start,
                    max_token_length=max_token_length,
                )
                for chunk in oversized:
                    quarantine_rows.append(
                        {"view_id": view_id, "chunk_range": [chunk, chunk + 1], "reason": "single_chunk_over_1500_tokens"}
                    )
                for start, stop, sequence, tokenized_length in windows:
                    index = (
                        f"duplexconv_edu0018__{timeline['source_id']}__"
                        f"ch{timeline['target_channel']:02d}__c{start:06d}-{stop:06d}"
                    )
                    if any((view_id, chunk) in exported_chunks for chunk in range(start, stop)):
                        raise ValueError(f"chunk exported more than once for {view_id}")
                    exported_chunks.update((view_id, chunk) for chunk in range(start, stop))
                    event_ids = _event_ids_for_window(timeline["event_assignments"], start, stop)
                    qwen_event_ids = sorted(
                        assignment["event_id"]
                        for assignment in timeline["event_assignments"]
                        if assignment["event_id"] in event_ids
                        and assignment["state_label_source"].startswith("openrouter_")
                    )
                    rows.append({"index": index, "sequence": sequence})
                    metadata_rows.append(
                        {
                            "index": index,
                            "view_id": view_id,
                            "source_id": timeline["source_id"],
                            "source_ntrack": timeline["source_ntrack"],
                            "target_channel": timeline["target_channel"],
                            "reference_channels": timeline["reference_channels"],
                            "chunk_range": [start, stop],
                            "chunk_count": stop - start,
                            "tokenized_length": tokenized_length,
                            "terminal_silence_padding_in_window": int(
                                timeline["terminal_silence_padding_chunks"] > 0
                                and stop == chunk_count
                            ),
                            "other_active_chunk_count": sum(
                                timeline["other_active_by_chunk"][start:stop]
                            ),
                            "overlap_chunk_count": sum(timeline["overlap_by_chunk"][start:stop]),
                            "event_ids": event_ids,
                            "qwen_event_ids": qwen_event_ids,
                            "asr_cache_signature": timeline["asr_cache_signature"],
                            "glm_cache_signature": glm["cache_signature"],
                            "sequence_sha256": hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
                        }
                    )

        if len({row["index"] for row in rows}) != len(rows):
            raise ValueError("model-ready indices are not globally unique")
        total_chunks = sum(item["effective_chunk_count"] for item in timelines)
        quarantined_chunk_ids = {
            (item["view_id"], chunk)
            for item in quarantine_rows
            for chunk in range(*item["chunk_range"])
        }
        if len(exported_chunks) + len(quarantined_chunk_ids) != total_chunks:
            raise ValueError("exported/quarantined chunks do not close")

        table = pa.Table.from_pylist(
            rows,
            schema=pa.schema([pa.field("index", pa.string()), pa.field("sequence", pa.string())]),
        )
        pq.write_table(table, temporary / "data" / "train-00000-of-00001.parquet", compression="zstd")
        with (temporary / "metadata" / "windows.jsonl").open("w", encoding="utf-8") as handle:
            for item in metadata_rows:
                handle.write(canonical_json(item) + "\n")
        with (temporary / "quarantine" / "chunks.jsonl").open("w", encoding="utf-8") as handle:
            for item in quarantine_rows:
                handle.write(canonical_json(item) + "\n")

        contract = {
            "schema_version": 1,
            "dataset_version": DATASET_VERSION,
            "sequence_profile": SEQUENCE_PROFILE,
            "columns": ["index", "sequence"],
            "prefix": PREFIX,
            "chunk_group": "audio_token*2 + incremental_asr + end_of_sentence + user_state",
            "max_token_length": max_token_length,
            "token_ids": expected_token_ids,
            "audio_token_raw_range": [0, 51865],
            "upstream_commit": upstream_commit,
            "timeline_profile": timelines[0]["timeline_profile"] if timelines else None,
            "glm_audio_profile": next(iter(glm_by_view.values()))["glm_audio_profile"] if glm_by_view else None,
        }
        (temporary / "contract.json").write_text(
            json.dumps(contract, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        state_counts = Counter(
            state
            for timeline in timelines
            for state in timeline["chunk_states"]
            if state is not None
        )
        stats = {
            "schema_version": 1,
            "dataset_version": DATASET_VERSION,
            "row_count": len(rows),
            "source_view_count": len(timelines),
            "total_effective_chunk_count": total_chunks,
            "exported_chunk_count": len(exported_chunks),
            "quarantined_chunk_count": len(quarantined_chunk_ids),
            "max_observed_tokenized_length": max(
                (item["tokenized_length"] for item in metadata_rows), default=0
            ),
            "rows_by_ntrack": dict(
                sorted(Counter(str(item["source_ntrack"]) for item in metadata_rows).items())
            ),
            "chunk_state_counts_before_window_quarantine": dict(sorted(state_counts.items())),
        }
        (temporary / "stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        checksum_lines = []
        for path in sorted(temporary.rglob("*")):
            if path.is_file():
                checksum_lines.append(f"{sha256_file(path)}  {path.relative_to(temporary)}\n")
        (temporary / "checksums.sha256").write_text("".join(checksum_lines), encoding="utf-8")
        temporary.rename(output_dir)
        return stats
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export official two-column Stage 3 Parquet.")
    parser.add_argument("--timeline-dir", type=Path, required=True)
    parser.add_argument("--glm-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--upstream-commit", required=True)
    parser.add_argument("--max-token-length", type=int, default=MAX_TOKEN_LENGTH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stats = export_model_ready(
        timeline_dir=args.timeline_dir,
        glm_dir=args.glm_dir,
        tokenizer_dir=args.tokenizer_dir,
        output_dir=args.output_dir,
        upstream_commit=args.upstream_commit,
        max_token_length=args.max_token_length,
    )
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
