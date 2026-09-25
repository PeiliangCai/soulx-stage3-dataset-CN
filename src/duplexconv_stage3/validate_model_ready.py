"""Validate model-ready Parquet and load it with the unmodified SoulX loader."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Sequence

from .source_scan import sha256_file


AUDIO_START = 151700
AUDIO_STOP = 203566
EOS_ID = 151674
STATE_IDS = {151676, 151677, 151678, 151680, 151681}
PREFIX_IDS = [151670, 151672]


def discover_parquet_files(model_ready_dir: Path) -> list[Path]:
    paths = sorted((model_ready_dir / "data").glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no Parquet files under {model_ready_dir / 'data'}")
    return paths


def discover_metadata_files(model_ready_dir: Path) -> list[Path]:
    legacy = model_ready_dir / "metadata" / "windows.jsonl"
    paths = [legacy] if legacy.exists() else sorted(
        (model_ready_dir / "metadata" / "by_shard").glob("*.windows.jsonl")
    )
    if not paths:
        raise FileNotFoundError(f"no window metadata under {model_ready_dir / 'metadata'}")
    return paths


def discover_quarantine_files(model_ready_dir: Path, kind: str) -> list[Path]:
    legacy = model_ready_dir / "quarantine" / f"{kind}.jsonl"
    suffix = "chunks.jsonl" if kind == "chunks" else "views.jsonl"
    return [legacy] if legacy.exists() else sorted(
        (model_ready_dir / "quarantine" / "by_shard").glob(f"*.{suffix}")
    )


def parse_sequence_ids(ids: Sequence[int]) -> dict[str, int]:
    if list(ids[:2]) != PREFIX_IDS:
        raise ValueError("sequence prefix mismatch")
    position = 2
    chunks = 0
    text_tokens = 0
    while position < len(ids):
        if position + 1 >= len(ids) or not (
            AUDIO_START <= ids[position] < AUDIO_STOP
            and AUDIO_START <= ids[position + 1] < AUDIO_STOP
        ):
            raise ValueError(f"chunk {chunks} does not start with exactly two audio tokens")
        position += 2
        while position < len(ids) and ids[position] != EOS_ID:
            if ids[position] >= AUDIO_START or ids[position] in STATE_IDS:
                raise ValueError(f"unexpected control/audio token in ASR text at chunk {chunks}")
            text_tokens += 1
            position += 1
        if position >= len(ids) or ids[position] != EOS_ID:
            raise ValueError(f"chunk {chunks} has no EOS")
        position += 1
        if position >= len(ids) or ids[position] not in STATE_IDS:
            raise ValueError(f"chunk {chunks} has no legal user state")
        position += 1
        chunks += 1
    return {"chunk_count": chunks, "text_token_count": text_tokens}


def validate_quarantine_closure(
    *,
    stats: dict[str, Any],
    metadata_rows: Sequence[dict[str, Any]],
    chunk_quarantine: Sequence[dict[str, Any]],
    source_view_quarantine: Sequence[dict[str, Any]],
    parsed_chunk_count: int,
) -> dict[str, int]:
    if parsed_chunk_count != stats["exported_chunk_count"]:
        raise ValueError("parsed/exported chunk count mismatch")
    chunk_ids = []
    for item in chunk_quarantine:
        start, stop = item["chunk_range"]
        if not isinstance(start, int) or not isinstance(stop, int) or not 0 <= start < stop:
            raise ValueError("invalid quarantined chunk range")
        chunk_ids.extend((item["view_id"], chunk) for chunk in range(start, stop))
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("chunk quarantine contains overlapping chunk IDs")
    if len(chunk_ids) != stats["quarantined_chunk_count"]:
        raise ValueError("chunk quarantine count differs from export stats")

    view_ids = [item["view_id"] for item in source_view_quarantine]
    if len(set(view_ids)) != len(view_ids):
        raise ValueError("source-view quarantine contains duplicate view IDs")
    metadata_view_ids = {item["view_id"] for item in metadata_rows}
    overlap = metadata_view_ids & set(view_ids)
    if overlap:
        raise ValueError(
            f"exported/source-view quarantine IDs overlap: {sorted(overlap)}"
        )
    source_view_quarantined_chunks = 0
    source_view_quarantined_events = 0
    for item in source_view_quarantine:
        chunk_count = item.get("original_chunk_count")
        event_count = item.get("event_count")
        if not isinstance(chunk_count, int) or chunk_count < 1:
            raise ValueError("invalid source-view quarantine chunk count")
        if not isinstance(event_count, int) or event_count < 0:
            raise ValueError("invalid source-view quarantine event count")
        if event_count != len(item.get("event_ids", [])):
            raise ValueError("source-view quarantine event closure failed")
        source_view_quarantined_chunks += chunk_count
        source_view_quarantined_events += event_count

    if "input_source_view_count" not in stats:
        if source_view_quarantine:
            raise ValueError(
                "legacy export stats cannot accompany source-view quarantine records"
            )
        if stats["exported_chunk_count"] + stats["quarantined_chunk_count"] != stats[
            "total_effective_chunk_count"
        ]:
            raise ValueError("legacy exported/quarantined chunk partition does not close")
        return {
            "chunk_quarantine_count": len(chunk_ids),
            "source_view_quarantine_count": 0,
            "source_view_quarantined_chunk_count": 0,
            "source_view_quarantined_event_count": 0,
        }

    if stats.get("source_view_quarantined_count", 0) != len(view_ids):
        raise ValueError("source-view quarantine count differs from export stats")
    if (
        stats.get("source_view_quarantined_chunk_count", 0)
        != source_view_quarantined_chunks
    ):
        raise ValueError("source-view quarantined chunks differ from export stats")
    if (
        stats.get("source_view_quarantined_event_count", 0)
        != source_view_quarantined_events
    ):
        raise ValueError("source-view quarantined events differ from export stats")
    if stats.get("input_source_view_count") != (
        stats["source_view_count"] + len(view_ids)
    ):
        raise ValueError("input source-view partition does not close")
    expected_input_chunks = (
        stats["exported_chunk_count"]
        + stats["quarantined_chunk_count"]
        + source_view_quarantined_chunks
    )
    if stats.get("input_total_chunk_count") != expected_input_chunks:
        raise ValueError("global input chunk partition does not close")
    if stats.get("total_quarantined_chunk_count") != (
        stats["quarantined_chunk_count"] + source_view_quarantined_chunks
    ):
        raise ValueError("total quarantined chunk count mismatch")
    return {
        "chunk_quarantine_count": len(chunk_ids),
        "source_view_quarantine_count": len(view_ids),
        "source_view_quarantined_chunk_count": source_view_quarantined_chunks,
        "source_view_quarantined_event_count": source_view_quarantined_events,
    }


def validate(
    *,
    model_ready_dir: Path,
    tokenizer_dir: Path,
    upstream_dir: Path,
    report_path: Path,
    random_sample_count: int = 20,
) -> dict[str, Any]:
    model_ready_dir = model_ready_dir.resolve(strict=True)
    tokenizer_dir = tokenizer_dir.resolve(strict=True)
    upstream_dir = upstream_dir.resolve(strict=True)
    report_path = report_path.absolute()
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite report: {report_path}")

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    checksum_failures = []
    for line in (model_ready_dir / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if sha256_file(model_ready_dir / relative) != expected:
            checksum_failures.append(relative)
    if checksum_failures:
        raise ValueError(f"checksum failures: {checksum_failures}")

    import pyarrow as pa

    parquet_files = discover_parquet_files(model_ready_dir)
    table = pa.concat_tables([pq.read_table(path) for path in parquet_files])
    if table.column_names != ["index", "sequence"]:
        raise ValueError(f"Parquet columns differ: {table.column_names}")
    rows = table.to_pylist()
    if len({row["index"] for row in rows}) != len(rows):
        raise ValueError("Parquet indices are not unique")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    lengths = []
    parsed_chunks = 0
    for row in rows:
        ids = tokenizer.encode(row["sequence"])
        if len(ids) > 1500:
            raise ValueError(f"sequence over 1500 tokens: {row['index']}")
        parsed = parse_sequence_ids(ids)
        parsed_chunks += parsed["chunk_count"]
        lengths.append(len(ids))

    metadata = {}
    metadata_rows = []
    metadata_files = discover_metadata_files(model_ready_dir)
    for metadata_file in metadata_files:
        for line in metadata_file.open(encoding="utf-8"):
            item = json.loads(line)
            if item["index"] in metadata:
                raise ValueError(f"duplicate metadata index: {item['index']}")
            metadata_rows.append(item)
            metadata[item["index"]] = item
    if set(metadata) != {row["index"] for row in rows}:
        raise ValueError("metadata and Parquet index sets differ")
    rng = random.Random(42)
    sampled = rng.sample(rows, min(random_sample_count, len(rows)))
    for row in sampled:
        item = metadata[row["index"]]
        if hashlib.sha256(row["sequence"].encode("utf-8")).hexdigest() != item[
            "sequence_sha256"
        ]:
            raise ValueError(f"sequence hash mismatch for {row['index']}")
        parsed = parse_sequence_ids(tokenizer.encode(row["sequence"]))
        if parsed["chunk_count"] != item["chunk_count"]:
            raise ValueError(f"round-trip chunk mismatch for {row['index']}")

    stats = json.loads((model_ready_dir / "stats.json").read_text(encoding="utf-8"))
    chunk_quarantine = [
        json.loads(line)
        for path in discover_quarantine_files(model_ready_dir, "chunks")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_view_quarantine = [
        json.loads(line)
        for path in discover_quarantine_files(model_ready_dir, "views")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    quarantine_closure = validate_quarantine_closure(
        stats=stats,
        metadata_rows=metadata_rows,
        chunk_quarantine=chunk_quarantine,
        source_view_quarantine=source_view_quarantine,
        parsed_chunk_count=parsed_chunks,
    )

    sys.path.insert(0, str(upstream_dir))
    from config.config import RunConfig
    from models.state_prediction_data import State_Prediction_Dataset

    config = RunConfig()
    config.model_config.model_name = str(tokenizer_dir)
    config.dataset_config.train_data_path = str(model_ready_dir / "data")
    config.dataset_config.split_size = 0.05
    config.dataset_config.max_token_length = 1500
    config.train_config.seed = 42
    official_train = State_Prediction_Dataset(config, "train")
    official_sample = official_train[0]
    if set(official_sample) != {
        "index",
        "input_id",
        "audio_mask",
        "label_text",
        "label_eos",
        "label_user_idle",
        "label_user_nonidle",
        "label_user_complete",
        "label_user_incomplete",
        "label_user_backchannel",
    }:
        raise ValueError("unmodified official loader sample contract changed")

    report = {
        "schema_version": 1,
        "status": "passed",
        "model_ready_dir": str(model_ready_dir),
        "row_count": len(rows),
        "parsed_chunk_count": parsed_chunks,
        "min_tokenized_length": min(lengths),
        "max_tokenized_length": max(lengths),
        "random_roundtrip_sample_count": len(sampled),
        "checksum_failure_count": 0,
        "parquet_columns": table.column_names,
        "parquet_file_count": len(parquet_files),
        "parquet_files": [str(path) for path in parquet_files],
        "metadata_file_count": len(metadata_files),
        "official_loader": {
            "source": str(upstream_dir / "models/state_prediction_data.py"),
            "train_split_length": len(official_train),
            "sample_index": official_sample["index"],
            "sample_token_length": len(official_sample["input_id"]),
        },
        "export_stats_sha256": sha256_file(model_ready_dir / "stats.json"),
        "exported_chunk_count": stats["exported_chunk_count"],
        "input_source_view_count": stats.get(
            "input_source_view_count", stats["source_view_count"]
        ),
        "source_view_count": stats["source_view_count"],
        "source_view_quarantined_count": quarantine_closure[
            "source_view_quarantine_count"
        ],
        "source_view_quarantined_chunk_count": quarantine_closure[
            "source_view_quarantined_chunk_count"
        ],
        "source_view_quarantined_event_count": quarantine_closure[
            "source_view_quarantined_event_count"
        ],
        "timeline_quarantined_chunk_count": quarantine_closure[
            "chunk_quarantine_count"
        ],
        "input_total_chunk_count": stats.get(
            "input_total_chunk_count", stats["total_effective_chunk_count"]
        ),
        "global_view_and_chunk_closure_passed": True,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-ready-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--random-sample-count", type=int, default=20)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate(
        model_ready_dir=args.model_ready_dir,
        tokenizer_dir=args.tokenizer_dir,
        upstream_dir=args.upstream_dir,
        report_path=args.report_path,
        random_sample_count=args.random_sample_count,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
