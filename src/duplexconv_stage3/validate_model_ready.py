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

    table = pq.read_table(model_ready_dir / "data" / "train-00000-of-00001.parquet")
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
    for line in (model_ready_dir / "metadata" / "windows.jsonl").open(encoding="utf-8"):
        item = json.loads(line)
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

    stats = json.loads((model_ready_dir / "stats.json").read_text(encoding="utf-8"))
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
        "official_loader": {
            "source": str(upstream_dir / "models/state_prediction_data.py"),
            "train_split_length": len(official_train),
            "sample_index": official_sample["index"],
            "sample_token_length": len(official_sample["input_id"]),
        },
        "export_stats_sha256": sha256_file(model_ready_dir / "stats.json"),
        "exported_chunk_count": stats["exported_chunk_count"],
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
