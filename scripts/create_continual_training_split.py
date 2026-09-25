#!/usr/bin/env python3
"""Freeze a leakage-free source-conversation train/validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from duplexconv_stage3.continual_training import (
    atomic_json_write,
    build_split_rows,
    load_window_metadata,
    sha256_file,
    utc_now,
)
from duplexconv_stage3.validate_model_ready import (
    discover_metadata_files,
    discover_parquet_files,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-ready-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = args.model_ready_root.resolve(strict=True)
    input_parquets = discover_parquet_files(root)
    metadata_paths = discover_metadata_files(root)
    output = args.output_dir.absolute()
    if output.exists():
        raise FileExistsError(f"split output already exists: {output}")

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.concat_tables(
        [pq.read_table(path, columns=["index", "sequence"]) for path in input_parquets]
    )
    indexes = table.column("index").to_pylist()
    sequences = table.column("sequence").to_pylist()
    metadata = {}
    for metadata_path in metadata_paths:
        current = load_window_metadata(metadata_path)
        overlap = set(metadata) & set(current)
        if overlap:
            raise ValueError(f"duplicate metadata indexes across shards: {sorted(overlap)[:10]}")
        metadata.update(current)
    train_rows, validation_rows, manifest = build_split_rows(
        indexes,
        sequences,
        metadata,
        args.validation_fraction,
        args.seed,
    )

    output.mkdir(parents=True)
    data_dir = output / "data"
    data_dir.mkdir()
    train_path = data_dir / "train.parquet"
    validation_path = data_dir / "validation.parquet"
    schema = pa.schema([("index", pa.string()), ("sequence", pa.string())])
    for path, rows in ((train_path, train_rows), (validation_path, validation_rows)):
        pq.write_table(
            pa.Table.from_pylist(
                [{"index": row["index"], "sequence": row["sequence"]} for row in rows],
                schema=schema,
            ),
            path,
            compression="zstd",
        )

    manifest.update(
        {
            "created_at_utc": utc_now(),
            "model_ready_root": str(root),
            "source_artifacts": {
                "parquet_files": [
                    {"path": str(path), "sha256": sha256_file(path)}
                    for path in input_parquets
                ],
                "window_metadata_files": [
                    {"path": str(path), "sha256": sha256_file(path)}
                    for path in metadata_paths
                ],
            },
            "artifacts": {
                "train": {
                    "path": str(train_path),
                    "sha256": sha256_file(train_path),
                    "bytes": train_path.stat().st_size,
                },
                "validation": {
                    "path": str(validation_path),
                    "sha256": sha256_file(validation_path),
                    "bytes": validation_path.stat().st_size,
                },
            },
        }
    )
    manifest_path = output / "split_manifest.json"
    atomic_json_write(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
