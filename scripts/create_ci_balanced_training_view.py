#!/usr/bin/env python3
"""Build a deterministic Complete/Incomplete active-row-balanced split view."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from duplexconv_stage3.ci_balance import select_ci_balanced_rows
from duplexconv_stage3.continual_training import (
    atomic_json_write,
    canonical_sha256,
    load_window_metadata,
    sha256_file,
    summarize_rows,
    utc_now,
)


def _verified_artifact_path(
    split_root: Path, manifest: dict, name: str
) -> Path:
    configured = split_root / "data" / f"{name}.parquet"
    artifact = manifest["artifacts"][name]
    recorded = Path(artifact["path"]).resolve(strict=True)
    actual = configured.resolve(strict=True)
    if recorded != actual:
        raise RuntimeError(f"{name} path differs from parent split manifest")
    if sha256_file(actual) != artifact["sha256"]:
        raise RuntimeError(f"{name} hash differs from parent split manifest")
    return actual


def _load_metadata(manifest: dict) -> dict[str, dict]:
    metadata = {}
    for artifact in manifest["source_artifacts"]["window_metadata_files"]:
        path = Path(artifact["path"]).resolve(strict=True)
        if sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"metadata hash differs from parent manifest: {path}")
        source_shard = path.name.split(".windows.jsonl", 1)[0]
        current = load_window_metadata(path)
        overlap = set(metadata) & set(current)
        if overlap:
            raise RuntimeError(f"duplicate metadata indexes: {sorted(overlap)[:10]}")
        for value in current.values():
            value["source_shard"] = source_shard
        metadata.update(current)
    return metadata


def _code_artifacts() -> list[dict[str, str]]:
    paths = [
        Path(__file__).resolve(),
        PROJECT_ROOT / "src" / "duplexconv_stage3" / "ci_balance.py",
    ]
    return [{"path": str(path), "sha256": sha256_file(path)} for path in paths]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-split-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    split_root = args.input_split_root.resolve(strict=True)
    output = args.output_dir.absolute()
    if output.exists():
        raise FileExistsError(f"balanced split output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    parent_manifest_path = split_root / "split_manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    train_path = _verified_artifact_path(split_root, parent_manifest, "train")
    validation_path = _verified_artifact_path(
        split_root, parent_manifest, "validation"
    )
    metadata = _load_metadata(parent_manifest)

    train_table = pq.read_table(train_path)
    if train_table.schema != pa.schema([("index", pa.string()), ("sequence", pa.string())]):
        raise RuntimeError(f"unexpected training schema: {train_table.schema}")
    rows = train_table.to_pylist()
    train_indexes = {row["index"] for row in rows}
    validation_table = pq.read_table(validation_path)
    validation_rows = validation_table.to_pylist()
    validation_indexes = {row["index"] for row in validation_rows}
    if train_indexes & validation_indexes:
        raise RuntimeError("parent split contains train/validation row leakage")
    required_indexes = train_indexes | validation_indexes
    if required_indexes - set(metadata):
        raise RuntimeError("parent split rows are missing window metadata")

    train_metadata = {index: metadata[index] for index in train_indexes}
    balanced_rows, balance_audit = select_ci_balanced_rows(
        rows, train_metadata, args.seed
    )
    balanced_indexes = {row["index"] for row in balanced_rows}
    if not balanced_indexes <= train_indexes:
        raise AssertionError("balanced view introduced unknown rows")

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent)
    )
    try:
        data_dir = staging / "data"
        data_dir.mkdir()
        staged_train = data_dir / "train.parquet"
        staged_validation = data_dir / "validation.parquet"
        pq.write_table(
            pa.Table.from_pylist(balanced_rows, schema=train_table.schema),
            staged_train,
            compression="zstd",
        )
        shutil.copyfile(validation_path, staged_validation)
        if sha256_file(staged_validation) != sha256_file(validation_path):
            raise AssertionError("validation byte-copy hash mismatch")

        selected_with_metadata = [
            {
                **row,
                "metadata": metadata[row["index"]],
            }
            for row in balanced_rows
        ]
        validation_with_metadata = [
            {
                **row,
                "metadata": metadata[row["index"]],
            }
            for row in validation_rows
        ]
        train_source_ids = sorted(
            {item["metadata"]["source_id"] for item in selected_with_metadata}
        )
        validation_source_ids = sorted(
            {item["metadata"]["source_id"] for item in validation_with_metadata}
        )
        leakage = sorted(set(train_source_ids) & set(validation_source_ids))
        if leakage:
            raise RuntimeError(f"source-conversation leakage: {leakage[:10]}")

        train_summary = summarize_rows(selected_with_metadata)
        validation_summary = summarize_rows(validation_with_metadata)
        if train_summary["row_count"] != balance_audit["output"]["row_count"]:
            raise AssertionError("independent training row summaries disagree")
        if train_summary["state_token_counts"] != balance_audit["output"][
            "state_token_counts"
        ]:
            raise AssertionError("independent state-token summaries disagree")

        final_train = output / "data" / "train.parquet"
        final_validation = output / "data" / "validation.parquet"
        manifest = {
            "schema_version": 2,
            "profile": (
                f"{parent_manifest['profile']}+{balance_audit['profile']}"
            ),
            "created_at_utc": utc_now(),
            "seed": args.seed,
            "validation_fraction_requested": parent_manifest.get(
                "validation_fraction_requested"
            ),
            "model_ready_root": parent_manifest.get("model_ready_root"),
            "parent_split": {
                "path": str(split_root),
                "manifest_path": str(parent_manifest_path),
                "manifest_sha256": sha256_file(parent_manifest_path),
                "split_identity_sha256": parent_manifest["split_identity_sha256"],
                "train_sha256": sha256_file(train_path),
                "validation_sha256": sha256_file(validation_path),
            },
            "balance": balance_audit,
            "code_artifacts": _code_artifacts(),
            "source_artifacts": parent_manifest["source_artifacts"],
            "train_source_ids": train_source_ids,
            "validation_source_ids": validation_source_ids,
            "source_leakage_count": 0,
            "train": train_summary,
            "validation": validation_summary,
            "artifacts": {
                "train": {
                    "path": str(final_train),
                    "sha256": sha256_file(staged_train),
                    "bytes": staged_train.stat().st_size,
                },
                "validation": {
                    "path": str(final_validation),
                    "sha256": sha256_file(staged_validation),
                    "bytes": staged_validation.stat().st_size,
                    "byte_identical_to_parent": True,
                },
            },
        }
        manifest["split_identity_sha256"] = canonical_sha256(
            {
                "profile": manifest["profile"],
                "seed": args.seed,
                "parent_split_identity_sha256": parent_manifest[
                    "split_identity_sha256"
                ],
                "train_rows": train_summary["row_index_identity_sha256"],
                "validation_rows": validation_summary["row_index_identity_sha256"],
            }
        )
        atomic_json_write(staging / "split_manifest.json", manifest)

        if output.exists():
            raise FileExistsError(f"balanced split output appeared during build: {output}")
        os.rename(staging, output)
        staging = None
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)

    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
