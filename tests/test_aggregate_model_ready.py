import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from duplexconv_stage3.aggregate_model_ready import aggregate, audit_aggregate


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def make_shard(root: Path, shard_id: str, index: str, source_id: str) -> dict:
    model = root / shard_id
    for relative in ("data", "metadata", "quarantine"):
        (model / relative).mkdir(parents=True, exist_ok=True)
    parquet = model / "data" / "train-00000-of-00001.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"index": index, "sequence": "sequence"}]), parquet
    )
    windows = model / "metadata" / "windows.jsonl"
    windows.write_text(
        json.dumps(
            {
                "index": index,
                "view_id": f"{source_id}/target-ch00",
                "source_id": source_id,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    chunks = model / "quarantine" / "chunks.jsonl"
    chunks.touch()
    views = model / "quarantine" / "views.jsonl"
    views.touch()
    contract = {
        "schema_version": 1,
        "columns": ["index", "sequence"],
        "prefix": "prefix",
        "sequence_profile": "profile",
        "chunk_group": "group",
        "max_token_length": 1500,
        "timeline_profile": "timeline",
        "glm_audio_profile": "glm",
        "audio_token_raw_range": [0, 1],
        "token_ids": {"a": 1},
        "upstream_commit": "commit",
        "dataset_version": shard_id,
        "index_prefix": shard_id,
    }
    stats = {
        "schema_version": 1,
        "dataset_version": shard_id,
        "row_count": 1,
        "source_view_count": 1,
        "exported_chunk_count": 1,
        "quarantined_chunk_count": 0,
        "total_effective_chunk_count": 1,
        "max_observed_tokenized_length": 8,
        "rows_by_ntrack": {"2": 1},
        "chunk_state_counts_before_window_quarantine": {"user_idle": 1},
    }
    write_json(model / "contract.json", contract)
    write_json(model / "stats.json", stats)
    checksum_paths = [model / "contract.json", model / "stats.json", parquet, windows, chunks, views]
    (model / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(model)}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    validation = root / f"{shard_id}.validation.json"
    write_json(
        validation,
        {
            "status": "passed",
            "checksum_failure_count": 0,
            "global_view_and_chunk_closure_passed": True,
            "export_stats_sha256": sha256(model / "stats.json"),
        },
    )
    closure = root / f"{shard_id}.gate.json"
    write_json(
        closure,
        {
            "gate_passed": True,
            "all_source_members_scored_exactly_twice": True,
            "selection_missing_count": 0,
            "selection_extra_count": 0,
            "model_ready_view_count": 1,
            "model_ready_source_id_count": 1,
            "audio_comparison": {
                "exact_audio_match_count": 0,
                "normalized_audio_match_count": 0,
                "content_exact_match_count": 0,
                "window_match_count": 0,
                "near_duplicate_gate_hit_count": 0,
            },
            "sha256": {
                "model_ready_windows": sha256(windows),
                "model_ready_stats": sha256(model / "stats.json"),
                "model_ready_validation": sha256(validation),
            },
        },
    )
    return {
        "shard_id": shard_id,
        "model_ready_dir": str(model),
        "validation_report": str(validation),
        "gate_d_closure": str(closure),
    }


class AggregateModelReadyTests(unittest.TestCase):
    def test_builds_hardlinked_multi_parquet_and_audits(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            shards = [
                make_shard(root, "Edu_0001", "one", "source-one"),
                make_shard(root, "Edu_0002", "two", "source-two"),
            ]
            config = root / "config.json"
            write_json(config, {"dataset_version": "aggregate-v1", "shards": shards})
            output = root / "aggregate"
            result = aggregate(config_path=config, output_dir=output)
            self.assertEqual(result["shard_count"], 2)
            data_files = sorted((output / "data").glob("*.parquet"))
            self.assertEqual(len(data_files), 2)
            source = Path(shards[0]["model_ready_dir"]) / "data" / "train-00000-of-00001.parquet"
            self.assertEqual(os.stat(data_files[0]).st_ino, os.stat(source).st_ino)
            report = audit_aggregate(
                config_path=config,
                aggregate_dir=output,
                report_path=root / "audit.json",
            )
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["row_count"], 2)

    def test_cross_shard_duplicate_index_is_rejected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            shards = [
                make_shard(root, "Edu_0001", "same", "source-one"),
                make_shard(root, "Edu_0002", "same", "source-two"),
            ]
            config = root / "config.json"
            write_json(config, {"dataset_version": "aggregate-v1", "shards": shards})
            with self.assertRaisesRegex(ValueError, "cross-shard duplicate indexes"):
                aggregate(config_path=config, output_dir=root / "aggregate")


if __name__ == "__main__":
    unittest.main()
