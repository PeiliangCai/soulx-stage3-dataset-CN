import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from duplexconv_stage3.validate_model_ready import (
    discover_metadata_files,
    discover_parquet_files,
    parse_sequence_ids,
    validate_quarantine_closure,
)


class ValidateModelReadyTests(unittest.TestCase):
    def test_discovers_sorted_multi_shard_inputs(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "data").mkdir()
            (root / "metadata" / "by_shard").mkdir(parents=True)
            for name in ("train-b.parquet", "train-a.parquet"):
                (root / "data" / name).touch()
            for name in ("B.windows.jsonl", "A.windows.jsonl"):
                (root / "metadata" / "by_shard" / name).touch()
            self.assertEqual(
                [path.name for path in discover_parquet_files(root)],
                ["train-a.parquet", "train-b.parquet"],
            )
            self.assertEqual(
                [path.name for path in discover_metadata_files(root)],
                ["A.windows.jsonl", "B.windows.jsonl"],
            )

    def test_valid_two_chunk_sequence(self):
        ids = [
            151670,
            151672,
            151700,
            151701,
            108386,
            151674,
            151681,
            151702,
            151703,
            151674,
            151676,
        ]
        self.assertEqual(parse_sequence_ids(ids), {"chunk_count": 2, "text_token_count": 1})

    def test_missing_second_audio_token_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_sequence_ids([151670, 151672, 151700, 151674, 151680])

    def test_global_source_view_and_chunk_closure(self):
        stats = {
            "exported_chunk_count": 5,
            "quarantined_chunk_count": 1,
            "total_effective_chunk_count": 6,
            "input_source_view_count": 2,
            "source_view_count": 1,
            "source_view_quarantined_count": 1,
            "source_view_quarantined_chunk_count": 3,
            "source_view_quarantined_event_count": 1,
            "input_total_chunk_count": 9,
            "total_quarantined_chunk_count": 4,
        }
        result = validate_quarantine_closure(
            stats=stats,
            metadata_rows=[{"view_id": "ok"}],
            chunk_quarantine=[{"view_id": "ok", "chunk_range": [5, 6]}],
            source_view_quarantine=[
                {
                    "view_id": "bad",
                    "original_chunk_count": 3,
                    "event_count": 1,
                    "event_ids": ["event"],
                }
            ],
            parsed_chunk_count=5,
        )
        self.assertEqual(result["source_view_quarantined_chunk_count"], 3)

    def test_overlapping_exported_and_quarantined_view_is_rejected(self):
        stats = {
            "exported_chunk_count": 1,
            "quarantined_chunk_count": 0,
            "total_effective_chunk_count": 1,
            "input_source_view_count": 2,
            "source_view_count": 1,
            "source_view_quarantined_count": 1,
            "source_view_quarantined_chunk_count": 1,
            "source_view_quarantined_event_count": 0,
            "input_total_chunk_count": 2,
            "total_quarantined_chunk_count": 1,
        }
        with self.assertRaises(ValueError):
            validate_quarantine_closure(
                stats=stats,
                metadata_rows=[{"view_id": "same"}],
                chunk_quarantine=[],
                source_view_quarantine=[
                    {
                        "view_id": "same",
                        "original_chunk_count": 1,
                        "event_count": 0,
                        "event_ids": [],
                    }
                ],
                parsed_chunk_count=1,
            )


if __name__ == "__main__":
    unittest.main()
