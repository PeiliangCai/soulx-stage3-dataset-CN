from __future__ import annotations

import unittest

from scripts.build_baidu_empty_file_compatibility import (
    infer_aggregate_directory_name,
)
from scripts.upload_duplexconv_baidu_release import (
    validate_compatibility_contract,
)


def zero_record(aggregate: str, index: int) -> dict[str, object]:
    return {
        "bytes": 0,
        "remote_path": (
            "/soulx-stage3-dataset-CN/datasets/release/model_ready/"
            f"{aggregate}/quarantine/by_shard/Edu_{index:04d}.views.jsonl"
        ),
    }


def supplemental(name: str) -> dict[str, object]:
    return {
        "category": "empty_file_compat",
        "remote_path": f"/release/receipts/empty_files_compat/{name}",
    }


class DynamicBaiduReleaseTests(unittest.TestCase):
    def test_infers_v1_and_v2_zero_file_aggregate_names(self) -> None:
        for count, aggregate in (
            (12, "edu0018_0045_stage3_zh_v1"),
            (21, "edu0001_0045_stage3_zh_v2"),
        ):
            records = [zero_record(aggregate, index) for index in range(count)]
            self.assertEqual(infer_aggregate_directory_name(records), aggregate)
            self.assertEqual(
                infer_aggregate_directory_name(records, aggregate), aggregate
            )

    def test_rejects_mixed_aggregate_zero_files(self) -> None:
        with self.assertRaises(ValueError):
            infer_aggregate_directory_name(
                [zero_record("aggregate_a", 1), zero_record("aggregate_b", 2)]
            )

    def test_uploader_accepts_dynamic_zero_file_counts(self) -> None:
        supplements = [
            supplemental("empty_files.tar"),
            supplemental("EMPTY_FILES.json"),
            supplemental("restore.py"),
        ]
        for count in (12, 21):
            zeros, archive = validate_compatibility_contract(
                [zero_record("aggregate", index) for index in range(count)],
                supplements,
            )
            self.assertEqual(len(zeros), count)
            self.assertTrue(str(archive["remote_path"]).endswith("empty_files.tar"))

    def test_uploader_rejects_missing_zero_file_contract(self) -> None:
        supplements = [
            supplemental("empty_files.tar"),
            supplemental("EMPTY_FILES.json"),
            supplemental("restore.py"),
        ]
        with self.assertRaises(ValueError):
            validate_compatibility_contract([], supplements)


if __name__ == "__main__":
    unittest.main()
