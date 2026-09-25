from __future__ import annotations

import unittest

from scripts.retire_baidu_release_after_verification import (
    NEW_ROOT,
    validate_roots,
    validate_verification_contract,
)


def valid_verification() -> dict[str, object]:
    return {
        "status": "passed",
        "core_manifest_record_count": 239,
        "supplemental_record_count": 3,
        "direct_remote_file_count": 221,
        "direct_remote_verified_file_count": 221,
        "represented_zero_byte_record_count": 21,
        "payload_bytes": 346099226326,
        "remote_overwrite_count": 0,
        "remote_delete_count": 0,
        "remote_file_results": [
            {
                "remote_path": f"{NEW_ROOT}/file-{index}",
                "status": "exact_byte_size_verified",
                "expected_bytes": index + 1,
                "actual_bytes": index + 1,
            }
            for index in range(221)
        ],
    }


def valid_migration() -> dict[str, object]:
    return {
        "status": "passed",
        "source_release_root": "/soulx-stage3-dataset-CN/datasets/duplexconv_edu0018_0045_stage3_zh_v1",
        "destination_release_root": NEW_ROOT,
        "raw_record_count": 29,
        "verified_raw_record_count": 29,
        "raw_payload_bytes": 211827029597,
        "verified_raw_payload_bytes": 211827029597,
        "remote_copy_created": False,
        "source_raw_absent": True,
        "destination_raw_present": True,
    }


class BaiduRetirementTests(unittest.TestCase):
    def test_accepts_exact_frozen_contract(self) -> None:
        validate_verification_contract(
            valid_verification(), valid_migration(), new_root=NEW_ROOT
        )

    def test_rejects_failed_or_incomplete_verification(self) -> None:
        verification = valid_verification()
        verification["direct_remote_verified_file_count"] = 220
        with self.assertRaises(ValueError):
            validate_verification_contract(
                verification, valid_migration(), new_root=NEW_ROOT
            )

    def test_rejects_remote_overwrite_or_wrong_root(self) -> None:
        verification = valid_verification()
        verification["remote_overwrite_count"] = 1
        with self.assertRaises(ValueError):
            validate_verification_contract(
                verification, valid_migration(), new_root=NEW_ROOT
            )
        verification = valid_verification()
        verification["remote_file_results"][0]["remote_path"] = "/wrong/file"
        with self.assertRaises(ValueError):
            validate_verification_contract(
                verification, valid_migration(), new_root=NEW_ROOT
            )

    def test_rejects_unapproved_roots(self) -> None:
        with self.assertRaises(ValueError):
            validate_roots("/wrong", NEW_ROOT)


if __name__ == "__main__":
    unittest.main()
