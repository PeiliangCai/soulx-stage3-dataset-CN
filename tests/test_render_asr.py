import unittest
from pathlib import Path
import json
import tempfile

from duplexconv_stage3.render_asr import (
    render_asr_directory,
    render_token_group,
    validate_asr_partition,
)


class RenderAsrTests(unittest.TestCase):
    def test_chinese_characters_are_contiguous(self):
        self.assertEqual(render_token_group(["你", "好"]), "你好")

    def test_adjacent_ascii_words_keep_space(self):
        self.assertEqual(render_token_group(["marketing", "data", "中", "文"]), "marketing data 中文")

    def test_chinese_before_ascii_needs_no_artificial_space(self):
        self.assertEqual(render_token_group(["是", "ABC"]), "是ABC")

    def test_result_and_quarantine_partition_must_be_disjoint(self):
        with self.assertRaises(ValueError):
            validate_asr_partition(
                [{"view_id": "same"}], [{"view_id": "same"}]
            )

    def test_directory_render_propagates_source_view_quarantine(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            result = {
                "view_id": "source/target-ch00",
                "chunk_count": 1,
                "tokens": [],
                "cache_signature": "asr-ok",
            }
            quarantine = {
                "view_id": "source/target-ch01",
                "source_id": "source",
                "cache_signature": "asr-bad",
                "reason": "strict timestamp failure",
            }
            (input_dir / "asr_results.jsonl").write_text(
                json.dumps(result) + "\n", encoding="utf-8"
            )
            (input_dir / "asr_quarantine.jsonl").write_text(
                json.dumps(quarantine) + "\n", encoding="utf-8"
            )
            (input_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "input_view_count": 2,
                        "passed_view_count": 1,
                        "quarantined_view_count": 1,
                    }
                ),
                encoding="utf-8",
            )
            summary = render_asr_directory(
                input_dir=input_dir, output_dir=output_dir
            )
            self.assertEqual(summary["rendered_view_count"], 1)
            self.assertEqual(summary["propagated_quarantined_view_count"], 1)
            propagated = json.loads(
                (output_dir / "asr_quarantine.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(propagated, quarantine)


if __name__ == "__main__":
    unittest.main()
