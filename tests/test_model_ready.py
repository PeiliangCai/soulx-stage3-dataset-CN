import unittest

from duplexconv_stage3.model_ready import (
    contiguous_usable_spans,
    filter_excluded_sources,
    greedy_windows,
    make_chunk_group,
    validate_model_ready_view_partition,
)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=True):
        return list(text)


class ModelReadyTests(unittest.TestCase):
    def test_source_exclusion_removes_every_view_of_the_conversation(self):
        records = [
            {"source_id": "keep", "view_id": "keep/target-ch00"},
            {"source_id": "drop", "view_id": "drop/target-ch00"},
            {"source_id": "drop", "view_id": "drop/target-ch01"},
        ]
        kept, removed = filter_excluded_sources(records, ["drop"])
        self.assertEqual([item["view_id"] for item in kept], ["keep/target-ch00"])
        self.assertEqual(
            [item["view_id"] for item in removed],
            ["drop/target-ch00", "drop/target-ch01"],
        )

    def test_usable_spans_split_at_quarantine(self):
        self.assertEqual(
            contiguous_usable_spans(["user_idle", None, "user_nonidle", "user_complete"]),
            [(0, 1), (2, 4)],
        )

    def test_chunk_group_has_exact_order(self):
        group = make_chunk_group([1, 2], "你好", "user_complete")
        self.assertEqual(
            group,
            "<|audio_1|><|audio_2|>你好<|end_of_sentence|><|user_complete|>",
        )

    def test_greedy_window_does_not_overlap(self):
        tokenizer = CharacterTokenizer()
        groups = ["abc", "def", "ghi"]
        prefix_length = len(tokenizer.encode("<|task_duplex_predict|><|punctuation_off|>"))
        windows, oversized = greedy_windows(
            tokenizer=tokenizer,
            groups=groups,
            span_start=10,
            max_token_length=prefix_length + 5,
        )
        self.assertEqual([(item[0], item[1]) for item in windows], [(10, 11), (11, 12), (12, 13)])
        self.assertEqual(oversized, [])

    def test_source_view_quarantine_provenance_must_be_identical(self):
        record = {
            "view_id": "bad",
            "original_chunk_count": 3,
            "event_count": 1,
            "event_ids": ["event"],
        }
        result = validate_model_ready_view_partition(
            timelines=[{"view_id": "ok"}],
            glm_records=[{"view_id": "ok"}],
            timeline_source_view_quarantine=[record],
            glm_source_view_quarantine=[dict(record)],
        )
        self.assertEqual(result, [record])
        with self.assertRaises(ValueError):
            validate_model_ready_view_partition(
                timelines=[{"view_id": "ok"}],
                glm_records=[{"view_id": "ok"}],
                timeline_source_view_quarantine=[record],
                glm_source_view_quarantine=[{**record, "event_count": 0}],
            )


if __name__ == "__main__":
    unittest.main()
