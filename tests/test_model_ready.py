import unittest

from duplexconv_stage3.model_ready import (
    contiguous_usable_spans,
    greedy_windows,
    make_chunk_group,
)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=True):
        return list(text)


class ModelReadyTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
