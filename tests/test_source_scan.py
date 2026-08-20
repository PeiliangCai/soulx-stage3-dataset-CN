import unittest

from duplexconv_stage3.source_scan import (
    activity_from_intervals,
    clip_terminal_end,
    interval_chunk_bounds,
    normalize_state,
    stable_event_id,
    stable_view_id,
    target_vs_rest_activity,
)


class SourceScanUnitTests(unittest.TestCase):
    def test_interval_chunk_bounds_are_half_open(self):
        self.assertEqual(interval_chunk_bounds(0, 160, 4), (0, 1))
        self.assertEqual(interval_chunk_bounds(160, 320, 4), (1, 2))
        self.assertEqual(interval_chunk_bounds(159, 161, 4), (0, 2))
        self.assertEqual(interval_chunk_bounds(640, 800, 4), (4, 4))

    def test_activity_marks_overlapping_chunks(self):
        self.assertEqual(
            activity_from_intervals([(10, 170), (480, 500)], 4),
            [True, True, False, True],
        )

    def test_three_channel_target_vs_rest_is_union_not_pairwise_views(self):
        channels = [
            [True, False, False, True],
            [False, True, False, True],
            [False, True, True, False],
        ]
        relation = target_vs_rest_activity(channels, 0)
        self.assertEqual(relation["target_active_by_chunk"], channels[0])
        self.assertEqual(
            relation["other_active_by_chunk"], [False, True, True, True]
        )
        self.assertEqual(
            relation["other_active_count_by_chunk"], [0, 2, 1, 1]
        )
        self.assertEqual(
            relation["overlap_by_chunk"], [False, False, False, True]
        )

    def test_stable_ids(self):
        self.assertEqual(
            stable_event_id("Edu--010456", 1, 2),
            "Edu--010456/ch01/event0002",
        )
        self.assertEqual(
            stable_view_id("Edu--010456", 1), "Edu--010456/target-ch01"
        )

    def test_official_state_normalization(self):
        self.assertEqual(normalize_state("<|wait|>"), "wait")
        self.assertIsNone(normalize_state(None))
        with self.assertRaises(ValueError):
            normalize_state("<|idle|>")

    def test_only_small_terminal_overhang_is_clipped(self):
        self.assertEqual(clip_terminal_end(1008, 1000), (1000, 8))
        self.assertEqual(clip_terminal_end(1000, 1000), (1000, 0))
        with self.assertRaises(ValueError):
            clip_terminal_end(1161, 1000)


if __name__ == "__main__":
    unittest.main()
