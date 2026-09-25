import unittest

from duplexconv_stage3.audio_leakage import Match
from duplexconv_stage3.audio_leakage_v2_2 import (
    _control_passes,
    next_power_of_two_strictly_above,
    split_sample_keys,
)


class AudioLeakageV22Test(unittest.TestCase):
    def test_next_power_of_two_is_strict(self):
        self.assertEqual(next_power_of_two_strictly_above(0), 1)
        self.assertEqual(next_power_of_two_strictly_above(8), 16)
        self.assertEqual(next_power_of_two_strictly_above(9), 16)
        self.assertEqual(next_power_of_two_strictly_above(16), 32)

    def test_split_is_stable_disjoint_and_grouped(self):
        keys = [f"sample/{value}" for value in range(20)] + ["sample/3"]
        first = split_sample_keys(keys)
        second = split_sample_keys(list(reversed(keys)))
        self.assertEqual(first, second)
        self.assertFalse(first[0].intersection(first[1]))
        self.assertEqual(first[0].union(first[1]), set(keys))

    def test_near_control_requires_similarity_frames_and_votes(self):
        good = Match(0, 0, 64, 0.655, 16)
        low_votes = Match(0, 0, 64, 0.90, 15)
        low_similarity = Match(0, 0, 64, 0.6549, 100)
        low_frames = Match(0, 0, 63, 0.90, 100)
        kwargs = {
            "similarity_threshold": 0.655,
            "minimum_aligned_frames": 64,
            "minimum_lsh_votes": 16,
        }
        self.assertTrue(_control_passes(good, **kwargs))
        self.assertFalse(_control_passes(low_votes, **kwargs))
        self.assertFalse(_control_passes(low_similarity, **kwargs))
        self.assertFalse(_control_passes(low_frames, **kwargs))


if __name__ == "__main__":
    unittest.main()
