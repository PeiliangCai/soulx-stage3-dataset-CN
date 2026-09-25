import unittest

import numpy as np

from duplexconv_stage3.audio_leakage import canonical_pcm16_mono
from duplexconv_stage3.audio_leakage_v2 import (
    CANONICAL_SAMPLE_RATE,
    compact_active_pcm16,
    content_window_hashes,
)


class AudioLeakageV2Test(unittest.TestCase):
    def test_pure_silence_emits_no_content_evidence(self):
        silence = np.zeros(CANONICAL_SAMPLE_RATE * 30, dtype=np.int16)
        content, stats = compact_active_pcm16(silence)
        self.assertEqual(content.size, 0)
        self.assertEqual(stats["active_frame_count"], 0)
        self.assertTrue(
            all(not values for values in content_window_hashes(content).values())
        )

    def test_zero_padding_does_not_change_compact_content(self):
        time = np.arange(3 * CANONICAL_SAMPLE_RATE) / CANONICAL_SAMPLE_RATE
        tone = (0.2 * np.sin(2 * np.pi * 337 * time)).astype(np.float32)
        padded = np.concatenate(
            (
                np.zeros(7 * CANONICAL_SAMPLE_RATE, dtype=np.float32),
                tone,
                np.zeros(11 * CANONICAL_SAMPLE_RATE, dtype=np.float32),
            )
        )
        original_pcm = canonical_pcm16_mono(tone, CANONICAL_SAMPLE_RATE)
        padded_pcm = canonical_pcm16_mono(padded, CANONICAL_SAMPLE_RATE)
        original_content, _ = compact_active_pcm16(original_pcm)
        padded_content, _ = compact_active_pcm16(padded_pcm)
        np.testing.assert_array_equal(original_content, padded_content)

    def test_different_active_signals_have_different_windows(self):
        time = np.arange(5 * CANONICAL_SAMPLE_RATE) / CANONICAL_SAMPLE_RATE
        left = (0.2 * np.sin(2 * np.pi * 337 * time)).astype(np.float32)
        right = (0.2 * np.sin(2 * np.pi * 719 * time)).astype(np.float32)
        left_content, _ = compact_active_pcm16(
            canonical_pcm16_mono(left, CANONICAL_SAMPLE_RATE)
        )
        right_content, _ = compact_active_pcm16(
            canonical_pcm16_mono(right, CANONICAL_SAMPLE_RATE)
        )
        left_hashes = {
            digest
            for values in content_window_hashes(left_content).values()
            for digest in values
        }
        right_hashes = {
            digest
            for values in content_window_hashes(right_content).values()
            for digest in values
        }
        self.assertTrue(left_hashes)
        self.assertFalse(left_hashes.intersection(right_hashes))


if __name__ == "__main__":
    unittest.main()
