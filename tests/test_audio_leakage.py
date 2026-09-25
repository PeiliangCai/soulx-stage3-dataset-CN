import io
import tempfile
import unittest
from pathlib import Path
import wave

import numpy as np

from duplexconv_stage3.audio_leakage import (
    Chromaprint,
    FingerprintIndex,
    bit_agreement,
    canonical_pcm16_mono,
    decode_fingerprint,
    encode_fingerprint,
    normalized_window_hashes,
)


def tone(sample_rate: int, seconds: float = 12.0) -> np.ndarray:
    time = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    carrier = np.sin(2 * np.pi * (220.0 + 20.0 * time) * time)
    envelope = 0.35 + 0.25 * np.sin(2 * np.pi * 1.7 * time)
    return (carrier * envelope).astype(np.float32)


class AudioLeakageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chromaprint = Chromaprint()

    def test_canonical_pcm_resampling_is_stable(self):
        left = canonical_pcm16_mono(tone(16_000), 16_000)
        right = canonical_pcm16_mono(tone(48_000), 48_000)
        self.assertLess(abs(len(left) - len(right)), 2)
        correlation = np.corrcoef(left[: min(len(left), len(right))], right[: min(len(left), len(right))])[0, 1]
        self.assertGreater(correlation, 0.999)

    def test_fingerprint_round_trip_and_self_match(self):
        audio = tone(16_000)
        fingerprint = self.chromaprint.fingerprint(audio, 16_000)
        restored = decode_fingerprint(encode_fingerprint(fingerprint))
        np.testing.assert_array_equal(fingerprint, restored)
        self.assertEqual(bit_agreement(fingerprint, restored), 1.0)

    def test_index_finds_gain_and_crop(self):
        audio = tone(16_000, 16.0)
        fingerprint = self.chromaprint.fingerprint(audio, 16_000)
        record = {"chromaprint_raw_u32_base64": encode_fingerprint(fingerprint)}
        index = FingerprintIndex([record])
        transformed = self.chromaprint.fingerprint(audio[int(1.5 * 16_000) :] * 0.4, 16_000)
        matches = index.query(transformed, min_aligned_frames=24)
        self.assertTrue(matches)
        self.assertEqual(matches[0].record_index, 0)
        self.assertGreater(matches[0].similarity, 0.95)

    def test_window_hashes_change_with_content(self):
        first = canonical_pcm16_mono(tone(16_000, 6.0), 16_000)
        second = first.copy()
        second[: 4 * 16_000] = 0
        self.assertNotEqual(normalized_window_hashes(first), normalized_window_hashes(second))


if __name__ == "__main__":
    unittest.main()
