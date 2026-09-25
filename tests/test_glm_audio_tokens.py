import unittest

import numpy as np

from duplexconv_stage3.glm_audio_tokens import (
    group_chunk_tokens,
    make_glm_cache_signature,
    split_audio_segments,
    validate_glm_input_partition,
)


class GlmAudioTokenTests(unittest.TestCase):
    def test_segments_preserve_every_sample_once(self):
        audio = np.arange(31 * 16000, dtype=np.float32)
        segments = split_audio_segments(audio)
        self.assertEqual([len(item) for item in segments], [30 * 16000, 16000])
        np.testing.assert_array_equal(np.concatenate(segments), audio)

    def test_exactly_two_tokens_per_chunk(self):
        self.assertEqual(group_chunk_tokens([1, 2, 3, 4], 2), [[1, 2], [3, 4]])

    def test_wrong_token_count_is_rejected(self):
        with self.assertRaises(ValueError):
            group_chunk_tokens([1, 2, 3], 2)

    def test_cache_signature_separates_identical_audio_views(self):
        timeline = {"terminal_silence_padding_chunks": 0}
        first = make_glm_cache_signature(
            manifest={"view_id": "source/target-ch00", "audio_sha256": "same"},
            timeline=timeline,
            model_signature="model",
        )
        second = make_glm_cache_signature(
            manifest={"view_id": "source/target-ch01", "audio_sha256": "same"},
            timeline=timeline,
            model_signature="model",
        )
        self.assertNotEqual(first, second)

    def test_upstream_quarantine_partitions_audio_manifests(self):
        manifests = [
            {"view_id": "ok", "source_id": "a", "chunk_count": 2},
            {"view_id": "bad", "source_id": "b", "chunk_count": 3},
        ]
        timelines, eligible = validate_glm_input_partition(
            manifests=manifests,
            timelines=[{"view_id": "ok"}],
            source_view_quarantine=[
                {
                    "view_id": "bad",
                    "source_id": "b",
                    "original_chunk_count": 3,
                }
            ],
        )
        self.assertEqual(set(timelines), {"ok"})
        self.assertEqual([item["view_id"] for item in eligible], ["ok"])

    def test_missing_manifest_partition_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_glm_input_partition(
                manifests=[{"view_id": "missing", "source_id": "a", "chunk_count": 1}],
                timelines=[],
                source_view_quarantine=[],
            )


if __name__ == "__main__":
    unittest.main()
