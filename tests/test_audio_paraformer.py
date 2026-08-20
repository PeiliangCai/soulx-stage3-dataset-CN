import unittest

import numpy as np

from duplexconv_stage3.audio_processing import (
    pad_to_chunk_boundary,
    resample_pcm16_48k_to_16k,
)
from duplexconv_stage3.paraformer_inference import (
    _token_has_activity_evidence,
    _validate_cached_result,
    make_cache_signature,
    parse_paraformer_result,
)


class AudioAndParaformerTests(unittest.TestCase):
    def test_resample_has_exact_ceil_one_third_length(self):
        source = np.arange(48001, dtype=np.int32)
        source = ((source % 2000) - 1000).astype(np.int16)
        result = resample_pcm16_48k_to_16k(source)
        self.assertEqual(len(result), 16001)
        self.assertEqual(result.dtype, np.dtype("int16"))

    def test_padding_is_exact_160_ms_boundary(self):
        source = np.zeros(2561, dtype=np.int16)
        padded, padding = pad_to_chunk_boundary(source)
        self.assertEqual(padding, 2559)
        self.assertEqual(len(padded), 5120)
        self.assertTrue(np.all(padded[-padding:] == 0))

    def test_token_is_emitted_once_by_end_timestamp(self):
        result = parse_paraformer_result(
            {"text": "你 好", "timestamp": [[0, 160], [160, 321]]},
            chunk_count=3,
            source_duration_ms=480,
        )
        self.assertEqual(result["chunk_asr_targets"], ["你", "", "好"])
        self.assertEqual([item["emit_chunk"] for item in result["tokens"]], [0, 2])

    def test_bad_timestamp_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_paraformer_result(
                {"text": "你 好", "timestamp": [[100, 200], [50, 250]]},
                chunk_count=2,
                source_duration_ms=320,
            )

    def test_activity_evidence_uses_token_interval_not_only_emit_chunk(self):
        view = {
            "chunk_count": 20,
            "target_active_by_chunk": [False] * 20,
        }
        view["target_active_by_chunk"][5] = True
        accepted, fallback_ids = _token_has_activity_evidence(
            {"start_ms": 800.0, "end_ms": 2400.0, "emit_chunk": 14},
            view,
            [],
        )
        self.assertTrue(accepted)
        self.assertEqual(fallback_ids, [])

    def test_activity_evidence_rejects_distant_token_interval(self):
        view = {
            "chunk_count": 20,
            "target_active_by_chunk": [False] * 20,
        }
        view["target_active_by_chunk"][2] = True
        accepted, fallback_ids = _token_has_activity_evidence(
            {"start_ms": 1600.0, "end_ms": 1760.0, "emit_chunk": 10},
            view,
            [],
        )
        self.assertFalse(accepted)
        self.assertEqual(fallback_ids, [])

    def test_cached_result_contract_rejects_wrong_signature(self):
        cached = {
            "cache_schema_version": 1,
            "cache_signature": "wrong",
            "view_id": "source/target-ch00",
            "audio_sha256": "audio",
            "asr_model_key_sha256": "model",
            "asr_supervision_profile": "paraformer-pseudolabel-v1",
            "token_emit_profile": "token-end-ceil-160ms-v1",
            "activity_evidence_profile": "token-interval-overlap-320ms-v2",
            "chunk_count": 1,
            "tokens": [],
            "chunk_asr_targets": [""],
        }
        with self.assertRaises(ValueError):
            _validate_cached_result(
                cached,
                cache_signature="expected",
                view_id="source/target-ch00",
                audio_sha256="audio",
                model_sha256="model",
            )

    def test_fallback_evidence_is_recorded_even_with_primary_activity(self):
        view = {
            "chunk_count": 20,
            "target_active_by_chunk": [False] * 20,
        }
        view["target_active_by_chunk"][5] = True
        accepted, fallback_ids = _token_has_activity_evidence(
            {"start_ms": 800.0, "end_ms": 960.0, "emit_chunk": 5},
            view,
            [
                {
                    "event_id": "source/ch00/event0001",
                    "effective_event_envelope_ms": [700.0, 1000.0],
                }
            ],
        )
        self.assertTrue(accepted)
        self.assertEqual(fallback_ids, ["source/ch00/event0001"])

    def test_cache_signature_separates_identical_audio_views(self):
        common = {"audio_sha256": "same"}
        first = make_cache_signature(
            audio={**common, "view_id": "source/target-ch00"},
            model_sha256="model",
            funasr_version="test",
        )
        second = make_cache_signature(
            audio={**common, "view_id": "source/target-ch01"},
            model_sha256="model",
            funasr_version="test",
        )
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
