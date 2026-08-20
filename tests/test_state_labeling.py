import os
from pathlib import Path
import tempfile
import unittest

from duplexconv_stage3.state_labeling import (
    build_request_record,
    load_api_key,
    validate_structured_labels,
)


def _view():
    return {
        "chunk_count": 4,
        "target_active_by_chunk": [False, True, True, False],
        "other_active_by_chunk": [True, True, False, False],
        "other_active_count_by_chunk": [1, 1, 0, 0],
        "overlap_by_chunk": [False, True, False, False],
    }


def _event(event_id="src/ch00/event0000", state=None):
    return {
        "event_id": event_id,
        "source_id": "src",
        "channel": 0,
        "ordinal": 0,
        "start_ms": 100,
        "end_ms": 300,
        "effective_event_envelope_ms": [100, 300],
        "text": "嗯",
        "official_state": state,
        "speaker_segments_ms": [[120, 280]],
    }


class StateLabelingTests(unittest.TestCase):
    def test_request_hides_target_and_has_stable_signature(self):
        event = _event()
        first = build_request_record(
            kind="connectivity",
            source_id="src",
            source_events=[event],
            target_event_ids=[event["event_id"]],
            views_by_channel={0: _view()},
        )
        second = build_request_record(
            kind="connectivity",
            source_id="src",
            source_events=[event],
            target_event_ids=[event["event_id"]],
            views_by_channel={0: _view()},
        )
        self.assertEqual(first["request_signature"], second["request_signature"])
        self.assertIn("TO_LABEL", first["messages"][1]["content"])

    def test_response_requires_exact_event_set(self):
        payload = {
            "labels": [
                {
                    "event_id": "a",
                    "state": "backchannel",
                    "confidence": 0.9,
                    "reason": "简短附和",
                }
            ]
        }
        self.assertEqual(validate_structured_labels(payload, ["a"])[0]["state"], "backchannel")
        with self.assertRaises(ValueError):
            validate_structured_labels(payload, ["b"])

    def test_env_file_must_be_0600_and_nonempty(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("OPENROUTER_API_KEY=test-key\n", encoding="utf-8")
            os.chmod(path, 0o600)
            self.assertEqual(load_api_key(path), "test-key")
            os.chmod(path, 0o644)
            with self.assertRaises(PermissionError):
                load_api_key(path)


if __name__ == "__main__":
    unittest.main()
