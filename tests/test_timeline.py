import unittest

from duplexconv_stage3.timeline import build_view_timeline


def make_view(chunk_count=3):
    return {
        "view_id": "source/target-ch00",
        "source_id": "source",
        "source_ntrack": 2,
        "target_channel": 0,
        "reference_channels": [1],
        "chunk_count": chunk_count,
        "target_active_by_chunk": [True] + [False] * (chunk_count - 1),
        "other_active_by_chunk": [False] * chunk_count,
        "other_active_count_by_chunk": [0] * chunk_count,
        "overlap_by_chunk": [False] * chunk_count,
    }


def make_event(state="complete", segments=None, fallback=False, event_id="source/ch00/event0000"):
    return {
        "event_id": event_id,
        "source_id": "source",
        "channel": 0,
        "final_state": state,
        "state_label_source": "test",
        "speaker_segments_ms": [[0.0, 160.0]] if segments is None else segments,
        "activity_fallback_candidate": fallback,
        "effective_event_envelope_ms": [0.0, 160.0],
    }


def make_asr(chunk_count=3, fallback_ids=None):
    return {
        "view_id": "source/target-ch00",
        "chunk_count": chunk_count,
        "chunk_asr_targets": [""] * chunk_count,
        "fallback_event_ids_with_token_evidence": fallback_ids or [],
        "asr_token_outside_target_activity": [],
        "tokens": [],
        "cache_signature": "asr",
    }


class TimelineTests(unittest.TestCase):
    def test_complete_uses_nonidle_then_terminal(self):
        timeline, quarantine = build_view_timeline(
            view=make_view(), events=[make_event()], asr=make_asr()
        )
        self.assertEqual(timeline["chunk_states"], ["user_nonidle", "user_complete", "user_idle"])
        self.assertEqual(quarantine, [])

    def test_backchannel_covers_active_chunks(self):
        timeline, _ = build_view_timeline(
            view=make_view(), events=[make_event(state="backchannel")], asr=make_asr()
        )
        self.assertEqual(timeline["chunk_states"][0], "user_backchannel")

    def test_terminal_at_end_adds_one_silent_chunk(self):
        view = make_view(1)
        timeline, _ = build_view_timeline(view=view, events=[make_event()], asr=make_asr(1))
        self.assertEqual(timeline["effective_chunk_count"], 2)
        self.assertEqual(timeline["terminal_silence_padding_chunks"], 1)
        self.assertEqual(timeline["chunk_states"], ["user_nonidle", "user_complete"])

    def test_fallback_without_asr_evidence_is_quarantined(self):
        event = make_event(segments=[], fallback=True)
        timeline, quarantine = build_view_timeline(
            view=make_view(), events=[event], asr=make_asr()
        )
        self.assertEqual(timeline["event_assignments"][0]["status"], "missing_paraformer_evidence")
        self.assertEqual(quarantine[0]["kind"], "fallback_event_without_paraformer_evidence")
        self.assertIsNone(timeline["chunk_states"][0])

    def test_terminal_collision_with_next_activity_is_not_guessed(self):
        view = make_view()
        view["target_active_by_chunk"][1] = True
        second = make_event(
            state="backchannel",
            segments=[[160.0, 320.0]],
            event_id="source/ch00/event0001",
        )
        timeline, quarantine = build_view_timeline(
            view=view, events=[make_event(), second], asr=make_asr()
        )
        self.assertIsNone(timeline["chunk_states"][1])
        self.assertTrue(any(item["kind"] == "state_claim_conflict" for item in quarantine))


if __name__ == "__main__":
    unittest.main()
