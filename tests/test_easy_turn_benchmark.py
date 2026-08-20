import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from duplexconv_stage3.easy_turn_benchmark import (
    EasyTurnSample,
    attach_raw_state_capture,
    evaluate_sample,
    load_audio_16k_mono,
    stream_chunks,
    summarize,
)


class FakeTurnModel:
    def __init__(self, states):
        self.states = list(states)
        self.index = 0

    def reset(self):
        self.index = 0

    def infer(self, *_args, **_kwargs):
        state = self.states[min(self.index, len(self.states) - 1)]
        self.index += 1
        return state, "", ""

    def process(self, chunk):
        state, _, _ = self.infer(chunk, None, None)
        return {"state": "nonidle" if state == "<|user_nonidle|>" else "idle"}


class EasyTurnBenchmarkTests(unittest.TestCase):
    def test_stream_chunks_pads_audio_and_adds_tail(self):
        audio = np.zeros(2_561, dtype=np.float32)
        chunks = list(stream_chunks(audio, 320))
        self.assertEqual(len(chunks), 4)
        self.assertEqual([is_tail for _, is_tail in chunks], [False, False, True, True])
        self.assertTrue(all(chunk.shape == (2_560,) for chunk, _ in chunks))

    def test_audio_is_downmixed_and_resampled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            sf.write(path, np.ones((2_400, 2), dtype=np.float32) * 0.25, 24_000)
            audio, info = load_audio_16k_mono(path)
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(audio.shape, (1_600,))
        self.assertEqual(info["source_channels"], 2)
        self.assertEqual(info["source_sample_rate"], 24_000)

    def test_terminal_state_after_nonidle_is_classified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.wav"
            sf.write(path, np.zeros(5_120, dtype=np.float32), 16_000)
            sample = EasyTurnSample("zh:test", "zh", "complete", path)
            model = FakeTurnModel(["<|user_nonidle|>", "<|user_complete|>"])
            attach_raw_state_capture(model)
            result = evaluate_sample(model, sample, 320)
        self.assertEqual(result["prediction"], "complete")
        self.assertTrue(result["correct"])

    def test_summary_is_macro_average(self):
        records = [
            {"label": "complete", "correct": True, "prediction": "complete", "total_elapsed_seconds": 1.0},
            {"label": "complete", "correct": False, "prediction": "incomplete", "total_elapsed_seconds": 1.0},
            {"label": "incomplete", "correct": True, "prediction": "incomplete", "total_elapsed_seconds": 1.0},
        ]
        result = summarize(records)
        self.assertAlmostEqual(result["macro_accuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()

