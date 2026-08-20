import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from duplexconv_stage3.easy_turn_benchmark import (
    EasyTurnSample,
    append_progress_record,
    attach_raw_state_capture,
    collect_runtime_environment,
    evaluate_sample,
    load_audio_16k_mono,
    load_or_create_progress,
    select_sample_subset,
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
    def test_runtime_environment_records_packages_and_git_identity(self):
        runtime = collect_runtime_environment(
            Path("third_party/SoulX-Duplug-inference-a0b9063").resolve()
        )
        self.assertEqual(runtime["packages"]["torch"], "2.6.0")
        self.assertEqual(runtime["packages"]["setuptools"], "78.1.1")
        self.assertTrue(runtime["official_git"]["commit"])
        self.assertIn("available", runtime["cuda"])

    def test_progress_journal_can_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.jsonl"
            metadata = {"language": "zh", "sample_count": 2}
            started_at, records = load_or_create_progress(path, metadata, False)
            self.assertEqual(records, [])
            append_progress_record(path, {"sample_id": "zh:one"})
            resumed_at, records = load_or_create_progress(path, metadata, True)
        self.assertEqual(resumed_at, started_at)
        self.assertEqual(records, [{"sample_id": "zh:one"}])

    def test_progress_journal_rejects_changed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.jsonl"
            load_or_create_progress(path, {"language": "zh"}, False)
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_or_create_progress(path, {"language": "en"}, True)

    def test_limit_per_label_selects_both_classes(self):
        samples = [
            EasyTurnSample(f"complete-{index}", "zh", "complete", Path("unused"))
            for index in range(3)
        ] + [
            EasyTurnSample(f"incomplete-{index}", "zh", "incomplete", Path("unused"))
            for index in range(3)
        ]
        selected = select_sample_subset(samples, limit_per_label=1)
        self.assertEqual([sample.label for sample in selected], ["complete", "incomplete"])

    def test_limit_modes_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            select_sample_subset([], limit=1, limit_per_label=1)

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
            result = evaluate_sample(
                model,
                sample,
                320,
                "complete-immediate-incomplete-provisional-v1",
            )
        self.assertEqual(result["prediction"], "complete")
        self.assertTrue(result["correct"])

    def test_incomplete_is_provisional_until_later_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.wav"
            sf.write(path, np.zeros(10_240, dtype=np.float32), 16_000)
            sample = EasyTurnSample("zh:test", "zh", "complete", path)
            model = FakeTurnModel(
                [
                    "<|user_nonidle|>",
                    "<|user_incomplete|>",
                    "<|user_nonidle|>",
                    "<|user_complete|>",
                ]
            )
            attach_raw_state_capture(model)
            result = evaluate_sample(
                model,
                sample,
                0,
                "complete-immediate-incomplete-provisional-v1",
            )
        self.assertEqual(result["prediction"], "complete")
        self.assertEqual(result["decision_chunk"], 3)
        self.assertEqual(len(result["trace"]), 4)

    def test_incomplete_remains_without_later_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.wav"
            sf.write(path, np.zeros(5_120, dtype=np.float32), 16_000)
            sample = EasyTurnSample("zh:test", "zh", "incomplete", path)
            model = FakeTurnModel(
                [
                    "<|user_nonidle|>",
                    "<|user_incomplete|>",
                    "<|user_idle|>",
                    "<|user_idle|>",
                ]
            )
            attach_raw_state_capture(model)
            result = evaluate_sample(
                model,
                sample,
                320,
                "complete-immediate-incomplete-provisional-v1",
            )
        self.assertEqual(result["prediction"], "incomplete")
        self.assertEqual(result["decision_chunk"], 1)
        self.assertEqual(len(result["trace"]), 4)

    def test_first_terminal_policy_stops_on_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.wav"
            sf.write(path, np.zeros(10_240, dtype=np.float32), 16_000)
            sample = EasyTurnSample("zh:test", "zh", "incomplete", path)
            model = FakeTurnModel(
                [
                    "<|user_nonidle|>",
                    "<|user_incomplete|>",
                    "<|user_nonidle|>",
                    "<|user_complete|>",
                ]
            )
            attach_raw_state_capture(model)
            result = evaluate_sample(model, sample, 0, "first-terminal-v1")
        self.assertEqual(result["prediction"], "incomplete")
        self.assertEqual(result["decision_chunk"], 1)
        self.assertEqual(len(result["trace"]), 2)

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
