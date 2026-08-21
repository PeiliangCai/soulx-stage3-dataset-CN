import unittest
from pathlib import Path
import tempfile

import torch

from duplexconv_stage3.continual_training import (
    build_split_rows,
    continuation_learning_rate,
    deterministic_group_split,
    select_lr_candidate,
)
from duplexconv_stage3.table3_protocol import (
    EXPECTED_OFFICIAL_CHECKPOINT_SHA256,
    EXPECTED_UPSTREAM_COMMIT,
)
from duplexconv_stage3.table3_reproduction import apply_continuation_checkpoint


class ContinualTrainingTest(unittest.TestCase):
    def test_group_split_is_deterministic_and_disjoint(self):
        source_ids = [f"s{value:03d}" for value in range(100)]
        first = deterministic_group_split(source_ids, 0.05, 42)
        second = deterministic_group_split(reversed(source_ids), 0.05, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first[1]), 5)
        self.assertFalse(set(first[0]) & set(first[1]))

    def test_all_views_of_a_source_stay_together(self):
        indexes = ["a0", "a1", "b0", "c0"]
        sequences = ["<|user_idle|>"] * len(indexes)
        metadata = {
            "a0": self._metadata("a", "a/ch0"),
            "a1": self._metadata("a", "a/ch1"),
            "b0": self._metadata("b", "b/ch0"),
            "c0": self._metadata("c", "c/ch0"),
        }
        train, validation, manifest = build_split_rows(
            indexes, sequences, metadata, 1 / 3, 42
        )
        split_by_index = {
            item["index"]: "train" for item in train
        } | {item["index"]: "validation" for item in validation}
        self.assertEqual(split_by_index["a0"], split_by_index["a1"])
        self.assertEqual(manifest["source_leakage_count"], 0)

    def test_continuation_lr_rewarms_then_decays_from_origin_offset(self):
        peak = 3.3333333333333335e-5
        self.assertLess(
            continuation_learning_rate(peak, 1800, 1, 5),
            continuation_learning_rate(peak, 1800, 5, 5),
        )
        self.assertLess(
            continuation_learning_rate(peak, 1800, 300, 5),
            continuation_learning_rate(peak, 1800, 5, 5),
        )

    def test_continuation_overlay_requires_exact_trainable_key_set(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
        model[1].weight.requires_grad = False
        model[1].bias.requires_grad = False
        expected = {
            name: torch.full_like(parameter, 3.0)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        payload = {
            "checkpoint_profile": "soulx-stage3-compact-evaluation-v2",
            "official_base_checkpoint_sha256": EXPECTED_OFFICIAL_CHECKPOINT_SHA256,
            "runtime_base_commit": EXPECTED_UPSTREAM_COMMIT,
            "local_step": 5,
            "trainable_state_dict": expected,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "step.pt"
            torch.save(payload, path)
            audit = apply_continuation_checkpoint(model, path)
        self.assertEqual(audit["status"], "accepted")
        self.assertEqual(audit["trainable_tensor_count"], 2)
        self.assertTrue(torch.equal(model[0].weight, expected["0.weight"]))

    def test_lr_selector_uses_lower_lr_for_one_percent_tie(self):
        def metrics(objective):
            return {
                "token_weighted_objective": objective,
                "heads": {
                    name: {"accuracy": 0.8}
                    for name in (
                        "idle",
                        "nonidle",
                        "user_complete",
                        "user_incomplete",
                        "user_backchannel",
                    )
                },
            }

        result = select_lr_candidate(
            [
                {"run_id": "low", "peak_lr": 1e-5, "baseline": metrics(1), "final": metrics(0.995)},
                {"run_id": "high", "peak_lr": 3e-5, "baseline": metrics(1), "final": metrics(0.99)},
            ]
        )
        self.assertEqual(result["selected_run_id"], "low")

    def test_lr_selector_prefers_longer_guard_safe_horizon(self):
        state_names = (
            "idle",
            "nonidle",
            "user_complete",
            "user_incomplete",
            "user_backchannel",
        )

        def metrics(objective, nonidle):
            return {
                "token_weighted_objective": objective,
                "heads": {
                    name: {"accuracy": nonidle if name == "nonidle" else 0.8}
                    for name in state_names
                },
            }

        baseline = metrics(2.0, 0.8)
        low_final = metrics(1.5, 0.7)
        high_final = metrics(1.0, 0.5)
        result = select_lr_candidate(
            [
                {
                    "run_id": "low",
                    "peak_lr": 1e-5,
                    "final_local_step": 20,
                    "baseline": baseline,
                    "final": low_final,
                    "checkpoints": [
                        {"local_step": 0, "metrics": baseline},
                        {"local_step": 10, "metrics": metrics(1.7, 0.76)},
                        {"local_step": 20, "metrics": low_final},
                    ],
                },
                {
                    "run_id": "high",
                    "peak_lr": 3e-5,
                    "final_local_step": 20,
                    "baseline": baseline,
                    "final": high_final,
                    "checkpoints": [
                        {"local_step": 0, "metrics": baseline},
                        {"local_step": 5, "metrics": metrics(1.2, 0.7)},
                        {"local_step": 20, "metrics": high_final},
                    ],
                },
            ]
        )
        self.assertTrue(result["eligibility_fallback_used"])
        self.assertEqual(result["selected_run_id"], "low")

    @staticmethod
    def _metadata(source_id, view_id):
        return {
            "source_id": source_id,
            "view_id": view_id,
            "source_ntrack": 2,
            "qwen_event_ids": [],
            "chunk_count": 1,
        }


if __name__ == "__main__":
    unittest.main()
