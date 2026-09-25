import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = PROJECT_ROOT / f"scripts/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OfficialGroupValidationTest(unittest.TestCase):
    def test_exact_token_weighting_ignores_empty_rows(self):
        runner = load_script("run_official_group_validation")
        model = SimpleNamespace(
            LOSS_HEADS=[SimpleNamespace(name="state")],
            val_losses={"state": [torch.tensor(2.0), torch.tensor(100.0), torch.tensor(4.0)]},
            val_accs={"state": [torch.tensor(0.5), torch.tensor(float("nan")), torch.tensor(1.0)]},
            val_valid_targets={"state": [torch.tensor(2.0), torch.tensor(0.0), torch.tensor(6.0)]},
        )
        result = runner.summarize_token_weighted_validation(model)
        self.assertEqual(result["valid_targets"], 8)
        self.assertAlmostEqual(result["objective"], 3.5)
        self.assertAlmostEqual(result["accuracy"], 0.875)
        self.assertEqual(result["profile"], "exact-per-row-target-weighted-v1")

    def test_selection_uses_guard_and_earlier_one_percent_tie(self):
        index = load_script("build_official_group_validation_index")

        def payload(step, objective, state_accuracy):
            heads = {
                name: {"token_weighted_accuracy": state_accuracy}
                for name in index.STATE_HEADS
            }
            return {
                "local_step": step,
                "estimated_total_optimizer_step": 1800 + step,
                "exact_token_weighted_metrics": {
                    "objective": objective,
                    "accuracy": 0.8,
                    "heads": heads,
                },
            }

        baseline = payload(0, 2.0, 0.8)
        result = index.select_checkpoint(
            baseline,
            [payload(1, 1.009, 0.76), payload(2, 1.0, 0.751), payload(3, 0.5, 0.70)],
        )
        self.assertEqual(result["selected_local_step"], 1)
        self.assertTrue(result["one_percent_earlier_tie_break_used"])
        self.assertFalse(result["candidates"][2]["eligible"])


if __name__ == "__main__":
    unittest.main()
