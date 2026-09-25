import importlib.util
import math
from pathlib import Path
import subprocess
import unittest

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = PROJECT_ROOT / "third_party/SoulX-Duplug-upstream"
RUNTIME = (
    PROJECT_ROOT / "runtimes/SoulX-Duplug-928b065-official-continual-v1"
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OfficialContinualRuntimeTest(unittest.TestCase):
    def test_upstream_checkout_remains_clean_and_pinned(self):
        head = subprocess.check_output(
            ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(UPSTREAM), "status", "--porcelain"], text=True
        )
        self.assertEqual(head, "928b06508ed2de1344208d06fb1f6fb2ebfb1df5")
        self.assertEqual(status, "")

    def test_empty_head_patch_is_finite_and_graph_connected(self):
        module = load_module(
            "official_continual_train_heads", RUNTIME / "models/_train_heads.py"
        )

        class Dummy(module.TokenHeadsMixin):
            lm_vocab_size = 5

        logits = torch.randn(1, 3, 5, dtype=torch.float16, requires_grad=True)
        labels = torch.full((1, 3), -100)
        loss = Dummy()._shifted_ce(logits, labels)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(torch.count_nonzero(logits.grad).item(), 0)

    def test_official_scheduler_formula_starts_at_estimated_step_1800(self):
        module = load_module(
            "official_continual_scheduler",
            RUNTIME / "utils/sparkvox/utils/scheduler.py",
        )
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=1.0e-4)
        optimizer.param_groups[0]["initial_lr"] = 1.0e-4
        scheduler = module.WarmupAnnealSteps(
            optimizer,
            warmup_step=200,
            anneal_steps=[100000],
            anneal_rate=0.5,
            final_lr=1.0e-6,
            last_epoch=1799,
        )
        expected = 1.0e-4 * math.sqrt(200 / 1801)
        self.assertEqual(scheduler.last_epoch, 1800)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], expected, places=12)

    def test_finetune_still_delegates_to_lightning_trainer_fit(self):
        source = (RUNTIME / "finetune.py").read_text(encoding="utf-8")
        model_source = (RUNTIME / "models/state_prediction_model.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("trainer.fit(model=model, datamodule=data)", source)
        self.assertIn("def configure_optimizers(self):", model_source)
        self.assertNotIn("optimizer.step()", source)
        self.assertNotIn("loss.backward()", source)


if __name__ == "__main__":
    unittest.main()
