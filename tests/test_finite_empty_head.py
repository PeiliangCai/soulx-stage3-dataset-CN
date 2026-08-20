import importlib.util
from pathlib import Path
import types
import unittest

import torch
from torch.nn import functional as F


RUNTIME_FILE = (
    Path(__file__).resolve().parents[1]
    / "runtimes/SoulX-Duplug-928b065-finite-empty-head-v2/models/_train_heads.py"
)
SPEC = importlib.util.spec_from_file_location("soulx_finite_train_heads", RUNTIME_FILE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DummyHeads(MODULE.TokenHeadsMixin):
    def __init__(self, vocab_size=5):
        self.lm_vocab_size = vocab_size
        self.train_config = types.SimpleNamespace(enable_switch_loss_rate=False)


class FiniteEmptyHeadTests(unittest.TestCase):
    def test_nonempty_loss_and_gradient_match_original_cross_entropy(self):
        labels = torch.tensor([[-100, 2, -100]])
        actual_logits = torch.randn(1, 3, 5, requires_grad=True)
        expected_logits = actual_logits.detach().clone().requires_grad_(True)
        actual = DummyHeads()._shifted_ce(actual_logits, labels)
        expected = F.cross_entropy(
            expected_logits[:, :-1, :].reshape(-1, 5),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        self.assertTrue(torch.equal(actual, expected))
        actual.backward()
        expected.backward()
        self.assertTrue(torch.equal(actual_logits.grad, expected_logits.grad))

    def test_all_ignored_returns_connected_fp32_zero_and_zero_gradient(self):
        logits = torch.randn(1, 3, 5, dtype=torch.float16, requires_grad=True)
        labels = torch.full((1, 3), -100)
        loss = DummyHeads()._shifted_ce(logits, labels)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(torch.count_nonzero(logits.grad).item(), 0)

    def test_nonfinite_anchor_cannot_create_inf_times_zero_nan(self):
        logits = torch.full((1, 2, 5), float("inf"), requires_grad=True)
        labels = torch.full((1, 2), -100)
        loss = DummyHeads()._shifted_ce(logits, labels)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_seven_head_weighted_total_is_finite_with_empty_heads(self):
        dummy = DummyHeads()
        heads = tuple(
            MODULE.LossHead(f"head{i}", f"label{i}", f"weight{i}") for i in range(7)
        )
        for i in range(7):
            setattr(dummy.train_config, f"weight{i}", 1.0)
        logits = torch.randn(1, 3, 5, requires_grad=True)
        preds = torch.argmax(logits, dim=-1)
        batch = {"switch_loss_rate": False}
        for i in range(7):
            batch[f"label{i}"] = torch.full((1, 3), -100)
        batch["label0"][0, 1] = 2
        total, losses, _ = dummy._compute_heads(logits, preds, batch, heads)
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(all(torch.isfinite(loss) for loss in losses.values()))
        total.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
