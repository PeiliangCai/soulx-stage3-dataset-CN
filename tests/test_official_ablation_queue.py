import copy
import unittest

from duplexconv_stage3.official_ablation_queue import (
    EXPECTED_BASE_SHA256,
    EXPECTED_CHECKPOINT_STEPS,
    EXPECTED_UPSTREAM_COMMIT,
    audit_variant_config,
    validate_group_validation_result,
    validate_training_manifest,
)


class OfficialAblationQueueTest(unittest.TestCase):
    def test_expected_ablation_differences_pass(self):
        reference = self._config()
        for name in ("B", "C", "D"):
            candidate = self._variant(reference, name)
            result = audit_variant_config(name, reference, candidate)
            self.assertEqual(result["group"], name)

    def test_unapproved_training_change_fails(self):
        reference = self._config()
        candidate = self._variant(reference, "B")
        candidate["train_config"]["learning_rate"] = 2e-4
        with self.assertRaisesRegex(RuntimeError, "unapproved"):
            audit_variant_config("B", reference, candidate)

    def test_complete_training_manifest_passes(self):
        manifest = self._training_manifest()
        result = validate_training_manifest(manifest, "split")
        self.assertEqual(result["optimizer_update_count"], 30)
        self.assertEqual(result["sample_exposure"], 17280)

    def test_effective_batch_drift_fails(self):
        manifest = self._training_manifest()
        manifest["updates"][4]["samples"] = 575
        with self.assertRaisesRegex(RuntimeError, "effective batch"):
            validate_training_manifest(manifest, "split")

    def test_step0_validation_rejects_optimizer(self):
        result = self._validation_result(0)
        result["optimizer_created"] = True
        with self.assertRaisesRegex(RuntimeError, "created an optimizer"):
            validate_group_validation_result(result, 0, "split", None)

    @staticmethod
    def _config():
        return {
            "model_config": {"task": "state_prediction"},
            "dataset_config": {
                "train_data_path": "/data/original/train.parquet",
                "validation_data_path": "/data/original/validation.parquet",
                "split_manifest_path": "/data/original/split_manifest.json",
                "batch_size": 1,
            },
            "train_config": {
                "total_steps": 30,
                "learning_rate": 1e-4,
                "accumulate_grad_batches": 576,
                "seed": 42,
                "origin_step_estimate": 1800,
                "continual_checkpoint_steps": EXPECTED_CHECKPOINT_STEPS,
                "user_complete_loss_rate": 0.24,
                "user_incomplete_loss_rate": 0.13,
                "continual_run_id": "A",
                "continual_audit_dir": "/training/A",
                "default_root_dir": "/training/A",
                "wandb_run_name": "A",
                "wandb_save_dir": "/training/A/wandb",
            },
        }

    @staticmethod
    def _variant(reference, name):
        result = copy.deepcopy(reference)
        train = result["train_config"]
        for key in (
            "continual_run_id",
            "continual_audit_dir",
            "default_root_dir",
            "wandb_run_name",
            "wandb_save_dir",
        ):
            train[key] = f"{train[key]}-{name}"
        if name in {"B", "D"}:
            train["user_complete_loss_rate"] = 0.185
            train["user_incomplete_loss_rate"] = 0.185
        if name in {"C", "D"}:
            data = result["dataset_config"]
            data["train_data_path"] = "/data/balanced_ci_v1/train.parquet"
            data["validation_data_path"] = "/data/balanced_ci_v1/validation.parquet"
            data["split_manifest_path"] = "/data/balanced_ci_v1/split_manifest.json"
        return result

    @staticmethod
    def _training_manifest():
        return {
            "status": "complete",
            "runtime_base_commit": EXPECTED_UPSTREAM_COMMIT,
            "base_checkpoint": {"sha256": EXPECTED_BASE_SHA256},
            "split": {"split_identity_sha256": "split"},
            "final_local_step": 30,
            "checkpoint_steps": EXPECTED_CHECKPOINT_STEPS,
            "updates": [
                {"local_step": step, "microbatches": 576, "samples": 576}
                for step in range(1, 31)
            ],
            "total_samples": 17280,
            "cuda_peak_memory_bytes": 1,
        }

    @staticmethod
    def _validation_result(step):
        heads = {
            name: {
                "token_weighted_cross_entropy": 1.0,
                "token_weighted_accuracy": 0.5,
            }
            for name in (
                "text",
                "eos",
                "idle",
                "nonidle",
                "user_complete",
                "user_incomplete",
                "user_backchannel",
            )
        }
        return {
            "status": "complete",
            "local_step": step,
            "base_checkpoint": {"sha256": EXPECTED_BASE_SHA256},
            "split_manifest": {"split_identity_sha256": "split"},
            "validation_data": {"row_count": 2061},
            "optimizer_created": False,
            "training_updates_performed": 0,
            "exact_token_weighted_metrics": {
                "profile": "exact-per-row-target-weighted-v1",
                "objective": 1.0,
                "accuracy": 0.5,
                "heads": heads,
            },
            "continuation_checkpoint": None,
        }


if __name__ == "__main__":
    unittest.main()
