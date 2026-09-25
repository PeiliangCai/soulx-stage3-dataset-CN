import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    path = PROJECT_ROOT / "scripts/run_abcd_table3_step5_10.py"
    spec = importlib.util.spec_from_file_location("abcd_table3_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AbcdTable3StepFiveTenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def test_group_pool_scales_to_three_devices(self):
        lock = threading.Lock()
        active_devices = set()
        maximum_active = 0

        def worker(group, device):
            nonlocal maximum_active
            with lock:
                self.assertNotIn(device, active_devices)
                active_devices.add(device)
                maximum_active = max(maximum_active, len(active_devices))
            time.sleep(0.03)
            with lock:
                active_devices.remove(device)
            return {"group": group, "device": device}

        results, assignments = self.runner.execute_group_pool(
            ["B", "C", "D"], ["0", "1", "2"], worker
        )
        self.assertEqual(sorted(results), ["B", "C", "D"])
        self.assertEqual(len(assignments), 3)
        self.assertEqual(maximum_active, 3)

    def test_group_pool_never_overlaps_one_device(self):
        lock = threading.Lock()
        active = 0
        maximum_active = 0

        def worker(group, device):
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.01)
            with lock:
                active -= 1
            return group

        results, _ = self.runner.execute_group_pool(
            ["B", "C", "D"], ["0"], worker
        )
        self.assertEqual(sorted(results), ["B", "C", "D"])
        self.assertEqual(maximum_active, 1)

    def test_failure_stops_later_unscheduled_group(self):
        called = []

        def worker(group, device):
            called.append(group)
            raise RuntimeError("intentional failure")

        with self.assertRaises(RuntimeError):
            self.runner.execute_group_pool(["B", "C", "D"], ["0"], worker)
        self.assertEqual(called, ["B"])

    def test_config_freezes_groups_steps_and_execution_waves(self):
        config = {
            "expected_steps": [5, 10],
            "groups": {"B": {}, "C": {}, "D": {}},
            "execution_waves": [
                ["baseline", "B5", "C5"],
                ["D5", "B10", "C10"],
                ["D10"],
            ],
            "table3_used_for_training_or_checkpoint_selection": False,
            "table2_enabled": False,
            "paid_api_enabled": False,
        }
        self.runner.validate_config(config)
        config["expected_steps"] = [5]
        with self.assertRaises(RuntimeError):
            self.runner.validate_config(config)

    def test_fresh_baseline_must_match_frozen_a_counts(self):
        expected = {
            "en/complete": {"correct": 251, "total": 318},
            "en/incomplete": {"correct": 268, "total": 299},
            "zh/complete": {"correct": 263, "total": 300},
            "zh/incomplete": {"correct": 241, "total": 300},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for key, row in expected.items():
                language, label = key.split("/")
                payload = {
                    "summary": {
                        "by_class": {
                            key: {
                                "correct": row["correct"],
                                "total": row["total"],
                                "accuracy": row["correct"] / row["total"],
                            }
                        }
                    }
                }
                (root / f"{language}-{label}.json").write_text(json.dumps(payload))
            gate = {"evaluation_mode": "baseline", "evidence_audit_passed": True}
            observed = self.runner.validate_baseline(root, expected, gate)
            self.assertEqual(observed["en/complete"]["correct"], 251)
            expected["en/complete"]["correct"] = 252
            with self.assertRaises(self.runner.BaselineMismatchError):
                self.runner.validate_baseline(root, expected, gate)

    def test_resume_identity_normalizes_integer_step_keys(self):
        controller_path = str(
            (PROJECT_ROOT / "scripts/run_abcd_table3_step5_10.py").resolve()
        )
        current = {
            "controller_files": [
                {"path": controller_path, "sha256": "new-controller-hash"}
            ],
            "groups": {"B": {5: {"sha256": "checkpoint-hash"}}},
        }
        stored = json.loads(json.dumps(current))
        stored["controller_files"][0]["sha256"] = (
            self.runner.LEGACY_RESUME_CONTROLLER_SHA256
        )
        migration = self.runner.validate_or_migrate_resume_identity(stored, current)
        self.assertEqual(migration["to_sha256"], "new-controller-hash")

    def test_resume_identity_rejects_unrelated_drift(self):
        controller_path = str(
            (PROJECT_ROOT / "scripts/run_abcd_table3_step5_10.py").resolve()
        )
        current = {
            "controller_files": [
                {"path": controller_path, "sha256": "new-controller-hash"}
            ],
            "groups": {"B": {5: {"sha256": "checkpoint-hash"}}},
        }
        stored = json.loads(json.dumps(current))
        stored["controller_files"][0]["sha256"] = (
            self.runner.LEGACY_RESUME_CONTROLLER_SHA256
        )
        stored["groups"]["B"]["5"]["sha256"] = "tampered"
        with self.assertRaises(RuntimeError):
            self.runner.validate_or_migrate_resume_identity(stored, current)


if __name__ == "__main__":
    unittest.main()
