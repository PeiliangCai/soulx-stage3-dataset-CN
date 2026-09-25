import importlib.util
from pathlib import Path
import threading
import time
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    path = PROJECT_ROOT / "scripts/run_official_table3_coarse_to_fine.py"
    spec = importlib.util.spec_from_file_location(
        "run_official_table3_coarse_to_fine_parallel_test", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OfficialTable3ParallelSchedulerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def test_resolve_devices_supports_explicit_and_auto_selection(self):
        self.assertEqual(self.runner.resolve_devices("0,1"), ["0", "1"])
        self.assertEqual(
            self.runner.resolve_devices(
                "auto", environment={"CUDA_VISIBLE_DEVICES": "3,7"}
            ),
            ["3", "7"],
        )
        self.assertEqual(
            self.runner.resolve_devices(
                "auto", environment={}, cuda_device_count=3
            ),
            ["0", "1", "2"],
        )
        with self.assertRaises(ValueError):
            self.runner.resolve_devices("0,0")

    def test_pool_scales_and_never_overlaps_work_on_one_device(self):
        lock = threading.Lock()
        active_devices = set()
        maximum_active = 0

        def worker(step, device):
            nonlocal maximum_active
            with lock:
                self.assertNotIn(device, active_devices)
                active_devices.add(device)
                maximum_active = max(maximum_active, len(active_devices))
            time.sleep(0.03)
            with lock:
                active_devices.remove(device)
            return {"step": step, "device": device}

        results, assignments = self.runner.execute_step_pool(
            [1, 2, 3, 4, 5], ["0", "1"], worker
        )

        self.assertEqual(sorted(results), [1, 2, 3, 4, 5])
        self.assertEqual(len(assignments), 5)
        self.assertEqual(maximum_active, 2)
        self.assertTrue(all(item["status"] == "complete" for item in assignments))

    def test_pool_preserves_single_gpu_serial_execution(self):
        lock = threading.Lock()
        active = 0
        maximum_active = 0

        def worker(step, device):
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.01)
            with lock:
                active -= 1
            return step

        results, _ = self.runner.execute_step_pool([1, 2, 3], ["0"], worker)

        self.assertEqual(sorted(results), [1, 2, 3])
        self.assertEqual(maximum_active, 1)

    def test_failure_prevents_submission_of_later_pending_steps(self):
        called = []

        def worker(step, device):
            called.append(step)
            raise RuntimeError("intentional test failure")

        with self.assertRaises(self.runner.ParallelStepFailure):
            self.runner.execute_step_pool([1, 2, 3], ["0"], worker)

        self.assertEqual(called, [1])


if __name__ == "__main__":
    unittest.main()
