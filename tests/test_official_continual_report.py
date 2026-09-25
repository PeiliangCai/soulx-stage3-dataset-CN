import importlib.util
from pathlib import Path
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_reporter():
    path = PROJECT_ROOT / "scripts/render_official_continual_report.py"
    spec = importlib.util.spec_from_file_location("render_official_continual_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_report_supervisor():
    path = PROJECT_ROOT / "scripts/run_official_report_after_evaluation.py"
    spec = importlib.util.spec_from_file_location(
        "run_official_report_after_evaluation", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_coarse_to_fine_runner():
    path = PROJECT_ROOT / "scripts/run_official_table3_coarse_to_fine.py"
    spec = importlib.util.spec_from_file_location(
        "run_official_table3_coarse_to_fine", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OfficialContinualReportTest(unittest.TestCase):
    def test_table3_rows_adds_frozen_baseline_before_checkpoint_grid(self):
        reporter = load_reporter()
        classes = {
            "en/complete": {
                "baseline_accuracy_percent": 80.0,
                "candidate_accuracy_percent": 81.0,
            },
            "en/incomplete": {
                "baseline_accuracy_percent": 90.0,
                "candidate_accuracy_percent": 88.0,
            },
            "zh/complete": {
                "baseline_accuracy_percent": 85.0,
                "candidate_accuracy_percent": 86.0,
            },
            "zh/incomplete": {
                "baseline_accuracy_percent": 75.0,
                "candidate_accuracy_percent": 74.0,
            },
        }
        payload = {
            "training_manifest": {"origin_step_estimate": 1800},
            "table3_index": {
                "checkpoints": [
                    {
                        "local_step": 1,
                        "estimated_total_optimizer_step": 1801,
                        "classes": classes,
                        "languages": {
                            "en": {"candidate_macro_accuracy_percent": 84.5},
                            "zh": {"candidate_macro_accuracy_percent": 80.0},
                        },
                        "almost_unchanged": True,
                        "obvious_decline_trigger": False,
                        "obvious_decline_confirmed": False,
                    }
                ]
            },
        }

        rows = reporter.table3_rows(payload)

        self.assertEqual([row["step"] for row in rows], [0, 1])
        self.assertEqual(rows[0]["estimated_total_step"], 1800)
        self.assertEqual(rows[0]["en_macro"], 85.0)
        self.assertEqual(rows[0]["zh_macro"], 80.0)
        self.assertEqual(rows[1]["en_incomplete"], 88.0)
        self.assertTrue(rows[1]["almost_unchanged"])

    def test_internal_rows_keeps_exact_token_weighted_profile_values(self):
        reporter = load_reporter()

        def exact(objective, accuracy):
            return {
                "objective": objective,
                "accuracy": accuracy,
                "heads": {"idle": {"token_weighted_accuracy": accuracy}},
            }

        payload = {
            "internal_validation": {
                "baseline": {"exact_token_weighted_metrics": exact(2.0, 0.7)},
                "checkpoints": [
                    {
                        "local_step": 1,
                        "exact_token_weighted_metrics": exact(1.9, 0.71),
                    }
                ],
            }
        }

        rows = reporter.internal_rows(payload)

        self.assertEqual([row["step"] for row in rows], [0, 1])
        self.assertEqual(rows[0]["objective"], 2.0)
        self.assertEqual(rows[1]["accuracy"], 0.71)
        self.assertEqual(rows[1]["heads"]["idle"]["token_weighted_accuracy"], 0.71)

    def test_report_supervisor_archives_interrupted_partial_outputs(self):
        supervisor = load_report_supervisor()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "report.md"
            output.write_text("partial", encoding="utf-8")
            manifest = {"identity": {"manifest": str(root / "state/manifest.json")}}

            supervisor.archive_partial_outputs(
                [output, root / "report.html"], manifest
            )

            self.assertFalse(output.exists())
            self.assertEqual(len(manifest["archived_partial_outputs"]), 1)
            archive = Path(manifest["archived_partial_outputs"][0]["archive"])
            self.assertEqual((archive / "report.md").read_text(encoding="utf-8"), "partial")

    def test_coarse_to_fine_refinement_is_frozen_on_step5_trigger(self):
        runner = load_coarse_to_fine_runner()

        def index(trigger):
            return {
                "checkpoints": [
                    {
                        "local_step": step,
                        "obvious_decline_trigger": trigger if step == 5 else False,
                    }
                    for step in runner.MANDATORY_ORDER
                ]
            }

        self.assertEqual(runner.choose_refinement_steps(index(False)), [])
        self.assertEqual(runner.choose_refinement_steps(index(True)), [2, 3])

    def test_coarse_to_fine_refinement_rejects_missing_mandatory_point(self):
        runner = load_coarse_to_fine_runner()
        with self.assertRaises(RuntimeError):
            runner.choose_refinement_steps(
                {"checkpoints": [{"local_step": 5, "obvious_decline_trigger": True}]}
            )


if __name__ == "__main__":
    unittest.main()
