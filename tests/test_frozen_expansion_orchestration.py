import importlib.util
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_frozen_expansion_shard_full_pipeline.py"
)
SPEC = importlib.util.spec_from_file_location("frozen_expansion_runner", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class FrozenExpansionOrchestrationTests(unittest.TestCase):
    def test_default_names_are_unchanged(self) -> None:
        names = [
            "full_pipeline_manifest_v1.json",
            "leakage_gate_v2_2",
            "source_scan.log",
        ]
        self.assertEqual(
            [RUNNER.variant_output_name(name, None) for name in names], names
        )

    def test_sanitized_variant_is_inserted_before_existing_version(self) -> None:
        self.assertEqual(
            RUNNER.variant_output_name("full_pipeline_manifest_v1.json", "sanitized"),
            "full_pipeline_manifest_sanitized_v1.json",
        )
        self.assertEqual(
            RUNNER.variant_output_name("leakage_gate_v2_2", "sanitized"),
            "leakage_gate_sanitized_v2_2",
        )

    def test_unversioned_log_gets_variant_and_v1(self) -> None:
        self.assertEqual(
            RUNNER.variant_output_name("source_scan.log", "sanitized"),
            "source_scan_sanitized_v1.log",
        )

    def test_valid_variant_is_accepted(self) -> None:
        self.assertEqual(RUNNER.validate_run_variant("sanitized"), "sanitized")
        self.assertEqual(RUNNER.validate_run_variant("source_fix_2"), "source_fix_2")

    def test_invalid_variant_is_rejected(self) -> None:
        for value in ("Sanitized", "../sanitized", "sanitized-v1", "sanitized__v1"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    RUNNER.validate_run_variant(value)

    def test_qwen_route_default_is_legacy_direct_and_proxy_is_explicit(self) -> None:
        self.assertEqual(RUNNER.QWEN_ROUTE, "direct-no-proxy-v2")
        self.assertEqual(
            RUNNER.validate_qwen_route("environment-proxy-aware-v2"),
            "environment-proxy-aware-v2",
        )
        with self.assertRaisesRegex(ValueError, "unsupported Qwen route"):
            RUNNER.validate_qwen_route("implicit-route")

    def _resume_fixture(self, result_file: Path):
        expected_stages = [
            "gate_a_metadata_identity",
            "source_scan",
            "frozen_leakage_gate_b_c",
            "prepare_qwen_requests",
            "qwen_connectivity",
            "qwen_calibration",
            "qwen_calibration_audit",
        ]
        state = {
            "status": "failed",
            "stage": "qwen_full_labeling",
            "contract_sha256": "contract-sha",
            "completed_stages": [{"stage": stage} for stage in expected_stages],
        }
        failed_labeling = {
            "status": "failed",
            "request_file_sha256": "request-sha",
            "request_count": 407,
            "target_event_count": 1778,
        }
        recovery = {
            "status": "complete",
            "request_file_sha256": "request-sha",
            "result_file": str(result_file),
            "result_file_sha256": RUNNER.sha256_file(result_file),
            "summary": {
                "request_count": 407,
                "result_count": 407,
                "target_event_count": 1778,
                "result_network_route_policy_counts": {
                    "direct-no-proxy-v2": 406,
                    "environment-proxy-aware-v2": 1,
                },
            },
        }
        return state, failed_labeling, recovery

    def test_qwen_resume_accepts_exact_closed_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result_file = Path(directory) / "full_results.jsonl"
            result_file.write_text('{"request_id":"one"}\n', encoding="utf-8")
            state, failed_labeling, recovery = self._resume_fixture(result_file)
            RUNNER.validate_resume_after_qwen_full(
                state,
                dict(state),
                failed_labeling,
                recovery,
                contract_sha256="contract-sha",
                request_file_sha256="request-sha",
                result_file=result_file,
                expected_route_counts={
                    "direct-no-proxy-v2": 406,
                    "environment-proxy-aware-v2": 1,
                },
            )

    def test_qwen_resume_rejects_pipeline_snapshot_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result_file = Path(directory) / "full_results.jsonl"
            result_file.write_text('{"request_id":"one"}\n', encoding="utf-8")
            state, failed_labeling, recovery = self._resume_fixture(result_file)
            snapshot = dict(state)
            snapshot["status"] = "running"
            with self.assertRaisesRegex(RuntimeError, "immutable snapshot"):
                RUNNER.validate_resume_after_qwen_full(
                    state,
                    snapshot,
                    failed_labeling,
                    recovery,
                    contract_sha256="contract-sha",
                    request_file_sha256="request-sha",
                    result_file=result_file,
                    expected_route_counts={
                        "direct-no-proxy-v2": 406,
                        "environment-proxy-aware-v2": 1,
                    },
                )

    def test_qwen_resume_rejects_incomplete_recovery_closure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result_file = Path(directory) / "full_results.jsonl"
            result_file.write_text('{"request_id":"one"}\n', encoding="utf-8")
            state, failed_labeling, recovery = self._resume_fixture(result_file)
            recovery["summary"]["result_count"] = 406
            with self.assertRaisesRegex(RuntimeError, "close every frozen full request"):
                RUNNER.validate_resume_after_qwen_full(
                    state,
                    dict(state),
                    failed_labeling,
                    recovery,
                    contract_sha256="contract-sha",
                    request_file_sha256="request-sha",
                    result_file=result_file,
                    expected_route_counts={
                        "direct-no-proxy-v2": 406,
                        "environment-proxy-aware-v2": 1,
                    },
                )

    def test_qwen_resume_rejects_unapproved_route_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result_file = Path(directory) / "full_results.jsonl"
            result_file.write_text('{"request_id":"one"}\n', encoding="utf-8")
            state, failed_labeling, recovery = self._resume_fixture(result_file)
            recovery["summary"]["result_network_route_policy_counts"] = {
                "direct-no-proxy-v2": 407
            }
            with self.assertRaisesRegex(RuntimeError, "differs from approval"):
                RUNNER.validate_resume_after_qwen_full(
                    state,
                    dict(state),
                    failed_labeling,
                    recovery,
                    contract_sha256="contract-sha",
                    request_file_sha256="request-sha",
                    result_file=result_file,
                    expected_route_counts={
                        "direct-no-proxy-v2": 406,
                        "environment-proxy-aware-v2": 1,
                    },
                )

    def test_route_count_arguments_are_exact(self) -> None:
        self.assertEqual(
            RUNNER.parse_route_count_arguments(
                [
                    "direct-no-proxy-v2=406",
                    "environment-proxy-aware-v2=1",
                ]
            ),
            {
                "direct-no-proxy-v2": 406,
                "environment-proxy-aware-v2": 1,
            },
        )
        with self.assertRaises(ValueError):
            RUNNER.parse_route_count_arguments(["unknown=407"])


if __name__ == "__main__":
    unittest.main()
