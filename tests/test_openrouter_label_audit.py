import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_openrouter_state_labels.py"
)
SPEC = importlib.util.spec_from_file_location("openrouter_label_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class OpenRouterLabelAuditTests(unittest.TestCase):
    def test_legacy_route_contract_requires_one_route_for_every_result(self) -> None:
        self.assertEqual(
            AUDIT.parse_expected_route_counts(
                [], default_route="direct-no-proxy-v2", result_count=407
            ),
            {"direct-no-proxy-v2": 407},
        )

    def test_mixed_route_contract_is_exact(self) -> None:
        self.assertEqual(
            AUDIT.parse_expected_route_counts(
                [
                    "direct-no-proxy-v2=406",
                    "environment-proxy-aware-v2=1",
                ],
                default_route="environment-proxy-aware-v2",
                result_count=407,
            ),
            {
                "direct-no-proxy-v2": 406,
                "environment-proxy-aware-v2": 1,
            },
        )

    def test_mixed_route_contract_rejects_wrong_total(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not sum"):
            AUDIT.parse_expected_route_counts(
                ["direct-no-proxy-v2=406"],
                default_route="environment-proxy-aware-v2",
                result_count=407,
            )


if __name__ == "__main__":
    unittest.main()
