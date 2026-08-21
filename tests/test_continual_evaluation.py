import unittest

from duplexconv_stage3.continual_evaluation import (
    bootstrap_delta_ci,
    exact_mcnemar_p_value,
)


class ContinualEvaluationTest(unittest.TestCase):
    def test_mcnemar_no_discordance(self):
        self.assertEqual(exact_mcnemar_p_value([True, False], [True, False]), 1.0)

    def test_mcnemar_one_sided_changes(self):
        value = exact_mcnemar_p_value([True] * 6, [False] * 6)
        self.assertAlmostEqual(value, 0.03125)

    def test_bootstrap_delta_has_percentage_point_units(self):
        self.assertEqual(bootstrap_delta_ci([1.0] * 10, 42, 100), [100.0, 100.0])


if __name__ == "__main__":
    unittest.main()
