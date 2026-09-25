import unittest

from duplexconv_stage3.ci_balance import (
    CATEGORY_BOTH,
    CATEGORY_COMPLETE_ONLY,
    CATEGORY_INCOMPLETE_ONLY,
    CATEGORY_NEITHER,
    ci_category,
    largest_remainder_allocation,
    select_ci_balanced_rows,
    sequence_state_counts,
)


class CompleteIncompleteBalanceTest(unittest.TestCase):
    def test_categories_use_exact_state_tokens(self):
        cases = {
            "<|user_complete|><|user_incomplete|>": CATEGORY_BOTH,
            "<|user_complete|><|user_idle|>": CATEGORY_COMPLETE_ONLY,
            "<|user_incomplete|><|user_nonidle|>": CATEGORY_INCOMPLETE_ONLY,
            "<|user_idle|><|user_backchannel|>": CATEGORY_NEITHER,
        }
        for sequence, expected in cases.items():
            self.assertEqual(ci_category(sequence_state_counts(sequence)), expected)

    def test_largest_remainder_closes_with_stable_tie_break(self):
        counts = {
            ("Edu_0002", "2", "no_backchannel"): 1,
            ("Edu_0001", "2", "no_backchannel"): 1,
            ("Edu_0003", "2", "no_backchannel"): 2,
        }
        allocation, _ = largest_remainder_allocation(counts, 2)
        self.assertEqual(sum(allocation.values()), 2)
        self.assertEqual(allocation[("Edu_0001", "2", "no_backchannel")], 1)
        self.assertEqual(allocation[("Edu_0003", "2", "no_backchannel")], 1)

    def test_selection_is_balanced_unique_and_order_preserving(self):
        rows = [
            self._row("both", "<|user_complete|><|user_incomplete|>"),
            self._row("c1", "<|user_complete|>"),
            self._row("n", "<|user_idle|>"),
            self._row("c2", "<|user_complete|><|user_backchannel|>"),
            self._row("i1", "<|user_incomplete|>"),
            self._row("c3", "<|user_complete|>"),
        ]
        metadata = {
            row["index"]: self._metadata(
                row["index"], "Edu_0002" if row["index"] == "c3" else "Edu_0001"
            )
            for row in rows
        }
        selected, audit = select_ci_balanced_rows(rows, metadata, 42)
        selected_again, audit_again = select_ci_balanced_rows(rows, metadata, 42)

        indexes = [row["index"] for row in selected]
        self.assertEqual(selected, selected_again)
        self.assertEqual(audit, audit_again)
        self.assertEqual(indexes, [row["index"] for row in rows if row["index"] in indexes])
        self.assertEqual(len(indexes), len(set(indexes)))
        self.assertIn("both", indexes)
        self.assertIn("i1", indexes)
        self.assertIn("n", indexes)
        self.assertEqual(len(indexes), 4)
        self.assertEqual(audit["output"]["complete_active_row_count"], 2)
        self.assertEqual(audit["output"]["incomplete_active_row_count"], 2)

    def test_selection_rejects_missing_metadata(self):
        rows = [self._row("c", "<|user_complete|>"), self._row("i", "<|user_incomplete|>")]
        with self.assertRaisesRegex(ValueError, "index sets differ"):
            select_ci_balanced_rows(rows, {"c": self._metadata("c", "Edu_0001")}, 42)

    @staticmethod
    def _row(index, sequence):
        return {"index": index, "sequence": sequence}

    @staticmethod
    def _metadata(index, shard):
        return {
            "source_shard": shard,
            "source_ntrack": 2,
            "source_id": f"source-{index}",
            "chunk_count": 1,
        }


if __name__ == "__main__":
    unittest.main()
