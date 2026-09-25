from __future__ import annotations

import unittest

from scripts.migrate_baidu_release_raw import migration_state


class BaiduRawMigrationTests(unittest.TestCase):
    def test_ready_and_resume_states_are_unambiguous(self) -> None:
        self.assertEqual(migration_state(True, False), "ready_to_move")
        self.assertEqual(
            migration_state(False, True), "already_moved_verify_resume"
        )

    def test_rejects_duplicate_or_missing_raw_trees(self) -> None:
        with self.assertRaises(RuntimeError):
            migration_state(True, True)
        with self.assertRaises(RuntimeError):
            migration_state(False, False)


if __name__ == "__main__":
    unittest.main()
