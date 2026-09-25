import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_frozen_expansion_shard_queue.py"
)
SPEC = importlib.util.spec_from_file_location("frozen_expansion_queue", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
QUEUE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUEUE)


class FrozenExpansionQueueTests(unittest.TestCase):
    def test_candidate_tar_path_is_shard_scoped(self) -> None:
        self.assertEqual(
            QUEUE.candidate_tar_for(7),
            QUEUE.DATA_ROOT
            / "work/gate_d_edu0007_v1/candidate_model_ready_views.tar",
        )

    def test_candidate_cleanup_audit_requires_exact_hash_and_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.tar"
            candidate.write_bytes(b"candidate")
            required = root / "model-ready"
            required.mkdir()
            closure = {
                "gate_passed": True,
                "sha256": {"candidate_tar": QUEUE.sha256_file(candidate)},
            }
            audit = QUEUE.audit_candidate_tar_cleanup(
                candidate, closure, required_inputs=[required]
            )
            self.assertEqual(audit["status"], "predelete_audit_passed")
            self.assertEqual(audit["candidate_tar_bytes"], len(b"candidate"))

            closure["sha256"]["candidate_tar"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "differs"):
                QUEUE.audit_candidate_tar_cleanup(
                    candidate, closure, required_inputs=[required]
                )

    def test_candidate_cleanup_audit_rejects_failed_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.tar"
            candidate.write_bytes(b"candidate")
            with self.assertRaisesRegex(RuntimeError, "passed Gate D"):
                QUEUE.audit_candidate_tar_cleanup(
                    candidate,
                    {"gate_passed": False, "sha256": {}},
                    required_inputs=[],
                )


if __name__ == "__main__":
    unittest.main()
