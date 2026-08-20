import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from duplexconv_stage3 import table3_gate
from duplexconv_stage3.table3_gate import audit_result
from duplexconv_stage3.table3_protocol import (
    EXPECTED_CHECKPOINT_LOAD_UNEXPECTED_KEYS,
    EXPECTED_MODEL_MANIFEST_SHA256,
    EXPECTED_OFFICIAL_CHECKPOINT_SHA256,
    EXPECTED_RUNTIME_VERSIONS,
    EXPECTED_UPSTREAM_COMMIT,
    EXPECTED_UPSTREAM_SCRIPT_SHA256,
    canonical_sha256,
    classify_trace,
    discover_samples,
    portable_inventory_identity,
    runtime_version_mismatches,
    summarize_records,
)
from duplexconv_stage3.table3_reproduction import (
    FreshAuditedASRCache,
    audit_checkpoint_compatibility,
)


class FakeASR:
    def __init__(self):
        self.calls = 0

    def recognize(self, _audio, sample_rate=16_000, **_kwargs):
        self.calls += 1
        return f"text-{sample_rate}-{self.calls}"


class Table3ProtocolTests(unittest.TestCase):
    def test_four_readout_rules_are_distinct_and_label_independent(self):
        trace = [
            {"state": "idle", "timestamp": [0.0, 0.16]},
            {"state": "wait", "timestamp": [0.16, 0.32]},
            {"state": "speak", "timestamp": [0.32, 0.48]},
            {"state": "wait", "timestamp": [0.48, 0.64]},
        ]
        self.assertEqual(
            classify_trace(trace, 0.35, "first-terminal-v1")["prediction"],
            "incomplete",
        )
        self.assertEqual(
            classify_trace(trace, 0.35, "last-terminal-v1")["prediction"],
            "incomplete",
        )
        self.assertEqual(
            classify_trace(trace, 0.35, "closest-to-file-endpoint-v1")[
                "prediction"
            ],
            "complete",
        )
        self.assertEqual(
            classify_trace(trace, 0.35, "first-at-or-after-file-endpoint-v1")[
                "prediction"
            ],
            "incomplete",
        )

    def test_first_after_endpoint_can_return_none(self):
        trace = [{"state": "speak", "timestamp": [0.16, 0.32]}]
        result = classify_trace(
            trace, 1.0, "first-at-or-after-file-endpoint-v1"
        )
        self.assertEqual(result["prediction"], "none")
        self.assertIsNone(result["selected_terminal"])

    def test_chinese_discovery_follows_released_list(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "complete/real").mkdir(parents=True)
            first = root / "complete/real/b.wav"
            second = root / "complete/real/a.wav"
            first.write_bytes(b"b")
            second.write_bytes(b"a")
            rows = [
                json.dumps({"wav": "./complete/real/b.wav"}),
                json.dumps({"wav": "./complete/real/a.wav"}),
            ]
            (root / "complete/complete_test.list").write_text("\n".join(rows))
            with patch.dict(
                "duplexconv_stage3.table3_protocol.EXPECTED_COUNTS",
                {("zh", "complete"): 2},
            ):
                samples = discover_samples(
                    root, "zh", "complete", "official-list"
                )
        self.assertEqual([sample.wav_path.name for sample in samples], ["b.wav", "a.wav"])

    def test_runtime_version_audit_reports_drift(self):
        packages = dict(EXPECTED_RUNTIME_VERSIONS)
        packages["transformers"] = "4.52.1"
        self.assertEqual(
            runtime_version_mismatches(packages),
            {"transformers": {"expected": "4.55.0", "actual": "4.52.1"}},
        )

    def test_fresh_asr_cache_records_text_and_hits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.jsonl"
            backend = FakeASR()
            cache = FreshAuditedASRCache(path, lambda: backend, "fake-v1")
            cache.begin_sample()
            audio = np.zeros(16, dtype=np.float32)
            self.assertEqual(cache.recognize(audio), "text-16000-1")
            self.assertEqual(cache.recognize(audio), "text-16000-1")
            self.assertEqual(cache.stats()["misses"], 1)
            self.assertEqual(cache.stats()["hits"], 1)
            self.assertEqual(len(cache.sample_calls), 2)
            with self.assertRaises(FileExistsError):
                FreshAuditedASRCache(path, lambda: backend, "fake-v1")

    def test_checkpoint_audit_accepts_only_the_tied_late_bound_alias(self):
        import torch

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.llm = torch.nn.Module()
                self.llm.base_model = torch.nn.Module()
                self.llm.base_model.model = torch.nn.Module()
                self.llm.base_model.model.model = torch.nn.Module()
                embedding = torch.nn.Embedding(4, 3)
                self.llm.base_model.model.model.embed_tokens = embedding
                self.llm.base_model.model.lm_head = torch.nn.Linear(3, 4, bias=False)
                self.llm.base_model.model.lm_head.weight = embedding.weight
                self.embed_tokens_func = embedding

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pt"
            model = TinyModel()
            torch.save(model.state_dict(), checkpoint_path)
            result = audit_checkpoint_compatibility(model, checkpoint_path)
            self.assertEqual(result["status"], "accepted")
            self.assertEqual(
                result["unexpected_keys_at_load"],
                ["embed_tokens_func.weight"],
            )

            forged = model.state_dict()
            forged["rogue.weight"] = torch.ones(1)
            torch.save(forged, checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "compatibility audit failed"):
                audit_checkpoint_compatibility(model, checkpoint_path)

    def _minimal_result(self, root: Path) -> Path:
        (root / "one.wav").write_bytes(b"x")
        trace = [{"state": "speak", "timestamp": [0.0, 0.16]}]
        trace_path = root / "trace.json"
        trace_path.write_text(json.dumps(trace))
        initialization_log = root / "initialization.log"
        initialization_log.write_text(
            "load_state_dict failed: Unexpected key(s) in state_dict: "
            '"embed_tokens_func.weight"\n'
        )
        asr_cache_path = root / "asr.jsonl"
        asr_cache_path.write_text(
            json.dumps(
                {
                    "key": "key",
                    "text": "hello",
                    "backend_identity": "fake-asr",
                }
            )
            + "\n"
        )
        primary = classify_trace(trace, 0.1, "last-terminal-v1")
        sensitivity = {
            rule: classify_trace(trace, 0.1, rule)
            for rule in (
                "first-terminal-v1",
                "closest-to-file-endpoint-v1",
                "first-at-or-after-file-endpoint-v1",
            )
        }
        record = {
            "sequence_index": 0,
            "sample_id": "en:complete:one",
            "language": "en",
            "label": "complete",
            "wav_path": str(root / "one.wav"),
            "wav_size": 1,
            "wav_sha256": "wav-hash",
            "audio_duration_seconds": 0.1,
            "trace_path": str(trace_path),
            "trace_sha256": "trace-hash",
            "log_path": str(root / "one.log"),
            "log_sha256": "log-hash",
            "trace": trace,
            "teacher_asr_calls": [
                {"key": "key", "text": "hello", "cache_hit": False}
            ],
            "llm_forward_audit": [
                {
                    "forward_index": 0,
                    "argmax_token_id": 1,
                    "state_logits": {
                        "complete": 1.0,
                        "incomplete": 0.0,
                        "backchannel": 0.0,
                        "idle": 0.0,
                        "nonidle": 0.0,
                    },
                }
            ],
            "primary_readout": primary,
            "sensitivity_readouts": sensitivity,
            "correct": True,
        }
        inventory = [
            {
                "sample_id": record["sample_id"],
                "wav_path": record["wav_path"],
                "size": 1,
                "sha256": "wav-hash",
            }
        ]
        payload = {
            "schema_version": 1,
            "status": "complete",
            "run_mode": "formal",
            "run_id": "audit",
            "language": "en",
            "label": "complete",
            "protocol": {
                "status": "frozen-candidate-v1",
                "primary_rule": "last-terminal-v1",
                "sensitivity_rules": [
                    "first-terminal-v1",
                    "closest-to-file-endpoint-v1",
                    "first-at-or-after-file-endpoint-v1",
                ],
                "post_silence_samples": 32000,
                "post_silence_seconds": 2.0,
                "far_field_filter": False,
                "sample_order": "official-os-walk",
                "resume_allowed": False,
                "autocast_enabled_by_runner": False,
            },
            "checkpoint": {
                "path": str(root / "checkpoint"),
                "sha256": EXPECTED_OFFICIAL_CHECKPOINT_SHA256,
                "load_audit": {
                    "policy": "known-late-bound-embedding-alias-v1",
                    "status": "accepted",
                    "missing_keys_at_load": [],
                    "unexpected_keys_at_load": list(
                        EXPECTED_CHECKPOINT_LOAD_UNEXPECTED_KEYS
                    ),
                    "allowed_unexpected_keys": list(
                        EXPECTED_CHECKPOINT_LOAD_UNEXPECTED_KEYS
                    ),
                    "final_missing_keys": [],
                    "final_unexpected_keys": [],
                    "shape_mismatches": {},
                    "checkpoint_alias_tied": True,
                    "loaded_model_alias_tied": True,
                },
            },
            "upstream": {
                "root": str(root),
                "commit": EXPECTED_UPSTREAM_COMMIT,
                "tree": "tree",
                "dirty": False,
                "inference_script": str(root / "duplex_inference.py"),
                "inference_script_sha256": EXPECTED_UPSTREAM_SCRIPT_SHA256,
            },
            "project": {},
            "runtime_environment": {
                "python": "3.10.20",
                "packages": dict(EXPECTED_RUNTIME_VERSIONS),
                "cuda": {"available": True},
            },
            "config": {"path": str(root / "config.yaml"), "sha256": "config"},
            "model_artifacts": {
                "llm": {
                    "manifest_sha256": EXPECTED_MODEL_MANIFEST_SHA256["llm"]
                },
                "speech_tokenizer": {
                    "manifest_sha256": EXPECTED_MODEL_MANIFEST_SHA256[
                        "speech_tokenizer"
                    ]
                },
            },
            "asr_model": {
                "manifest_sha256": EXPECTED_MODEL_MANIFEST_SHA256["asr_en"]
            },
            "initialization_log": {
                "path": str(initialization_log),
                "sha256": "initialization",
            },
            "asr_cache": {
                "path": str(asr_cache_path),
                "sha256": "asr",
                "backend_identity": "fake-asr",
                "entries": 1,
                "hits": 0,
                "misses": 1,
            },
            "dataset": {
                "root": str(root),
                "inventory": inventory,
                "inventory_sha256": canonical_sha256(inventory),
                "portable_identity_sha256": portable_inventory_identity(
                    inventory, root
                ),
                "source_class_identity_sha256": portable_inventory_identity(
                    inventory, root
                ),
            },
            "records": [record],
            "summary": summarize_records([record]),
        }
        path = root / "result.json"
        path.write_text(json.dumps(payload))
        return path

    def test_gate_recomputes_summary_instead_of_trusting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._minimal_result(root)

            def git_output(command, text=True):
                return "" if "status" in command else EXPECTED_UPSTREAM_COMMIT + "\n"

            with patch.dict(
                table3_gate.EXPECTED_COUNTS, {("en", "complete"): 1}
            ), patch.dict(
                table3_gate.PAPER_CORRECT, {("en", "complete"): 1}
            ), patch.dict(
                table3_gate.EXPECTED_DATASET_IDENTITY_SHA256,
                {
                    ("en", "complete"): portable_inventory_identity(
                        [
                            {
                                "sample_id": "en:complete:one",
                                "wav_path": str(root / "one.wav"),
                                "size": 1,
                                "sha256": "wav-hash",
                            }
                        ],
                        root,
                    )
                },
            ), patch.object(
                table3_gate, "verify_file"
            ), patch.object(
                table3_gate, "verify_directory_manifest"
            ), patch.object(
                table3_gate, "verify_project_identity"
            ), patch.object(
                table3_gate.subprocess, "check_output", side_effect=git_output
            ):
                _payload, report = audit_result(path, "en", "complete")
                self.assertEqual(report["actual_correct"], 1)

                forged = json.loads(path.read_text())
                forged["summary"]["by_class"]["en/complete"]["correct"] = 0
                path.write_text(json.dumps(forged))
                with self.assertRaisesRegex(ValueError, "summary"):
                    audit_result(path, "en", "complete")


if __name__ == "__main__":
    unittest.main()
