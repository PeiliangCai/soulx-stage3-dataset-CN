#!/usr/bin/env bash
set -euo pipefail

repo=/root/SoulX-stage3-dataset
python_bin=/root/autodl-tmp/conda_envs/soulx-duplug-official/bin/python
data=/root/autodl-tmp/dataset/duplexconv
paraformer_dir="$data/processed/paraformer_edu0038_v1"
decision="$data/reports/expansion_v1/Edu_0038/post_paraformer_relay_decision.json"

cd "$repo"

while tmux has-session -t soulx_edu0038_paraformer 2>/dev/null; do
  sleep 10
done

if [[ ! -f "$paraformer_dir/summary.json" ]]; then
  "$python_bin" - "$decision" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({
    "schema_version": 1,
    "status": "stopped_paraformer_missing_summary",
}, indent=2) + "\n")
PY
  exit 31
fi

if ! "$python_bin" - "$paraformer_dir/summary.json" "$decision" <<'PY'
import json
import pathlib
import sys

summary_path = pathlib.Path(sys.argv[1])
decision_path = pathlib.Path(sys.argv[2])
summary = json.loads(summary_path.read_text())
input_views = summary.get("input_view_count")
passed_views = summary.get("passed_view_count")
quarantined_views = summary.get("quarantined_view_count")
allow = input_views == 1008 and passed_views == 1008 and quarantined_views == 0
decision_path.parent.mkdir(parents=True, exist_ok=True)
decision_path.write_text(json.dumps({
    "schema_version": 1,
    "status": "continue_zero_strict_quarantine" if allow else "stopped_for_strict_quarantine_review",
    "input_view_count": input_views,
    "passed_view_count": passed_views,
    "quarantined_view_count": quarantined_views,
    "rule": "Continue automatically only for exact 1008=1008+0 Paraformer closure.",
}, indent=2) + "\n")
raise SystemExit(0 if allow else 42)
PY
then
  exit 42
fi

rendered="$data/processed/paraformer_rendered_edu0038_v1"
timelines="$data/processed/timelines_edu0038_v1"
glm="$data/processed/glm_audio_tokens_edu0038_v1"
glm_cache="$data/cache/glm_audio_tokens_edu0038_v1"
model_ready="$data/model_ready/edu0038_stage3_zh_v1"
validation="$data/reports/expansion_v1/Edu_0038/model_ready_validation_v1.json"
gate_work="$data/work/gate_d_edu0038_v1"
gate_output="$data/reports/expansion_v1/Edu_0038/leakage_gate_d_v1"
gate_closure="$data/reports/expansion_v1/Edu_0038/gate_d_closure_v1.json"

for output in "$rendered" "$timelines" "$glm" "$glm_cache" "$model_ready" "$gate_work" "$gate_output"; do
  if [[ -e "$output" ]]; then
    echo "refusing existing output: $output" >&2
    exit 32
  fi
done

"$python_bin" scripts/render_paraformer_text.py \
  --input-dir "$paraformer_dir" \
  --output-dir "$rendered"

"$python_bin" scripts/build_timelines.py \
  --scan-dir "$data/work/source_scan_edu0038_v1" \
  --state-dir "$data/processed/state_labels_edu0038_v1" \
  --asr-dir "$rendered" \
  --output-dir "$timelines"

"$python_bin" scripts/extract_glm_audio_tokens.py \
  --upstream-dir "$repo/third_party/SoulX-Duplug-upstream" \
  --model-dir "$repo/pretrained_models/glm-4-voice-tokenizer" \
  --audio-dir "$data/processed/target_audio_edu0038_v1" \
  --timeline-dir "$timelines" \
  --output-dir "$glm" \
  --cache-dir "$glm_cache" \
  --device cuda:0 \
  --batch-size 16 \
  --all

"$python_bin" scripts/export_model_ready.py \
  --timeline-dir "$timelines" \
  --glm-dir "$glm" \
  --tokenizer-dir "$repo/pretrained_models/Qwen3-0.6B-expand_vocab_v2" \
  --output-dir "$model_ready" \
  --upstream-commit 928b06508ed2de1344208d06fb1f6fb2ebfb1df5 \
  --max-token-length 1500 \
  --dataset-version duplexconv_edu0038_v1_stage3_zh_v1 \
  --index-prefix duplexconv_edu0038_v1

"$python_bin" scripts/validate_model_ready.py \
  --model-ready-dir "$model_ready" \
  --tokenizer-dir "$repo/pretrained_models/Qwen3-0.6B-expand_vocab_v2" \
  --upstream-dir "$repo/third_party/SoulX-Duplug-upstream" \
  --report-path "$validation" \
  --random-sample-count 20

mkdir -p "$gate_work"
"$python_bin" scripts/build_model_ready_gate_d_tar.py \
  --model-ready-dir "$model_ready" \
  --target-audio-dir "$data/processed/target_audio_edu0038_v1" \
  --selection-list "$gate_work/selection.txt" \
  --output-tar "$gate_work/model_ready_views.tar" \
  --manifest "$gate_work/input_manifest.json"

"$python_bin" scripts/run_audio_leakage_gate_v2_2.py score-tar \
  --candidate-tar "$gate_work/model_ready_views.tar" \
  --benchmark-records /root/autodl-tmp/dataset/soulx_duplug_eval/reports/benchmark_leakage_denylist_v2/benchmark_identity_manifest.jsonl \
  --frozen-config "$repo/configs/benchmark_audio_leakage_frozen_v2_2.json" \
  --output-dir "$gate_output"

"$python_bin" scripts/audit_model_ready_gate_d_closure.py \
  --model-ready-dir "$model_ready" \
  --model-ready-validation "$validation" \
  --target-audio-manifest "$data/processed/target_audio_edu0038_v1/audio_manifest.jsonl" \
  --selection-list "$gate_work/selection.txt" \
  --candidate-tar "$gate_work/model_ready_views.tar" \
  --gate-output-dir "$gate_output" \
  --benchmark-records /root/autodl-tmp/dataset/soulx_duplug_eval/reports/benchmark_leakage_denylist_v2/benchmark_identity_manifest.jsonl \
  --frozen-config "$repo/configs/benchmark_audio_leakage_frozen_v2_2.json" \
  --expected-view-count 1008 \
  --expected-source-count 500 \
  --output "$gate_closure"

"$python_bin" - "$decision" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text())
value["status"] = "complete_all_automatic_zero_quarantine_gates_passed"
path.write_text(json.dumps(value, indent=2) + "\n")
PY
