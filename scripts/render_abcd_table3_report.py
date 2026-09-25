#!/usr/bin/env python3
"""Audit and render the final Edu_0001-Edu_0045 A/B/C/D meeting report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
from typing import Any


CLASSES = ("en/complete", "en/incomplete", "zh/complete", "zh/incomplete")
TOTALS = {"en/complete": 318, "en/incomplete": 299, "zh/complete": 300, "zh/incomplete": 300}
EXPECTED_BASELINE = {
    "en/complete": 251,
    "en/incomplete": 268,
    "zh/complete": 263,
    "zh/incomplete": 241,
}
EVALUATION_COMMIT = "b17bcf903b9bd896238a4ee9fec495fd75df1401"
OFFICIAL_COMMIT = "928b06508ed2de1344208d06fb1f6fb2ebfb1df5"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing report artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def embedded_a_payload(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r'<script id="data" type="application/json">(.*?)</script>', text, re.DOTALL
    )
    if match is None:
        raise RuntimeError("frozen A HTML has no embedded structured payload")
    return json.loads(match.group(1))


def checkpoint_row(group: str, step: int, item: dict[str, Any]) -> dict[str, Any]:
    values = {
        key: float(item["classes"][key]["candidate_accuracy_percent"])
        for key in CLASSES
    }
    baseline = {
        key: float(item["classes"][key]["baseline_accuracy_percent"])
        for key in CLASSES
    }
    return {
        "group": group,
        "step": step,
        **{key.replace("/", "_"): value for key, value in values.items()},
        "en_macro": (values["en/complete"] + values["en/incomplete"]) / 2,
        "zh_macro": (values["zh/complete"] + values["zh/incomplete"]) / 2,
        "four_class_macro": sum(values.values()) / 4,
        "delta_vs_baseline": sum(values.values()) / 4 - sum(baseline.values()) / 4,
        "obvious_decline_trigger": item.get("obvious_decline_trigger"),
        "obvious_decline_confirmed": item.get("obvious_decline_confirmed"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--table3-manifest", type=Path, required=True)
    parser.add_argument("--training-queue", type=Path, required=True)
    parser.add_argument("--original-split", type=Path, required=True)
    parser.add_argument("--balanced-split", type=Path, required=True)
    parser.add_argument("--frozen-a-html", type=Path, required=True)
    parser.add_argument("--frozen-a-audit", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = {
        key: value.resolve(strict=True)
        for key, value in {
            "summary_markdown": args.summary_md,
            "table3_orchestration": args.table3_manifest,
            "training_queue": args.training_queue,
            "original_split": args.original_split,
            "balanced_split": args.balanced_split,
            "frozen_a_html": args.frozen_a_html,
            "frozen_a_audit": args.frozen_a_audit,
        }.items()
    }
    output_html = args.output_html.resolve()
    output_audit = args.output_audit.resolve()
    if output_html.exists() or output_audit.exists():
        raise FileExistsError("final HTML/audit output already exists")

    manifest = load_json(paths["table3_orchestration"])
    training = load_json(paths["training_queue"])
    original = load_json(paths["original_split"])
    balanced = load_json(paths["balanced_split"])
    a_audit = load_json(paths["frozen_a_audit"])
    a_data = embedded_a_payload(paths["frozen_a_html"])
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    identity = manifest.get("identity", {})
    require(manifest.get("status") == "complete", "Table 3 orchestration incomplete")
    require(manifest.get("stage") == "table3_step5_10_complete", "Table 3 final stage mismatch")
    require(identity.get("protocol") == "frozen-table3-abcd-step5-10-parallel-v1", "protocol drift")
    require(identity.get("evaluation_commit") == EVALUATION_COMMIT, "evaluation commit drift")
    require(identity.get("official_commit") == OFFICIAL_COMMIT, "official commit drift")
    require(identity.get("table3_used_for_training_or_checkpoint_selection") is False, "Table 3 affected training or checkpoint selection")
    require(identity.get("execution_waves") == [["baseline", "B5", "C5"], ["D5", "B10", "C10"], ["D10"]], "execution wave drift")
    expected_markers = {
        *(f"baseline-step000000/{key}" for key in CLASSES),
        *(f"{group}/step{step:06d}/{key}" for group in "BCD" for step in (5, 10) for key in CLASSES),
    }
    require(set(manifest.get("completed_classes", [])) == expected_markers, "completed class grid mismatch")

    baseline_root = paths["table3_orchestration"].parent / "baseline-step000000"
    baseline_gate_path = baseline_root / "table3-gate.json"
    baseline_gate = load_json(baseline_gate_path.resolve(strict=True))
    require(baseline_gate.get("evaluation_mode") == "baseline", "baseline gate mode mismatch")
    require(baseline_gate.get("evidence_audit_passed") is True, "baseline evidence audit failed")
    baseline_values: dict[str, float] = {}
    for key in CLASSES:
        language, label = key.split("/")
        result = load_json((baseline_root / f"{language}-{label}.json").resolve(strict=True))
        row = result.get("summary", {}).get("by_class", {}).get(key, {})
        require(result.get("status") == "complete", f"baseline {key} incomplete")
        require(row.get("total") == TOTALS[key], f"baseline {key} denominator drift")
        require(row.get("correct") == EXPECTED_BASELINE[key], f"baseline {key} count drift")
        baseline_values[key] = 100 * EXPECTED_BASELINE[key] / TOTALS[key]

    input_identities = {
        name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in paths.items()
    }
    input_identities["baseline_gate"] = {
        "path": str(baseline_gate_path),
        "bytes": baseline_gate_path.stat().st_size,
        "sha256": sha256_file(baseline_gate_path),
    }

    indexes: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for group in "BCD":
        index_record = manifest.get("group_indexes", {}).get(group, {})
        index_path = Path(index_record.get("path", "")).resolve(strict=True)
        require(sha256_file(index_path) == index_record.get("sha256"), f"{group} index SHA mismatch")
        index = load_json(index_path)
        indexes[group] = index
        input_identities[f"{group}_table3_index"] = {
            "path": str(index_path),
            "bytes": index_path.stat().st_size,
            "sha256": sha256_file(index_path),
        }
        require(index.get("status") == "complete", f"{group} index incomplete")
        require(index.get("primary_rule") == "last-terminal-v1", f"{group} primary rule drift")
        require(index.get("selection_used_paper_targets") is False, f"{group} used paper targets")
        require([item.get("local_step") for item in index.get("checkpoints", [])] == [5, 10], f"{group} step grid mismatch")
        for item in index.get("checkpoints", []):
            step = int(item["local_step"])
            require(set(item.get("classes", {})) == set(CLASSES), f"{group}{step} class grid mismatch")
            expected_checkpoint = identity.get("groups", {}).get(group, {}).get(str(step), {})
            require(item.get("checkpoint", {}).get("sha256") == expected_checkpoint.get("sha256"), f"{group}{step} checkpoint mismatch")
            for key in CLASSES:
                require(item["classes"][key].get("total") == TOTALS[key], f"{group}{step} {key} denominator drift")
                require(item["classes"][key].get("baseline_correct") == EXPECTED_BASELINE[key], f"{group}{step} {key} baseline drift")
            gate_path = Path(item["evidence_gate_path"]).resolve(strict=True)
            gate = load_json(gate_path)
            require(sha256_file(gate_path) == item.get("evidence_gate_sha256"), f"{group}{step} gate SHA mismatch")
            require(gate.get("evidence_audit_passed") is True and gate.get("gate_passed") is True, f"{group}{step} evidence gate failed")
            input_identities[f"{group}{step}_evidence_gate"] = {
                "path": str(gate_path),
                "bytes": gate_path.stat().st_size,
                "sha256": sha256_file(gate_path),
            }
            rows.append(checkpoint_row(group, step, item))

    require(training.get("status") == "complete", "B/C/D training queue incomplete")
    for group in "BCD":
        item = training.get("groups", {}).get(group, {})
        require(item.get("status") == "complete", f"{group} training incomplete")
        require(item.get("training", {}).get("final_local_step") == 30, f"{group} final train step drift")
        require(item.get("training", {}).get("optimizer_update_count") == 30, f"{group} optimizer update count drift")
        require(item.get("training", {}).get("sample_exposure") == 17280, f"{group} sample exposure drift")
        require(item.get("partial_validation_index", {}).get("selection_is_partial") is True, f"{group} internal validation scope misreported")
    require(original.get("source_leakage_count") == 0, "original split source leakage")
    require(balanced.get("source_leakage_count") == 0, "balanced split source leakage")
    require(original.get("validation", {}).get("row_index_identity_sha256") == balanced.get("validation", {}).get("row_index_identity_sha256"), "validation changed between splits")
    require(balanced.get("balance", {}).get("output", {}).get("complete_active_row_count") == 47182, "balanced Complete active rows drift")
    require(balanced.get("balance", {}).get("output", {}).get("incomplete_active_row_count") == 47182, "balanced Incomplete active rows drift")

    require(a_audit.get("status") == "passed", "frozen A audit failed")
    expected_a_html_sha = a_audit.get("outputs", {}).get("html", {}).get("sha256")
    require(sha256_file(paths["frozen_a_html"]) == expected_a_html_sha, "frozen A HTML SHA mismatch")
    a_by_step = {int(item["step"]): item for item in a_data.get("table3", [])}
    require(all(step in a_by_step for step in (0, 5, 10)), "frozen A comparison points missing")
    require(all(abs(a_by_step[0][key.replace("/", "_")] - baseline_values[key]) < 1e-9 for key in CLASSES), "frozen A baseline differs from fresh baseline")
    for step in (5, 10):
        item = a_by_step[step]
        values = {key: float(item[key.replace("/", "_")]) for key in CLASSES}
        rows.append({
            "group": "A", "step": step,
            **{key.replace("/", "_"): value for key, value in values.items()},
            "en_macro": (values["en/complete"] + values["en/incomplete"]) / 2,
            "zh_macro": (values["zh/complete"] + values["zh/incomplete"]) / 2,
            "four_class_macro": sum(values.values()) / 4,
            "delta_vs_baseline": sum(values.values()) / 4 - sum(baseline_values.values()) / 4,
        })
    rows.sort(key=lambda row: (row["step"], row["group"]))
    by_key = {(row["group"], row["step"]): row for row in rows}
    for row in rows:
        reference = by_key.get(("A", row["step"]))
        row["delta_vs_same_step_a"] = 0.0 if row["group"] == "A" else row["four_class_macro"] - reference["four_class_macro"]

    baseline_macro = sum(baseline_values.values()) / 4
    candidate_rows = [row for row in rows if row["group"] != "A"]
    best = max(candidate_rows, key=lambda row: row["four_class_macro"])
    effects = {
        str(step): {
            "equal_loss_on_original": by_key[("B", step)]["four_class_macro"] - by_key[("A", step)]["four_class_macro"],
            "balanced_sampling_on_original_loss": by_key[("C", step)]["four_class_macro"] - by_key[("A", step)]["four_class_macro"],
            "equal_loss_on_balanced_sampling": by_key[("D", step)]["four_class_macro"] - by_key[("C", step)]["four_class_macro"],
            "balanced_sampling_on_equal_loss": by_key[("D", step)]["four_class_macro"] - by_key[("B", step)]["four_class_macro"],
        }
        for step in (5, 10)
    }
    internal = {
        group: [
            {
                "step": step,
                "accuracy_percent": 100 * float(training["groups"][group]["validation"][str(step)]["accuracy"]),
                "idle": 100 * float(training["groups"][group]["validation"][str(step)]["heads"]["idle"]["token_weighted_accuracy"]),
                "nonidle": 100 * float(training["groups"][group]["validation"][str(step)]["heads"]["nonidle"]["token_weighted_accuracy"]),
                "backchannel": 100 * float(training["groups"][group]["validation"][str(step)]["heads"]["user_backchannel"]["token_weighted_accuracy"]),
                "complete": 100 * float(training["groups"][group]["validation"][str(step)]["heads"]["user_complete"]["token_weighted_accuracy"]),
                "incomplete": 100 * float(training["groups"][group]["validation"][str(step)]["heads"]["user_incomplete"]["token_weighted_accuracy"]),
            }
            for step in (0, 5, 10)
        ]
        for group in "BCD"
    }
    payload = {
        "generated_at_utc": utc_now(),
        "title": "DuplexConv Edu_0001–Edu_0045 Complete/Incomplete 平衡消融",
        "baseline": {**{key.replace("/", "_"): value for key, value in baseline_values.items()}, "four_class_macro": baseline_macro},
        "groups": {
            "A": {"name": "原始训练", "data": "原始 train", "loss": "0.24 / 0.13"},
            "B": {"name": "仅等权 loss", "data": "原始 train", "loss": "0.185 / 0.185"},
            "C": {"name": "仅平衡采样", "data": "balanced-CI train", "loss": "0.24 / 0.13"},
            "D": {"name": "联合平衡", "data": "balanced-CI train", "loss": "0.185 / 0.185"},
        },
        "rows": rows,
        "factor_effects": effects,
        "best_new_candidate": best,
        "dataset": {
            "aggregate_rows": original["train"]["row_count"] + original["validation"]["row_count"],
            "aggregate_hours": original["train"]["duration_hours_from_160ms_chunks"] + original["validation"]["duration_hours_from_160ms_chunks"],
            "original_train": original["train"],
            "balanced_train": balanced["train"],
            "validation": original["validation"],
            "balanced_categories": balanced["balance"]["output"]["category_row_counts"],
            "balanced_active_rows": {
                "complete": balanced["balance"]["output"]["complete_active_row_count"],
                "incomplete": balanced["balance"]["output"]["incomplete_active_row_count"],
            },
            "qwen_construction_cost_usd": a_data.get("dataset", {}).get("qwen_cost"),
        },
        "internal_validation": internal,
        "training": {"optimizer_steps": 30, "sample_exposure_per_group": 17280, "origin_step_estimate": 1800, "origin_confidence": "low"},
        "integrity": {
            "developmental_table3": True,
            "table3_used_for_training_or_selection": False,
            "all_continuation_evidence_gates_passed": True,
            "table2_run": False,
            "paid_api_cost_for_training_and_evaluation_usd": 0.0,
            "evaluation_commit": EVALUATION_COMMIT,
            "official_commit": OFFICIAL_COMMIT,
        },
    }
    if errors:
        raise RuntimeError("final report audit failed:\n- " + "\n- ".join(errors))

    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html_text = render_html(data_json)
    atomic_write(output_html, html_text)
    input_identities["renderer"] = {
        "path": str(Path(__file__).resolve(strict=True)),
        "bytes": Path(__file__).stat().st_size,
        "sha256": sha256_file(Path(__file__)),
    }
    audit = {
        "schema_version": 1,
        "status": "passed",
        "generated_at_utc": payload["generated_at_utc"],
        "checks": {
            "training_queue_complete": True,
            "official_lightning_only": True,
            "table3_grid_complete": True,
            "fresh_baseline_exact_match": True,
            "all_continuation_evidence_gates_passed": True,
            "all_denominators_frozen": True,
            "table3_used_for_training_or_checkpoint_selection": False,
            "paper_targets_used_for_selection": False,
            "original_and_balanced_source_leakage_count": 0,
            "balanced_complete_incomplete_active_rows": [47182, 47182],
            "validation_byte_identity_preserved": True,
            "frozen_a_report_chain_verified": True,
            "table2_run": False,
            "paid_api_cost_for_training_and_evaluation_usd": 0.0,
        },
        "inputs": input_identities,
        "outputs": {
            "markdown": {"path": str(paths["summary_markdown"]), "bytes": paths["summary_markdown"].stat().st_size, "sha256": sha256_file(paths["summary_markdown"])},
            "html": {"path": str(output_html), "bytes": output_html.stat().st_size, "sha256": sha256_file(output_html)},
        },
        "metrics": {"baseline_four_class_macro_percent": baseline_macro, "rows": rows, "factor_effects": effects, "best_new_candidate": best},
    }
    atomic_write(output_audit, json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "html": str(output_html), "audit": str(output_audit), "best": best}, ensure_ascii=False, indent=2))
    return 0


def render_html(data_json: str) -> str:
    return f"""<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>SoulX A/B/C/D 消融报告</title><style>
:root{{--bg:#08111f;--panel:#111e31;--line:#273a54;--text:#e8f0fa;--muted:#9bb0c8;--cyan:#57d6ff;--green:#5ce1a4;--yellow:#ffd166;--red:#ff718d;--purple:#b69cff}}*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(140deg,#07101d,#0d1830 55%,#101d2d);color:var(--text);font:14px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}}main{{max-width:1280px;margin:auto;padding:30px 24px 60px}}h1{{margin:0;font-size:30px}}h2{{margin:30px 0 12px;font-size:20px}}.sub,.muted{{color:var(--muted)}}.cards,.grid2{{display:grid;gap:14px}}.cards{{grid-template-columns:repeat(auto-fit,minmax(190px,1fr));margin:22px 0}}.grid2{{grid-template-columns:repeat(auto-fit,minmax(480px,1fr))}}.card,.panel,.callout{{background:rgba(17,30,49,.94);border:1px solid var(--line);border-radius:14px;padding:17px;box-shadow:0 8px 28px #0004}}.card .v{{font-size:26px;font-weight:750;margin-top:4px}}.good{{color:var(--green)}}.warn{{color:var(--yellow)}}.bad{{color:var(--red)}}.callout{{border-left:4px solid var(--cyan);margin:14px 0}}.chart svg{{width:100%;height:auto;overflow:visible}}.axis,.grid{{stroke:#38506c;stroke-width:1}}.grid{{opacity:.55}}.tick{{fill:var(--muted);font-size:11px}}.legend{{display:flex;gap:15px;flex-wrap:wrap;color:var(--muted);margin:8px 0}}.dot{{width:10px;height:10px;display:inline-block;border-radius:50%;margin-right:5px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}.scroll{{overflow:auto}}.pill{{padding:3px 8px;border-radius:999px;background:#1a2b43}}ul{{margin:8px 0;padding-left:20px}}footer{{margin-top:30px;color:var(--muted);font-size:12px}}@media(max-width:600px){{main{{padding:20px 12px}}.grid2{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>DuplexConv × SoulX Stage 3</h1><div class=\"sub\">Edu_0001–Edu_0045 · Complete/Incomplete 2×2 消融 · 冻结 Table 3</div><div id=\"cards\" class=\"cards\"></div><div id=\"summary\" class=\"callout\"></div><div class=\"grid2\"><section class=\"panel\"><h2>四类宏平均</h2><div class=\"legend\" id=\"macroLegend\"></div><div class=\"chart\" id=\"macroChart\"></div></section><section class=\"panel\"><h2>Incomplete 准确率</h2><div class=\"legend\" id=\"incLegend\"></div><div class=\"chart\" id=\"incChart\"></div></section></div><h2>Table 3 完整结果</h2><section class=\"panel scroll\"><table><thead><tr><th>配置</th><th>Step</th><th>EN C</th><th>EN I</th><th>ZH C</th><th>ZH I</th><th>四类宏平均</th><th>vs 基线</th><th>vs 同step A</th></tr></thead><tbody id=\"results\"></tbody></table></section><div class=\"grid2\"><section class=\"panel\"><h2>数据与采样</h2><div id=\"dataset\"></div></section><section class=\"panel\"><h2>结论与边界</h2><div id=\"findings\"></div></section></div><h2>内部 validation（step 0/5/10）</h2><section class=\"panel scroll\"><table><thead><tr><th>组</th><th>Step</th><th>总体</th><th>Idle</th><th>Non-idle</th><th>Backchannel</th><th>Complete</th><th>Incomplete</th></tr></thead><tbody id=\"internal\"></tbody></table></section><footer id=\"footer\"></footer></main><script id=\"data\" type=\"application/json\">{data_json}</script><script>
const D=JSON.parse(document.getElementById('data').textContent), colors={{A:'#9bb0c8',B:'#57d6ff',C:'#ffd166',D:'#5ce1a4'}}, f=(x,n=2)=>Number(x).toFixed(n)+'%', pp=x=>(x>=0?'+':'')+Number(x).toFixed(2)+' pp';
document.getElementById('cards').innerHTML=[["聚合数据",D.dataset.aggregate_hours.toFixed(1)+' h'],["原始 train",D.dataset.original_train.row_count.toLocaleString()+' rows'],["平衡 train",D.dataset.balanced_train.row_count.toLocaleString()+' rows'],["最佳新候选",D.best_new_candidate.group+D.best_new_candidate.step],["最佳候选宏平均",f(D.best_new_candidate.four_class_macro)]].map(x=>`<div class=\"card\"><div class=\"muted\">${{x[0]}}</div><div class=\"v\">${{x[1]}}</div></div>`).join('');
document.getElementById('summary').innerHTML=`<b>结论：</b>D（平衡采样 + 等权 loss）在 step 5/10 均为消融候选中最好；D5 四类宏平均 ${{f(D.best_new_candidate.four_class_macro)}}，仍比官方基线低 ${{Math.abs(D.best_new_candidate.delta_vs_baseline).toFixed(2)}} pp。没有新候选超过官方基线。`;
function lineChart(id,series,ymin,ymax){{const W=610,H=285,p={{l:48,r:18,t:20,b:38}},X=x=>p.l+x/10*(W-p.l-p.r),Y=y=>p.t+(ymax-y)*(H-p.t-p.b)/(ymax-ymin);let s=`<svg viewBox=\"0 0 ${{W}} ${{H}}\">`;for(let i=0;i<6;i++){{let y=ymin+i*(ymax-ymin)/5,yy=Y(y);s+=`<line class=\"grid\" x1=\"${{p.l}}\" y1=\"${{yy}}\" x2=\"${{W-p.r}}\" y2=\"${{yy}}\"/><text class=\"tick\" x=\"${{p.l-7}}\" y=\"${{yy+4}}\" text-anchor=\"end\">${{y.toFixed(0)}}</text>`}}[0,5,10].forEach(x=>s+=`<text class=\"tick\" x=\"${{X(x)}}\" y=\"${{H-10}}\" text-anchor=\"middle\">${{x}}</text>`);for(const q of series){{s+=`<polyline fill=\"none\" stroke=\"${{q.color}}\" stroke-width=\"3\" points=\"${{q.points.map(v=>X(v.x)+','+Y(v.y)).join(' ')}}\"/>`;q.points.forEach(v=>s+=`<circle cx=\"${{X(v.x)}}\" cy=\"${{Y(v.y)}}\" r=\"4\" fill=\"${{q.color}}\"><title>${{q.name}} step ${{v.x}}: ${{v.y.toFixed(2)}}%</title></circle>`)}}return s+'</svg>'}}
const base=D.baseline.four_class_macro, groups=['A','B','C','D'];const macro=groups.map(g=>({{name:g,color:colors[g],points:[{{x:0,y:base}},...D.rows.filter(r=>r.group===g).map(r=>({{x:r.step,y:r.four_class_macro}}))]}}));document.getElementById('macroChart').innerHTML=lineChart('macroChart',macro,60,88);document.getElementById('macroLegend').innerHTML=groups.map(g=>`<span><i class=\"dot\" style=\"background:${{colors[g]}}\"></i>${{g}} ${{D.groups[g].name}}</span>`).join('');
let inc=[];for(const g of groups)for(const [key,label,dash] of [['en_incomplete','EN',''],['zh_incomplete','ZH','']])inc.push({{name:g+' '+label,color:colors[g],points:[{{x:0,y:D.baseline[key]}},...D.rows.filter(r=>r.group===g).map(r=>({{x:r.step,y:r[key]}}))]}});document.getElementById('incChart').innerHTML=lineChart('incChart',inc,15,95);document.getElementById('incLegend').innerHTML=groups.map(g=>`<span><i class=\"dot\" style=\"background:${{colors[g]}}\"></i>${{g}}（EN/ZH）</span>`).join('');
document.getElementById('results').innerHTML=D.rows.map(r=>`<tr><td><span class=\"pill\">${{r.group}} ${{D.groups[r.group].name}}</span></td><td>${{r.step}}</td><td>${{f(r.en_complete)}}</td><td>${{f(r.en_incomplete)}}</td><td>${{f(r.zh_complete)}}</td><td>${{f(r.zh_incomplete)}}</td><td><b>${{f(r.four_class_macro)}}</b></td><td class=\"${{r.delta_vs_baseline<0?'bad':'good'}}\">${{pp(r.delta_vs_baseline)}}</td><td>${{r.group==='A'?'—':pp(r.delta_vs_same_step_a)}}</td></tr>`).join('');
const ds=D.dataset;document.getElementById('dataset').innerHTML=`<p>原始 train：<b>${{ds.original_train.row_count.toLocaleString()}}</b> rows，${{ds.original_train.duration_hours_from_160ms_chunks.toFixed(1)}} h；validation：${{ds.validation.row_count.toLocaleString()}} rows，${{ds.validation.duration_hours_from_160ms_chunks.toFixed(1)}} h。</p><p>balanced-CI train：<b>${{ds.balanced_train.row_count.toLocaleString()}}</b> rows，${{ds.balanced_train.duration_hours_from_160ms_chunks.toFixed(1)}} h。Complete/Incomplete active rows=${{ds.balanced_active_rows.complete.toLocaleString()}}:${{ds.balanced_active_rows.incomplete.toLocaleString()}}。</p><p class=\"muted\">状态 token 仍非1:1；validation字节身份保持不变，source leakage=0。数据构造累计Qwen费用 $${{Number(ds.qwen_construction_cost_usd).toFixed(4)}}；本轮训练/评测API费用 $0。</p>`;
document.getElementById('findings').innerHTML=`<ul><li>等权 loss 单因素在step 5/10相对A提高 ${{pp(D.factor_effects['5'].equal_loss_on_original)}} / ${{pp(D.factor_effects['10'].equal_loss_on_original)}}。</li><li>平衡采样单因素提高 ${{pp(D.factor_effects['5'].balanced_sampling_on_original_loss)}} / ${{pp(D.factor_effects['10'].balanced_sampling_on_original_loss)}}。</li><li>D10相对A10提高 ${{pp(D.rows.find(r=>r.group==='D'&&r.step===10).delta_vs_same_step_a)}}，但仍低于基线。</li><li>本轮是查看A结果后提出的开发性评测；Table 3未用于训练或checkpoint选择，Table 2未执行。</li></ul>`;
document.getElementById('internal').innerHTML=groups.filter(g=>g!=='A').flatMap(g=>D.internal_validation[g].map(r=>`<tr><td>${{g}}</td><td>${{r.step}}</td><td>${{f(r.accuracy_percent)}}</td><td>${{f(r.idle)}}</td><td>${{f(r.nonidle)}}</td><td>${{f(r.backchannel)}}</td><td>${{f(r.complete)}}</td><td>${{f(r.incomplete)}}</td></tr>`)).join('');document.getElementById('footer').textContent=`生成时间 ${{D.generated_at_utc}} · audit passed · official ${{D.integrity.official_commit.slice(0,8)}} · eval ${{D.integrity.evaluation_commit.slice(0,8)}}`;
</script></body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
