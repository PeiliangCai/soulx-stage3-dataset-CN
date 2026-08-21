#!/usr/bin/env python3
"""Render the meeting Markdown and self-contained HTML continuation report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from duplexconv_stage3.continual_training import atomic_json_write, sha256_file, utc_now


CLASS_KEYS = ("en/complete", "en/incomplete", "zh/complete", "zh/incomplete")
BASELINE_FILES = {
    "en/complete": "en-complete.json",
    "en/incomplete": "en-incomplete.json",
    "zh/complete": "zh-complete.json",
    "zh/incomplete": "zh-incomplete.json",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def build_report_data(args) -> dict[str, Any]:
    formal_root = args.formal_run.resolve(strict=True)
    manifest_path = formal_root / "run_manifest.json"
    manifest = load_json(manifest_path)
    split = load_json(args.split_manifest.resolve(strict=True))
    lr_selection = load_json(args.lr_selection.resolve(strict=True))
    validation = load_jsonl(formal_root / "validation_metrics.jsonl")
    training = load_jsonl(formal_root / "training_steps.jsonl")
    table3 = (
        load_json(args.table3_index.resolve(strict=True))
        if args.table3_index and args.table3_index.exists()
        else {"status": "pending", "checkpoints": []}
    )
    baseline_root = args.table3_baseline.resolve(strict=True)
    baseline_classes = {}
    for key, filename in BASELINE_FILES.items():
        payload = load_json(baseline_root / filename)
        summary = payload["summary"]["by_class"][key]
        baseline_classes[key] = {
            "correct": summary["correct"],
            "total": summary["total"],
            "accuracy_percent": 100 * summary["accuracy"],
        }
    baseline_languages = {
        language: 0.5
        * (
            baseline_classes[f"{language}/complete"]["accuracy_percent"]
            + baseline_classes[f"{language}/incomplete"]["accuracy_percent"]
        )
        for language in ("en", "zh")
    }
    expected_steps = manifest["checkpoint_steps"]
    table3_by_step = {item["local_step"]: item for item in table3.get("checkpoints", [])}
    checkpoint_rows = []
    for step in expected_steps:
        if step == 0:
            checkpoint_rows.append(
                {
                    "step": 0,
                    "estimated_total": manifest["origin_step_estimate"],
                    "lr": None,
                    "en_complete": baseline_classes["en/complete"]["accuracy_percent"],
                    "en_incomplete": baseline_classes["en/incomplete"]["accuracy_percent"],
                    "en_macro": baseline_languages["en"],
                    "zh_complete": baseline_classes["zh/complete"]["accuracy_percent"],
                    "zh_incomplete": baseline_classes["zh/incomplete"]["accuracy_percent"],
                    "zh_macro": baseline_languages["zh"],
                    "en_delta": 0.0,
                    "zh_delta": 0.0,
                    "status": "官方发布权重基线",
                    "almost_unchanged": True,
                }
            )
            continue
        row = table3_by_step.get(step)
        validation_row = next((item for item in validation if item["local_step"] == step), None)
        if row is None:
            checkpoint_rows.append(
                {
                    "step": step,
                    "estimated_total": manifest["origin_step_estimate"] + step,
                    "lr": validation_row["learning_rate"] if validation_row else None,
                    "status": "Table 3 待评测",
                }
            )
            continue
        checkpoint_rows.append(
            {
                "step": step,
                "estimated_total": row["estimated_total_optimizer_step"],
                "lr": validation_row["learning_rate"] if validation_row else None,
                "en_complete": row["classes"]["en/complete"]["candidate_accuracy_percent"],
                "en_incomplete": row["classes"]["en/incomplete"]["candidate_accuracy_percent"],
                "en_macro": row["languages"]["en"]["candidate_macro_accuracy_percent"],
                "zh_complete": row["classes"]["zh/complete"]["candidate_accuracy_percent"],
                "zh_incomplete": row["classes"]["zh/incomplete"]["candidate_accuracy_percent"],
                "zh_macro": row["languages"]["zh"]["candidate_macro_accuracy_percent"],
                "en_delta": row["languages"]["en"]["delta_percentage_points"],
                "zh_delta": row["languages"]["zh"]["delta_percentage_points"],
                "status": (
                    "明显下降（已确认）"
                    if row["obvious_decline_confirmed"]
                    else "基本不变"
                    if row["almost_unchanged"]
                    else "灰区/下降触发待确认"
                ),
                "almost_unchanged": row["almost_unchanged"],
                "obvious_decline_trigger": row["obvious_decline_trigger"],
                "obvious_decline_confirmed": row["obvious_decline_confirmed"],
                "zh_subgroups": row["zh_subgroups"],
            }
        )
    completed_rows = [row for row in checkpoint_rows if row.get("en_macro") is not None]
    stable_rows = [row for row in completed_rows if row.get("almost_unchanged")]
    eligible_best = [row for row in completed_rows if row.get("en_delta", 0) >= -1.0]
    best = max(eligible_best, key=lambda row: (row.get("zh_macro", -1), -row["step"])) if eligible_best else None
    first_decline = next(
        (row for row in completed_rows if row.get("obvious_decline_confirmed")), None
    )
    return {
        "generated_at_utc": utc_now(),
        "formal_root": str(formal_root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest": manifest,
        "split": split,
        "lr_selection": lr_selection,
        "training": training,
        "validation": validation,
        "table3": table3,
        "baseline_classes": baseline_classes,
        "baseline_languages": baseline_languages,
        "checkpoint_rows": checkpoint_rows,
        "summary": {
            "last_stable_step": stable_rows[-1]["step"] if stable_rows else None,
            "best_step": best["step"] if best else None,
            "best_zh_macro": best.get("zh_macro") if best else None,
            "first_confirmed_decline_step": first_decline["step"] if first_decline else None,
            "table3_completed_count": len(completed_rows) - 1,
            "table3_expected_continuation_count": len(expected_steps) - 1,
        },
    }


def render_markdown(data: dict[str, Any]) -> str:
    manifest = data["manifest"]
    split = data["split"]
    selection = data["lr_selection"]
    summary = data["summary"]
    status = "已完成" if manifest["status"] == "complete" and summary["table3_completed_count"] == summary["table3_expected_continuation_count"] else "进行中"
    lines = [
        "# SoulX-Duplug Stage 3 中文续训练实验与性能评估",
        "",
        f"更新时间：{data['generated_at_utc']}  ",
        "用途：课题组会议/导师汇报  ",
        f"实验状态：**{status}**",
        "",
        "## 1. 结论摘要",
        "",
        f"- 正式训练状态：`{manifest['status']}`；已记录 optimizer step：{manifest.get('step_records', 0)}/{manifest['max_steps']}。",
        f"- Table 3 已完成续训练 checkpoint：{summary['table3_completed_count']}/{summary['table3_expected_continuation_count']}。",
        f"- 验证集选择的 peak LR：`{selection['selected_peak_lr']:.8g}`；选择过程未读取 Table 3。",
        f"- 最后一个“基本不变”点：{summary['last_stable_step'] if summary['last_stable_step'] is not None else 'TBD'}。",
        f"- 满足 EN 宏平均下降不超过 1pp 时，ZH 宏平均最高点：step {summary['best_step'] if summary['best_step'] is not None else 'TBD'}（{fmt(summary['best_zh_macro'])}%）。",
        f"- 首个已确认明显下降点：{summary['first_confirmed_decline_step'] if summary['first_confirmed_decline_step'] is not None else '尚未观察到/TBD'}。",
        "",
        "> 未完成的 checkpoint 保持为 TBD；报告生成器不会用预期值、论文值或相邻 step 插值填充实验结果。",
        "",
        "## 2. 起始状态与“续训练”定义",
        "",
        f"- 官方 Bilingual 权重 SHA-256：`{manifest['base_checkpoint']['sha256']}`。",
        "- 发布权重不含 global_step、AdamW、scheduler、AMP scaler，因此这是从模型参数继续微调，不是 optimizer 的精确 resume。",
        f"- 公开 Stage 3 配置 `total_steps=1800`，故将起始 step 估计为 {manifest['origin_step_estimate']}（低置信度），不能表述为已证实的官方 checkpoint step。",
        f"- 本地 batch=1、梯度累积={manifest['gradient_accumulation']}，有效 batch={manifest['local_effective_batch']}；官方参考全局有效 batch={manifest['official_reference_global_effective_batch']}。",
        f"- 每个本地 step 对应约 {manifest['official_sample_equivalent_step_per_local_step']:.3f} 个官方 sample-equivalent step；报告同时保留本地 optimizer step、累计 micro-batch 与 epoch-equivalent。",
        f"- 可训练参数：{manifest['trainable_parameter_count']:,}；总参数：{manifest['total_parameter_count']:,}。",
        "",
        "## 3. 数据集与会话级切分",
        "",
        "训练源为 DuplexConv `Edu_0018`：500 个同步多轨会话（495 个双声道、5 个三声道），展开为 1,005 个 target-speaker views。三声道不丢弃：每次只输入一个目标声道，其他声道聚合为关系证据，不把多路 audio token 放入同一 sequence。",
        "",
        "状态映射：官方 complete/incomplete/backchannel 原样映射；11 个 WAIT 映射为 complete；1,599 个缺失状态由固定 `qwen3-235b-a22b-instruct-2507` 补标。Paraformer 只用于中文伪转录/时间戳构造，不参与模型训练。",
        "",
        "| Split | 源会话 | target views | rows | 160ms chunks | 估算时长(h) | Qwen 补标事件 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Train | {split['train']['source_conversation_count']} | {split['train']['target_view_count']} | {split['train']['row_count']} | {split['train']['chunk_count']} | {split['train']['duration_hours_from_160ms_chunks']:.3f} | {split['train']['unique_qwen_labeled_event_count']} |",
        f"| Validation | {split['validation']['source_conversation_count']} | {split['validation']['target_view_count']} | {split['validation']['row_count']} | {split['validation']['chunk_count']} | {split['validation']['duration_hours_from_160ms_chunks']:.3f} | {split['validation']['unique_qwen_labeled_event_count']} |",
        "",
        f"切分协议：`{split['profile']}`，seed={split['seed']}，source leakage={split['source_leakage_count']}，split identity=`{split['split_identity_sha256']}`。同一 WAV 的全部声道视角和窗口只属于一个 split。",
        "",
        "## 4. LR 校准与正式训练配置",
        "",
        "两档校准都从官方权重重新初始化，并使用相同训练顺序和固定验证集；Table 3 不参与 LR 选择。失格规则为任一状态头相对 step 0 下降超过 5pp；最终验证目标差异不超过 1% 时选择较低 LR。",
        "",
        "| Candidate | Peak LR | step 20 validation objective | state macro | 合格 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for candidate in selection["candidates"]:
        lines.append(
            f"| `{candidate['run_id']}` | {candidate['peak_lr']:.8g} | {candidate['final']['token_weighted_objective']:.6f} | {100*candidate['final']['state_macro_accuracy']:.3f}% | {'是' if candidate['eligible'] else '否'} |"
        )
    lines.extend(
        [
            "",
            f"选择原因：{selection['selection_reason']}。正式 LR 采用 5-step 新 AdamW 重热身，并按估计原 step=1800 进行 offset inverse-square-root 衰减。",
            "",
            "## 5. 训练期 validation 变化",
            "",
            "| Local step | 估计总 step | LR | token-weighted objective | state macro ACC | epoch-equivalent |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    training_by_step = {item["local_step"]: item for item in data["training"]}
    for row in data["validation"]:
        train = training_by_step.get(row["local_step"], {})
        lines.append(
            f"| {row['local_step']} | {row['estimated_total_optimizer_step']} | {row['learning_rate']:.8g} | {row['metrics']['token_weighted_objective']:.6f} | {100*row['metrics']['state_macro_accuracy']:.3f}% | {fmt(train.get('epoch_equivalent'))} |"
        )
    lines.extend(
        [
            "",
            "## 6. Table 3 checkpoint sweep",
            "",
            "主规则始终为 `last-terminal-v1`；样本、seed、顺序、推理核心、尾部静音和 Teacher-ASR 固定。每个 checkpoint 四类结果都经过独立证据 gate；论文目标没有传入推理 runner，也不用于选择 checkpoint。",
            "",
            "“基本不变”：EN/ZH macro 各下降不超过 1pp，且任一 class 下降不超过 2pp。“明显下降触发”：任一语言 macro 下降超过 3pp，或任一 class 下降超过 5pp；必须在下一个预注册点仍触发才确认。",
            "",
            "| Local step | 估计总 step | LR | EN C | EN I | EN Macro | ΔEN | ZH C | ZH I | ZH Macro | ΔZH | 判定 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in data["checkpoint_rows"]:
        lines.append(
            "| {step} | {total} | {lr} | {enc} | {eni} | {enm} | {end} | {zhc} | {zhi} | {zhm} | {zhd} | {status} |".format(
                step=row["step"],
                total=row["estimated_total"],
                lr=fmt(row.get("lr"), 8),
                enc=fmt(row.get("en_complete")),
                eni=fmt(row.get("en_incomplete")),
                enm=fmt(row.get("en_macro")),
                end=fmt(row.get("en_delta")),
                zhc=fmt(row.get("zh_complete")),
                zhi=fmt(row.get("zh_incomplete")),
                zhm=fmt(row.get("zh_macro")),
                zhd=fmt(row.get("zh_delta")),
                status=row["status"],
            )
        )
    lines.extend(
        [
            "",
            "中文固定 600 条是发布方完整测试集，不是本项目随机抽样；每个 checkpoint 另报 complete/incomplete × real/synthetic 四个固定子组。差异显著性使用同一样本的 paired bootstrap 95% CI 和 exact McNemar，而不是把两次准确率当独立样本。",
            "",
            "## 7. 可恢复性、资源与异常记录",
            "",
            f"- 训练状态：`{manifest['status']}`；AMP overflow 次数：{len(manifest['amp_overflows'])}。",
            f"- 峰值 CUDA allocated：{manifest.get('cuda_peak_memory_bytes', 0)/1024**3:.3f} GiB。",
            f"- GPU：{manifest['environment']['gpu']}；Python：{manifest['environment']['python']}；Torch：{manifest['environment']['packages']['torch']}；CUDA：{manifest['environment']['torch_cuda']}。",
            "- 每个预注册点保存仅含 118 个可训练 tensor 的评测快照；`latest_resume.pt` 额外保存 AdamW、GradScaler 和确定性数据流 cursor。",
            f"- 正式 run manifest：`{data['manifest_path']}`，SHA-256=`{data['manifest_sha256']}`。",
            "",
            "## 8. 限制与汇报口径",
            "",
            "1. 起始 1800 step 是依据公开配置的低置信度估计，不是官方权重元数据。",
            "2. 本地有效 batch=72，只有官方参考全局有效 batch=576 的 1/8；因此必须同时报告 local optimizer step 和 sample-equivalent step。",
            "3. 当前 Table 3 样本级读出规则是已审计候选协议，数值与论文基本一致，但仍缺作者发布的样本级计分脚本确认。",
            "4. Table 3 是外部 checkpoint 测试，不用于 LR 调参或逐 step 反向选择；Full-Duplex-Bench 系统级复现应只在 step 0、推荐点、首个下降点和最终点执行。",
            "",
        ]
    )
    return "\n".join(lines)


def render_html(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SoulX Stage 3 续训练仪表盘</title>
<style>
:root{{--bg:#08111f;--panel:#101d31;--panel2:#14243b;--text:#eef5ff;--muted:#9bb0ca;--blue:#55a7ff;--cyan:#42d8c2;--orange:#ffb45b;--red:#ff6b7a;--line:#27405f}}
*{{box-sizing:border-box}} body{{margin:0;background:linear-gradient(145deg,#07101d,#0d1930 55%,#07111f);color:var(--text);font:14px/1.55 Inter,system-ui,-apple-system,"Segoe UI",sans-serif}}
.wrap{{max-width:1440px;margin:auto;padding:28px}} h1{{font-size:28px;margin:0 0 4px}} h2{{font-size:18px;margin:0 0 14px}} .sub{{color:var(--muted);margin-bottom:24px}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin-bottom:18px}} .card,.panel{{background:linear-gradient(160deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:14px;box-shadow:0 12px 34px #0004}}
.card{{padding:16px}} .label{{color:var(--muted);font-size:12px}} .value{{font-size:25px;font-weight:750;margin-top:4px}} .panel{{padding:18px;margin-bottom:18px;overflow:auto}} .two{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
table{{width:100%;border-collapse:collapse;white-space:nowrap}} th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right}} th:first-child,td:first-child{{text-align:left}} th{{color:#bed2eb;font-size:12px}} tr:last-child td{{border:0}}
.pill{{display:inline-block;padding:3px 9px;border-radius:999px;background:#24415d;color:#dcecff;font-size:12px}} .ok{{background:#174d47;color:#8df1d7}} .warn{{background:#594324;color:#ffd18e}} .bad{{background:#5a2832;color:#ffacb6}}
svg{{width:100%;height:300px;overflow:visible}} .axis{{stroke:#46617f;stroke-width:1}} .tick{{fill:#8fa8c4;font-size:11px}} .legend{{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted)}} .dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}}
.note{{color:var(--muted)}} code{{color:#a9d4ff}} @media(max-width:900px){{.grid{{grid-template-columns:1fr 1fr}}.two{{grid-template-columns:1fr}}}} @media(max-width:560px){{.wrap{{padding:14px}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main class="wrap">
<h1>SoulX-Duplug Stage 3 中文续训练</h1><div class="sub" id="subtitle"></div><section class="grid" id="cards"></section>
<section class="two"><div class="panel"><h2>训练期 validation</h2><div id="valChart"></div></div><div class="panel"><h2>Table 3 Macro ACC</h2><div id="table3Chart"></div></div></section>
<section class="panel"><h2>预注册 checkpoint 网格</h2><table><thead><tr><th>Step</th><th>LR</th><th>EN C</th><th>EN I</th><th>EN Macro</th><th>ΔEN</th><th>ZH C</th><th>ZH I</th><th>ZH Macro</th><th>ΔZH</th><th>判定</th></tr></thead><tbody id="checkpointTable"></tbody></table></section>
<section class="two"><div class="panel"><h2>数据与切分</h2><div id="dataset"></div></div><div class="panel"><h2>可审计性与限制</h2><div id="audit"></div></div></section>
</main><script id="reportData" type="application/json">{payload}</script><script>
const D=JSON.parse(document.getElementById('reportData').textContent); const M=D.manifest,S=D.summary;
const f=(v,n=2)=>v==null?'—':Number(v).toFixed(n); const status=M.status==='complete'&&S.table3_completed_count===S.table3_expected_continuation_count?'已完成':'进行中';
document.getElementById('subtitle').textContent=`生成时间 ${{D.generated_at_utc}} · ${{status}} · 主规则 last-terminal-v1`;
const cards=[['正式训练',`${{M.step_records||0}} / ${{M.max_steps}} step`],['Table 3',`${{S.table3_completed_count}} / ${{S.table3_expected_continuation_count}} checkpoint`],['最后稳定点',S.last_stable_step??'TBD'],['推荐候选',S.best_step==null?'TBD':`step ${{S.best_step}} · ZH ${{f(S.best_zh_macro)}}%`]];
document.getElementById('cards').innerHTML=cards.map(x=>`<div class="card"><div class="label">${{x[0]}}</div><div class="value">${{x[1]}}</div></div>`).join('');
function chart(id,series,yLabel){{const all=series.flatMap(s=>s.data).filter(p=>p.y!=null); if(!all.length){{document.getElementById(id).innerHTML='<p class="note">结果待生成</p>';return}} const W=640,H=260,P=42,xs=all.map(p=>p.x),ys=all.map(p=>p.y),xmin=Math.min(...xs),xmax=Math.max(...xs)||1,ymin=Math.min(...ys),ymax=Math.max(...ys); const pad=Math.max((ymax-ymin)*.15,.01),lo=ymin-pad,hi=ymax+pad; const X=x=>P+(x-xmin)/(xmax-xmin||1)*(W-2*P),Y=y=>H-P-(y-lo)/(hi-lo)*(H-2*P); let svg=`<svg viewBox="0 0 ${{W}} ${{H}}"><line class="axis" x1="${{P}}" y1="${{H-P}}" x2="${{W-P}}" y2="${{H-P}}"/><line class="axis" x1="${{P}}" y1="${{P}}" x2="${{P}}" y2="${{H-P}}"/>`; for(let i=0;i<5;i++){{let y=lo+(hi-lo)*i/4;svg+=`<text class="tick" x="${{P-7}}" y="${{Y(y)+4}}" text-anchor="end">${{f(y)}}</text>`}} series.forEach(s=>{{const pts=s.data.filter(p=>p.y!=null);svg+=`<polyline fill="none" stroke="${{s.color}}" stroke-width="3" points="${{pts.map(p=>`${{X(p.x)}},${{Y(p.y)}}`).join(' ')}}"/>`;pts.forEach(p=>svg+=`<circle cx="${{X(p.x)}}" cy="${{Y(p.y)}}" r="4" fill="${{s.color}}"><title>step ${{p.x}}: ${{f(p.y,4)}}</title></circle>`);}}); svg+=`<text class="tick" x="${{W/2}}" y="${{H-5}}" text-anchor="middle">Local optimizer step</text><text class="tick" x="12" y="${{H/2}}" transform="rotate(-90 12 ${{H/2}})" text-anchor="middle">${{yLabel}}</text></svg><div class="legend">${{series.map(s=>`<span><i class="dot" style="background:${{s.color}}"></i>${{s.name}}</span>`).join('')}}</div>`;document.getElementById(id).innerHTML=svg}}
chart('valChart',[{{name:'Token objective',color:'#55a7ff',data:D.validation.map(r=>({{x:r.local_step,y:r.metrics.token_weighted_objective}}))}},{{name:'State macro',color:'#42d8c2',data:D.validation.map(r=>({{x:r.local_step,y:r.metrics.state_macro_accuracy}}))}}],'value');
chart('table3Chart',[{{name:'EN Macro %',color:'#55a7ff',data:D.checkpoint_rows.map(r=>({{x:r.step,y:r.en_macro}}))}},{{name:'ZH Macro %',color:'#ffb45b',data:D.checkpoint_rows.map(r=>({{x:r.step,y:r.zh_macro}}))}}],'ACC (%)');
document.getElementById('checkpointTable').innerHTML=D.checkpoint_rows.map(r=>{{let c=r.status.includes('基本')?'ok':r.status.includes('明显')?'bad':r.status.includes('待')?'warn':'';return `<tr><td>${{r.step}}</td><td>${{f(r.lr,8)}}</td><td>${{f(r.en_complete)}}</td><td>${{f(r.en_incomplete)}}</td><td>${{f(r.en_macro)}}</td><td>${{f(r.en_delta)}}</td><td>${{f(r.zh_complete)}}</td><td>${{f(r.zh_incomplete)}}</td><td>${{f(r.zh_macro)}}</td><td>${{f(r.zh_delta)}}</td><td><span class="pill ${{c}}">${{r.status}}</span></td></tr>`}}).join('');
const sp=D.split;document.getElementById('dataset').innerHTML=`<p><b>Train</b> ${{sp.train.source_conversation_count}} 会话 · ${{sp.train.row_count}} rows · ${{f(sp.train.duration_hours_from_160ms_chunks,3)}} h</p><p><b>Validation</b> ${{sp.validation.source_conversation_count}} 会话 · ${{sp.validation.row_count}} rows · ${{f(sp.validation.duration_hours_from_160ms_chunks,3)}} h</p><p>Source leakage: <b>${{sp.source_leakage_count}}</b></p><p class="note">split identity<br><code>${{sp.split_identity_sha256}}</code></p>`;
document.getElementById('audit').innerHTML=`<p>官方 checkpoint 不含 optimizer/global_step；起始 ${{M.origin_step_estimate}} 为低置信度估计。</p><p>本地有效 batch ${{M.local_effective_batch}}，官方参考 ${{M.official_reference_global_effective_batch}}；每个 local step≈${{f(M.official_sample_equivalent_step_per_local_step,3)}} sample-equivalent step。</p><p>AMP overflow: <b>${{M.amp_overflows.length}}</b> · Peak CUDA: <b>${{f((M.cuda_peak_memory_bytes||0)/1073741824,3)}} GiB</b></p><p class="note">未完成结果不插值；Table 3 不参与 LR 选择。</p>`;
</script></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-run", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--lr-selection", type=Path, required=True)
    parser.add_argument("--table3-baseline", type=Path, required=True)
    parser.add_argument("--table3-index", type=Path)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    args = parser.parse_args()
    data = build_report_data(args)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(data) + "\n", encoding="utf-8")
    args.output_html.write_text(render_html(data) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "markdown": str(args.output_md),
                "markdown_sha256": sha256_file(args.output_md),
                "html": str(args.output_html),
                "html_sha256": sha256_file(args.output_html),
                "status": data["manifest"]["status"],
                "table3_completed": data["summary"]["table3_completed_count"],
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
