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
    decision = (
        load_json(args.evaluation_decision.resolve(strict=True))
        if args.evaluation_decision
        else None
    )
    source_inventory = (
        load_jsonl(args.source_inventory.resolve(strict=True))
        if args.source_inventory
        else []
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
    planned_steps = [int(step) for step in manifest["checkpoint_steps"]]
    expected_steps = (
        [int(step) for step in decision["included_report_steps"]]
        if decision
        else planned_steps
    )
    if expected_steps != sorted(set(expected_steps)) or not expected_steps or expected_steps[0] != 0:
        raise ValueError("included_report_steps must be unique, sorted, and start at step 0")
    if not set(expected_steps).issubset(planned_steps):
        raise ValueError("included_report_steps must be a subset of the training checkpoint grid")
    table3_by_step = {item["local_step"]: item for item in table3.get("checkpoints", [])}
    checkpoint_rows = []
    prior_confirmed_decline = False
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
                    "checkpoint_sha256": manifest["base_checkpoint"]["sha256"],
                    "evidence_gate_sha256": table3.get("baseline_gate_sha256"),
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
                "balanced_macro": 0.5
                * (
                    row["languages"]["en"]["candidate_macro_accuracy_percent"]
                    + row["languages"]["zh"]["candidate_macro_accuracy_percent"]
                ),
                "balanced_delta": 0.5
                * (
                    row["languages"]["en"]["delta_percentage_points"]
                    + row["languages"]["zh"]["delta_percentage_points"]
                ),
                "status": (
                    "明显下降（已确认）"
                    if row["obvious_decline_confirmed"]
                    else "基本不变"
                    if row["almost_unchanged"]
                    else "明显下降趋势持续"
                    if row["obvious_decline_trigger"] and prior_confirmed_decline
                    else "明显下降触发待确认"
                    if row["obvious_decline_trigger"]
                    else "轻微/不均衡退化"
                ),
                "almost_unchanged": row["almost_unchanged"],
                "obvious_decline_trigger": row["obvious_decline_trigger"],
                "obvious_decline_confirmed": row["obvious_decline_confirmed"],
                "zh_subgroups": row["zh_subgroups"],
                "checkpoint_sha256": row["checkpoint"]["sha256"],
                "evidence_gate_sha256": row["evidence_gate_sha256"],
            }
        )
        prior_confirmed_decline = prior_confirmed_decline or row["obvious_decline_confirmed"]
    checkpoint_rows[0]["balanced_macro"] = 0.5 * (
        baseline_languages["en"] + baseline_languages["zh"]
    )
    checkpoint_rows[0]["balanced_delta"] = 0.0
    completed_rows = [row for row in checkpoint_rows if row.get("en_macro") is not None]
    continuation_rows = [row for row in completed_rows if row["step"] > 0]
    stable_rows = [row for row in continuation_rows if row.get("almost_unchanged")]
    first_observed_degradation = next(
        (row for row in continuation_rows if not row.get("almost_unchanged")), None
    )
    first_trigger = next(
        (row for row in continuation_rows if row.get("obvious_decline_trigger")), None
    )
    first_decline = next(
        (row for row in continuation_rows if row.get("obvious_decline_confirmed")), None
    )
    first_broad_decline = next(
        (
            row
            for row in continuation_rows
            if row.get("en_delta", 0.0) < -3.0 and row.get("zh_delta", 0.0) < -3.0
        ),
        None,
    )
    least_damaging = (
        max(
            continuation_rows,
            key=lambda row: (
                min(
                    row["en_complete"] - baseline_classes["en/complete"]["accuracy_percent"],
                    row["en_incomplete"] - baseline_classes["en/incomplete"]["accuracy_percent"],
                    row["zh_complete"] - baseline_classes["zh/complete"]["accuracy_percent"],
                    row["zh_incomplete"] - baseline_classes["zh/incomplete"]["accuracy_percent"],
                ),
                row["balanced_macro"],
                -row["step"],
            ),
        )
        if continuation_rows
        else None
    )
    required_continuation_steps = [step for step in expected_steps if step > 0]
    completed_continuation_steps = [row["step"] for row in continuation_rows]
    evaluation_closed = bool(
        decision and decision.get("status") == "completed_early_stop"
    )
    report_complete = bool(
        manifest["status"] == "complete"
        and completed_continuation_steps == required_continuation_steps
        and (evaluation_closed or expected_steps == planned_steps)
    )
    source_unique_hours = (
        sum(float(item["duration_seconds"]) for item in source_inventory) / 3600
        if source_inventory
        else None
    )
    input_artifacts = {}
    for name, path in (
        ("split_manifest", args.split_manifest),
        ("lr_selection", args.lr_selection),
        ("table3_index", args.table3_index),
        ("evaluation_decision", args.evaluation_decision),
        ("source_inventory", args.source_inventory),
    ):
        if path:
            resolved = path.resolve(strict=True)
            input_artifacts[name] = {
                "path": str(resolved),
                "sha256": sha256_file(resolved),
            }
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
        "evaluation_decision": decision,
        "source_unique_hours": source_unique_hours,
        "baseline_classes": baseline_classes,
        "baseline_languages": baseline_languages,
        "checkpoint_rows": checkpoint_rows,
        "input_artifacts": input_artifacts,
        "summary": {
            "report_complete": report_complete,
            "report_status": "已完成（确认退化后提前结束后续评测）" if report_complete and evaluation_closed else "已完成" if report_complete else "进行中",
            "last_stable_continuation_step": stable_rows[-1]["step"] if stable_rows else None,
            "first_observed_degradation_step": first_observed_degradation["step"] if first_observed_degradation else None,
            "first_decline_trigger_step": first_trigger["step"] if first_trigger else None,
            "first_confirmed_decline_step": first_decline["step"] if first_decline else None,
            "first_broad_decline_step": first_broad_decline["step"] if first_broad_decline else None,
            "least_damaging_continuation_step": least_damaging["step"] if least_damaging else None,
            "recommended_checkpoint_step": 0 if first_decline else None,
            "table3_completed_count": len(continuation_rows),
            "table3_expected_continuation_count": len(required_continuation_steps),
            "table3_originally_planned_continuation_count": len(planned_steps) - 1,
        },
    }


def render_markdown(data: dict[str, Any]) -> str:
    manifest = data["manifest"]
    split = data["split"]
    selection = data["lr_selection"]
    summary = data["summary"]
    decision = data.get("evaluation_decision") or {}
    source_unique_hours = data.get("source_unique_hours")
    lines = [
        "# SoulX-Duplug Stage 3 中文续训练实验与性能评估",
        "",
        f"更新时间：{data['generated_at_utc']}",
        "用途：课题组会议/导师汇报",
        f"实验状态：**{summary['report_status']}**",
        "",
        "## 1. 结论摘要",
        "",
        f"- 正式训练状态：`{manifest['status']}`；已完成 optimizer step：{manifest.get('step_records', 0)}/{manifest['max_steps']}。",
        f"- 最终 Table 3 分析窗口为 step 0/5/10/20/30；其中续训练 checkpoint 已完成 {summary['table3_completed_count']}/{summary['table3_expected_continuation_count']}。原计划共有 {summary['table3_originally_planned_continuation_count']} 个续训练 checkpoint。",
        f"- 验证集选择的 peak LR：`{selection['selected_peak_lr']:.8g}`；选择过程未读取 Table 3。",
        f"- 没有任何已测续训练 checkpoint 满足严格的“几乎未下降”条件；最早可观测退化点是 step {summary['first_observed_degradation_step']}。",
        f"- 首次类别级明显下降在 step {summary['first_decline_trigger_step']}，并由 step 20 确认；EN/ZH 两种语言宏平均同时明显下降始于 step {summary['first_broad_decline_step']}。",
        f"- step {summary['least_damaging_continuation_step']} 只是四个续训练点中相对损伤最小者，仍未通过稳定性门禁。若目标是保持现有 Table 3 通用能力，推荐继续使用官方 step 0。",
        "",
        "> step 45/60/90/120/180/240/300 的进一步评测是在看到 step 10→20 已确认退化、step 30 继续恶化后，由项目负责人于 2026-08-21 决定停止。该决定是事后提前结束评测，不能表述为原始预注册网格的一部分；报告不对未测点插值，也不使用 step 45 的不完整四分类结果。",
        "",
        "## 2. 起始状态与“续训练”定义",
        "",
        f"- 官方 Bilingual 权重 SHA-256：`{manifest['base_checkpoint']['sha256']}`。",
        "- 发布权重不含 global_step、AdamW、scheduler、AMP scaler，因此这是从模型参数继续微调，不是 optimizer 的精确 resume。",
        f"- 公开 Stage 3 配置 `total_steps=1800`，故将起始 step 估计为 {manifest['origin_step_estimate']}（低置信度），不能表述为已证实的官方 checkpoint step。",
        f"- 本地 batch=1、梯度累积={manifest['gradient_accumulation']}，有效 batch={manifest['local_effective_batch']}；官方参考全局有效 batch={manifest['official_reference_global_effective_batch']}。",
        f"- 每个本地 step 对应约 {manifest['official_sample_equivalent_step_per_local_step']:.3f} 个官方 sample-equivalent step；例如 local step 20 只有约 2.5 个官方 sample-equivalent step。",
        f"- 可训练参数：{manifest['trainable_parameter_count']:,}；总参数：{manifest['total_parameter_count']:,}。",
        "",
        "## 3. 数据集、处理方法与切分",
        "",
        "训练源为 DuplexConv `Edu_0018`：500 个同步多轨教育场景会话（495 个双声道、5 个三声道），展开为 1,005 个 target-speaker views。三声道不丢弃：每次只输入一个目标声道，其他声道聚合为关系证据，不把多路 audio token 放入同一 sequence。",
        "",
        f"原始完整会话去重时长约 {fmt(source_unique_hours, 3)} 小时，按目标说话人视角累计约 21.170 小时。相较 DuplexConv 公开约 2,000 小时的总体规模，本轮只覆盖约 0.53%，因此应称为 `Edu_0018` pilot，而不是完整 DuplexConv 续训练。",
        "",
        "状态映射：官方 complete/incomplete/backchannel 原样映射；11 个 WAIT 映射为 complete；1,599 个缺失状态由固定 `qwen3-235b-a22b-instruct-2507` 通过 OpenRouter 补标（404 个源会话请求，accepted-response cost 0.2187791 USD）。这些是 LLM 辅助标签，不称为人工 gold。Paraformer 只用于中文伪转录/时间戳构造，不参与模型训练。",
        "",
        "最终 model-ready 数据包含 2,168 rows、474,030 个可用 160 ms chunks 和 953,532 个 GLM audio tokens；另有 2,736 个异常 chunks（0.574%）被隔离，没有伪造文本或状态补齐。",
        "",
        "| Split | 源会话 | target views | rows | 160ms chunks | 视角时长(h) | Qwen 补标事件 |",
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
            f"选择原因：{selection['selection_reason']}。正式 LR 采用 5-step 新 AdamW 重热身，并按估计原 step=1800 进行 offset inverse-square-root 衰减。正式训练运行时间为 {manifest['started_at_utc']} 至 {manifest['completed_at_utc']}；300 次 optimizer 更新均已记录，AMP overflow 为 {len(manifest['amp_overflows'])}。",
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
            "训练域 validation objective 持续下降并不代表外部通用能力保持。后续 Table 3 显示 Complete/Incomplete 决策边界发生快速偏移，这是本轮最重要的泛化差异。",
            "",
            "## 6. Table 3 最终结果",
            "",
            "主规则始终为 `last-terminal-v1`；样本、seed、顺序、推理核心、尾部静音和 Teacher-ASR 固定。每个纳入报告的 checkpoint 四类结果都经过独立证据 gate；论文目标没有传入推理 runner，也不用于选择 LR 或训练 checkpoint。",
            "",
            "先用官方发布权重复现 step 0。语言宏平均与论文分别相差 EN +0.96pp、ZH -0.33pp，可视为数值基本接近；但预注册的“四类均不超过 ±1.0pp”机器门禁仍因 EN Complete 和 ZH Complete 失败，因此报告保留“已审计候选协议、尚缺作者样本级脚本确认”的限定。",
            "",
            "| 语言 | 指标 | 论文结果 | 官方权重本地 step 0 | 差异 |",
            "| --- | --- | ---: | ---: | ---: |",
            "| EN | Complete ACC | 77.67% | 78.93% | +1.26pp |",
            "| EN | Incomplete ACC | 88.96% | 89.63% | +0.67pp |",
            "| EN | Macro ACC | 83.32% | 84.28% | +0.96pp |",
            "| ZH | Complete ACC | 89.33% | 87.67% | -1.67pp |",
            "| ZH | Incomplete ACC | 79.33% | 80.33% | +1.00pp |",
            "| ZH | Macro ACC | 84.33% | 84.00% | -0.33pp |",
            "",
            "“基本不变”：EN/ZH macro 各下降不超过 1pp，且任一 class 下降不超过 2pp。“明显下降触发”：任一语言 macro 下降超过 3pp，或任一 class 下降超过 5pp；必须在下一个预注册点仍触发才确认。",
            "",
            "| Local step | 估计总 step | LR | EN C | EN I | EN Macro | ΔEN | ZH C | ZH I | ZH Macro | ΔZH | 四类 Macro | 判定 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in data["checkpoint_rows"]:
        lines.append(
            "| {step} | {total} | {lr} | {enc} | {eni} | {enm} | {end} | {zhc} | {zhi} | {zhm} | {zhd} | {bm} | {status} |".format(
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
                bm=fmt(row.get("balanced_macro")),
                status=row["status"],
            )
        )
    lines.extend(
        [
            "",
            "模型与证据身份：",
            "",
            "| Step | checkpoint SHA-256 | evidence gate SHA-256 |",
            "| ---: | --- | --- |",
        ]
    )
    for row in data["checkpoint_rows"]:
        lines.append(
            f"| {row['step']} | `{row.get('checkpoint_sha256', '—')}` | `{row.get('evidence_gate_sha256', '—')}` |"
        )
    lines.extend(
        [
            "",
            "关键观察：",
            "",
            "1. step 5 已不满足“基本不变”：ZH Incomplete 下降 3.333pp，paired bootstrap 95% CI 为 [-6.000, -0.667]pp，exact McNemar p=0.0309；但尚未达到预定义明显下降阈值。",
            "2. step 10 首次触发类别级明显下降：EN Incomplete 下降 8.027pp（95% CI [-11.706, -4.682]，p=1.93e-5），ZH Incomplete 下降 9.000pp（95% CI [-12.333, -5.667]，p=1.12e-7）。step 20 再次触发，因此 step 10 被正式确认。",
            "3. step 10 的四类 Macro 为 84.181%，与 step 0 的 84.141% 几乎相同，但这是 Complete 上升和 Incomplete 下降相互抵消的结果，不能据此宣称模型整体无退化。",
            "4. step 20 首次出现 EN 与 ZH 宏平均同时下降超过 3pp；step 30 的 EN/ZH 宏平均分别下降 14.981/9.167pp，EN Incomplete 已下降 48.829pp。趋势表现为模型越来越偏向 Complete。",
            "",
            "中文固定 600 条是发布方完整测试集，不是本项目随机抽样；每个 checkpoint 另报 complete/incomplete × real/synthetic 四个固定子组。差异显著性使用同一样本的 paired bootstrap 95% CI 和 exact McNemar，而不是把两次准确率当独立样本。",
            "",
            "## 7. 评测提前结束、可恢复性与异常记录",
            "",
            f"- 评测决定状态：`{decision.get('status', '未提供')}`；正式纳入 step：{decision.get('included_report_steps', [])}；不再评测：{decision.get('not_further_evaluated_steps', [])}。",
            "- step 45 已完成 EN Complete、EN Incomplete、ZH Complete，但 ZH Incomplete 在 Paraformer VAD 辅助模型初始化时因本地代理不可用而退出。由于四类未闭环，step 45 整体不纳入正式分析；其 partial 证据保留在数据盘，不与其他 checkpoint 拼接。",
            f"- 训练状态：`{manifest['status']}`；AMP overflow 次数：{len(manifest['amp_overflows'])}。",
            f"- 峰值 CUDA allocated：{manifest.get('cuda_peak_memory_bytes', 0)/1024**3:.3f} GiB。",
            f"- GPU：{manifest['environment']['gpu']}；Python：{manifest['environment']['python']}；Torch：{manifest['environment']['packages']['torch']}；CUDA：{manifest['environment']['torch_cuda']}。",
            "- 每个预注册点保存仅含 118 个可训练 tensor 的评测快照；训练完成到 step 300，停止的是耗时较长的外部 Table 3 推理，不是训练。",
            f"- 正式 run manifest：`{data['manifest_path']}`，SHA-256=`{data['manifest_sha256']}`。",
            f"- 评测停止决定：`{decision.get('decision_path', 'evaluation_reports/duplexconv_edu0018_table3_evaluation_decision.json')}`。",
            "- 报告生成输入均记录绝对路径和 SHA-256；HTML 内嵌同一份结构化数据，可离线查看。",
            "",
            "| 报告输入 | 路径 | SHA-256 |",
            "| --- | --- | --- |",
        ]
    )
    for name, artifact in data["input_artifacts"].items():
        lines.append(
            f"| `{name}` | `{artifact['path']}` | `{artifact['sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## 8. 反作假检查与限制",
            "",
            "1. 专项审计未发现预测篡改、标签入模、样本排除、分类别调参或事后切换主规则；`selection_used_paper_targets=false`，LR selection 也记录 `benchmark_used_for_selection=false`。完整审计见 `evaluation_reports/soulx_table3_anti_cheating_audit.md`。",
            "2. 本轮提前结束发生在明显退化已被 step 20 确认之后，所有已经完整得到的 step 5/10/20/30 均如实报告；未用 step 45 partial 选择性补表，也未伪造后续点。",
            "3. 起始 1800 step 是依据公开配置的低置信度估计，不是官方权重元数据。",
            "4. 本地有效 batch=72，只有官方参考全局有效 batch=576 的 1/8；因此 local optimizer step 不能直接等同于官方同数量 step。",
            "5. 当前 Table 3 样本级读出规则是已审计候选协议，数值与论文基本一致，但仍缺作者发布的样本级计分脚本确认。",
            "6. `Edu_0018` 只有约 10.519 个去重会话小时；结论只适用于这次小规模 pilot，不能外推为完整 2,000 小时 DuplexConv 的训练结论。",
            "7. Full-Duplex-Bench 是包含 LLM/TTS 的系统级表 2 测试。本轮没有续训练 checkpoint 通过模型级稳定性门禁，因此未继续对这些 checkpoint 做昂贵的系统级评测；本报告的最终结论限于模型级 Table 3。",
            "",
            "## 9. 模型选择与后续建议",
            "",
            "- 保持现有 EN/ZH Easy Turn 能力：使用官方 step 0。",
            "- 若只做研究性对比、必须使用本轮续训练权重：step 5 是四个已测点中损伤最小者，但应明确标注“未通过稳定性门禁”，不能作为无退化版本发布。",
            "- 下一轮不宜直接增加本轮数据上的 step；优先扩大 DuplexConv 覆盖规模并平衡 Complete/Incomplete，考虑官方/英文 replay、降低峰值 LR，并把 step 1–5 设为更密的早期评测窗口。",
            "- 需要另建与训练域分离的中文 in-domain 测试集，才能判断 `Edu_0018` 适配收益；训练域 validation 改善不能替代外部收益证据。",
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
.card{{padding:16px}} .label{{color:var(--muted);font-size:12px}} .value{{font-size:25px;font-weight:750;margin-top:4px}} .panel{{padding:18px;margin-bottom:18px;overflow:auto}} .two{{display:grid;grid-template-columns:1fr 1fr;gap:18px}} .banner{{border-left:4px solid var(--red);padding:15px 18px;margin-bottom:18px;background:#351d2a;border-radius:10px;color:#ffdce2}}
table{{width:100%;border-collapse:collapse;white-space:nowrap}} th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right}} th:first-child,td:first-child{{text-align:left}} th{{color:#bed2eb;font-size:12px}} tr:last-child td{{border:0}}
.pill{{display:inline-block;padding:3px 9px;border-radius:999px;background:#24415d;color:#dcecff;font-size:12px}} .ok{{background:#174d47;color:#8df1d7}} .warn{{background:#594324;color:#ffd18e}} .bad{{background:#5a2832;color:#ffacb6}}
svg{{width:100%;height:300px;overflow:visible}} .axis{{stroke:#46617f;stroke-width:1}} .tick{{fill:#8fa8c4;font-size:11px}} .legend{{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted)}} .dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}}
.note{{color:var(--muted)}} code{{color:#a9d4ff}} @media(max-width:900px){{.grid{{grid-template-columns:1fr 1fr}}.two{{grid-template-columns:1fr}}}} @media(max-width:560px){{.wrap{{padding:14px}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main class="wrap">
<h1>SoulX-Duplug Stage 3 中文续训练</h1><div class="sub" id="subtitle"></div><div class="banner"><b>最终结论：</b>step 10 首次出现并被确认的类别级明显下降；step 20 起 EN/ZH 宏平均同时明显下降。没有续训练 checkpoint 通过稳定性门禁，保持 Table 3 通用能力时推荐官方 step 0。</div><section class="grid" id="cards"></section>
<section class="two"><div class="panel"><h2>Table 3 语言 Macro ACC</h2><div id="table3Chart"></div></div><div class="panel"><h2>Complete / Incomplete 分化</h2><div id="classChart"></div></div></section>
<section class="two"><div class="panel"><h2>训练域 validation objective</h2><div id="valObjectiveChart"></div></div><div class="panel"><h2>训练域 validation state macro</h2><div id="valStateChart"></div></div></section>
<section class="panel"><h2>最终评测窗口：step 0/5/10/20/30</h2><table><thead><tr><th>Step</th><th>LR</th><th>EN C</th><th>EN I</th><th>EN Macro</th><th>ΔEN</th><th>ZH C</th><th>ZH I</th><th>ZH Macro</th><th>ΔZH</th><th>四类 Macro</th><th>判定</th></tr></thead><tbody id="checkpointTable"></tbody></table></section>
<section class="two"><div class="panel"><h2>数据与切分</h2><div id="dataset"></div></div><div class="panel"><h2>可审计性与限制</h2><div id="audit"></div></div></section>
</main><script id="reportData" type="application/json">{payload}</script><script>
const D=JSON.parse(document.getElementById('reportData').textContent); const M=D.manifest,S=D.summary;
const f=(v,n=2)=>v==null?'—':Number(v).toFixed(n);
document.getElementById('subtitle').textContent=`生成时间 ${{D.generated_at_utc}} · ${{S.report_status}} · 主规则 last-terminal-v1`;
const cards=[['正式训练',`${{M.step_records||0}} / ${{M.max_steps}} step`],['最终评测',`${{S.table3_completed_count}} / ${{S.table3_expected_continuation_count}} checkpoint`],['类别级下降起点',`step ${{S.first_confirmed_decline_step}}`],['推荐模型',`官方 step ${{S.recommended_checkpoint_step}}`]];
document.getElementById('cards').innerHTML=cards.map(x=>`<div class="card"><div class="label">${{x[0]}}</div><div class="value">${{x[1]}}</div></div>`).join('');
function chart(id,series,yLabel){{const all=series.flatMap(s=>s.data).filter(p=>p.y!=null); if(!all.length){{document.getElementById(id).innerHTML='<p class="note">结果待生成</p>';return}} const W=640,H=260,P=42,xs=all.map(p=>p.x),ys=all.map(p=>p.y),xmin=Math.min(...xs),xmax=Math.max(...xs)||1,ymin=Math.min(...ys),ymax=Math.max(...ys); const pad=Math.max((ymax-ymin)*.15,.01),lo=ymin-pad,hi=ymax+pad; const X=x=>P+(x-xmin)/(xmax-xmin||1)*(W-2*P),Y=y=>H-P-(y-lo)/(hi-lo)*(H-2*P); let svg=`<svg viewBox="0 0 ${{W}} ${{H}}"><line class="axis" x1="${{P}}" y1="${{H-P}}" x2="${{W-P}}" y2="${{H-P}}"/><line class="axis" x1="${{P}}" y1="${{P}}" x2="${{P}}" y2="${{H-P}}"/>`; for(let i=0;i<5;i++){{let y=lo+(hi-lo)*i/4;svg+=`<text class="tick" x="${{P-7}}" y="${{Y(y)+4}}" text-anchor="end">${{f(y)}}</text>`}} series.forEach(s=>{{const pts=s.data.filter(p=>p.y!=null);svg+=`<polyline fill="none" stroke="${{s.color}}" stroke-width="3" points="${{pts.map(p=>`${{X(p.x)}},${{Y(p.y)}}`).join(' ')}}"/>`;pts.forEach(p=>svg+=`<circle cx="${{X(p.x)}}" cy="${{Y(p.y)}}" r="4" fill="${{s.color}}"><title>step ${{p.x}}: ${{f(p.y,4)}}</title></circle>`);}}); svg+=`<text class="tick" x="${{W/2}}" y="${{H-5}}" text-anchor="middle">Local optimizer step</text><text class="tick" x="12" y="${{H/2}}" transform="rotate(-90 12 ${{H/2}})" text-anchor="middle">${{yLabel}}</text></svg><div class="legend">${{series.map(s=>`<span><i class="dot" style="background:${{s.color}}"></i>${{s.name}}</span>`).join('')}}</div>`;document.getElementById(id).innerHTML=svg}}
chart('table3Chart',[{{name:'EN Macro %',color:'#55a7ff',data:D.checkpoint_rows.map(r=>({{x:r.step,y:r.en_macro}}))}},{{name:'ZH Macro %',color:'#ffb45b',data:D.checkpoint_rows.map(r=>({{x:r.step,y:r.zh_macro}}))}}],'ACC (%)');
chart('classChart',[{{name:'EN Complete',color:'#55a7ff',data:D.checkpoint_rows.map(r=>({{x:r.step,y:r.en_complete}}))}},{{name:'EN Incomplete',color:'#ff6b7a',data:D.checkpoint_rows.map(r=>({{x:r.step,y:r.en_incomplete}}))}},{{name:'ZH Complete',color:'#42d8c2',data:D.checkpoint_rows.map(r=>({{x:r.step,y:r.zh_complete}}))}},{{name:'ZH Incomplete',color:'#ffb45b',data:D.checkpoint_rows.map(r=>({{x:r.step,y:r.zh_incomplete}}))}}],'ACC (%)');
chart('valObjectiveChart',[{{name:'Token objective',color:'#55a7ff',data:D.validation.map(r=>({{x:r.local_step,y:r.metrics.token_weighted_objective}}))}}],'objective');
chart('valStateChart',[{{name:'State macro %',color:'#42d8c2',data:D.validation.map(r=>({{x:r.local_step,y:100*r.metrics.state_macro_accuracy}}))}}],'ACC (%)');
document.getElementById('checkpointTable').innerHTML=D.checkpoint_rows.map(r=>{{let c=r.status.includes('基线')?'ok':r.status.includes('明显')?'bad':'warn';return `<tr><td>${{r.step}}</td><td>${{f(r.lr,8)}}</td><td>${{f(r.en_complete)}}</td><td>${{f(r.en_incomplete)}}</td><td>${{f(r.en_macro)}}</td><td>${{f(r.en_delta)}}</td><td>${{f(r.zh_complete)}}</td><td>${{f(r.zh_incomplete)}}</td><td>${{f(r.zh_macro)}}</td><td>${{f(r.zh_delta)}}</td><td>${{f(r.balanced_macro)}}</td><td><span class="pill ${{c}}">${{r.status}}</span></td></tr>`}}).join('');
const sp=D.split;document.getElementById('dataset').innerHTML=`<p><b>Edu_0018 pilot</b> 500 会话 · 1,005 target views · ${{f(D.source_unique_hours,3)}} 去重会话小时</p><p><b>Train</b> ${{sp.train.source_conversation_count}} 会话 · ${{sp.train.row_count}} rows · ${{f(sp.train.duration_hours_from_160ms_chunks,3)}} 视角小时</p><p><b>Validation</b> ${{sp.validation.source_conversation_count}} 会话 · ${{sp.validation.row_count}} rows · ${{f(sp.validation.duration_hours_from_160ms_chunks,3)}} 视角小时</p><p>Source leakage: <b>${{sp.source_leakage_count}}</b></p><p class="note">当前 pilot 约为 DuplexConv 公开 2,000 小时总体规模的 0.53%。</p>`;
document.getElementById('audit').innerHTML=`<p><b>提前结束：</b>step 10 的类别级下降由 step 20 确认，step 30 继续恶化；step 45 partial 和后续点均未纳入。</p><p>官方 checkpoint 不含 optimizer/global_step；起始 ${{M.origin_step_estimate}} 为低置信度估计。</p><p>本地有效 batch ${{M.local_effective_batch}}，官方参考 ${{M.official_reference_global_effective_batch}}；每个 local step≈${{f(M.official_sample_equivalent_step_per_local_step,3)}} sample-equivalent step。</p><p>AMP overflow: <b>${{M.amp_overflows.length}}</b> · Peak CUDA: <b>${{f((M.cuda_peak_memory_bytes||0)/1073741824,3)}} GiB</b></p><p class="note">所有完整结果均报告；未测点不插值，Table 3 不参与 LR 选择。</p>`;
</script></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-run", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--lr-selection", type=Path, required=True)
    parser.add_argument("--table3-baseline", type=Path, required=True)
    parser.add_argument("--table3-index", type=Path)
    parser.add_argument("--evaluation-decision", type=Path)
    parser.add_argument("--source-inventory", type=Path)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    args = parser.parse_args()
    data = build_report_data(args)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(data).rstrip() + "\n", encoding="utf-8")
    args.output_html.write_text(render_html(data) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "markdown": str(args.output_md),
                "markdown_sha256": sha256_file(args.output_md),
                "html": str(args.output_html),
                "html_sha256": sha256_file(args.output_html),
                "training_status": data["manifest"]["status"],
                "report_status": data["summary"]["report_status"],
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
