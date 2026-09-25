#!/usr/bin/env python3
"""Render a dynamic audited SoulX official-continuation meeting report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence


EXPECTED_CLASSES = ("en/complete", "en/incomplete", "zh/complete", "zh/incomplete")
STATE_HEADS = ("idle", "nonidle", "user_complete", "user_incomplete", "user_backchannel")
OFFICIAL_BASE_SHA256 = (
    "b0703dea0b1dbb1cd51e6e7b6514c60907ea4d4b6752cecc3f71cb6445650dbe"
)
UPSTREAM_COMMIT = "928b06508ed2de1344208d06fb1f6fb2ebfb1df5"
EVALUATION_COMMIT = "b17bcf903b9bd896238a4ee9fec495fd75df1401"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return payload


def atomic_write(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def audit_inputs(paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = {name: path.resolve(strict=True) for name, path in paths.items()}
    payload = {name: load_json(path) for name, path in resolved.items()}
    training = payload["training_manifest"]
    pretrain_step0 = payload["pretrain_step0"]
    internal = payload["internal_validation"]
    posttrain = payload["posttrain_orchestration"]
    table3_orchestration = payload["table3_orchestration"]
    table3 = payload["table3_index"]
    aggregate = payload["aggregate_manifest"]
    states = payload["state_provenance"]
    split = payload["split_manifest"]
    execution = payload["execution_manifest"]
    training_steps = training.get("checkpoint_steps", [])
    orchestration_identity = table3_orchestration.get("identity", {})
    coarse_to_fine = (
        orchestration_identity.get("profile")
        == "table3-endpoint-first-coarse-to-fine-v1"
    )
    evaluated_steps = (
        table3_orchestration.get("evaluated_steps", [])
        if coarse_to_fine
        else orchestration_identity.get("expected_steps", [])
    )
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(training.get("status") == "complete", "official Lightning training is incomplete")
    require(training.get("runtime_base_commit") == UPSTREAM_COMMIT, "training upstream commit drift")
    require(training.get("base_checkpoint", {}).get("sha256") == OFFICIAL_BASE_SHA256, "base checkpoint identity drift")
    require(training.get("exact_optimizer_resume") is False, "released checkpoint was misreported as exact optimizer resume")
    require(training.get("origin_step_estimate_confidence") == "low", "origin-step confidence drift")
    require(training_steps == [1, 2, 3, 5, 10, 20, 30], "unexpected training checkpoint grid")
    require(training.get("final_local_step") == 30, "training did not finish at local step 30")
    updates = training.get("updates", [])
    require([row.get("local_step") for row in updates] == list(range(1, 31)), "optimizer update sequence drift")
    require(all(row.get("microbatches") == 576 and row.get("samples") == 576 for row in updates), "effective-batch audit drift")

    split_identity = split.get("split_identity_sha256")
    require(split.get("source_leakage_count") == 0, "source-conversation leakage is non-zero")
    require(training.get("split", {}).get("split_identity_sha256") == split_identity, "training split identity mismatch")
    require(pretrain_step0.get("status") == "complete", "pre-training step 0 is incomplete")
    require(pretrain_step0.get("local_step") == 0, "pre-training baseline is not step 0")
    require(pretrain_step0.get("optimizer_created") is False, "pre-training baseline created an optimizer")
    require(pretrain_step0.get("training_updates_performed") == 0, "pre-training baseline performed updates")
    require(pretrain_step0.get("backward_calls_performed") == 0, "pre-training baseline performed backward")
    require(pretrain_step0.get("split_manifest", {}).get("split_identity_sha256") == split_identity, "pre-training split identity mismatch")

    require(internal.get("status") == "complete", "exact internal-validation index is incomplete")
    require(internal.get("expected_steps") == training_steps, "internal-validation grid mismatch")
    require(internal.get("table3_used_for_selection") is False, "Table 3 was used for checkpoint selection")
    require(internal.get("split_identity_sha256") == split_identity, "internal-validation split identity mismatch")
    require(internal.get("baseline", {}).get("exact_token_weighted_metrics", {}).get("profile") == "exact-per-row-target-weighted-v1", "exact token-weighted baseline is missing")

    require(posttrain.get("status") == "complete", "post-training evaluation coordinator is incomplete")
    require(table3_orchestration.get("status") == "complete", "Table 3 orchestration is incomplete")
    require(orchestration_identity.get("evaluation_commit") == EVALUATION_COMMIT, "Table 3 evaluation commit drift")
    require(orchestration_identity.get("baseline", {}).get("mode") == "external_read_only", "Table 3 baseline was not reused read-only")
    if coarse_to_fine:
        mandatory = orchestration_identity.get("mandatory_evaluation_order")
        refinement = table3_orchestration.get("refinement_steps", [])
        require(mandatory == [1, 30, 10, 5, 20], "coarse-to-fine mandatory order drift")
        require(orchestration_identity.get("conditional_refinement_order") == [2, 3], "coarse-to-fine refinement order drift")
        require(orchestration_identity.get("non_monotonicity_assumed") is False, "coarse-to-fine incorrectly assumes monotonicity")
        require(orchestration_identity.get("table3_used_for_checkpoint_selection") is False, "Table 3 affected checkpoint selection")
        require(evaluated_steps == sorted(set(mandatory + refinement)), "coarse-to-fine evaluated grid mismatch")
        require(table3_orchestration.get("omitted_steps") == sorted(set(training_steps) - set(evaluated_steps)), "coarse-to-fine omitted-step audit mismatch")
    else:
        require(evaluated_steps == training_steps, "Table 3 orchestration grid mismatch")
    require(len(table3_orchestration.get("completed_classes", [])) == len(evaluated_steps) * 4, "Table 3 completed-class count mismatch")
    require(table3.get("status") == "complete", "Table 3 index is incomplete")
    require(table3.get("primary_rule") == "last-terminal-v1", "Table 3 primary rule drift")
    require(table3.get("selection_used_paper_targets") is False, "paper targets affected selection")
    require([row.get("local_step") for row in table3.get("checkpoints", [])] == evaluated_steps, "Table 3 index grid mismatch")

    for row in table3.get("checkpoints", []):
        step = row.get("local_step")
        require(sorted(row.get("classes", {})) == sorted(EXPECTED_CLASSES), f"step {step} Table 3 classes incomplete")
        expected_checkpoint = training.get("checkpoints", {}).get(str(step), {})
        require(row.get("checkpoint", {}).get("sha256") == expected_checkpoint.get("sha256"), f"step {step} checkpoint SHA mismatch")
        gate_path_value = row.get("evidence_gate_path")
        if not gate_path_value:
            errors.append(f"step {step} evidence gate path missing")
            continue
        gate_path = Path(gate_path_value).resolve(strict=True)
        gate = load_json(gate_path)
        require(sha256_file(gate_path) == row.get("evidence_gate_sha256"), f"step {step} evidence gate SHA mismatch")
        require(gate.get("evidence_audit_passed") is True and gate.get("gate_passed") is True, f"step {step} evidence gate failed")

    baseline_gate_path = Path(table3["baseline_root"]) / "table3-gate.json"
    baseline_gate = load_json(baseline_gate_path.resolve(strict=True))
    require(baseline_gate.get("evaluation_mode") == "baseline", "baseline gate mode drift")
    require(baseline_gate.get("evidence_audit_passed") is True, "baseline evidence audit failed")
    require(sha256_file(baseline_gate_path) == table3.get("baseline_gate_sha256"), "baseline gate SHA mismatch")

    aggregate_stats = aggregate.get("aggregate_stats", {})
    require(aggregate.get("shard_count") == 45, "aggregate shard count drift")
    require(aggregate.get("all_input_gate_d_closures_passed") is True, "aggregate Gate D closure failed")
    require(aggregate.get("all_input_loader_validations_passed") is True, "aggregate loader validation failed")
    require(aggregate.get("cross_shard_source_overlap_count") == 0, "aggregate cross-shard source overlap")
    require(aggregate_stats.get("row_count") == 101395, "aggregate row count drift")
    require(states.get("status") == "passed" and states.get("shard_count") == 45, "state provenance aggregate failed")
    require(sum(states.get("state_source_counts", {}).values()) == states.get("event_count"), "state provenance counts do not close")
    require(execution.get("training_runtime", {}).get("old_custom_training_flow_used") is False, "custom optimizer loop was used")

    internal_by_step = {row["local_step"]: row for row in internal.get("checkpoints", [])}
    for step in training_steps:
        item = internal_by_step.get(step)
        if item is None:
            errors.append(f"internal-validation step {step} missing")
            continue
        require(item.get("continuation_checkpoint", {}).get("sha256") == training["checkpoints"][str(step)]["sha256"], f"internal step {step} checkpoint SHA mismatch")
        exact = item.get("exact_token_weighted_metrics", {})
        require(exact.get("profile") == "exact-per-row-target-weighted-v1", f"internal step {step} exact profile missing")
        require(math.isfinite(exact.get("objective", float("nan"))), f"internal step {step} objective non-finite")

    if errors:
        raise RuntimeError("report input audit failed:\n- " + "\n- ".join(errors))

    payload["baseline_gate"] = baseline_gate
    payload["optimizer_updates"] = updates
    payload["training_validations"] = training.get("validations", [])
    payload["training_steps"] = training_steps
    payload["evaluated_steps"] = evaluated_steps
    payload["coarse_to_fine"] = coarse_to_fine
    input_identities = {
        name: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for name, path in resolved.items()
    }
    renderer_path = Path(__file__).resolve(strict=True)
    input_identities["renderer"] = {
        "path": str(renderer_path),
        "sha256": sha256_file(renderer_path),
        "bytes": renderer_path.stat().st_size,
    }
    audit = {
        "schema_version": 1,
        "status": "passed",
        "generated_at_utc": utc_now(),
        "checks": {
            "official_lightning_training_complete": True,
            "old_custom_training_flow_used": False,
            "exact_optimizer_resume": False,
            "origin_step_estimate_confidence": "low",
            "pretrain_step0_no_training": True,
            "exact_internal_validation_complete": True,
            "table3_used_for_selection": False,
            "table3_baseline_reused_read_only": True,
            "table3_evaluated_steps": evaluated_steps,
            "table3_coarse_to_fine": coarse_to_fine,
            "all_table3_continuation_gates_passed": True,
            "baseline_evidence_audit_passed": True,
            "baseline_accuracy_gate_passed": baseline_gate.get("accuracy_gate_passed"),
            "source_conversation_leakage_count": 0,
            "aggregate_gate_d_closed": True,
        },
        "inputs": input_identities,
    }
    return payload, audit


def table3_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    sweep = payload["table3_index"]
    first = sweep["checkpoints"][0]
    baseline = {name: value["baseline_accuracy_percent"] for name, value in first["classes"].items()}
    rows = [{
        "step": 0,
        "estimated_total_step": payload["training_manifest"]["origin_step_estimate"],
        "en_complete": baseline["en/complete"], "en_incomplete": baseline["en/incomplete"],
        "zh_complete": baseline["zh/complete"], "zh_incomplete": baseline["zh/incomplete"],
        "en_macro": (baseline["en/complete"] + baseline["en/incomplete"]) / 2,
        "zh_macro": (baseline["zh/complete"] + baseline["zh/incomplete"]) / 2,
        "almost_unchanged": True, "decline_trigger": False, "decline_confirmed": False,
    }]
    for item in sweep["checkpoints"]:
        classes = item["classes"]
        rows.append({
            "step": item["local_step"], "estimated_total_step": item["estimated_total_optimizer_step"],
            "en_complete": classes["en/complete"]["candidate_accuracy_percent"],
            "en_incomplete": classes["en/incomplete"]["candidate_accuracy_percent"],
            "zh_complete": classes["zh/complete"]["candidate_accuracy_percent"],
            "zh_incomplete": classes["zh/incomplete"]["candidate_accuracy_percent"],
            "en_macro": item["languages"]["en"]["candidate_macro_accuracy_percent"],
            "zh_macro": item["languages"]["zh"]["candidate_macro_accuracy_percent"],
            "almost_unchanged": item["almost_unchanged"],
            "decline_trigger": item["obvious_decline_trigger"],
            "decline_confirmed": item["obvious_decline_confirmed"],
        })
    return rows


def internal_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    index = payload["internal_validation"]
    exact_rows = [(0, index["baseline"]["exact_token_weighted_metrics"])]
    exact_rows.extend((item["local_step"], item["exact_token_weighted_metrics"]) for item in index["checkpoints"])
    return [{"step": step, "objective": exact["objective"], "accuracy": exact["accuracy"], "heads": exact["heads"]} for step, exact in exact_rows]


def pp(value: float) -> str:
    return f"{100 * value:.2f}%"


def render_markdown(payload: dict[str, Any], audit: dict[str, Any]) -> str:
    training = payload["training_manifest"]
    execution = payload["execution_manifest"]
    aggregate = payload["aggregate_manifest"]["aggregate_stats"]
    states = payload["state_provenance"]
    split = payload["split_manifest"]
    internal = payload["internal_validation"]
    t3_rows = table3_rows(payload)
    i_rows = internal_rows(payload)
    selected = internal["selection"]["selected_local_step"]
    first_trigger = next((row for row in t3_rows if row["decline_trigger"]), None)
    first_confirmed = next((row for row in t3_rows if row["decline_confirmed"]), None)
    effective_hours = aggregate["exported_chunk_count"] * 0.16 / 3600
    final_recommendation = (
        "内部 validation 预注册规则保留官方发布模型（step 0）。"
        if selected == 0
        else f"内部 validation 预注册规则选择 local step {selected}；该选择在查看 Table 3 前完成。"
    )
    failed_training_attempt = next(
        (
            item
            for item in execution.get("training_attempts", [])
            if item.get("attempt") == 1
        ),
        None,
    )
    recovery_note = (
        "- 正式训练第一次尝试已完成 local step 10 及其 validation，但 Lightning/fsspec "
        "在提交原生 `last.ckpt` 时把事务临时文件写入系统 `/tmp`，因系统盘空间不足而失败；"
        "模型数值未失败，step 1/2/3/5/10 compact checkpoint 与日志完整归档。由于没有产生 "
        "optimizer/scheduler/global-step checkpoint，不能精确从 step 10 恢复，因此在训练配置、数据、"
        "模型和学习率均不变的前提下从发布权重重跑，并只将 `TMPDIR/TEMP/TMP` 改到数据盘。"
        if failed_training_attempt
        else "- 未记录正式训练基础设施重试。"
    )
    if payload["coarse_to_fine"]:
        table3_protocol_note = (
            "评测使用 Easy Turn Testset（Table 3）和预注册的 endpoint-first "
            "coarse-to-fine-v1 协议：固定粗测顺序为 1→30→10→5→20；仅当 step 5 "
            "触发明显下降时补测 2、3。该规则不假设性能单调，且只决定追加哪些评测点，"
            "不参与 checkpoint 选择。`last-terminal-v1`、EN/ZH ASR、样本顺序、阈值、"
            "规则和逐样本配对统计保持不变。Full-Duplex-Bench（Table 2）不在本阶段范围内。"
        )
        table3_result_heading = "### 5.2 粗测—细化 checkpoint 结果"
        table3_integrity_note = (
            "- 训练与内部 validation 的七点网格保持冻结；Table 3 在查看 step 1 汇总结果前"
            "改为预注册粗测—细化协议。固定粗测点全部评测，2/3 是否追加只由 step 5 的"
            "冻结明显下降规则决定；没有按论文目标或候选准确率任意删点。若 2/3 未被触发而"
            "省略，仍存在漏掉局部窄幅非单调波动的限制，不能把未测点表述为性能不变。"
        )
    else:
        table3_protocol_note = (
            "评测使用 Easy Turn Testset（Table 3），固定 `last-terminal-v1`、相同 EN/ZH "
            "ASR、样本顺序、阈值、规则与逐样本配对统计。Full-Duplex-Bench（Table 2）"
            "不在本阶段评测范围内。"
        )
        table3_result_heading = "### 5.2 七个 checkpoint 结果"
        table3_integrity_note = (
            "- 七点网格、会话级 split、学习率、标签、Table 3 样本/规则/阈值在结果前冻结。"
            "所有点均评测，未按结果删除 checkpoint。"
        )
    lines = [
        "# DuplexConv Edu_0001–Edu_0045：SoulX Stage 3 官方流程续训练与评估", "",
        f"生成时间：{audit['generated_at_utc']}  ",
        "状态：正式会议汇报报告；训练、内部 validation 与 Table 3 证据审计通过。", "",
        "## 1. 结论摘要", "", f"- {final_recommendation}",
        f"- Table 3 首次明显下降触发：{('local step ' + str(first_trigger['step'])) if first_trigger else '未出现'}；相邻预注册点确认：{('local step ' + str(first_confirmed['step'])) if first_confirmed else '未确认'}。",
        "- Table 3 只用于训练后独立性能描述，没有用于选择 checkpoint、学习率、样本、标签、阈值或解码规则。",
        "- 官方发布权重不含 optimizer、scheduler、GradScaler 和 Trainer global step；起点 1800 只是依据公开 Stage 3 配置的低置信度估计，不是精确断点恢复。", "",
        "## 2. 续训练数据集", "",
        "数据来自 DuplexConv 中文教育会话 `Edu_0001`–`Edu_0045`。多轨会话被展开为说话人视角：每条训练序列只放目标声道的 audio token，其他声道仅提供正在说话、重叠和 backchannel 等关系信息；不混音，也不把多路 audio token 塞进同一序列。", "",
        f"- 45 个冻结 shard；{aggregate['source_conversation_count']:,} 个源会话；{aggregate['source_view_count']:,} 个目标视角。",
        f"- {aggregate['row_count']:,} 个训练窗口；{aggregate['exported_chunk_count']:,} 个有效 160 ms chunk；按目标视角 chunk 计约 {effective_hours:.3f} 小时。该值是训练暴露时长，不等同于去重后的原始录音时长。",
        f"- 多轨行分布：2 轨 {aggregate['rows_by_ntrack']['2']:,}，3 轨 {aggregate['rows_by_ntrack']['3']:,}，4 轨 {aggregate['rows_by_ntrack']['4']:,}。",
        f"- 隔离：窗口级 {aggregate['quarantined_chunk_count']:,} chunks；源视角级 {aggregate['source_view_quarantined_count']:,} views/{aggregate['source_view_quarantined_chunk_count']:,} chunks；最终跨 shard source/index/view 重叠均为 0，45 个 Gate D 均通过。",
        "- 目标声道文本与时间戳由固定 Paraformer 生成；GLM-4-Voice tokenizer 每 160 ms chunk 生成两个 audio token。", "",
        "### 2.1 状态标签处理", "",
        "SoulX Stage 3 使用 `idle`、`nonidle`、`backchannel`、`complete`、`incomplete` 五类状态 token。DuplexConv 官方给出的终态标签保留原义；`WAIT` 确定性映射为 `complete`；官方缺失的终态由固定 `Qwen3-235B-A22B-Instruct-2507` 通过 OpenRouter 补标。Qwen 结果有独立 provenance，不表述为人工 gold。", "",
        f"- 终态事件 {states['event_count']:,}：官方标签 {states['state_source_counts']['duplexconv_official_llm_assisted']:,}，WAIT→Complete {states['state_source_counts']['deterministic_wait_to_complete']:,}，Qwen 补标 {states['state_source_counts']['openrouter_qwen3_235b_a22b_instruct_2507']:,}。",
        f"- 最终事件分布：Complete {states['final_state_distribution']['complete']:,}，Incomplete {states['final_state_distribution']['incomplete']:,}，Backchannel {states['final_state_distribution']['backchannel']:,}。",
        f"- Qwen 共 {states['qwen']['request_count']:,} 个请求，accepted-response 累计成本 {states['qwen']['accepted_response_usage']['cost']:.6f} USD；这是 45 个 shard 跨多个处理日的累计值，不是本次训练或当天费用。训练与评测 API 费用为 0。", "",
        "### 2.2 冻结划分", "",
        f"按完整源会话进行 seed=42 的 98/2 划分：Train {split['train']['source_conversation_count']:,} 会话/{split['train']['row_count']:,} rows/{split['train']['duration_hours_from_160ms_chunks']:.3f}h；Validation {split['validation']['source_conversation_count']:,} 会话/{split['validation']['row_count']:,} rows/{split['validation']['duration_hours_from_160ms_chunks']:.3f}h；source leakage={split['source_leakage_count']}。",
        f"split identity：`{split['split_identity_sha256']}`。", "",
        "## 3. 官方续训练方法", "",
        f"官方上游 commit 固定为 `{training['runtime_base_commit']}`。训练入口是 `{training['training_flow']['entrypoint']}` → Lightning `Trainer.fit()` → `training_step()` → `configure_optimizers()`；没有使用先前自定义 optimizer loop。", "",
        "补丁仅包括：全 `-100` 空 head 的有限零损失修复、严格基础权重加载/初始化顺序、冻结会话级 split 审计、scheduler 起点偏移、训练审计与 compact checkpoint。官方上游目录保持 clean，补丁只存在于独立 runtime 副本。", "",
        f"- 基础权重 SHA-256：`{training['base_checkpoint']['sha256']}`。",
        f"- local batch=1，梯度累积={training['trainer']['accumulate_grad_batches']}，单卡 world size={training['trainer']['world_size']}，有效 batch=576。",
        f"- AdamW；weight decay={training['optimizer']['weight_decay']}；scheduler=`{training['scheduler']['class']}`；续训起始 LR={training['optimizer']['lr_at_fit_start']:.12f}。",
        f"- local step 1–30 对应估计总 step 1801–1830；总暴露 {training['total_samples']:,} 个样本，约占首个 epoch 的 17.4%，没有循环读取。",
        f"- 可训练参数 {training['trainable_parameter_count']:,}/{training['total_parameter_count']:,}，{training['trainable_tensor_count']} 个可训练张量。", "",
        "## 4. 训练域精确内部 validation", "",
        "选择规则在 Table 3 前冻结：指标必须有限；五个状态 head 的精确 token-weighted accuracy 相比 step 0 均不得下降超过 5 pp；合格点中选精确 token-weighted objective 最低者，1% 内并列取更早 step；若无人合格则保留 step 0。", "",
        "| Local step | 估计总 step | Token-weighted objective | 总 accuracy | Idle | Non-idle | Backchannel | Complete | Incomplete | Guard |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    selection_by_step = {row["local_step"]: row for row in internal["selection"]["candidates"]}
    for row in i_rows:
        step = row["step"]
        heads = row["heads"]
        guard = "基线" if step == 0 else ("通过" if selection_by_step[step]["guard_passed"] else "拒绝")
        lines.append(f"| {step} | {1800 + step} | {row['objective']:.6f} | {pp(row['accuracy'])} | {pp(heads['idle']['token_weighted_accuracy'])} | {pp(heads['nonidle']['token_weighted_accuracy'])} | {pp(heads['user_backchannel']['token_weighted_accuracy'])} | {pp(heads['user_complete']['token_weighted_accuracy'])} | {pp(heads['user_incomplete']['token_weighted_accuracy'])} | {guard} |")
    lines.extend(["", f"预注册内部选择结果：local step **{selected}**。原因：{internal['selection']['reason']}。", "",
        "说明：训练审计中的 `mean_training_step_loss` 是 Lightning 在梯度累积语义下返回的缩放 loss，约为未缩放值的 1/576，不用于 checkpoint 选择；上表由每行 loss×有效 token 数重新聚合，才是精确 token-weighted 指标。", "",
        "## 5. Table 3 冻结评测", "",
        table3_protocol_note, "",
        "### 5.1 论文值与本机官方权重基线", "",
        "| 类别 | 论文正确数/总数 | 论文准确率 | 本机正确数/总数 | 本机准确率 | 差值 | ±1 pp |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for check in payload["baseline_gate"]["checks"]:
        lines.append(f"| {check['language'].upper()} {check['label'].title()} | {check['paper_correct']}/{check['total']} | {check['paper_accuracy_percent']:.2f}% | {check['actual_correct']}/{check['total']} | {check['accuracy_percent']:.2f}% | {check['delta_percentage_points']:+.2f} pp | {'通过' if check['within_one_percentage_point'] else '未通过'} |")
    lines.extend(["", f"原 ±1 pp 准确率门禁实际为 `{'PASSED' if payload['baseline_gate']['accuracy_gate_passed'] else 'FAILED'}`；证据完整性审计为 PASSED。失败结果如实保留，本机 step 0 只能作为冻结配对基线，不表述为严格复现通过。", "",
        table3_result_heading, "",
        "| Local step | 估计总 step | EN C | EN I | EN Macro | ZH C | ZH I | ZH Macro | 判定 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in t3_rows:
        if row["step"] == 0: status = "官方权重基线"
        elif row["decline_confirmed"]: status = "明显下降（相邻点确认）"
        elif row["decline_trigger"]: status = "明显下降触发"
        elif row["almost_unchanged"]: status = "几乎未下降"
        else: status = "不均衡变化"
        lines.append(f"| {row['step']} | {row['estimated_total_step']} | {row['en_complete']:.2f}% | {row['en_incomplete']:.2f}% | {row['en_macro']:.2f}% | {row['zh_complete']:.2f}% | {row['zh_incomplete']:.2f}% | {row['zh_macro']:.2f}% | {status} |")
    lines.extend(["", "Table 3 的稳定/下降定义由冻结统计索引给出：两种语言 macro 均不低于基线 1 pp 且四类均不低于 2 pp，才称为“几乎未下降”；任一语言 macro 下降超过 3 pp 或任一类下降超过 5 pp 触发“明显下降”，且需下一个预注册点继续触发才确认。", "",
        "## 6. 实验完整性与限制", "",
        "- 官方上游和 Table 3 runtime 均固定 commit 且保持 clean；训练补丁副本与评测副本隔离。",
        table3_integrity_note,
        "- 第一次 step 0 尝试因 Hugging Face cache 默认写系统盘而在验证前失败；未上 GPU、未创建 optimizer、未训练。临时缓存随后改道数据盘，失败日志完整保留。",
        recovery_note,
        "- 起点 1800 是低置信度估计；这次实验不能回答恢复官方 optimizer 状态后的精确续训轨迹。",
        "- 本报告只覆盖训练域 validation 与论文 Table 3；未把 Table 2 Full-Duplex-Bench 结果混入结论。",
        "- 训练有效时长按目标视角统计，同一源会话的不同目标声道会重复贡献会话时长，因此不能写成去重后的原始录音小时数。", "",
        "## 7. 关键产物", "",
        f"- 训练 manifest：`{audit['inputs']['training_manifest']['path']}`",
        f"- 精确内部 validation：`{audit['inputs']['internal_validation']['path']}`",
        f"- Table 3 index：`{audit['inputs']['table3_index']['path']}`",
        f"- 数据聚合 manifest：`{audit['inputs']['aggregate_manifest']['path']}`",
        f"- 状态 provenance：`{audit['inputs']['state_provenance']['path']}`", "",
        f"审计状态：`{audit['status']}`。",
    ])
    return "\n".join(lines) + "\n"


def render_html(payload: dict[str, Any], audit: dict[str, Any]) -> str:
    aggregate = payload["aggregate_manifest"]["aggregate_stats"]
    states = payload["state_provenance"]
    internal = payload["internal_validation"]
    t3_rows = table3_rows(payload)
    i_rows = internal_rows(payload)
    selection_by_step = {row["local_step"]: row for row in internal["selection"]["candidates"]}
    for row in i_rows:
        row["guard"] = None if row["step"] == 0 else selection_by_step[row["step"]]["guard_passed"]
    data = {
        "generated_at": audit["generated_at_utc"], "audit_status": audit["status"],
        "dataset": {"sources": aggregate["source_conversation_count"], "views": aggregate["source_view_count"], "rows": aggregate["row_count"], "hours": aggregate["exported_chunk_count"] * 0.16 / 3600, "events": states["event_count"], "qwen_cost": states["qwen"]["accepted_response_usage"]["cost"]},
        "selected_step": internal["selection"]["selected_local_step"], "table3": t3_rows, "internal": i_rows,
        "paper": payload["baseline_gate"]["checks"], "baseline_accuracy_gate_passed": payload["baseline_gate"]["accuracy_gate_passed"],
        "coarse_to_fine": payload["coarse_to_fine"],
        "omitted_steps": payload["table3_orchestration"].get("omitted_steps", []),
    }
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    template = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SoulX Stage 3 续训练评估</title><style>
:root{--bg:#07101f;--panel:#101c31;--panel2:#152540;--line:#2b4164;--text:#eef4ff;--muted:#9db0ce;--cyan:#58d9ff;--green:#63e6ad;--red:#ff758b;--amber:#ffd166}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 10% 0,#162a4b 0,#07101f 44%);color:var(--text);font:14px/1.55 Inter,system-ui,-apple-system,"Segoe UI",sans-serif}.wrap{max-width:1320px;margin:auto;padding:30px 22px 70px}h1{font-size:31px;margin:0 0 4px}.sub{color:var(--muted)}h2{margin:31px 0 12px;font-size:20px}.cards{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:22px 0}.card,.panel{background:linear-gradient(145deg,var(--panel),var(--panel2));border:1px solid var(--line);border-radius:14px;box-shadow:0 14px 34px #0005}.card{padding:15px}.k{text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-size:11px}.v{font-size:23px;font-weight:750;margin-top:4px}.charts{display:grid;grid-template-columns:1fr 1fr;gap:14px}.panel{padding:16px}.chart{height:300px}.legend{display:flex;flex-wrap:wrap;gap:14px;color:var(--muted);font-size:12px}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}svg{width:100%;height:255px;overflow:visible}.axis{stroke:#5f759c}.grid{stroke:#263a5c}.label{fill:#9db0ce;font-size:10px}.series{fill:none;stroke-width:2.6}.point{stroke:#081122;stroke-width:2}table{width:100%;border-collapse:collapse;background:#0e192c}th,td{padding:9px;border-bottom:1px solid #263a5c;text-align:right;white-space:nowrap}th{background:#172640;color:#b9c8df}th:first-child,td:first-child{text-align:left}.tablewrap{overflow:auto}.tag{padding:3px 8px;border-radius:999px;background:#263a5c}.good{color:var(--green)}.bad{color:var(--red)}.warn{color:var(--amber)}.callout{margin:14px 0;padding:12px 15px;border-left:4px solid var(--amber);background:#2b2518;border-radius:8px}.foot{margin-top:25px;color:var(--muted);font-size:12px}@media(max-width:950px){.cards{grid-template-columns:repeat(2,1fr)}.charts{grid-template-columns:1fr}}@media(max-width:520px){.cards{grid-template-columns:1fr}}
</style></head><body><main class="wrap"><h1>DuplexConv × SoulX Stage 3</h1><div class="sub">Edu_0001–Edu_0045 · 官方 Lightning · 精确内部 validation · Table 3</div><section id="cards" class="cards"></section><div id="gate" class="callout"></div><section class="charts"><div class="panel"><h2>Table 3 语言宏准确率</h2><div class="legend"><span><i class="dot" style="background:#58d9ff"></i>EN Macro</span><span><i class="dot" style="background:#63e6ad"></i>ZH Macro</span></div><div id="macro" class="chart"></div></div><div class="panel"><h2>Table 3 Complete / Incomplete</h2><div class="legend"><span><i class="dot" style="background:#58d9ff"></i>EN C</span><span><i class="dot" style="background:#ff758b"></i>EN I</span><span><i class="dot" style="background:#63e6ad"></i>ZH C</span><span><i class="dot" style="background:#ffd166"></i>ZH I</span></div><div id="classes" class="chart"></div></div></section><h2>精确 token-weighted 内部 validation</h2><section class="charts"><div class="panel"><div id="objective" class="chart"></div></div><div class="panel"><div id="states" class="chart"></div></div></section><h2>Table 3 明细</h2><div class="panel tablewrap"><table><thead><tr><th>Step</th><th>估计总 step</th><th>EN C</th><th>EN I</th><th>EN Macro</th><th>ZH C</th><th>ZH I</th><th>ZH Macro</th><th>状态</th></tr></thead><tbody id="t3table"></tbody></table></div><h2>内部 validation 明细</h2><div class="panel tablewrap"><table><thead><tr><th>Step</th><th>Objective</th><th>总体 acc</th><th>Idle</th><th>Non-idle</th><th>Backchannel</th><th>Complete</th><th>Incomplete</th><th>Guard</th></tr></thead><tbody id="itable"></tbody></table></div><h2>论文值与本机基线</h2><div class="panel tablewrap"><table><thead><tr><th>类别</th><th>论文</th><th>本机</th><th>差值</th><th>±1 pp</th></tr></thead><tbody id="paper"></tbody></table></div><div class="foot" id="foot"></div></main><script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),T=D.table3,I=D.internal,F=n=>Number(n).toFixed(2)+'%',P=n=>(100*n).toFixed(2)+'%';const trigger=T.find(x=>x.decline_trigger),confirmed=T.find(x=>x.decline_confirmed);document.getElementById('cards').innerHTML=[['源会话',D.dataset.sources.toLocaleString(),'' ],['训练暴露',D.dataset.hours.toFixed(1)+' h',''],['状态事件',D.dataset.events.toLocaleString(),'' ],['内部选择','step '+D.selected_step,'good'],['首次下降',trigger?'step '+trigger.step:'无',trigger?'warn':'good']].map(x=>`<div class="card"><div class="k">${x[0]}</div><div class="v ${x[2]}">${x[1]}</div></div>`).join('');const coarseNote=D.coarse_to_fine?` 采用预注册粗测—细化顺序；省略点：${D.omitted_steps.length?D.omitted_steps.join('/'):'无'}。未测点不能表述为性能不变，稀疏网格可能漏掉局部窄幅非单调波动。`:'';document.getElementById('gate').innerHTML=`<b>实验边界：</b>内部 validation 在 Table 3 前选择 step ${D.selected_step}；Table 3 不参与选点。官方模型 ±1 pp 准确率门禁为 <span class="${D.baseline_accuracy_gate_passed?'good':'bad'}">${D.baseline_accuracy_gate_passed?'PASSED':'FAILED'}</span>，证据审计仍为 PASSED。${confirmed?'明显下降由 step '+confirmed.step+' 确认。':''}${coarseNote}`;
function chart(id,rows,series,percent=true){const W=610,H=250,p={l:52,r:18,t:15,b:34},vals=rows.flatMap(r=>series.map(s=>r[s.key])).filter(Number.isFinite),lo=Math.min(...vals),hi=Math.max(...vals),pad=Math.max((hi-lo)*.12,percent?1:.01),ymin=lo-pad,ymax=hi+pad,x=i=>p.l+(rows.length===1?0:(i*(W-p.l-p.r)/(rows.length-1))),y=v=>p.t+(ymax-v)*(H-p.t-p.b)/(ymax-ymin);let q=`<svg viewBox="0 0 ${W} ${H}">`;for(let j=0;j<5;j++){const v=ymin+j*(ymax-ymin)/4,yy=y(v);q+=`<line class="grid" x1="${p.l}" y1="${yy}" x2="${W-p.r}" y2="${yy}"/><text class="label" x="${p.l-7}" y="${yy+4}" text-anchor="end">${percent?v.toFixed(1):v.toFixed(3)}</text>`}rows.forEach((r,i)=>q+=`<text class="label" x="${x(i)}" y="${H-9}" text-anchor="middle">${r.step}</text>`);series.forEach(s=>{q+=`<polyline class="series" stroke="${s.color}" points="${rows.map((r,i)=>x(i)+','+y(r[s.key])).join(' ')}"/>`;rows.forEach((r,i)=>q+=`<circle class="point" fill="${s.color}" cx="${x(i)}" cy="${y(r[s.key])}" r="4"><title>step ${r.step} ${s.name}: ${r[s.key]}</title></circle>`)});document.getElementById(id).innerHTML=q+'</svg>'}
chart('macro',T,[{key:'en_macro',name:'EN Macro',color:'#58d9ff'},{key:'zh_macro',name:'ZH Macro',color:'#63e6ad'}]);chart('classes',T,[{key:'en_complete',name:'EN C',color:'#58d9ff'},{key:'en_incomplete',name:'EN I',color:'#ff758b'},{key:'zh_complete',name:'ZH C',color:'#63e6ad'},{key:'zh_incomplete',name:'ZH I',color:'#ffd166'}]);chart('objective',I.map(r=>({step:r.step,objective:r.objective})),[{key:'objective',name:'objective',color:'#ffd166'}],false);const IS=I.map(r=>({step:r.step,idle:100*r.heads.idle.token_weighted_accuracy,nonidle:100*r.heads.nonidle.token_weighted_accuracy,backchannel:100*r.heads.user_backchannel.token_weighted_accuracy,complete:100*r.heads.user_complete.token_weighted_accuracy,incomplete:100*r.heads.user_incomplete.token_weighted_accuracy}));chart('states',IS,[{key:'idle',name:'idle',color:'#58d9ff'},{key:'nonidle',name:'nonidle',color:'#b58cff'},{key:'backchannel',name:'backchannel',color:'#ffd166'},{key:'complete',name:'complete',color:'#63e6ad'},{key:'incomplete',name:'incomplete',color:'#ff758b'}]);
document.getElementById('t3table').innerHTML=T.map(r=>{let s=r.step===0?['官方基线','good']:r.decline_confirmed?['下降确认','bad']:r.decline_trigger?['下降触发','bad']:r.almost_unchanged?['几乎未下降','good']:['不均衡','warn'];return `<tr><td>${r.step}</td><td>${r.estimated_total_step}</td><td>${F(r.en_complete)}</td><td>${F(r.en_incomplete)}</td><td>${F(r.en_macro)}</td><td>${F(r.zh_complete)}</td><td>${F(r.zh_incomplete)}</td><td>${F(r.zh_macro)}</td><td><span class="tag ${s[1]}">${s[0]}</span></td></tr>`}).join('');document.getElementById('itable').innerHTML=I.map(r=>{const h=r.heads,g=r.step===0?['基线','']:r.guard?['通过','good']:['拒绝','bad'];return `<tr><td>${r.step}</td><td>${r.objective.toFixed(6)}</td><td>${P(r.accuracy)}</td><td>${P(h.idle.token_weighted_accuracy)}</td><td>${P(h.nonidle.token_weighted_accuracy)}</td><td>${P(h.user_backchannel.token_weighted_accuracy)}</td><td>${P(h.user_complete.token_weighted_accuracy)}</td><td>${P(h.user_incomplete.token_weighted_accuracy)}</td><td><span class="tag ${g[1]}">${g[0]}</span></td></tr>`}).join('');document.getElementById('paper').innerHTML=D.paper.map(x=>`<tr><td>${x.language.toUpperCase()} ${x.label}</td><td>${x.paper_correct}/${x.total} (${F(x.paper_accuracy_percent)})</td><td>${x.actual_correct}/${x.total} (${F(x.accuracy_percent)})</td><td>${x.delta_percentage_points>=0?'+':''}${x.delta_percentage_points.toFixed(2)} pp</td><td><span class="tag ${x.within_one_percentage_point?'good':'bad'}">${x.within_one_percentage_point?'通过':'未通过'}</span></td></tr>`).join('');document.getElementById('foot').textContent=`生成时间 ${D.generated_at} · audit ${D.audit_status} · 训练/评测 API 费用 $0 · 数据构造 accepted-response 累计 Qwen 成本 $${D.dataset.qwen_cost.toFixed(6)}`;
</script></body></html>"""
    return template.replace("__DATA__", encoded)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training_manifest", "pretrain_step0", "internal_validation", "posttrain_orchestration", "table3_orchestration", "table3_index", "aggregate_manifest", "state_provenance", "split_manifest", "execution_manifest"):
        parser.add_argument("--" + name.replace("_", "-"), dest=name, type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_paths = {name: getattr(args, name) for name in ("training_manifest", "pretrain_step0", "internal_validation", "posttrain_orchestration", "table3_orchestration", "table3_index", "aggregate_manifest", "state_provenance", "split_manifest", "execution_manifest")}
    payload, audit = audit_inputs(input_paths)
    markdown = render_markdown(payload, audit)
    dashboard = render_html(payload, audit)
    audit["outputs"] = {
        "markdown": {"path": str(args.output_md.absolute()), "sha256": sha256_text(markdown)},
        "html": {"path": str(args.output_html.absolute()), "sha256": sha256_text(dashboard)},
    }
    atomic_write(args.output_md.absolute(), markdown)
    atomic_write(args.output_html.absolute(), dashboard)
    atomic_write(args.output_audit.absolute(), json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": "complete", "outputs": {"markdown": str(args.output_md.absolute()), "html": str(args.output_html.absolute()), "audit": str(args.output_audit.absolute())}}, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
