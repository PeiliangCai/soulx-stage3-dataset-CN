# DuplexConv Edu_0018：SoulX Stage 3 官方流程续训练与 Table 3 评估

生成时间：2026-08-22T10:14:22.161640+00:00  
状态：正式报告；官方 Lightning 训练与 step 0/1/2/3/5 Table 3 证据审计通过。

## 1. 汇报结论

本轮证明了小规模 DuplexConv 中文续训练能够显著改变 SoulX 的终态决策，但没有得到一个同时保持 Complete 与 Incomplete 平衡的续训练 checkpoint。Complete 持续提高，Incomplete 同时下降，说明模型发生了决策偏置迁移，不能将 validation loss 或 Complete 单项上涨表述为整体性能提升。

- 最早的明显下降触发：local step 2。
- 由相邻 checkpoint 确认的下降点：local step 2。
- 满足预注册“几乎未下降”条件的续训练点：无。
- 推荐交付模型：若目标是保持中英文 Complete/Incomplete 的平衡通用能力，仍推荐官方发布模型（local step 0）；续训练 checkpoint 只适合继续研究，不应替代官方模型。

## 2. 续训练数据

数据来自 DuplexConv `Edu_0018`：500 个同步多轨中文教育会话，495 个双声道、5 个三声道。每个声道分别作为目标说话人视角，其他声道只提供 activity/overlap 关系证据，不混音、不在一个 sequence 中放入多路 audio token。

- 去重源会话时长：约 10.519 小时；target-view 时长：约 21.170 小时。
- model-ready：1,005 views、2,168 rows、474,030 可用 chunks、2,736 隔离 chunks。
- split：Train 475 会话/2,066 rows；Validation 25 会话/102 rows；source leakage=0。
- 状态事件：8,505；官方三状态 6,895，WAIT→Complete 11，Qwen 补标 1,599。
- Qwen：`Qwen3-235B-A22B-Instruct-2507`，404 个会话级请求，accepted-response cost=0.218779 USD。
- 文本与时间戳由固定 Paraformer 从目标声道音频生成；GLM-4-Voice tokenizer 为每个 160 ms chunk 生成两个 audio token。官方/Qwen/WAIT 标签均保留独立 provenance，LLM 标签不称为人工 gold。

## 3. 官方续训练起点与方法

官方发布权重不含 optimizer、scheduler、GradScaler 或 Trainer global step，因此只能从模型权重继续训练，不能精确恢复官方 optimizer 现场。公开 Stage 3 配置的 `total_steps=1800` 被用作低置信度起点估计；本轮 local step 1–5 对应估计总 step 1801–1805。

- 官方上游基准 commit：`928b06508ed2de1344208d06fb1f6fb2ebfb1df5`；上游保持 clean，补丁位于独立 runtime。
- 训练入口：`finetune.py::train` → 官方 Lightning `Trainer.fit()` / `training_step()` / `configure_optimizers()`。
- 单卡 microbatch=1，梯度累积=576，名义有效 batch=576。
- AdamW + `utils.sparkvox.utils.scheduler.WarmupAnnealSteps`；起始 continuation LR=0.000033324078。
- 实际样本更新量：576/576/576/338/576；step 4 是官方 Lightning epoch-tail 更新，没有人工循环重复补齐。
- 可训练参数 13,505,536，共 118 tensors；峰值 CUDA allocated=17.883 GiB。

## 4. Table 3 正式结果

评测使用同一冻结 `last-terminal-v1` 候选协议、相同样本顺序、相同 ASR、相同推理代码和逐样本配对统计；训练 checkpoint 没有参与规则或样本选择。

### 4.1 论文值与本机官方权重基线

| 类别 | 论文正确数/总数 | 论文准确率 | 本机正确数/总数 | 本机准确率 | 差值 | ±1 pp 门禁 |
|---|---:|---:|---:|---:|---:|---|
| EN Complete | 247/318 | 77.67% | 251/318 | 78.93% | +1.26 pp | 未通过 |
| EN Incomplete | 266/299 | 88.96% | 268/299 | 89.63% | +0.67 pp | 通过 |
| ZH Complete | 268/300 | 89.33% | 263/300 | 87.67% | -1.67 pp | 未通过 |
| ZH Incomplete | 238/300 | 79.33% | 241/300 | 80.33% | +1.00 pp | 通过 |

原预定义的四类均在 ±1 pp 内这一机器门禁实际为 `FAILED`（EN Complete 与 ZH Complete 超界），不能将本机 baseline 写成严格复现通过。项目负责人在获知差异和 `last-terminal-v1` 尚未经作者确认后，仅授权把它作为本项目内部的冻结配对 step 0；后续 checkpoint 都与这个固定 step 0 比较。

### 4.2 续训练 checkpoint 配对结果

| Local step | 估计总 step | EN C | EN I | EN Macro | ZH C | ZH I | ZH Macro | 判定 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 1800 | 78.93% | 89.63% | 84.28% | 87.67% | 80.33% | 84.00% | 官方发布权重基线 |
| 1 | 1801 | 85.22% | 84.95% | 85.08% | 89.67% | 77.33% | 83.50% | 不均衡变化 |
| 2 | 1802 | 91.51% | 82.61% | 87.06% | 91.00% | 72.67% | 81.83% | 明显下降（由下一点确认） |
| 3 | 1803 | 96.23% | 74.58% | 85.40% | 90.67% | 69.67% | 80.17% | 明显下降（由下一点确认） |
| 5 | 1805 | 98.43% | 61.54% | 79.98% | 90.67% | 62.67% | 76.67% | 明显下降触发 |

预注册判据：EN/ZH macro 均不低于基线 1 pp 且任一类不低于 2 pp 才算“几乎未下降”；任一语言 macro 下降超过 3 pp 或任一类下降超过 5 pp 为触发，并需下一 checkpoint 再次触发才能确认。索引中的 `decline_confirmed` 标在被下一点确认的前一 checkpoint 上。

## 5. 官方 validation 与外部 benchmark 的区别

| Step | Validation loss | 官方 val_acc | Idle | Non-idle | Complete | Incomplete | Backchannel |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.427645 | 55.279% | 95.955% | 86.224% | 28.599% | 19.091% | 18.142% |
| 1 | 2.996503 | 55.212% | 95.658% | 84.420% | 30.649% | 19.091% | 17.701% |
| 2 | 2.582847 | 55.525% | 94.990% | 81.914% | 32.698% | 23.333% | 16.966% |
| 3 | 2.230164 | 55.859% | 94.275% | 78.481% | 36.798% | 26.061% | 16.493% |
| 5 | 1.862548 | 59.029% | 91.870% | 69.352% | 58.881% | 39.697% | 14.512% |

`val_acc` 是官方七 head 汇总口径，不等于 Table 3 四分类宏准确率。validation loss 下降与 Table 3 Incomplete 退化同时发生，说明训练域 validation 不能替代外部 benchmark。

## 6. 实验诚信与可复现性

- Table 3 运行前冻结样本、规则、尾静音、ASR、checkpoint 网格和推理 commit。
- `selection_used_paper_targets=false`；没有根据论文目标或当前结果改读出规则。
- baseline 与所有 continuation gate 均通过逐样本 evidence audit；每个 checkpoint 恰含四类完整结果。
- 训练/validation source-conversation leakage=0；运行配置与输入审计确认 Table 3 和 Full-Duplex-Bench 文件未被直接用于训练、补标或规则选择。跨语料 PCM/近重复排查尚未覆盖未下载的英文 Full-Duplex-Bench，因此不宣称已完成全 benchmark 音频级零泄漏证明。
- 官方发布模型的本机候选基线与论文基本接近，但 last-terminal 样本级协议仍未获作者确认，不能写成作者官方评测脚本。
- 旧自定义 optimizer-loop 结果只保留为 pilot 对照，不与本报告的官方 Lightning 结果混算。

## 7. 限制与下一步

1. `Edu_0018` 只有约 10.519 个去重源小时，规模太小，且状态分布不足以支撑泛化结论。
2. 起点 1800 来自公开配置，是低置信度估计；本轮不是官方 optimizer 精确 resume。
3. 当前只完成 Table 3 模型级 Easy Turn；没有将续训练点放行到 Full-Duplex-Bench 系统级评测。
4. 下一阶段扩大 DuplexConv 数据规模，并建立独立内部开发集；Table 3 继续锁定为最终外部测试，不能用于 shard、标签、LR 或处理规则选择。扩展数据放行前还需取得并固定英文 Full-Duplex-Bench，以补齐跨语料音频级泄漏门禁。

## 8. 证据身份

- `training_manifest`：`927f369cd122c5a5e7eece11f23bd54f4a39025793810747bf998378d3cacbe7` — `/root/SoulX-stage3-dataset/checkpoints/duplexconv_edu0018_official_continual_v1/run_manifest.json`
- `optimizer_updates`：`1464020840686e44408bc76c098c434d64e92c9cbaa486a34d7fcaaafdb4c793` — `/root/SoulX-stage3-dataset/checkpoints/duplexconv_edu0018_official_continual_v1/optimizer_updates.jsonl`
- `validation_metrics`：`d613a0bbfcbb51c974ceb9267d6fba897cb670442d08586948dd31e21a117f6d` — `/root/SoulX-stage3-dataset/checkpoints/duplexconv_edu0018_official_continual_v1/validation_metrics.jsonl`
- `group_validation_step0`：`5cca4646650bef4d2acf35c144dbb6f1b7d213669bdf377806abe829f72422a2` — `/root/autodl-tmp/dataset/duplexconv/work/official_group_validation_step0_v1/result.json`
- `orchestration_manifest`：`54130270361aa9858f7fb073132e5e63c46bf425435e9643783753d8bd125081` — `/root/autodl-tmp/dataset/soulx_duplug_eval/table3_official_continual_sweep/official_lightning_v1_b17bcf/orchestration_manifest.json`
- `sweep_index`：`9c192fd0e7fda41ae1499fb41bab11d53cd849d07f9fb78649a2d48d626bcd5c` — `/root/autodl-tmp/dataset/soulx_duplug_eval/table3_official_continual_sweep/official_lightning_v1_b17bcf/sweep_index.json`
- `state_summary`：`9723f05afd4b4b13ee3f9681c4d1acbc90130a8f009489bc6128a4c85d688167` — `/root/autodl-tmp/dataset/duplexconv/work/state_labels_v1/summary.json`
- `model_ready_stats`：`502ff369d14cf2d1d77122afdaa8b1e500bf3794455e8305fb5fd5d51d03b63b` — `/root/autodl-tmp/dataset/duplexconv/model_ready/edu0018_stage3_zh_v1/stats.json`
- `split_manifest`：`b4874ccf3e263ec4bf56257f5850e676b66528647f06407caf9a3ff0819d6210` — `/root/autodl-tmp/dataset/duplexconv/splits/edu0018_stage3_zh_v1_seed42_group95_5/split_manifest.json`
- `report_renderer`：`c2326f3505605700ff44f294e75d1abd7ce4518bdf3ccd45ad8ca048a0aa0145` — `/root/SoulX-stage3-dataset/scripts/render_official_continual_report.py`
- `group_validation_runner`：`e22fab0e80ed19c6eb4d1377c1292a4254f83f8ceb850fcca9af37fa519a6ed4` — `/root/SoulX-stage3-dataset/scripts/run_official_group_validation.py`

报告审计状态：`passed`。所有输入在生成时重新计算 SHA-256。
