# SoulX Table 3 候选复现反作假与数据选择审计

审计日期：2026-08-21

审计对象：`formal-candidate-v1-ac8fcf1` 四类全量结果、生成这些结果的项目 commit `ac8fcf1`、官方 training-code commit `928b065`、官方权重、固定 EN/ZH Easy Turn 资产及严格 gate。

## 1. 审计结论

未发现为提高准确率而修改预测、按真实标签控制模型输出、排除错误样本、以敏感性规则替换主规则、在不同类别使用不同推理配置或事后直接改汇总数值的代码级作假。四类 1,217 条样本全部计分，`no_decision` 全部计错，完整证据 gate 已通过。

仍有一项方法学限制：`last-terminal-v1` 在本机正式运行前已固定，但它的来源包含已经比较过四种规则和论文目标的历史 bundle，因此不是完全盲选的官方协议。当前结果应继续称为“冻结候选协议独立复现”，不宣称作者的样本级评测脚本已获得证实。这是协议不确定性，不是本次 runner 伪造结果。

## 2. 反作假检查证据

| 检查项 | 审计结果 |
| --- | --- |
| 正式运行前固定代码 | runner 与主规则首次固定于 `5dd6cc5`；ASR 空结果修复于 `ac8fcf1`，四类正式运行均在该提交后开始，且运行时 worktree 为 clean |
| 修复是否改动计分规则 | `5dd6cc5..ac8fcf1` 只补齐官方 ASR 异常返回空文本的语义、增加结构化证据和 diagnostic slice；未改 `PRIMARY_RULE`、标签映射、论文目标或数据身份 |
| 标签是否进入模型 | `label` 只用于找到官方类别目录、记录 provenance 和推理后计算 `prediction == label`；官方推理函数只接收配置、模型、WAV 和 Teacher-ASR，不接收真实标签或参考转写 |
| 论文指标是否参与预测 | `PAPER_CORRECT` 已从共用 protocol 移入仅在推理后运行的 gate 模块；模型 runner 和 `classify_trace` 不导入、不接收论文目标 |
| 是否挑样本 | formal 模式禁止 diagnostic slice，要求每类固定数量和固定数据哈希；1,217 条 inventory 与 1,217 条 records 顺序和身份逐条闭环，无排除 |
| 是否事后换规则 | 唯一主规则为 `last-terminal-v1`；first/closest/first-after 只存为 sensitivity readout，gate 从原始 trace 重算并强制主规则身份 |
| 是否按类别调参 | 同一语言的 Complete/Incomplete 必须共用同一 config hash、ASR manifest 和模型 manifest；四类各用新进程、seed 42 和独立空 ASR cache |
| 是否复用有利缓存/中断结果 | formal 模式禁止 resume，输出、trace 目录和 ASR cache 必须不存在；第一次 partial 未拼接到修复后结果 |
| 是否可从底层证据复算 | gate 已重算每条 trace 的主/敏感性读出、correctness、summary、ASR cache 命中顺序、state logits 有限性、WAV/trace/log/model/code hash；`evidence_audit_passed=true` |
| 同 seed 可重复性 | 旧批次与修复后批次的 EN 318+299 条及 ZH Complete 前 119 条，共 736 条的主预测逐条完全一致 |

官方推理函数内部使用 seed 控制的微小随机声缓冲区。为保证官方权重与后续 checkpoint 的 paired comparison 公平，以后必须保持相同 seed、样本顺序和每类新进程，不得为某个 checkpoint 挑选更有利的 seed。

## 3. 中文 600 条是否随机选择

不是本项目随机选择。它们是 ASLP 发布的固定 Easy Turn Testset 中全部 Complete 和 Incomplete 样本：

| 类别 | 真人录音 | 合成语音 | 合计 |
| --- | ---: | ---: | ---: |
| Complete | 150 | 150 | 300 |
| Incomplete | 150 | 150 | 300 |
| 合计 | 300 | 300 | 600 |

runner 不做抽样，而是按发布包 `.list` 顺序读取全部 600 条。两个 list 的 SHA-256 分别为：

```text
complete/complete_test.list:
0b35356353848bf25200a2fc0f6d5e25f2a3e86a1baa5b0e1a6e4cfe86ee26a3

incomplete/incomplete_real_test.list:
01b2465c0bbeb9b434b955d15274f7f74a03bec18ba22f30981ba555b191e476
```

发布包中没有第二组可互换的 Complete/Incomplete 600 条测试池。Backchannel 和 Wait 是不同任务标签，不能冒充第二组 Table 3 Complete/Incomplete 测试集。因此本次不进行伪随机的“再抽 600 条”。

不额外运行模型即可从已固定的逐样本预测得到发布时就存在的真人/合成子组结果：

| 类别 | 真人录音 | 合成语音 |
| --- | ---: | ---: |
| Complete | 138/150（92.00%） | 125/150（83.33%） |
| Incomplete | 129/150（86.00%） | 112/150（74.67%） |

总体结果不是由运行时挑选某个子组得到的；但合成语音明显难于真人录音，后续 checkpoint 报告必须继续分层报告，防止 macro 结果掩盖子组退化。

## 4. 废弃实现清理

最初基于部署 `TurnModel` 服务语义的 Easy Turn 评测实现不是当前 Table 3 候选协议。为避免两套脚本被混用，已从代码仓库删除：

```text
scripts/run_easy_turn_benchmark.py
src/duplexconv_stage3/easy_turn_benchmark.py
tests/test_easy_turn_benchmark.py
configs/soulx_official_easy_turn_eval.yaml
configs/soulx_official_easy_turn_en_eval.yaml
configs/soulx_easy_turn_zh_no_farfield_diagnostic.yaml
```

现在只保留 `run_table3_reproduction.py`、`table3_protocol.py`、`table3_reproduction.py`、`table3_gate.py`、候选 EN/ZH 固定配置和对应测试。数据盘中旧 diagnostic 的结果 JSON 保留为历史审计证据，不是可执行的仓库代码，也不参与后续基线或 checkpoint 汇总。

## 5. 对续训练的影响

完成上述审计和清理后，项目的下一阶段是正式续训练准备，但不应立即盲跑 300 step。启动正式 run 前还必须：

1. 按完整源会话生成并冻结 group-aware train/validation split；
2. 只使用 train/validation 对 `1.0e-5` 和 `3.33e-5` 做最多 20 optimizer step 的短程 LR 校准，不用 Easy Turn 选 LR；
3. 冻结唯一训练配置、有效 batch、AMP、checkpoint 网格和停止规则；
4. 为续训 checkpoint 实现不放宽数据/协议哈希的 paired 评测入口；官方 step-0 gate 仍保持不可变；
5. 由项目负责人明确接受当前 Table 3 候选协议作为本地 step-0 paired baseline，同时保留它未获得作者协议确认的限制。

官方发布权重没有 `global_step`、optimizer、scheduler 或 AMP scaler，所以不能精确 resume 论文训练状态。公开 Stage 3 重实现配置的 `total_steps=1800`，因此后续报告应固定记录：

```text
origin_step_estimate = 1800
estimate_confidence = low
continuation_optimizer_step = 本地从 0 重新计数
estimated_total_step = 1800 + continuation_optimizer_step
```

若只连接公开 scheduler 的数值语义，step 1,800 的参考 LR 约为 `3.33e-5`。但由于 optimizer/scheduler 状态不存在，实际必须新建 AdamW，不能把本地 trainer 的 `global_step` 伪装成已真正 resume 至 1,800。
