# DuplexConv Edu_0001–Edu_0045：SoulX Stage 3 官方流程续训练与评估

生成时间：2026-08-30T05:06:31.836519+00:00  
状态：正式会议汇报报告；训练、内部 validation 与 Table 3 证据审计通过。

## 1. 结论摘要

- 内部 validation 预注册规则选择 local step 1；该选择在查看 Table 3 前完成。
- Table 3 首次明显下降触发：local step 2；相邻预注册点确认：local step 2。
- Table 3 只用于训练后独立性能描述，没有用于选择 checkpoint、学习率、样本、标签、阈值或解码规则。
- 官方发布权重不含 optimizer、scheduler、GradScaler 和 Trainer global step；起点 1800 只是依据公开 Stage 3 配置的低置信度估计，不是精确断点恢复。

## 2. 续训练数据集

数据来自 DuplexConv 中文教育会话 `Edu_0001`–`Edu_0045`。多轨会话被展开为说话人视角：每条训练序列只放目标声道的 audio token，其他声道仅提供正在说话、重叠和 backchannel 等关系信息；不混音，也不把多路 audio token 塞进同一序列。

- 45 个冻结 shard；22,032 个源会话；44,332 个目标视角。
- 101,395 个训练窗口；22,352,510 个有效 160 ms chunk；按目标视角 chunk 计约 993.445 小时。该值是训练暴露时长，不等同于去重后的原始录音时长。
- 多轨行分布：2 轨 99,021，3 轨 2,206，4 轨 168。
- 隔离：窗口级 142,382 chunks；源视角级 35 views/20,743 chunks；最终跨 shard source/index/view 重叠均为 0，45 个 Gate D 均通过。
- 目标声道文本与时间戳由固定 Paraformer 生成；GLM-4-Voice tokenizer 每 160 ms chunk 生成两个 audio token。

### 2.1 状态标签处理

SoulX Stage 3 使用 `idle`、`nonidle`、`backchannel`、`complete`、`incomplete` 五类状态 token。DuplexConv 官方给出的终态标签保留原义；`WAIT` 确定性映射为 `complete`；官方缺失的终态由固定 `Qwen3-235B-A22B-Instruct-2507` 通过 OpenRouter 补标。Qwen 结果有独立 provenance，不表述为人工 gold。

- 终态事件 396,950：官方标签 319,767，WAIT→Complete 846，Qwen 补标 76,337。
- 最终事件分布：Complete 217,725，Incomplete 86,452，Backchannel 92,773。
- Qwen 共 17,948 个请求，accepted-response 累计成本 9.083424 USD；这是 45 个 shard 跨多个处理日的累计值，不是本次训练或当天费用。训练与评测 API 费用为 0。

### 2.2 冻结划分

按完整源会话进行 seed=42 的 98/2 划分：Train 21,591 会话/99,334 rows/972.995h；Validation 441 会话/2,061 rows/20.450h；source leakage=0。
split identity：`a2599191a146f0cc37c7cf068cfe2d8cbbbb4906ef3b9a6a04912e3af92dc6fb`。

## 3. 官方续训练方法

官方上游 commit 固定为 `928b06508ed2de1344208d06fb1f6fb2ebfb1df5`。训练入口是 `finetune.py::train` → Lightning `Trainer.fit()` → `training_step()` → `configure_optimizers()`；没有使用先前自定义 optimizer loop。

补丁仅包括：全 `-100` 空 head 的有限零损失修复、严格基础权重加载/初始化顺序、冻结会话级 split 审计、scheduler 起点偏移、训练审计与 compact checkpoint。官方上游目录保持 clean，补丁只存在于独立 runtime 副本。

- 基础权重 SHA-256：`b0703dea0b1dbb1cd51e6e7b6514c60907ea4d4b6752cecc3f71cb6445650dbe`。
- local batch=1，梯度累积=576，单卡 world size=1，有效 batch=576。
- AdamW；weight decay=0.01；scheduler=`utils.sparkvox.utils.scheduler.WarmupAnnealSteps`；续训起始 LR=0.000033324078。
- local step 1–30 对应估计总 step 1801–1830；总暴露 17,280 个样本，约占首个 epoch 的 17.4%，没有循环读取。
- 可训练参数 13,505,536/953,154,816，118 个可训练张量。

## 4. 训练域精确内部 validation

选择规则在 Table 3 前冻结：指标必须有限；五个状态 head 的精确 token-weighted accuracy 相比 step 0 均不得下降超过 5 pp；合格点中选精确 token-weighted objective 最低者，1% 内并列取更早 step；若无人合格则保留 step 0。

| Local step | 估计总 step | Token-weighted objective | 总 accuracy | Idle | Non-idle | Backchannel | Complete | Incomplete | Guard |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 1800 | 0.844982 | 84.40% | 97.26% | 86.99% | 19.00% | 29.00% | 10.47% | 基线 |
| 1 | 1801 | 0.844320 | 83.96% | 97.13% | 85.23% | 18.69% | 30.13% | 10.72% | 通过 |
| 2 | 1802 | 0.847982 | 83.45% | 96.90% | 82.97% | 18.40% | 33.06% | 12.21% | 通过 |
| 3 | 1803 | 0.861401 | 82.80% | 96.41% | 79.73% | 18.12% | 40.43% | 17.78% | 拒绝 |
| 5 | 1805 | 0.937068 | 80.56% | 93.95% | 70.49% | 17.36% | 61.03% | 32.16% | 拒绝 |
| 10 | 1810 | 1.220783 | 73.12% | 80.78% | 50.66% | 15.14% | 77.44% | 33.33% | 拒绝 |
| 20 | 1820 | 1.053396 | 77.80% | 93.33% | 54.50% | 13.34% | 81.39% | 31.78% | 拒绝 |
| 30 | 1830 | 0.847138 | 81.69% | 95.29% | 68.10% | 11.92% | 81.27% | 25.71% | 拒绝 |

预注册内部选择结果：local step **1**。原因：earliest eligible checkpoint within 1% of the lowest exact token-weighted validation objective。

说明：训练审计中的 `mean_training_step_loss` 是 Lightning 在梯度累积语义下返回的缩放 loss，约为未缩放值的 1/576，不用于 checkpoint 选择；上表由每行 loss×有效 token 数重新聚合，才是精确 token-weighted 指标。

## 5. Table 3 冻结评测

评测使用 Easy Turn Testset（Table 3）和预注册的 endpoint-first coarse-to-fine-v1 协议：固定粗测顺序为 1→30→10→5→20；仅当 step 5 触发明显下降时补测 2、3。该规则不假设性能单调，且只决定追加哪些评测点，不参与 checkpoint 选择。`last-terminal-v1`、EN/ZH ASR、样本顺序、阈值、规则和逐样本配对统计保持不变。Full-Duplex-Bench（Table 2）不在本阶段范围内。

### 5.1 论文值与本机官方权重基线

| 类别 | 论文正确数/总数 | 论文准确率 | 本机正确数/总数 | 本机准确率 | 差值 | ±1 pp |
|---|---:|---:|---:|---:|---:|---|
| EN Complete | 247/318 | 77.67% | 251/318 | 78.93% | +1.26 pp | 未通过 |
| EN Incomplete | 266/299 | 88.96% | 268/299 | 89.63% | +0.67 pp | 通过 |
| ZH Complete | 268/300 | 89.33% | 263/300 | 87.67% | -1.67 pp | 未通过 |
| ZH Incomplete | 238/300 | 79.33% | 241/300 | 80.33% | +1.00 pp | 通过 |

原 ±1 pp 准确率门禁实际为 `FAILED`；证据完整性审计为 PASSED。失败结果如实保留，本机 step 0 只能作为冻结配对基线，不表述为严格复现通过。

### 5.2 粗测—细化 checkpoint 结果

| Local step | 估计总 step | EN C | EN I | EN Macro | ZH C | ZH I | ZH Macro | 判定 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 1800 | 78.93% | 89.63% | 84.28% | 87.67% | 80.33% | 84.00% | 官方权重基线 |
| 1 | 1801 | 82.39% | 84.95% | 83.67% | 88.67% | 77.67% | 83.17% | 不均衡变化 |
| 2 | 1802 | 90.25% | 84.62% | 87.43% | 89.67% | 74.00% | 81.83% | 明显下降（相邻点确认） |
| 3 | 1803 | 96.54% | 76.92% | 86.73% | 92.00% | 69.00% | 80.50% | 明显下降（相邻点确认） |
| 5 | 1805 | 97.80% | 62.21% | 80.00% | 91.67% | 64.33% | 78.00% | 明显下降（相邻点确认） |
| 10 | 1810 | 97.80% | 22.41% | 60.10% | 94.67% | 53.67% | 74.17% | 明显下降（相邻点确认） |
| 20 | 1820 | 92.14% | 31.77% | 61.96% | 96.67% | 53.00% | 74.83% | 明显下降（相邻点确认） |
| 30 | 1830 | 93.40% | 23.08% | 58.24% | 98.33% | 52.33% | 75.33% | 明显下降触发 |

Table 3 的稳定/下降定义由冻结统计索引给出：两种语言 macro 均不低于基线 1 pp 且四类均不低于 2 pp，才称为“几乎未下降”；任一语言 macro 下降超过 3 pp 或任一类下降超过 5 pp 触发“明显下降”，且需下一个预注册点继续触发才确认。

## 6. 实验完整性与限制

- 官方上游和 Table 3 runtime 均固定 commit 且保持 clean；训练补丁副本与评测副本隔离。
- 训练与内部 validation 的七点网格保持冻结；Table 3 在查看 step 1 汇总结果前改为预注册粗测—细化协议。固定粗测点全部评测，2/3 是否追加只由 step 5 的冻结明显下降规则决定；没有按论文目标或候选准确率任意删点。若 2/3 未被触发而省略，仍存在漏掉局部窄幅非单调波动的限制，不能把未测点表述为性能不变。
- 第一次 step 0 尝试因 Hugging Face cache 默认写系统盘而在验证前失败；未上 GPU、未创建 optimizer、未训练。临时缓存随后改道数据盘，失败日志完整保留。
- 正式训练第一次尝试已完成 local step 10 及其 validation，但 Lightning/fsspec 在提交原生 `last.ckpt` 时把事务临时文件写入系统 `/tmp`，因系统盘空间不足而失败；模型数值未失败，step 1/2/3/5/10 compact checkpoint 与日志完整归档。由于没有产生 optimizer/scheduler/global-step checkpoint，不能精确从 step 10 恢复，因此在训练配置、数据、模型和学习率均不变的前提下从发布权重重跑，并只将 `TMPDIR/TEMP/TMP` 改到数据盘。
- 起点 1800 是低置信度估计；这次实验不能回答恢复官方 optimizer 状态后的精确续训轨迹。
- 本报告只覆盖训练域 validation 与论文 Table 3；未把 Table 2 Full-Duplex-Bench 结果混入结论。
- 训练有效时长按目标视角统计，同一源会话的不同目标声道会重复贡献会话时长，因此不能写成去重后的原始录音小时数。

## 7. 关键产物

- 训练 manifest：`/root/autodl-tmp/dataset/duplexconv/training/edu0001_0045_official_continual_v1/run_manifest.json`
- 精确内部 validation：`/root/autodl-tmp/dataset/duplexconv/evaluation/edu0001_0045_official_continual_v1/group_validation/index.json`
- Table 3 index：`/root/autodl-tmp/dataset/soulx_duplug_eval/table3_official_continual_sweep/edu0001_0045_official_continual_v1_b17bcf/sweep_index.json`
- 数据聚合 manifest：`/root/autodl-tmp/dataset/duplexconv/aggregates/edu0001_0045_stage3_zh_v2/aggregate_manifest.json`
- 状态 provenance：`/root/autodl-tmp/dataset/duplexconv/training/edu0001_0045_official_continual_v1/dataset_state_provenance_summary.json`

审计状态：`passed`。
