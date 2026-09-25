# DuplexConv Edu_0018：官方 Lightning 续训练审计

更新时间：2026-08-22  
状态：官方训练、完整 validation step 0/1/2/3/5 与 Table 3 step 0/1/2/3/5 均已完成；正式会议报告及审计 JSON 已生成。

## 1. 本次纠正的结论

本次正式训练不再使用项目自定义 optimizer loop。它实际运行 SoulX 官方训练入口、官方 LightningModule 与官方优化器构造：

```text
finetune.py::train
  -> pytorch_lightning.Trainer.fit
  -> State_Prediction_Model.training_step
  -> State_Prediction_Model.configure_optimizers
  -> AdamW + WarmupAnnealSteps
```

官方上游提交 `928b065` 保持 clean；所有补丁位于独立运行时 `runtimes/SoulX-Duplug-928b065-official-continual-v1`。

## 2. 续训练起点的含义

官方发布 `.pth` 只有679个模型 tensor，不含 AdamW、scheduler、AMP scaler 或 Trainer global step。因此：

- 公开 Stage 3 配置 `total_steps=1800` 只用于低置信度估计起点；
- 新建官方 AdamW，无法恢复官方训练结束时的 moments；
- 使用同一个官方 `WarmupAnnealSteps` 公式，将 scheduler 定位到估计 step 1800；
- 第一次续训练更新使用 LR `3.332407793031351e-5`；
- 这应称为“从发布模型权重继续训练”，不能称为 optimizer 精确 resume。

## 3. 数据与 batch

- Train：475个源会话、2,066 rows、约19.921视角小时；
- Validation：25个源会话、102 rows、约1.147视角小时；
- source-conversation leakage：0；
- split identity：`0f5060afcf27857af17b99755775921851125fdf7b96dbf6e233fbfb967c1f2b`；
- 单卡 microbatch：1；
- 梯度累积：576；
- 名义全局有效 batch：576，与公开配置 `1 × 72 × 8 = 576` 对齐。

官方 Lightning 在 epoch 末尾执行尾部更新，因此五次 optimizer update 的实际样本数为：

| Local step | 估计总 step | 实际样本数 | 累计样本数 | 使用 LR |
|---:|---:|---:|---:|---:|
| 1 | 1801 | 576 | 576 | 0.000033324078 |
| 2 | 1802 | 576 | 1,152 | 0.000033314830 |
| 3 | 1803 | 576 | 1,728 | 0.000033305590 |
| 4 | 1804 | 338 | 2,066 | 0.000033296358 |
| 5 | 1805 | 576 | 2,642 | 0.000033287133 |

step4 的338条是官方 Trainer 的 epoch-tail 行为，未跨 epoch 人工拼接，也未隐藏。

## 4. Group-aware validation

| Local step | Validation loss | 官方 `val_acc` | Idle | Non-idle | Complete | Incomplete | Backchannel |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.427645 | 55.279% | 95.955% | 86.224% | 28.599% | 19.091% | 18.142% |
| 1 | 2.996503 | 55.212% | 95.658% | 84.420% | 30.649% | 19.091% | 17.701% |
| 2 | 2.582847 | 55.525% | 94.990% | 81.914% | 32.698% | 23.333% | 16.966% |
| 3 | 2.230164 | 55.859% | 94.275% | 78.481% | 36.798% | 26.061% | 16.493% |
| 5 | 1.862548 | 59.029% | 91.870% | 69.352% | 58.881% | 39.697% | 14.512% |

这里的 `val_acc` 是官方模型对七个 head 的汇总口径，不等同于 Table 3 四分类准确率，也不等同于旧自定义报告中的 token-weighted state macro accuracy。

step 0 已经按单独确认的 validation-only 方案补跑：完整读取冻结 validation 的102行，调用官方 Lightning `Trainer.validate()`，没有 optimizer、backward 或训练更新。Lightning 默认2-batch sanity check仍被排除，旧自定义 runner 的 step0 也不与本表混用。

趋势上，validation loss 持续下降，Complete/Incomplete 提高，但 Idle、Non-idle 和 Backchannel 同时下降；因此不能只依据总 `val_acc` 宣称模型整体改善，仍需冻结的 Table 3 进行模型级比较。

## 5. AMP、显存与恢复现场

- 可训练参数：13,505,536，共118个 tensor；
- 总参数：953,154,816；
- CUDA peak allocated：19,202,004,480 bytes（约17.883 GiB）；
- AMP scale：65,536，growth tracker=5；五次 update 未触发 scale 回退；
- `last.ckpt`：4,005,077,122 bytes；
- `last.ckpt` SHA-256：`68a574976593b57339cbf5bb8792d370ac185d5130e8a3488d69a3037226baa9`；
- `last.ckpt` global step=5、epoch=1；
- 含679个模型 tensor、1个 AdamW state（118组参数状态）、1个 scheduler state（last_epoch=1805）、AMP scaler 和 Callback 状态。

这意味着今后可以从本次 step5 进行 Lightning 精确恢复；它不改变“官方发布权重无法精确恢复到 step1800 optimizer 现场”的事实。

## 6. 可评测 checkpoint

| Local step | 大小 | SHA-256 | Table 3 overlay 严格加载 |
|---:|---:|---|---|
| 1 | 54,065,116 B | `aff1ec1f3a0c17432a031491d1c4e61be65e409044209c20f53a47c3b51e6c23` | accepted |
| 2 | 54,065,116 B | `60efa07fd837043a361b70e652bde75b1954667d39f2bee7fe6e36653169dbea` | accepted |
| 3 | 54,065,116 B | `8251939ded8dff6412f4a0006a6bb72efdefa0c5697fafc6cd9850f11aeb35dd` | accepted |
| 5 | 54,065,116 B | `180d6d455c665847ad4772eb10f2dee5488a030526b555e4dd147cf7014170b3` | accepted |

四个 overlay 都通过正式 Table 3 加载器检查：118个 trainable keys 完整，无额外键、形状错误或非有限 tensor。

## 7. 主要产物

- 正式 manifest：`checkpoints/duplexconv_edu0018_official_continual_v1/run_manifest.json`
- optimizer 更新日志：同目录 `optimizer_updates.jsonl`
- validation 日志：同目录 `validation_metrics.jsonl`
- 完整恢复 checkpoint：同目录 `last.ckpt`
- 紧凑评测 checkpoint：同目录 `checkpoints/evaluation/`
- 正式配置：`configs/duplexconv_edu0018_stage3_official_continual_v1.yaml`
- 补丁说明：`runtimes/SoulX-Duplug-928b065-official-continual-v1/OFFICIAL_CONTINUAL_PATCH.md`

## 8. Table 3 结论与正式报告

- step 0/1/2/3/5 的四类 Table 3 共20项评测均完成，所有逐样本 evidence gate 通过；
- 原官方权重 baseline 的 ±1 pp 论文一致性门禁仍为 `FAILED`（EN Complete 与 ZH Complete 超界），没有改写或隐藏；
- local step 1 已出现不均衡变化；step 2 首次触发明显下降并由 step 3 确认；step 3 又由 step 5 确认；没有续训练 checkpoint 满足预注册“几乎未下降”条件；
- 正式 Markdown：`evaluation_reports/duplexconv_stage3_official_continual_final.md`；
- 自包含 HTML：`evaluation_reports/duplexconv_stage3_official_continual_final.html`；
- 报告输入审计：`evaluation_reports/duplexconv_stage3_official_continual_final_audit.json`；
- 旧自定义训练结果继续仅作为 pilot 对照，不能作为官方流程结果。
