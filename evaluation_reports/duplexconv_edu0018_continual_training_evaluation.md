# SoulX-Duplug Stage 3 中文续训练实验与性能评估

更新时间：2026-08-21T13:22:58.705540+00:00
用途：课题组会议/导师汇报
实验状态：**已完成（确认退化后提前结束后续评测）**

## 1. 结论摘要

- 正式训练状态：`complete`；已完成 optimizer step：300/300。
- 最终 Table 3 分析窗口为 step 0/5/10/20/30；其中续训练 checkpoint 已完成 4/4。原计划共有 11 个续训练 checkpoint。
- 验证集选择的 peak LR：`1e-05`；选择过程未读取 Table 3。
- 没有任何已测续训练 checkpoint 满足严格的“几乎未下降”条件；最早可观测退化点是 step 5。
- 首次类别级明显下降在 step 10，并由 step 20 确认；EN/ZH 两种语言宏平均同时明显下降始于 step 20。
- step 5 只是四个续训练点中相对损伤最小者，仍未通过稳定性门禁。若目标是保持现有 Table 3 通用能力，推荐继续使用官方 step 0。

> step 45/60/90/120/180/240/300 的进一步评测是在看到 step 10→20 已确认退化、step 30 继续恶化后，由项目负责人于 2026-08-21 决定停止。该决定是事后提前结束评测，不能表述为原始预注册网格的一部分；报告不对未测点插值，也不使用 step 45 的不完整四分类结果。

## 2. 起始状态与“续训练”定义

- 官方 Bilingual 权重 SHA-256：`b0703dea0b1dbb1cd51e6e7b6514c60907ea4d4b6752cecc3f71cb6445650dbe`。
- 发布权重不含 global_step、AdamW、scheduler、AMP scaler，因此这是从模型参数继续微调，不是 optimizer 的精确 resume。
- 公开 Stage 3 配置 `total_steps=1800`，故将起始 step 估计为 1800（低置信度），不能表述为已证实的官方 checkpoint step。
- 本地 batch=1、梯度累积=72，有效 batch=72；官方参考全局有效 batch=576。
- 每个本地 step 对应约 0.125 个官方 sample-equivalent step；例如 local step 20 只有约 2.5 个官方 sample-equivalent step。
- 可训练参数：13,505,536；总参数：953,154,816。

## 3. 数据集、处理方法与切分

训练源为 DuplexConv `Edu_0018`：500 个同步多轨教育场景会话（495 个双声道、5 个三声道），展开为 1,005 个 target-speaker views。三声道不丢弃：每次只输入一个目标声道，其他声道聚合为关系证据，不把多路 audio token 放入同一 sequence。

原始完整会话去重时长约 10.519 小时，按目标说话人视角累计约 21.170 小时。相较 DuplexConv 公开约 2,000 小时的总体规模，本轮只覆盖约 0.53%，因此应称为 `Edu_0018` pilot，而不是完整 DuplexConv 续训练。

状态映射：官方 complete/incomplete/backchannel 原样映射；11 个 WAIT 映射为 complete；1,599 个缺失状态由固定 `qwen3-235b-a22b-instruct-2507` 通过 OpenRouter 补标（404 个源会话请求，accepted-response cost 0.2187791 USD）。这些是 LLM 辅助标签，不称为人工 gold。Paraformer 只用于中文伪转录/时间戳构造，不参与模型训练。

最终 model-ready 数据包含 2,168 rows、474,030 个可用 160 ms chunks 和 953,532 个 GLM audio tokens；另有 2,736 个异常 chunks（0.574%）被隔离，没有伪造文本或状态补齐。

| Split | 源会话 | target views | rows | 160ms chunks | 视角时长(h) | Qwen 补标事件 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 475 | 955 | 2066 | 448222 | 19.921 | 1134 |
| Validation | 25 | 50 | 102 | 25808 | 1.147 | 63 |

切分协议：`source-conversation-group-aware-random-v1`，seed=42，source leakage=0，split identity=`0f5060afcf27857af17b99755775921851125fdf7b96dbf6e233fbfb967c1f2b`。同一 WAV 的全部声道视角和窗口只属于一个 split。

## 4. LR 校准与正式训练配置

两档校准都从官方权重重新初始化，并使用相同训练顺序和固定验证集；Table 3 不参与 LR 选择。失格规则为任一状态头相对 step 0 下降超过 5pp；最终验证目标差异不超过 1% 时选择较低 LR。

| Candidate | Peak LR | step 20 validation objective | state macro | 合格 |
| --- | ---: | ---: | ---: | --- |
| `calibration-lr1e-5-seed42-v1` | 1e-05 | 1.913779 | 56.680% | 否 |
| `calibration-lr3p33e-5-seed42-v1` | 3.3333333e-05 | 1.731448 | 53.576% | 否 |

选择原因：all candidates failed the final guard; selected longest guard-safe horizon, then lower LR。正式 LR 采用 5-step 新 AdamW 重热身，并按估计原 step=1800 进行 offset inverse-square-root 衰减。正式训练运行时间为 2026-08-21T02:41:40.249676+00:00 至 2026-08-21T04:06:55.628100+00:00；300 次 optimizer 更新均已记录，AMP overflow 为 0。

## 5. 训练期 validation 变化

| Local step | 估计总 step | LR | token-weighted objective | state macro ACC | epoch-equivalent |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1800 | 0 | 3.925068 | 50.715% | — |
| 5 | 1805 | 9.98614e-06 | 3.451332 | 50.594% | 0.174 |
| 10 | 1810 | 9.9723374e-06 | 2.703682 | 50.923% | 0.348 |
| 20 | 1820 | 9.9449032e-06 | 1.913598 | 56.678% | 0.697 |
| 30 | 1830 | 9.9176941e-06 | 1.789306 | 55.250% | 1.045 |
| 45 | 1845 | 9.877296e-06 | 1.722838 | 55.097% | 1.568 |
| 60 | 1860 | 9.8373875e-06 | 1.669028 | 56.115% | 2.091 |
| 90 | 1890 | 9.7590007e-06 | 1.594009 | 57.740% | 3.136 |
| 120 | 1920 | 9.6824584e-06 | 1.542596 | 57.757% | 4.182 |
| 180 | 1980 | 9.5346259e-06 | 1.475095 | 59.126% | 6.273 |
| 240 | 2040 | 9.3933644e-06 | 1.430206 | 60.400% | 8.364 |
| 300 | 2100 | 9.258201e-06 | 1.395761 | 63.080% | 10.455 |

训练域 validation objective 持续下降并不代表外部通用能力保持。后续 Table 3 显示 Complete/Incomplete 决策边界发生快速偏移，这是本轮最重要的泛化差异。

## 6. Table 3 最终结果

主规则始终为 `last-terminal-v1`；样本、seed、顺序、推理核心、尾部静音和 Teacher-ASR 固定。每个纳入报告的 checkpoint 四类结果都经过独立证据 gate；论文目标没有传入推理 runner，也不用于选择 LR 或训练 checkpoint。

先用官方发布权重复现 step 0。语言宏平均与论文分别相差 EN +0.96pp、ZH -0.33pp，可视为数值基本接近；但预注册的“四类均不超过 ±1.0pp”机器门禁仍因 EN Complete 和 ZH Complete 失败，因此报告保留“已审计候选协议、尚缺作者样本级脚本确认”的限定。

| 语言 | 指标 | 论文结果 | 官方权重本地 step 0 | 差异 |
| --- | --- | ---: | ---: | ---: |
| EN | Complete ACC | 77.67% | 78.93% | +1.26pp |
| EN | Incomplete ACC | 88.96% | 89.63% | +0.67pp |
| EN | Macro ACC | 83.32% | 84.28% | +0.96pp |
| ZH | Complete ACC | 89.33% | 87.67% | -1.67pp |
| ZH | Incomplete ACC | 79.33% | 80.33% | +1.00pp |
| ZH | Macro ACC | 84.33% | 84.00% | -0.33pp |

“基本不变”：EN/ZH macro 各下降不超过 1pp，且任一 class 下降不超过 2pp。“明显下降触发”：任一语言 macro 下降超过 3pp，或任一 class 下降超过 5pp；必须在下一个预注册点仍触发才确认。

| Local step | 估计总 step | LR | EN C | EN I | EN Macro | ΔEN | ZH C | ZH I | ZH Macro | ΔZH | 四类 Macro | 判定 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 1800 | — | 78.931 | 89.632 | 84.281 | 0.000 | 87.667 | 80.333 | 84.000 | 0.000 | 84.141 | 官方发布权重基线 |
| 5 | 1805 | 0.00000999 | 83.333 | 87.291 | 85.312 | 1.031 | 88.333 | 77.000 | 82.667 | -1.333 | 83.989 | 轻微/不均衡退化 |
| 10 | 1810 | 0.00000997 | 92.453 | 81.605 | 87.029 | 2.748 | 91.333 | 71.333 | 81.333 | -2.667 | 84.181 | 明显下降（已确认） |
| 20 | 1820 | 0.00000994 | 97.799 | 61.873 | 79.836 | -4.446 | 91.333 | 63.667 | 77.500 | -6.500 | 78.668 | 明显下降（已确认） |
| 30 | 1830 | 0.00000992 | 97.799 | 40.803 | 69.301 | -14.981 | 92.000 | 57.667 | 74.833 | -9.167 | 72.067 | 明显下降趋势持续 |

模型与证据身份：

| Step | checkpoint SHA-256 | evidence gate SHA-256 |
| ---: | --- | --- |
| 0 | `b0703dea0b1dbb1cd51e6e7b6514c60907ea4d4b6752cecc3f71cb6445650dbe` | `e013f7ee866498a7797f5226cb0558be42c41675cc76ef8732461877d87336c9` |
| 5 | `30fab3f72af4c23ac4194a546b3a041753e8cf398996203526405d0b3cdb02e7` | `8e8582053d98372cc054873cc69961e625de288febeda0853e6f9105d67fda30` |
| 10 | `4bec2628ecd1ea659dc9f1590cdfc4931ccf7d88e6648ec8eb439fa6cacf0230` | `027b971e2e6b81d3348933ee10c5af1ddb6052100c0ead8ede990a91c7c91335` |
| 20 | `0ba2cb7333a7d12f91bede80bbea78ee6d0c9f4e5531a8f99a16251c9d62ebf9` | `4cc750d9cbf495e683bfaa8d3288eb2dcd8b502f0ec0491481b3be937d6dfec0` |
| 30 | `26fe02223468e1e7ca262979e5cec87364c2a0d3e716a843756ee1e7c75b3ef8` | `c8987463fd052f39be4546c7752037786479ce0a7ecdd566549c73d095974fe1` |

关键观察：

1. step 5 已不满足“基本不变”：ZH Incomplete 下降 3.333pp，paired bootstrap 95% CI 为 [-6.000, -0.667]pp，exact McNemar p=0.0309；但尚未达到预定义明显下降阈值。
2. step 10 首次触发类别级明显下降：EN Incomplete 下降 8.027pp（95% CI [-11.706, -4.682]，p=1.93e-5），ZH Incomplete 下降 9.000pp（95% CI [-12.333, -5.667]，p=1.12e-7）。step 20 再次触发，因此 step 10 被正式确认。
3. step 10 的四类 Macro 为 84.181%，与 step 0 的 84.141% 几乎相同，但这是 Complete 上升和 Incomplete 下降相互抵消的结果，不能据此宣称模型整体无退化。
4. step 20 首次出现 EN 与 ZH 宏平均同时下降超过 3pp；step 30 的 EN/ZH 宏平均分别下降 14.981/9.167pp，EN Incomplete 已下降 48.829pp。趋势表现为模型越来越偏向 Complete。

中文固定 600 条是发布方完整测试集，不是本项目随机抽样；每个 checkpoint 另报 complete/incomplete × real/synthetic 四个固定子组。差异显著性使用同一样本的 paired bootstrap 95% CI 和 exact McNemar，而不是把两次准确率当独立样本。

## 7. 评测提前结束、可恢复性与异常记录

- 评测决定状态：`completed_early_stop`；正式纳入 step：[0, 5, 10, 20, 30]；不再评测：[45, 60, 90, 120, 180, 240, 300]。
- step 45 已完成 EN Complete、EN Incomplete、ZH Complete，但 ZH Incomplete 在 Paraformer VAD 辅助模型初始化时因本地代理不可用而退出。由于四类未闭环，step 45 整体不纳入正式分析；其 partial 证据保留在数据盘，不与其他 checkpoint 拼接。
- 训练状态：`complete`；AMP overflow 次数：0。
- 峰值 CUDA allocated：18.733 GiB。
- GPU：NVIDIA vGPU-32GB；Python：3.10.20；Torch：2.6.0；CUDA：12.4。
- 每个预注册点保存仅含 118 个可训练 tensor 的评测快照；训练完成到 step 300，停止的是耗时较长的外部 Table 3 推理，不是训练。
- 正式 run manifest：`/root/SoulX-stage3-dataset/checkpoints/duplexconv_edu0018_continual_formal_v1/run_manifest.json`，SHA-256=`b80c3db0cb83eed1e67fab04d4571d58070fd2fa4c92da935d40839e94147b58`。
- 评测停止决定：`evaluation_reports/duplexconv_edu0018_table3_evaluation_decision.json`。
- 报告生成输入均记录绝对路径和 SHA-256；HTML 内嵌同一份结构化数据，可离线查看。

| 报告输入 | 路径 | SHA-256 |
| --- | --- | --- |
| `split_manifest` | `/root/autodl-tmp/dataset/duplexconv/splits/edu0018_stage3_zh_v1_seed42_group95_5/split_manifest.json` | `b4874ccf3e263ec4bf56257f5850e676b66528647f06407caf9a3ff0819d6210` |
| `lr_selection` | `/root/autodl-tmp/dataset/duplexconv/continual_training/lr_selection_v2.json` | `56a30722df519967c023287d9e7c68facc185887ea25737caec9ebc801b39897` |
| `table3_index` | `/root/autodl-tmp/dataset/soulx_duplug_eval/table3_continual_sweep/formal_v1_b17bcf9/sweep_index.json` | `91c871d4855bf44b4f8c4a42797d723f8b8a1954cc857e6df6e107ecde4c34cd` |
| `evaluation_decision` | `/root/SoulX-stage3-dataset/evaluation_reports/duplexconv_edu0018_table3_evaluation_decision.json` | `d02dc57641440857f6a33351c9ffd06dd7a1cf6051b685ae3cc9f0229d3b92c7` |
| `source_inventory` | `/root/autodl-tmp/dataset/duplexconv/work/source_scan_v1/source_inventory.jsonl` | `bb6a31cf53f9cab69f72fec2198d9e57ef086ab8510bf127580e47feab5cc840` |

## 8. 反作假检查与限制

1. 专项审计未发现预测篡改、标签入模、样本排除、分类别调参或事后切换主规则；`selection_used_paper_targets=false`，LR selection 也记录 `benchmark_used_for_selection=false`。完整审计见 `evaluation_reports/soulx_table3_anti_cheating_audit.md`。
2. 本轮提前结束发生在明显退化已被 step 20 确认之后，所有已经完整得到的 step 5/10/20/30 均如实报告；未用 step 45 partial 选择性补表，也未伪造后续点。
3. 起始 1800 step 是依据公开配置的低置信度估计，不是官方权重元数据。
4. 本地有效 batch=72，只有官方参考全局有效 batch=576 的 1/8；因此 local optimizer step 不能直接等同于官方同数量 step。
5. 当前 Table 3 样本级读出规则是已审计候选协议，数值与论文基本一致，但仍缺作者发布的样本级计分脚本确认。
6. `Edu_0018` 只有约 10.519 个去重会话小时；结论只适用于这次小规模 pilot，不能外推为完整 2,000 小时 DuplexConv 的训练结论。
7. Full-Duplex-Bench 是包含 LLM/TTS 的系统级表 2 测试。本轮没有续训练 checkpoint 通过模型级稳定性门禁，因此未继续对这些 checkpoint 做昂贵的系统级评测；本报告的最终结论限于模型级 Table 3。

## 9. 模型选择与后续建议

- 保持现有 EN/ZH Easy Turn 能力：使用官方 step 0。
- 若只做研究性对比、必须使用本轮续训练权重：step 5 是四个已测点中损伤最小者，但应明确标注“未通过稳定性门禁”，不能作为无退化版本发布。
- 下一轮不宜直接增加本轮数据上的 step；优先扩大 DuplexConv 覆盖规模并平衡 Complete/Incomplete，考虑官方/英文 replay、降低峰值 LR，并把 step 1–5 设为更密的早期评测窗口。
- 需要另建与训练域分离的中文 in-domain 测试集，才能判断 `Edu_0018` 适配收益；训练域 validation 改善不能替代外部收益证据。
