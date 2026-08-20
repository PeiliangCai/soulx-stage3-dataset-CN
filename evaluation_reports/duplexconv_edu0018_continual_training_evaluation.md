# SoulX-Duplug Stage 3 中文续训练模型性能评估

更新时间：2026-08-20  
用途：课题组会议/导师汇报  
状态：工作稿。数据集已构造，benchmark 资产已固定；官方模型基线、正式续训练和 checkpoint 评测尚未完成。所有 `TBD` 必须用真实实验结果回填，不得用预期值代替。

## 1. 汇报摘要

### 1.1 工作目标

本项目使用 DuplexConv `Edu_0018` 构造 SoulX-Duplug Stage 3 中文状态预测训练数据，并从官方 `SoulX-Duplug-0.6B-Bilingual` 权重继续微调。核心问题不是只看训练 loss 是否下降，而是回答：

1. 新中文数据能否改善或保持中文对话状态预测能力；
2. 在多少个 continuation optimizer step 内，模型在论文 benchmark 上几乎不下降；
3. 从哪个 step 开始出现明显下降或英文能力遗忘；
4. 哪个 checkpoint 最适合作为最终交付模型。

### 1.2 当前进展

| 项目 | 当前状态 |
| --- | --- |
| DuplexConv 源数据、状态补标、Paraformer、GLM token 和 Stage 3 导出 | 已完成 |
| 官方 loader、NaN 修复、真实 5-step 训练和 checkpoint 重载 | 已通过 |
| 官方论文、benchmark、推理代码和权重 step 元数据核对 | 已完成 |
| EN/ZH Easy Turn 与 ZH Full-Duplex-Bench 资产固定 | 已完成 |
| 官方 checkpoint 的论文指标复现 | TBD |
| group-aware split 与 LR 校准 | TBD |
| 正式 continuation step sweep | TBD |
| 最后无明显下降点、首次明显下降点和推荐模型 | TBD |

### 1.3 最终结论

> TBD：在实验全部完成后，用 3–5 句话给出官方基线是否复现、中文收益、英文遗忘、最后稳定 step、首次明显下降 step 和推荐 checkpoint。

## 2. 基础模型与续训练定义

基础模型：`Soul-AILab/SoulX-Duplug-0.6B` 的 Bilingual checkpoint。SoulX-Duplug 是流式状态预测模块；Paraformer 只在中文训练数据构造和中文推理 teacher forcing 中提供 ASR 文本，不是本项目要训练的模型。

固定版本：

```text
官方模型仓库 revision：61701c4ab8193cc1ee2220d3848872ac6c720142
官方 Bilingual pth SHA-256：b0703dea0b1dbb1cd51e6e7b6514c60907ea4d4b6752cecc3f71cb6445650dbe
官方训练代码 commit：928b06508ed2de1344208d06fb1f6fb2ebfb1df5
官方推理代码 commit：a0b9063843df69619b087b95b74597b2176910b8
训练 runtime：SoulX-Duplug-928b065-finite-empty-head-v2
```

本项目的“续训练”是从官方权重继续微调，不是训练状态的精确 resume。发布的 `.pth` 只含 679 个模型 tensor，不含 `global_step`、AdamW、scheduler 或 AMP scaler。

## 3. 续训练数据集

### 3.1 数据集介绍与规模

训练源为 DuplexConv 官方同步多轨教育场景子集 `Edu_0018`：

| 项目 | 数量 |
| --- | ---: |
| 完整同步多轨 WAV | 500 |
| 双声道 WAV | 495 |
| 三声道 WAV | 5 |
| target-speaker views | 1,005 |
| target views 总时长 | 约 21.169878 小时 |
| 官方 metadata 事件 | 8,505 |

双声道文件生成两个 target-speaker 视角。三声道不丢弃，每个声道分别作为目标用户，其他声道聚合为 `rest`，因此仍只向 SoulX 输入一条目标音频流；其他说话人的活动仅用于判断重叠、backchannel 等离线关系，不把多路 audio token 塞入同一 sequence。

### 3.2 状态标签来源与映射

| DuplexConv 来源 | 数量 | SoulX 监督 | 处理方式 |
| --- | ---: | --- | --- |
| 官方 `complete` | 4,570 | `<|user_complete|>` | 保留官方标签 |
| 官方 `incomplete` | 1,527 | `<|user_incomplete|>` | 保留官方标签 |
| 官方 `backchannel` | 798 | `<|user_backchannel|>` | 保留官方标签 |
| 官方 `wait` | 11 | `<|user_complete|>` | 按项目约定确定性映射 |
| 官方缺失 state | 1,599 | complete/incomplete/backchannel | 固定 Qwen 模型补标 |

缺失状态使用固定模型 `qwen3-235b-a22b-instruct-2507`，通过 OpenRouter 以结构化输出补标；共处理 1,599 个事件、404 个源会话请求，accepted-response cost 为 0.2187791 USD。Qwen 标签属于 LLM 辅助标签，不称为人工 gold，也不覆盖已有官方状态。

训练 sequence 的五种正式状态为：

```text
<|user_idle|>
<|user_nonidle|>
<|user_backchannel|>
<|user_complete|>
<|user_incomplete|>
```

其中 complete/incomplete/backchannel 是事件级终止状态；nonidle/idle 按目标声道的活动、ASR token 与时间线构造。其他声道活动只作关系证据，不新增第六种状态。

### 3.3 ASR、时间戳与 Stage 3 格式

每个目标声道先转为 16 kHz 单声道，由固定 Paraformer 生成伪转录和 token 时间戳。时间戳只决定 ASR 文本在哪个 160 ms chunk 首次发射，不反向更改状态标签。

每个 160 ms chunk 写入：

```text
2 个 GLM-4-Voice audio token
+ 当前 chunk 新增的 Paraformer 文本
+ <|end_of_sentence|>
+ 1 个 SoulX 用户状态 token
```

### 3.4 最终 model-ready 数据

| 项目 | 数量 |
| --- | ---: |
| 原始时间线 chunk | 476,763 |
| 新增 terminal 静音决策 chunk | 3 |
| 可用 chunk | 474,030 |
| 隔离 chunk | 2,736（0.574%） |
| GLM audio token | 953,532 |
| 最大序列长度 | 1,500 tokens |
| 最终训练 rows | 2,168 |

2,736 个异常 chunk 被隔离，没有为了保留数量而伪造 ASR 或状态。最终 2,168 条 sequence 已全部通过官方 loader、语法、长度和随机逐 chunk 回解验收。

### 3.5 Train/validation 划分

状态：TBD。

正式实验必须按完整源会话 group-aware 切分：同一 WAV 的所有声道视角和相邻窗口只能位于同一 split，避免随机按 row 切分造成泄漏。这里回填 train/validation 的会话、视角、rows、时长、声道和状态分布。

## 4. 官方模型原训练 step 估计与本地学习率

官方发布的 Stage 3 重实现配置给出：

```text
total_steps = 1800
learning_rate = 1e-4
warmup_steps = 200
anneal_steps = 100000
batch_size = 1
accumulate_grad_batches = 72
num_gpu_per_node = 8
```

据此暂将官方 checkpoint 记为：

```text
origin_step_estimate = 1800
estimate_confidence = low
```

该值只是基于公开重实现配置的估计，论文没有报告实际 checkpoint step，权重也没有 step 元数据，因此不能写成已证实事实。若按官方 inverse-square-root scheduler 把 1,800 当作原位置，参考 LR 约为 `3.33e-5`。

由于 optimizer 动量不可恢复，本地训练必须重新初始化 AdamW。本项目会在不读取 benchmark 的前提下，用 group-aware validation 对 `1e-5` 与 `3.33e-5` 做最多 20 optimizer step 的短程校准，然后冻结唯一正式配置。

正式实验参数：

| 参数 | 最终值 |
| --- | --- |
| origin step estimate | 1,800（低置信度） |
| selected peak LR | TBD |
| LR scheduler / offset | TBD |
| optimizer | AdamW |
| batch size | 1 |
| gradient accumulation | 暂定 72，TBD |
| effective batch | TBD |
| precision / initial scale | FP16 mixed / 16,384 |
| trainable parameters | 13,505,536 |
| LoRA | r=32, alpha=64, dropout=0.1 |
| projector | trainable |
| GLM speech tokenizer | frozen |
| seed | 42 |

所有表格中的 `step` 均指 optimizer update，不是 micro-batch。报告同时给出累计 micro-batch、样本曝光量和 epoch-equivalent。

## 5. 论文 benchmark 复现

### 5.1 为什么先复现官方模型

续训练前先用同一 runner 测试官方 checkpoint。只有官方模型能复现论文结果，后续 checkpoint 与 step 0 的差值才有可解释性。基线不通过时禁止正式续训练。

### 5.2 模型级 benchmark：Bilingual Easy Turn

固定测试资产：

| 测试集 | 固定 revision | 类别/数量 | 本地事实 |
| --- | --- | --- | --- |
| SoulX-Duplug Easy Turn EN | `f6e50e8...` | Complete 318、Incomplete 299 | 617 个 24 kHz 单声道 WAV |
| ASLP Easy Turn ZH | `5812651...` | Complete 300、Incomplete 300 | 600 WAV，含 16/24/48 kHz 和 34 个双声道文件 |

评测前只做协议规定的单声道化和 16 kHz 重采样，不做响度或语义相关的 test-set 调参。按官方推理参数以 160 ms chunk 模拟在线输入；中文 teacher ASR 为 Paraformer，英文为 SenseVoice Small。

论文目标和本地复现：

| 语言 | 指标 | 论文 | 官方权重本地结果 | 差异 |
| --- | --- | ---: | ---: | ---: |
| EN | Complete ACC | 77.67%（约 247/318） | TBD | TBD |
| EN | Incomplete ACC | 88.96%（约 266/299） | TBD | TBD |
| EN | Macro Avg. ACC | 83.32% | TBD | TBD |
| ZH | Complete ACC | 89.33%（268/300） | TBD | TBD |
| ZH | Incomplete ACC | 79.33%（238/300） | TBD | TBD |
| ZH | Macro Avg. ACC | 84.33% | TBD | TBD |

延迟：论文给出 240 ms 理论延迟和 L20 上 205 ms 部署测量。本机硬件不同，因此回填本机 median、p90、p95、首个有效 state latency、样本实时率和 CUDA 配置，不把硬件差异误判为模型退化。

### 5.3 系统级 benchmark：Full-Duplex-Bench

论文系统由 SoulX-Duplug、Qwen2.5-7B-Instruct 和 IndexTTS-1.5 组成。主要指标：

- Pause Handling：TOR 越低越好；
- Turn Taking：TOR 越高越好，RL 越低越好；
- User Backchannel：RsR 越高越好；
- User Interruption v1：TOR 越高越好，RL 越低越好；
- User Interruption v1.5：RpR 越高越好，SL/RL 越低越好；
- Overall：turn-management accuracy 越高越好，latency 越低越好。

论文中文官方结果：

| Pause TOR ↓ | Turn TOR ↑ | Turn RL ↓ | Backchannel RsR ↑ | Interrupt v1 TOR ↑ | Interrupt v1 RL ↓ | v1.5 RpR ↑ | v1.5 SL ↓ | v1.5 RL ↓ | Overall ACC ↑ | Overall latency ↓ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.038 | 0.994 | 0.767 | 0.800 | 0.994 | 1.089 | 0.830 | 0.380 | 1.150 | 0.916 | 0.847 |

本地复现表：TBD。随机组件至少重复 3 次，报告 mean/std、seed、失败率和系统版本。

## 6. Continuation step 与性能变化

预注册 checkpoint：

```text
0, 5, 10, 20, 30, 45, 60, 90, 120, 180, 240, 300 optimizer steps
```

Easy Turn 结果表：

| Local step | Estimated total step | LR | ZH Complete | ZH Incomplete | ZH Macro | EN Complete | EN Incomplete | EN Macro | 相对 step 0 | 判定 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | ≈1,800 | — | TBD | TBD | TBD | TBD | TBD | TBD | — | 官方基线 |
| 5 | ≈1,805 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 10 | ≈1,810 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 20 | ≈1,820 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 30 | ≈1,830 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 45 | ≈1,845 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 60 | ≈1,860 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 90 | ≈1,890 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 120 | ≈1,920 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 180 | ≈1,980 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 240 | ≈2,040 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 300 | ≈2,100 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

每个点还需报告 train/validation loss、五状态 accuracy、每 head 有效目标数、预测状态分布、AMP overflow/跳步、梯度范数、累计曝光量、耗时和峰值显存。

## 7. 退化判定与统计方法

以同一 runner 得到的官方 step 0 逐样本预测为配对基线：

- 几乎未下降：EN、ZH macro 降幅均 ≤1.0 个百分点，任一单类降幅 ≤2.0 个百分点；
- 明显下降：任一语言 macro 降幅 >3.0 个百分点，或任一单类降幅 >5.0 个百分点，并在相邻下一个 checkpoint 再次出现；
- 中间范围标为灰区，结合 95% paired bootstrap CI、McNemar 检验、validation 和预测分布解释；
- 中文提升与英文遗忘分别报告，不能只报一个合并平均数。

最终回填：

```text
最后一个几乎未下降的 step 区间：TBD
首次明显下降的 step：TBD
validation 最佳 step：TBD
推荐交付 checkpoint：TBD
推荐原因：TBD
```

预注册网格用于描述 step–性能曲线，不能看完 test 结果再改变 LR 或重新定义阈值。超参数和 checkpoint 主选择依据为 group-aware validation；论文 benchmark 用于冻结后的对比和汇报。

## 8. 重要工程修复与验证

官方 Stage 3 对某一训练样本未涉及的状态 head 使用全 `-100` 标签。直接调用 cross-entropy 会产生 NaN。本项目 runtime 对空 head 返回与计算图相连的有限 FP32 0，保留原有 `-100` 屏蔽语义，不伪造标签。

真实 5-step 预检：

| 项目 | 结果 |
| --- | --- |
| 成功 optimizer updates | 5 |
| 可训练参数 | 13,505,536 |
| FP16 scale | 65,536 经 2 次可恢复 overflow 降至 16,384 |
| CUDA peak memory | 8,144,328,192 bytes |
| projector 参数变化 | L2 0.0207257 |
| 紧凑 checkpoint | 162,241,270 bytes |
| checkpoint SHA-256 | `f12a49a392f7ad319e56c4cb75f21ce8f027f2fb76d9ea88c7799762efa6c4de` |
| checkpoint 重载 | 通过 |

该 5-step run 只证明训练链路可用，不代表正式模型性能。

## 9. 可复现性记录

最终文档附上：

- 训练源、评测集、模型、代码 revision 和 SHA-256；
- 完整配置和命令；
- GPU、driver、CUDA、PyTorch、Transformers、PEFT、FunASR/ModelScope 版本；
- train/validation split manifest 与泄漏审计；
- 每个 checkpoint 的 step、LR、有效 batch、累计样本、epoch-equivalent 和 hash；
- 每条 benchmark 样本的预测、时序、正确性与耗时；
- 随机 seed、重复次数、均值/标准差、置信区间和失败样本；
- 任何相对官方代码的补丁及其必要性。

## 10. 局限与风险

1. 官方权重不含原始 optimizer/scheduler/global step，`1,800` 只是低置信度估计。
2. DuplexConv 状态中有 1,599 个 Qwen 辅助标签，不能等同人工标注。
3. 训练 ASR 文本由 Paraformer 生成，存在伪标签错误。
4. Easy Turn 是固定 test set；多 checkpoint 评测可能造成隐性 test overfitting，因此使用预注册网格并禁止据其修改超参数。
5. Full-Duplex-Bench 的系统结果受 LLM、TTS、ASR、网络/调度和随机性共同影响，不宜单独归因于 SoulX checkpoint。
6. 本机 GPU 与论文 L20/H20 环境不同，吞吐和实测延迟不可直接横向比较。

## 11. 会议结论页

> TBD：实验完成后将本节整理为一页，包括一张 step–performance 曲线、一张关键 checkpoint 对比表、三条结论、两条局限和最终模型路径/hash。
