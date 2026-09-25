# SoulX Table 2 评测仓库静态审计报告 v1

审计时间：2026-09-12（UTC）  
审计结论：**Table 2 复现门禁不通过；停止进入基线评测。**  
诚信结论：**未发现直接伪造或按论文目标值修改预测的静态代码证据，但不能据此证明仓库从未发生过结果驱动调参。**

## 1. 审计对象与边界

- 仓库：`https://github.com/time-northern/soulx-duplug-eval.git`
- 固定 commit：`ce779713513f049b020019c8eab7d71f1080131d`
- 本地基准副本：`third_party/soulx-duplug-eval-ce779713`
- Git 状态：detached HEAD、工作树干净、sparse checkout 仅展开代码
- 派生运行副本：`runtimes/soulx_table2_eval_ce779713/SoulX-Duplug-Eval`
- 派生代码改动：无；仅新增来源说明、校验清单及数据软链接
- 未执行：依赖安装、模型下载、GPU 推理、正式评测、付费 API

审计对照的一手来源：

- [SoulX-Duplug 论文（arXiv:2603.14877）](https://arxiv.org/html/2603.14877)
- [Full-Duplex-Bench 官方仓库](https://github.com/DanielLin94144/Full-Duplex-Bench)

## 2. 执行摘要

该仓库不是论文 Table 2 的完整复现实现。仓库 README 明确将其定义为
`SoulX state-only evaluation`：它只读取 SoulX-Duplug 的逐块状态输出，不加载
Qwen2.5-7B-Instruct，不运行 IndexTTS-1.5，也没有 TTS 取消链路。默认流程计算的是
自定义的 interruption、rejection、background-noise 以及 Easy-Turn/VAD 指标。

论文 Table 2 则评测完整模块化全双工系统：

```text
SoulX-Duplug + Qwen2.5-7B-Instruct + IndexTTS-1.5
```

并在 Bilingual Full-Duplex-Bench 上计算 Pause Handling、Turn Taking、User
Backchannel、User Interruption v1/v1.5 及 Overall Score。两者的系统边界、场景、
行为观测量和指标均不等价。因此，即使该仓库运行成功，其结果也不能写成论文
Table 2 复现结果，更不能用于判断官方模型是否达到论文 Table 2。

静态代码中未发现以下直接造假模式：

- 按 checkpoint 文件名或模型名切换有利评分规则；
- 硬编码 SoulX 论文 Table 2 的目标分数；
- 将样本标签直接写入模型预测；
- 手工覆盖单条预测结果；
- 失败样本自动重试至成功后只保留最好结果；
- 调用付费 LLM/API 充当隐藏裁判。

但仓库存在证据可覆写、数据身份未冻结、测试不自包含等严重可复现性缺口。故审计
不能给出“实现无问题”的结论。

## 3. 与论文 Table 2 的逐项对照

| 论文 Table 2 要求 | 被审计仓库实际实现 | 结论 |
| --- | --- | --- |
| SoulX-Duplug 状态控制 | `service.model.TurnModel` 或官方 state inference | 部分具备 |
| Qwen2.5-7B-Instruct 响应生成 | 无相关加载、调用或输出 | 缺失 |
| IndexTTS-1.5 语音合成与取消 | README 明确“不包含 TTS 取消链路” | 缺失 |
| 模拟在线完整系统 | 仅对 state model 做 160 ms chunk 推理 | 不等价 |
| Pause Handling v1 TOR | 配置和入口中均无 `pause_handling` | 缺失 |
| Turn Taking v1 TOR / RL | 配置和入口中均无 `turn_taking` | 缺失 |
| User Interruption v1 TOR / RL | 只检查标注窗口内是否出现 public `nonidle` | 指标不等价 |
| User Backchannel v1.5 RsR | 只检查 raw `backchannel` 且无 public `nonidle` | 指标不等价 |
| User Interruption v1.5 RpR / SL / RL | 无助手停止、恢复、响应音频链路 | 缺失 |
| Overall ACC | 无论文公式 `(1-Pause TOR, Turn TOR, RsR, RpR)` 聚合 | 缺失 |
| Overall Latency | 无论文中所有 RL/SL 的均值 | 缺失 |

论文将 Table 2 定义为系统级测试；该仓库的 `current_time` 只覆盖音频块可用时间与
state-model 处理时间，不包含 LLM 首包、TTS 首包、助手实际开始发声或停止发声，
因此不能替代 Table 2 的 RL/SL。

## 4. 数据审计

### 4.1 Git 仓库实际附带的数据

仓库 commit 随 Git 提交的是：

| 数据 | 文件数 | 逻辑字节数 | 在该仓库中的用途 |
| --- | ---: | ---: | --- |
| HumDial-FDBench | 2,394 | 728,797,180 | 中文 background speech / talking-to-other 可选诊断 |
| SID-Bench | 501 | 697,860,578 | 自定义 background-noise 拒识指标 |
| 合计 | 2,895 | 1,426,657,758 | 不是完整 Table 2 数据集 |

仓库没有提交以下配置所引用的数据：

- `Full-Duplex-Bench-en`
- `Full-Duplex-Bench-zh`
- `Easy-Turn-Testset-en`
- `Easy-Turn-Testset-zh`

因此不能把 Git 仓库附带的约 1.1 GiB 物理占用笼统称为“Table 2 数据集”。其中
HumDial 大部分不进入默认正式场景，SID-Bench 也不是论文 Table 2 的五组主指标数据。

### 4.2 数据盘迁移与软链接

仓库附带数据已复制至：

```text
/root/autodl-tmp/dataset/soulx_duplug_eval/repo_bundled_ce779713/
```

迁移前后完整 SHA-256 清单一致：

```text
文件数：2,895
逻辑字节：1,426,657,758
SOURCE_FILES.sha256 的 SHA-256：
009460d293a9b63bcd253f20c4d637f9166e0e0cccfc4e8f39b9b787a22244ae
```

运行副本已创建 5 个数据链接：HumDial、SID-Bench、中文 Full-Duplex-Bench、
英文 Easy-Turn、中文 Easy-Turn。英文 Full-Duplex-Bench 当前仅有已验证 ZIP，
没有伪装成可用解压目录，故暂不链接。

### 4.3 数据完整性缺口

1. FDB 正式发现逻辑只要求目录下至少存在一个严格命名的 `input.wav`；没有核对论文或
   官方数据集预期总数，也没有记录逐文件哈希。一个内部一致但缺样本的目录仍可能被
   标为完整运行。
2. SID-Bench 检查标注与 WAV 集合的一致性，但没有冻结官方 500 条的身份哈希；若标注
   与音频一起被一致地裁剪，仍可能通过。
3. manifest 记录 checkpoint 路径和字节数，不记录 checkpoint SHA-256；同一路径内容
   被替换后无法从 manifest 检出。
4. 原始评测配置、代码 commit、数据清单和推理证据之间没有统一的不可变哈希闭包。
5. HumDial background-speech 对非恰好两个 speech segment 的样本直接 `continue`；
   README 披露了一个具体排除样本，但代码没有把这类排除写入错误清单。该场景当前是
   可选诊断，不影响本次“不能复现 Table 2”的主结论。

## 5. 实验诚信检查

### 5.1 未发现的直接造假模式

- 全仓搜索未发现论文 Table 2 的 SoulX 目标数值硬编码。
- 未发现按 checkpoint 名、权重名或目标模型名改变评分公式的分支。
- Easy-Turn 的 `label` 用于选择数据目录、验证样本数和写结果元数据；模型配置和模型
  调用未读取该标签来生成状态预测。
- FDB 时间戳只用于确定评分窗口，没有作为模型输入。
- 单条常规模型推理抛异常会使整个运行失败，并把 manifest 标记为不完整；未见自动
  丢弃该条后继续生成正式分数。
- `--limit-per-dataset` 只能生成 diagnostic subset，manifest 不会标为正式完整运行。
- `run_silero_vad_oracle.sh` 不被默认 `run_all.sh` 调用，当前 Easy-Turn 主规则也明确
  不使用 endpoint oracle。

以上只能说明固定 commit 的静态代码中没有这些直接路径，不能证明阈值或规则在提交前
没有根据已见结果调整。仓库只有三个 commit，整个 7,220 行框架集中在单一提交中，
缺少足够的规则演化历史来审计是否存在事后调参。

### 5.2 高风险可变证据机制

仓库宣称推理 JSONL 为“不可变证据”，但同时提供三个原地替换入口：

- `rerun_interruption.py`
- `rerun_background_noise.py`
- `rerun_vad.py` / `replace_scenario_inference()`

这些入口能够替换完整 run 中的单一场景证据并重新生成报告，却没有：

- 保存旧 evidence 的 SHA-256 或只读快照；
- 记录执行时的代码 commit / code SHA；
- 建立 append-only rerun 历史；
- 在新 run ID 下保存新证据；
- 对所有场景统一重跑。

其中 background-noise rerun 还能按当前配置改变 duration cap、尾静音和样本数量，
随后直接更新原 manifest。该设计不等于已经发生造假，但允许结果查看后的选择性协议
更新而缺乏可审计轨迹，不能用于严格论文复现。

### 5.3 静默失败

Paraformer 与 SenseVoice 的 ASR wrapper 捕获任意异常后返回空字符串，不增加
manifest error count，也不记录结构化失败样本。这不会直接抬高分数，但会让推理错误
被混入模型行为，降低结果的可解释性和可复现性。

## 6. 代码与测试检查

| 检查 | 结果 |
| --- | --- |
| Python AST 解析 | 25/25 文件通过 |
| 根目录 Shell `bash -n` | 7/7 文件通过 |
| Easy-Turn 独立单元测试 | 5/5 通过 |
| 主测试模块 | 导入失败，36 个测试未运行 |
| 固定副本 Git fsck | 通过 |
| 固定副本 Git 状态 | detached HEAD、干净 |
| 运行副本与固定副本代码逐字节比较 | 一致 |
| 5 个数据软链接 | 均可解析 |

主测试模块失败原因不是 GPU 或模型依赖，而是代码在导入阶段强制加载仓库外文件：

```text
<evaluation-repo-parent>/scripts/search_stage3_loss_weights.py
```

当前仓库及本项目均没有该脚本，因此该仓库的测试套件不自包含。为了避免制造虚假通过，
本次没有创建占位脚本，也没有修改测试跳过失败项。

## 7. 资源与依赖结论

从代码只能确认该 state-only 框架需要：

- SoulX-Duplug 推理/训练代码及 checkpoint；
- Qwen3-0.6B expand-vocab backbone 与 GLM-4-Voice tokenizer；
- 中文 Paraformer、英文 SenseVoice；
- PyTorch、Transformers、ModelScope/FunASR、SoundFile、soxr、PyYAML 等。

它不包含 Qwen2.5-7B-Instruct 和 IndexTTS-1.5 的加载或服务代码，因此无法从本仓库
可靠估计真正 Table 2 的显存、磁盘、并发服务结构或总运行时。当前 32 GiB 单卡能否
同时承载论文完整链路也不能据此承诺。

本阶段实际资源：GPU 0、付费 API 0、运行代码约 356 KiB；固定 Git 副本因历史中包含
音频对象仍占约 367 MiB。sparse checkout 后系统盘可用空间由约 3.1 GiB 回升至约
4.2 GiB。数据盘新增仓库附带数据约 1.2 GiB 物理占用。

## 8. 门禁结论与后续建议

### 8.1 当前门禁

```text
Table 2 repository static audit: FAILED / STOP
```

失败原因是**评测范围与论文 Table 2 不匹配**，不是因为已观察到的分数好或坏。
因此禁止直接进入“使用官方 SoulX 权重跑 Table 2 基线”阶段。

### 8.2 可保留价值

该仓库可在另行冻结和加固后作为补充的 state-only robustness diagnostic，尤其用于：

- interruption 窗口内 nonidle 检出；
- backchannel state 行为；
- SID-Bench 背景噪声误触发；
- Easy-Turn/Table 3 终态诊断。

这些结果必须使用其真实指标名，不能称为论文 Table 2。

### 8.3 下一步需重新审批的方案

1. 请仓库提供方给出真正包含 Qwen2.5-7B、IndexTTS-1.5、完整系统事件日志以及
   Table 2 五组任务/十一列指标的实现；或
2. 以 Full-Duplex-Bench 官方 v1/v1.5 代码为指标基准，在独立 runtime 中为官方
   SoulX 模块化系统编写适配器；
3. 在任何候选 checkpoint 进入 Table 2 前，冻结官方模型基线的样本清单、数据哈希、
   checkpoint SHA、超时/失败分母、重试规则、场景公式和允许复现容差；
4. 修复必须新建派生 runtime 和新版本协议，不修改本次固定审计副本。

上述任何一条都属于新执行阶段，必须先提交具体文件、依赖、模型、资源、耗时与验证
方案，并由项目负责人确认。
