# DuplexConv Edu_0001–Edu_0045：Complete/Incomplete 平衡 A/B/C/D 实验持续总结

更新日期：2026-09-13  
文档状态：A/B/C/D 消融训练及冻结 Table 3 step 5/10 已完成；会议汇报最终版  
当前阶段：完整结果与证据门禁已冻结；最终 HTML 与 audit JSON 同步生成  
HTML：`evaluation_reports/duplexconv_edu0001_0045_abcd_experiment_summary.html`

## 1. 研究问题

使用原始 Edu_0001–Edu_0045 数据续训练后，Table 3 的 Complete 准确率随 step 上升，而 Incomplete 准确率明显下降。本轮用 2×2 消融区分两个可能因素：

1. Complete 与 Incomplete 状态 head 在训练窗口中的激活次数不平衡；
2. 训练配置把 Complete/Incomplete loss rate 设为 `0.24/0.13`，进一步偏向 Complete。

本轮目标是比较四类 Table 3 的等权宏平均和各类别变化，不以“模型更愿意回答”的工程偏好作为选择依据。

## 2. 四种配置

| 组别 | 训练数据 | Complete/Incomplete loss rate | 单一变量 | 当前状态 |
|---|---|---:|---|---|
| A | 原始冻结 train | 0.24 / 0.13 | 首轮对照 | 已完成 |
| B | 原始冻结 train | 0.185 / 0.185 | 只改变 loss rate | 30-step训练及0/5/10部分内部validation完成 |
| C | balanced-CI train | 0.24 / 0.13 | 只改变采样 | 30-step训练及0/5/10部分内部validation完成 |
| D | balanced-CI train | 0.185 / 0.185 | 同时改变两项 | 30-step训练及0/5/10部分内部validation完成 |

`0.185/0.185`保持两类终态 loss rate 总和为`0.37`，避免同时改变终态任务相对其他 head 的总强度。B/C/D 均必须从同一个官方发布 checkpoint 开始，不能从 A 或其他实验的 checkpoint 接续。

## 3. 数据对比

### 3.1 训练视图

| 指标 | 原始视图（A/B） | balanced-CI 视图（C/D） |
|---|---:|---:|
| 训练窗口 | 99,334 | 67,957 |
| 目标视角 chunk 时长 | 972.995 h | 681.562 h |
| Complete+Incomplete 窗口 | 40,261 | 40,261 |
| Complete-only 窗口 | 38,298 | 6,921 |
| Incomplete-only 窗口 | 6,921 | 6,921 |
| 两者均不含窗口 | 13,854 | 13,854 |
| Complete-head-active 窗口 | 78,559 | 47,182 |
| Incomplete-head-active 窗口 | 47,182 | 47,182 |
| Complete token | 210,359 | 128,903 |
| Incomplete token | 78,037 | 78,037 |
| Complete/Incomplete token 比 | 2.696:1 | 1.652:1 |
| 源会话数 | 21,591 | 19,850 |

这里的“1:1 均衡”专指状态 head 被激活的窗口数`47,182:47,182`，不表示 token 数也达到 1:1。官方补丁版训练代码在每个样本内对每个 head 的有效 token 先取 mean cross entropy，且 local batch 为 1，因此窗口级 head 激活次数是主要采样口径；token 数仍完整报告。

### 3.2 抽样规则

- 保留全部 Complete+Incomplete、Incomplete-only 和两者均不含窗口；
- 从 38,298 条 Complete-only 窗口中选择 6,921 条，不做有放回过采样，不制造重复行；
- 按`Edu shard × source_ntrack × Backchannel 是否存在`分层，以最大余数法分配精确名额；
- 层内按`SHA256(seed=42 + NUL + row index)`升序选择，结果不依赖 Python 随机数版本；
- 选中窗口保持父 Parquet 中的原始行顺序；
- 删除 31,377 条 Complete-only 窗口；选中 Complete-only index identity 为`46bf6c02c98589af0fd979f57bf0ae1a18ec29fb2cbabbaf474714839aa3394a`。

### 3.3 验证集

四组都使用同一份自然分布验证集：441 个完整源会话、2,061 条窗口、20.450 小时。balanced-CI 产物中的验证 Parquet 是父产物的字节级副本，SHA-256 均为：

```text
6bc82ad10fd7c4e4baa0b0902faec589029e6673e70cb802ca699680fbd017fb
```

训练/验证完整源会话泄漏均为 0；没有从 validation 回流训练样本。

## 4. 固定训练方法

四组除数据采样、Complete/Incomplete loss rate 和各自输出标识外，保持以下条件一致：

- 官方上游基准 commit：`928b06508ed2de1344208d06fb1f6fb2ebfb1df5`；
- 补丁 runtime：`runtimes/SoulX-Duplug-928b065-official-continual-v1`；
- 官方 Lightning 调用链：`finetune.py → Trainer.fit() → training_step() → configure_optimizers()`；
- 官方发布基础权重 SHA-256：`b0703dea0b1dbb1cd51e6e7b6514c60907ea4d4b6752cecc3f71cb6445650dbe`；
- LoRA `r=32`、`alpha=64`、`dropout=0.1`，projector 可训练；
- local batch=1、梯度累积=576、单卡有效 batch=576；
- seed=42、30 个 local optimizer step；checkpoint=`1/2/3/5/10/20/30`；
- 公开`total_steps=1800`只作为起点的低置信度估计，不宣称恢复官方 optimizer、scheduler、GradScaler 或 Trainer global step；
- 保留既有空 head 有限零损失、严格 checkpoint 加载、冻结会话级 split、scheduler 起点及审计/checkpoint 补丁；
- 不使用旧的自定义 optimizer loop。

所有训练 index 均不以`fe`开头，B/C/D loader 门禁实测`switch_loss_active_train_rows=0`，因此`*_loss_rate_switched`不生效。

## 5. A 组既有结果

### 5.1 内部 validation

内部 validation 在查看 Table 3 前按冻结规则选择 local step 1。主要轨迹如下：

| Local step | 总 accuracy | Complete | Incomplete | Guard |
|---:|---:|---:|---:|---|
| 0 | 84.40% | 29.00% | 10.47% | 基线 |
| 1 | 83.96% | 30.13% | 10.72% | 通过 |
| 2 | 83.45% | 33.06% | 12.21% | 通过 |
| 3 | 82.80% | 40.43% | 17.78% | 拒绝 |
| 5 | 80.56% | 61.03% | 32.16% | 拒绝 |
| 10 | 73.12% | 77.44% | 33.33% | 拒绝 |
| 20 | 77.80% | 81.39% | 31.78% | 拒绝 |
| 30 | 81.69% | 81.27% | 25.71% | 拒绝 |

### 5.2 Table 3

| Local step | EN Complete | EN Incomplete | EN Macro | ZH Complete | ZH Incomplete | ZH Macro |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 78.93% | 89.63% | 84.28% | 87.67% | 80.33% | 84.00% |
| 1 | 82.39% | 84.95% | 83.67% | 88.67% | 77.67% | 83.17% |
| 2 | 90.25% | 84.62% | 87.43% | 89.67% | 74.00% | 81.83% |
| 3 | 96.54% | 76.92% | 86.73% | 92.00% | 69.00% | 80.50% |
| 5 | 97.80% | 62.21% | 80.00% | 91.67% | 64.33% | 78.00% |
| 10 | 97.80% | 22.41% | 60.10% | 94.67% | 53.67% | 74.17% |
| 20 | 92.14% | 31.77% | 61.96% | 96.67% | 53.00% | 74.83% |
| 30 | 93.40% | 23.08% | 58.24% | 98.33% | 52.33% | 75.33% |

A 组结果说明 Complete/Incomplete 发生显著方向性分化，但不能单凭相关性判定是采样比例还是 loss rate 导致，B/C/D 用于做单变量归因。

## 6. 当前门禁状态

2026-09-12 的数据构造和静态/loader 门禁已完成：

- 新采样模块、CLI 和 4 个针对性测试；连同 6 个既有续训测试共 10 项通过；
- 原始输入 Parquet、45 份 metadata 和父 manifest 哈希验证通过；
- 67,957-row balanced-CI 产物以临时目录构造，全部闭包通过后原子发布；
- A/B/C/D YAML 机械差异门禁通过，未发现未披露超参数漂移；
- B/C/D 分别通过补丁版官方`State_Prediction_DataModule.setup("fit")`；
- B loader：99,334/2,061 rows，split identity=`a2599191...dc6fb`；
- C/D loader：67,957/2,061 rows，split identity=`c63fbb08...314177`；
- 三组 source leakage=0、Parquet 哈希和 row identity 均通过；
- 在该构造与 loader 门禁阶段没有构建模型、optimizer 或 Trainer，没有使用 GPU、付费 API，也没有启动训练或 Table 3；后续获批的 B/C/D 训练完成情况见第 11 节。

loader 首次门禁发现 Hugging Face 会把压缩 Parquet 展开成临时 Arrow cache；在观测到约 1.3 GiB、超过原估算后按审批规范主动终止并自动清理。负责人批准峰值 3 GiB 后重跑，观测到的最大缓存约 1.4 GiB，三组通过后自动清理，无缓存残留。

## 7. 实验诚信边界

提出 B/C/D 之前已经查看过 A 的 Table 3 结果。因此，B/C/D 的 Table 3 必须标记为“事后假设驱动的开发性评测”，不能宣称是完全独立、无偏的最终测试。不得修改 Table 3 样本、标签、ASR、解码规则、终态规则、尾静音、阈值、分母或失败处理来提高结果。

Table 2 暂按负责人指令延期，不在当前实现与执行范围内。未来若恢复，应先完成协议审计和官方权重基线门禁，再对预先选定的候选做一次冻结确认性评测；不得根据 Table 2 返回调参。

## 8. 关键产物与身份

| 产物 | SHA-256 |
|---|---|
| A config | `08f276af19e469355a78f3b12fe1e4a75791a247406a9b54532aef3403f39a1b` |
| B config | `6c659bbf97e860cd668eb5ed79e77e9160e27f562fee4a17a723c4e20dd5a569` |
| C config | `b14e91e7bbe3ad986641c60b7c781cedf83a57899f76ea4a0724237824613557` |
| D config | `513a439e51717e10c286b9438ca3e8d671e30075234334d04e39f2dd53dc64b9` |
| balance module | `13ba6223b67b6f523d33bcac2fe9aa2f69cc53eda691602e80962f8339a58c5b` |
| builder CLI | `59b7dbd30586d5555ce124bdab6343d9bcb5c919ed83e01c8b30299881cfaf10` |
| balanced train Parquet | `cf8e4a59d2aef793e81bac0cf734b4dd75a8f59371457d3838903f656675c432` |
| balanced validation Parquet | `6bc82ad10fd7c4e4baa0b0902faec589029e6673e70cb802ca699680fbd017fb` |
| balanced split manifest | `b7b162df52ea1ebce29c129d1b7a131163c4f657254eb562627ee89669e8ca73` |

balanced-CI 数据目录：

```text
/root/autodl-tmp/dataset/duplexconv/splits/edu0001_0045_stage3_zh_v2_seed42_group98_2_balanced_ci_v1/
```

## 9. 后续阶段

B/C/D 官方 Lightning 30-step 训练已经完成；均保存 checkpoint `1/2/3/5/10/20/30`，本阶段外部精确内部 validation 按负责人调整后的范围只执行 step `0/5/10`。这三个点形成部分轨迹，不得表述为七点完整选择结论。官方训练内部固定在 step `10/20/30`触发的 validation 未改变。

已授权并运行：B/C/D 使用完全相同的冻结 Table 3 协议各评测 step 5、10。当前三卡调度首先并行运行新鲜官方基线、B5、C5；新基线精确匹配冻结 A 基线后才调度其余点。

尚未授权：

1. 对 step `1/2/3/20/30`补做外部精确内部 validation；
2. B/C/D 的其他 Table 3 checkpoint；
3. Table 2；
4. 根据本轮 Table 3 结果返回修改训练数据、loss rate、checkpoint权重或评测规则。

每一阶段仍须先按`agent_governance/EXECUTION_APPROVAL_PROTOCOL.md`披露执行方案并获得明确确认。本文件用于辅助任务连续性，不能替代负责人的最新对话指令。

## 10. B/C/D 训练队列启动记录（2026-09-13）

- 负责人批准单 GPU 按 B→C→D 严格串行，每组从相同官方发布权重独立开始；
- 随后将外部内部-validation范围收缩为 step 0、5、10；训练仍保存七个冻结 checkpoint，训练内置 validation 仍为 step 10/20/30；
- preflight 验证 runtime commit=`928b06508...`、tracked diff=`b1d55ba7...`、审计文件、基础权重、配置、split、输出无冲突和 30 GiB 空间门禁；实际可用 100.43 GiB；
- 新增的调度/门禁代码连同相关既有测试共 15 项通过；
- tmux=`soulx_abcd_train_bcd_v1`；live manifest=`/root/autodl-tmp/dataset/duplexconv/training/edu0001_0045_abcd_queue_v1/orchestration_manifest.json`；
- 启动时实际状态为 B step 0 validation；Table 2/Table 3 均未启用；
- B step 0 已通过：精确 token-weighted accuracy=`0.8439856750`、objective=`0.8449824423`；optimizer created=false、training updates=0；随后自动进入 B 官方训练；
- 队列任一身份、有限性、optimizer step、有效 batch、checkpoint 或磁盘门禁失败即停止，不自动重试、恢复、删除或进入下一组。

## 11. B/C/D 训练与部分内部 validation 完成记录（2026-09-12 UTC）

队列于`2026-09-12T16:10:09Z`开始、`2026-09-12T21:13:02Z`完成，墙钟时间约 5 小时 2 分 53 秒。B、C、D 分别耗时约 1:40:44、1:41:39、1:40:31。每组均满足：

- 从相同官方发布基础权重独立开始，而非从 A 或前一组接续；
- 使用既有补丁版官方 Lightning 调用链完成 30 次 optimizer update；
- 实际暴露 17,280 条训练样本，有效 batch=576；
- 生成 step `1/2/3/5/10/20/30`七个 compact checkpoint，逐个 tensor audit 全部通过；
- 外部精确内部 validation 仅执行 step `0/5/10`；
- 峰值 CUDA allocated memory：B=`19,231,065,600` bytes，C/D=`19,230,263,808` bytes，约 17.91 GiB；
- 未运行 Table 2、Table 3，未调用付费 API；完成后 tmux/相关进程均退出，GPU 空闲。

三组共用的 step 0 是同一官方基础权重，其 token-weighted accuracy=`84.3986%`、Complete=`29.0002%`、Incomplete=`10.4709%`。部分轨迹如下：

| 组别 | Local step | 总 accuracy | Idle | Non-idle | Backchannel | Complete | Incomplete | 5pp状态保护门禁 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 共用基线 | 0 | 84.40% | 97.26% | 86.99% | 19.00% | 29.00% | 10.47% | 基线 |
| B：等权 | 5 | 80.56% | 93.92% | 70.71% | 17.29% | 57.22% | 36.80% | 未通过 |
| B：等权 | 10 | 73.42% | 81.67% | 51.93% | 14.99% | 67.17% | 48.76% | 未通过 |
| C：均衡采样 | 5 | 80.51% | 93.97% | 70.62% | 17.33% | 57.95% | 35.38% | 未通过 |
| C：均衡采样 | 10 | 73.25% | 81.18% | 50.47% | 15.08% | 70.08% | 44.80% | 未通过 |
| D：联合 | 5 | 80.54% | 94.00% | 70.74% | 17.32% | 53.01% | 42.50% | 未通过 |
| D：联合 | 10 | 73.20% | 80.96% | 50.65% | 14.90% | 55.30% | 62.27% | 未通过 |

直接观察到：三种干预都在 step 5 和 10 同时提高了内部验证的 Complete 与 Incomplete accuracy；D 对 Incomplete 的提升最大，但其 Complete 提升相对较小。与此同时，总 accuracy、Idle、Non-idle 和 Backchannel 均下降，step 5 的 Non-idle 已下降约 16.25–16.37 个百分点，超过预注册的单状态 head 最大下降 5 个百分点门槛。因此三个**部分**选择索引都返回 step 0，理由均为“没有 continuation checkpoint 通过 5pp state-head guard”。

这只能说明 step 5/10 不满足原先用于保护全部状态 head 的内部门禁，不能推导 B/C/D 在 Table 3 上的 Complete/Incomplete 指标，也不能把 step 0 写成七点完整选择结果。负责人随后批准三组统一评测 step 5、10；执行记录见第12节。

关键证据 SHA-256：

| 证据 | SHA-256 |
|---|---|
| 队列 orchestration manifest | `1b8684717c4f7fb3a9d36ebb121079b623d8a9628601e786edc634b67749ed51` |
| B training manifest | `65d35a758f93c97394a67af8d023b568a8e3d74d1ff2c4e54aade27372765f4d` |
| C training manifest | `7e21063def424637402920ecc77e673c9a04393f7debe715b3ea54b9af6f8969` |
| D training manifest | `fc989b81e22c35383537df88759330696884ad833ceaa57dc0bb924857563a19` |
| B step 5/10 partial index | `9e7112caca1e1ecfb63cac6a5824416a4768d16ac46bc18ea7828140531eeef1` |
| C step 5/10 partial index | `328b13165b1e744bcb11b109192f49d1874238d0a741905806b787b95a3dbd90` |
| D step 5/10 partial index | `7c14423b6ae1d862604e77b285254b92c506b49720b41ea3450e14f2b2552416` |

三组训练目录各约 4.2 GiB；完成时数据盘可用约 86 GiB。项目 checkpoints 目录中的 B/C/D 软链接均已建立并指向对应数据盘目录。

## 12. B/C/D Step 5/10 开发性 Table 3 启动记录（2026-09-13）

- 负责人批准只评测 B/C/D 的 step 5、10，并在确认当前有三张 GPU 后要求并行执行；
- 评测固定为 EN Complete 318、EN Incomplete 299、ZH Complete 300、ZH Incomplete 300，沿用同一`last-terminal-v1`、ASR、样本、阈值、分母及配对统计；
- 新鲜官方 step 0 基线的精确门禁为`251/318、268/299、263/300、241/300`；负责人允许其与 B5/C5 同时运行，若不一致则记录原因并删除仅属于本轮的 B5/C5 临时结果；
- 三卡波次固定为：`baseline+B5+C5`、`D5+B10+C10`、`D10`。每张 GPU 同时只运行一个 checkpoint，该 checkpoint 内四类串行；
- Table 3 runtime commit=`b17bcf903b9bd896238a4ee9fec495fd75df1401`，官方 upstream commit=`928b06508ed2de1344208d06fb1f6fb2ebfb1df5`；
- 配置 SHA-256=`522ddcd9a1daa6343354651e44ca2a08549912eec29a0c14b49a840e56327f64`，调度器 SHA-256=`a046fa967d340a2b4edb954d4f5c842e5685c1f19670ac01d9b08bfa33133523`，5项针对性测试通过；
- 第一次启动在任何模型初始化或样本推理前因英文数据`official-os-walk`身份漂移停止。独立检查确认617个WAV的文件名、字节和哈希完全一致，差异仅来自目录项遍历顺序；原始ZIP完整性通过，按ZIP顺序临时解压后 Complete/Incomplete 冻结身份均精确匹配；
- 当前英文目录已由验证通过的新目录接替，原目录保留在`Easy-Turn-Testset-en-order-drift-backup-20260913`。第一次失败产物和日志完整归档，没有覆盖；
- 正式任务于`2026-09-13T02:29:31Z`进入首轮：GPU0=baseline、GPU1=B5、GPU2=C5。tmux=`soulx_abcd_table3_5_10_v1`；live manifest=`/root/autodl-tmp/dataset/soulx_duplug_eval/table3_abcd_step5_10_v1/orchestration_manifest.json`；
- 本阶段不运行 Table 2、不评测其他 checkpoint、不调用付费 API。预计墙钟约6–7小时，新增磁盘低于约0.6GiB。

### 12.1 首轮门禁与中断恢复记录（2026-09-13）

- 新鲜 step 0 基线四类全部完成并精确匹配冻结 A：EN Complete=`251/318`、EN Incomplete=`268/299`、ZH Complete=`263/300`、ZH Incomplete=`241/300`；证据审计通过，因此 B5/C5 结果有效且允许进入第二轮。
- 服务器随后在第二轮 EN Complete 中断：D5=`164/318`、B10=`163/318`、C10=`179/318`。这些文件的内部`status=running`，未被计作正式结果。
- 首次恢复在推理前暴露控制器缺陷：首次写入 JSON 后 checkpoint 的整数键变成字符串，旧实现恢复时却与内存整数键直接比较，导致无真实漂移也误报 identity mismatch。该失败没有归档或覆盖评测输出。
- 控制器补丁将身份统一为 JSON round-trip 规范形式，并只允许已知旧控制器 SHA-256=`a046fa96...9333523`迁移到新 SHA-256=`230c3103...8bdbf6`；任何其他字段漂移仍硬失败。新增回归与既有调度测试共11项通过，迁移详情写入 live manifest 的`identity_migrations`。
- `2026-09-13T04:46:39Z`显式`--resume`成功；三份中断半成品各自移动到`interrupted-retry-*`留证目录后，从对应类别头部重跑。第一轮12份完整结果按身份校验直接复用。当前第二轮为GPU0=D5、GPU1=B10、GPU2=C10，之后自动执行D10。
- 第二次服务器中断发生于停止监控器最后一次仍观测到D5/B10/C10三个active assignment之后，并非“只剩一张GPU”门禁触发。此时三组EN Complete、EN Incomplete、ZH Complete均已完成；ZH Incomplete分别为D5尚无首条持久记录、B10=`59/300`、C10=`72/300`。`2026-09-13T06:04:55Z`再次显式恢复，复用全部完整类别，三份ZH Incomplete半成品分别留证归档后重跑；新版独立监控器`/root/autodl-tmp/dataset/soulx_duplug_eval/table3_abcd_step5_10_v1/stop_at_one_active_receipt_resume2.json`继续执行负责人要求的单任务停止策略。

### 12.2 单任务停止点与当前开发性结果（2026-09-13）

- D5、B10、C10随后完成全部四类和continuation evidence gate，B/C/D索引分别完整到`5+10、5+10、5`。
- 控制器于`2026-09-13T06:32:44Z`进入最后的D10单任务。监控器连续三次确认仅`D10@GPU0`活跃，并于`06:32:57Z`按负责人要求终止主tmux；D10仅开始EN Complete初始化，持久样本记录为0。当前无评测进程，三张GPU均空闲。
- 监控回执最初停在`stopping`：第一版进程枚举误把承载两个session的tmux server计为评测进程，结束server时也结束了监控自身。事后核对确认主任务和所有子进程均已退出；回执据实闭合为`stopped_reconciled`，监控实现已限定为Python控制器/评测进程。该问题不影响任何已经完整的类别结果。

当前完成点的冻结Table 3结果如下；四类宏平均为四个类别accuracy的等权平均：

| 组别/step | EN Complete | EN Incomplete | ZH Complete | ZH Incomplete | 四类宏平均 | 相对基线 |
|---|---:|---:|---:|---:|---:|---:|
| 官方基线 | 78.93% | 89.63% | 87.67% | 80.33% | 84.14% | — |
| B5 等权loss | 96.86% | 69.57% | 89.00% | 69.33% | 81.19% | -2.95pp |
| B10 等权loss | 94.97% | 38.46% | 89.67% | 63.67% | 71.69% | -12.45pp |
| C5 平衡采样 | 97.17% | 66.89% | 89.00% | 67.33% | 80.10% | -4.04pp |
| C10 平衡采样 | 96.23% | 33.78% | 90.67% | 61.33% | 70.50% | -13.64pp |
| D5 联合平衡 | 94.03% | 75.25% | 87.00% | 70.67% | 81.74% | -2.41pp |

截至该停止点，D5是已评测候选中四类宏平均最高者，但仍低于官方基线；B/C在step10均出现比step5更严重的Incomplete下降。D10尚未评测，因此不能宣称四种配置的step5/10网格完整，也不能生成完整最终结论。

### 12.3 单 GPU 恢复 D10（2026-09-13）

- 负责人随后将容器缩容为单张RTX 4080 SUPER，并明确要求继续执行；因此此前“只剩一个GPU先停止”指令由本次新指令覆盖，旧停止监控不再启动。
- 恢复前复核配置、控制器、辅助脚本、Table 3 runtime commit、官方upstream commit和checkpoint身份均无漂移；数据盘可用约86GiB。
- `2026-09-13T06:55:44Z`使用同一控制器显式`--resume --devices 0`。B5/B10/C5/C10/D5按身份完整复用；此前D10只留下的空process log/trace目录进入`interrupted-retry-*`留证目录。
- 当前仅实际执行D10，tmux=`soulx_abcd_table3_d10_single_v1`；EN Complete已开始持久写入。完成四类与证据门禁后再生成完整MD/HTML/audit。

## 13. 最终 Table 3 结果与结论

### 13.1 完整结果

`2026-09-13T08:40:48Z`，D10 四类、continuation evidence gate 和最终 D 组索引完成；控制器以`status=complete`、`stage=table3_step5_10_complete`正常退出。B/C/D 均已完成 step 5、10 的冻结四类网格，当前无 tmux、评测进程或 GPU 占用。

四类宏平均定义为 EN Complete、EN Incomplete、ZH Complete、ZH Incomplete 四个类别准确率的等权平均，不按样本量加权。

| 组别 | Step | EN Complete | EN Incomplete | ZH Complete | ZH Incomplete | 四类宏平均 | 相对官方基线 | 相对同 step A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 官方基线 | 0 | 78.93% | 89.63% | 87.67% | 80.33% | 84.14% | — | — |
| A 原始训练 | 5 | 97.80% | 62.21% | 91.67% | 64.33% | 79.00% | -5.14pp | — |
| B 仅等权loss | 5 | 96.86% | 69.57% | 89.00% | 69.33% | 81.19% | -2.95pp | +2.19pp |
| C 仅平衡采样 | 5 | 97.17% | 66.89% | 89.00% | 67.33% | 80.10% | -4.04pp | +1.10pp |
| D 联合平衡 | 5 | 94.03% | 75.25% | 87.00% | 70.67% | **81.74%** | **-2.41pp** | **+2.73pp** |
| A 原始训练 | 10 | 97.80% | 22.41% | 94.67% | 53.67% | 67.14% | -17.01pp | — |
| B 仅等权loss | 10 | 94.97% | 38.46% | 89.67% | 63.67% | 71.69% | -12.45pp | +4.56pp |
| C 仅平衡采样 | 10 | 96.23% | 33.78% | 90.67% | 61.33% | 70.50% | -13.64pp | +3.37pp |
| D 联合平衡 | 10 | 90.88% | 54.85% | 82.00% | 75.00% | **75.68%** | **-8.46pp** | **+8.55pp** |

### 13.2 对两个假设的回答

1. **原始 loss 倾向确实是原因之一。** 在不改变数据的条件下，B 相对 A 的四类宏平均在 step 5/10 分别提高`+2.19/+4.56pp`。主要收益来自 Incomplete，代价是 Complete 有所回落。
2. **Complete/Incomplete active-row 失衡也是原因之一。** 在不改变 loss rate 的条件下，C 相对 A 在 step 5/10 分别提高`+1.10/+3.37pp`；同样表现为以部分 Complete 准确率换取 Incomplete 恢复。
3. **两者联合最好，但不是无代价修复。** D 相对 A 在 step 5/10 分别提高`+2.73/+8.55pp`，且在两个 step 都是 B/C/D 中宏平均最高者。step 10 的联合收益尤其明显，但 D10 的 EN Incomplete 仍比基线低`34.78pp`，ZH Complete 也比基线低`5.67pp`。
4. **等权 loss 的单因素宏平均贡献大于平衡采样。** B-A 在两个 step 都大于 C-A；但 D 的结果表明采样仍提供额外收益，不能据此认为采样无效。
5. **训练更久仍会恶化整体折中。** D 从 step 5 到 step 10 的四类宏平均下降`6.05pp`。其中 ZH Incomplete 从`70.67%`升至`75.00%`，但 EN Incomplete、EN Complete、ZH Complete 同时下降，说明不能假设各语言/类别随 step 单调一致。

### 13.3 模型选择结论

- 若主目标是本实验预注册的四类等权宏平均，**D5 是 B/C/D 六个新候选中最佳点**，为`81.74%`。
- D5 仍低于官方 step 0 基线`2.41pp`，也没有通过训练阶段用于保护全部状态 head 的 5pp 内部门禁。因此本实验没有产生可替代官方基线/既有 A 组选点的无条件胜者。
- 若只比较同 step 的消融效果，D 在 step 5 和 step 10 均为最优，说明“平衡采样 + Complete/Incomplete 等权 loss”最能缓解原始训练的方向性偏移。
- 不应选择 D10：虽然它显著优于 A10，且 ZH Incomplete 较 D5 更高，但四类宏平均明显低于 D5，内部 validation 的 Idle、Non-idle、Backchannel 保护门禁也未通过。

### 13.4 诚信边界与限制

- A 的结果在提出 B/C/D 假设之前已经被查看，因此本轮属于**事后假设驱动的开发性评测**，不能当作独立最终测试。
- Table 3 没有用于修改任何已训练 checkpoint、样本、标签、ASR、终态规则、阈值或分母；六个 B/C/D 点使用完全相同协议，失败和中断均保留证据。
- balanced-CI 实现的是 Complete-head-active 与 Incomplete-head-active row 数`47,182:47,182`，不是状态 token 的 1:1；其 Complete/Incomplete token 数仍为`128,903:78,037≈1.652:1`。
- 训练只做30个 local optimizer step，外部精确内部 validation 只测 step 0/5/10；不能把内部 validation 写成七点完整选择轨迹。
- 官方发布权重不含 optimizer、scheduler、scaler 和精确 global step；`1800`只是根据公开训练配置得到的低置信度起点估计。
- 本轮没有执行 Table 2，也没有调用付费 API。若未来恢复 Table 2，必须先完成官方基线复现门禁并对预先选定的单一候选做一次确认性测试，不能根据 Table 2 返回调参。

### 13.5 最终证据身份

| 证据 | SHA-256 |
|---|---|
| Table 3 orchestration manifest | `7a1a8dc842384b84f57eec110f6b74a42f17d1cca6065571e3c08938c7209d97` |
| 新鲜基线 evidence gate | `ec4a75cdcc116b64de7a0eae1e20fcf133b8bf7b8f88c1b594d1ca08cd198a29` |
| B step 5/10 index | `f4feef545a47e47ccdfcf2c832dc81a697e23872d006790b070ae9b16f5b68e4` |
| C step 5/10 index | `4a09ea9a89c9ee6ef8c64fcdf85c5023c6613e5d594a4b7045deb967984faebb` |
| D step 5/10 index | `0c7c8858b7094dff7a5cb000f331ae061fbd53be1ac91b60fcae43d133e6bd43` |
| B/C/D training queue manifest | `1b8684717c4f7fb3a9d36ebb121079b623d8a9628601e786edc634b67749ed51` |
| 冻结 A 最终 HTML | `d30c139358fa5891c67ba1e39fd05d4389ec7484e5ab26477499e6e713c0692b` |

机器可审计的完整输入/输出哈希、逐 gate 检查与结构化指标见`evaluation_reports/duplexconv_edu0001_0045_abcd_experiment_summary_audit.json`。
