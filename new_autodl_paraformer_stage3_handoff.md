# SoulX Stage 3 中文平替：新 AutoDL 服务器完整交接文档

更新时间：2026-08-20  
文档用途：把本项目交接给一个**没有旧对话记忆、且可能没有旧服务器文件**的新 Codex 窗口。  
当前唯一优先目标：尽快得到一份真实可训练的数据，完成一次会保存 checkpoint 的 SoulX Stage 3 续训练，并观察实际训练表现。

## 0. 新窗口应如何使用本文

把本文完整交给新服务器上的 Codex，并同时告诉它：

> 先完整阅读交接文档。第一轮只核对新服务器已有的代码、数据、模型、磁盘、GPU 和网络代理，不下载、不删除、不覆盖、不训练；然后给出与本文一致的实施计划，等我确认后继续。

新 Codex 不得假设本文中的旧服务器路径在新服务器仍然存在。所有外部下载、百度网盘操作、长时间数据处理和正式训练，仍应先列出范围、预计磁盘、网络方式、输出路径和验收标准，取得项目负责人确认后再执行。

本文记录的是当前技术决定和恢复点。旧文档中若仍要求对 `Edu_0018` 完成 125 条人工 A/B/C 审核，或仍把“fa-zh + Paraformer 一致性门禁”写成当前唯一下一步，以本文 2026-08-20 的决定为准。

## 1. 最新决定

项目负责人已经决定采用一个面向快速效果验证的实验路径：

1. DuplexConv 的训练 ASR 文本和 token 时间戳直接由 Paraformer 根据目标声道音频生成。
2. 不再让 fa-zh 对官方文本做强制对齐。
3. 不再用“官方文本/fa-zh 与 Paraformer 是否一致”决定 accepted/review/rejected。
4. 暂停 125 条人工 A/B/C 音频审核；旧审核工作台只保留为历史证据。
5. 暂不增加第二个审核模型。等真实续训练显示该数据路线有价值后，再决定是否用审核模型提高伪标签质量。
6. DuplexConv 原始数据提供的 `complete`、`incomplete`、`backchannel` 状态类别继续作为状态监督来源；Paraformer 只生成 ASR 文本和时间戳，绝不根据识别文本重写状态类别或状态时刻。
7. 保留必要的确定性校验。格式损坏、时间戳非法、数组不闭合或明显无法形成训练监督的样本自动隔离，不进入训练；这不等于恢复人工审核或第二模型门禁。
8. 先处理已经下载和物理验收的 `Edu_0018`，不下载下一个 DuplexConv tar。
9. 新版本不得覆盖旧的 strict v1、冻结 SmoothConv 或任何审计证据。
10. 得到新 DuplexConv model-ready 后，与冻结 SmoothConv model-ready 合成一个新的 pilot 训练目录，进行真实 Stage 3 续训练。

这个实验路径固定命名为：

```text
asr_supervision_profile = paraformer-pseudolabel-v1
```

建议的新产物名为：

```text
processed/duplexconv_edu_0018_2ch_paraformer_pseudolabel_v1
model_ready/duplexconv_edu_0018_2ch_paraformer_pseudolabel_v1
model_ready/cn_stage3_pilot_smoothconv2ch_duplex0018_paraformer_v1
```

这些产物必须被称为“Paraformer 伪标签实验数据”，不能称为人工 gold、官方转录训练数据或已经过审核模型验证的数据。

## 2. 项目背景：到底在训练什么

本项目补充的是 SoulX-Duplug 的 Stage 3：`Duplex State Prediction Fine-tuning`，不是普通 ASR 预训练。

模型的正式五状态只有：

```text
<|user_idle|>
<|user_nonidle|>
<|user_backchannel|>
<|user_complete|>
<|user_incomplete|>
```

每个 160 ms chunk 在最终 `sequence` 中必须对应：

```text
2 个 GLM-4-Voice audio token
+ 当前 chunk 新增的 ASR 文本
+ <|end_of_sentence|>
+ 1 个用户状态 token
```

完整序列为：

```text
<|task_duplex_predict|><|punctuation_off|>
(2 audio tokens + 增量 ASR + EOS + state) × N
```

最终官方 loader 只读取 Hugging Face/Parquet 目录中的两个字段：

```text
index: string
sequence: string
```

本项目的 WAV、manifest、tar shard 和关系元数据是可审计中间层；它们本身不能直接交给官方 Stage 3 loader。每个新数据集都必须再完成 GLM-4-Voice audio tokenizer、完整 chunk 组切窗、`index/sequence` Parquet 导出和官方 loader 验收。

## 3. 双声道数据为什么最终仍是单声道训练记录

对一个同步双人双轨会话 A/B，应构造两个目标视角：

```text
A-as-user：模型音频只使用 A，B 只用于离线关系判断
B-as-user：模型音频只使用 B，A 只用于离线关系判断
```

参考轨道用于离线判断 overlap、对方是否持续持有话轮、backchannel、打断和事件后话轮走向，但不进入官方 `sequence`。

状态时间规则为：

```text
普通目标语音的有效发声 chunk        -> user_nonidle
发声结束后的第一个决策 chunk         -> user_complete / user_incomplete
backchannel 的有效发声跨度           -> user_backchannel
其余无目标语义发声的 chunk           -> user_idle
```

`complete/incomplete` 不得覆盖仍有目标语音的最后一个发声块。`backchannel` 也不得先标 `nonidle` 再只在最后一块标 backchannel，否则官方运行时可能错误触发 barge-in。

连续窗口可能含多个状态事件，因此顶层 `terminal_state` 可以为 `null`；所有局部 complete/incomplete/backchannel 事件应保存在 `source_fields.state_events`。

## 4. WAIT 的固定结论

Easy-Turn 和 SmoothConv 中已试听的 WAIT 是完整停止指令，例如“请立即停止讲话”。在 SoulX 五状态体系内固定处理为：

```text
WAIT 发声阶段       -> user_nonidle
WAIT 发声后决策位置 -> user_complete
```

`user_nonidle` 负责让下游立即停止当前 LLM/TTS/播放；`user_complete` 表示停止指令已经说完。停止后是否保持沉默、暂停还是结束会话，应由下游文本意图和会话策略决定。

DuplexConv 当前 `Edu_0018` 严格范围没有 WAIT；该数据域仍保持 `wait_policy=exclude`，不要把 Easy-Turn/SmoothConv 的映射未经试听外推到新 DuplexConv tar。

## 5. 当前数据状态

### 5.1 SmoothConv：已完成并冻结

当前采用的是 SmoothConv 的 2ch 双人连续视角，不是旧的 utterance 片段批次。

已完成闭环：

```text
1,638 份同步双轨源音频
-> 2,280 条目标视角候选
-> 713 accepted + 1,347 review + 220 rejected
-> accepted 8.777556 小时、197,495 个 160 ms chunk
-> 744 条官方 index/sequence Parquet 记录
-> 未修改官方 loader 验收通过
```

冻结中间层的旧服务器参考路径：

```text
/root/autodl-tmp/soulx-duplug-stage3-cn-replacement/processed/smoothconv_2ch_continuous_v1
```

冻结 model-ready 的旧服务器参考路径：

```text
/root/autodl-tmp/soulx-duplug-stage3-cn-replacement/model_ready/smoothconv_2ch_continuous_v1
```

百度网盘已经上传并逐相对路径/精确字节验收：

```text
/soulx-duplug-stage3-cn-replacement/processed/smoothconv_2ch_continuous_v1
/soulx-duplug-stage3-cn-replacement/model_ready/smoothconv_2ch_continuous_v1
/soulx-duplug-stage3-cn-replacement/receipts/smoothconv_2ch_continuous_v1_v1
```

其中 model-ready 目录只有 8 个文件、1,587,204 字节。若新服务器只需要合并并训练，优先下载 model-ready 即可；不需要为了训练重新下载 1.80 GB 的 processed 或完整 SmoothConv 原始音频。

SmoothConv 已由项目负责人试听确认并冻结。不得原地重跑、改标签或覆盖 `smoothconv_2ch_continuous_v1`。

### 5.2 DuplexConv `Edu_0018`：旧 strict v1 已完成，但下一步改走伪标签新版本

已下载的官方音频 tar：

```text
Edu/audios/Edu_0018.tar
字节数：7,317,340,160
SHA-256：4f2a65763e20206a58e1de95fc646749f9ebd6b9bfc94beae677f99beb1c60fd
```

旧服务器路径：

```text
/root/autodl-tmp/soulx-duplug-stage3-cn-replacement/datasets/duplexconv_audio/Edu/audios/Edu_0018.tar
```

已经物理验收的 85 文件严格双轨提取快照：

```text
/root/autodl-tmp/soulx-duplug-stage3-cn-replacement/datasets/duplexconv_2ch_extracted/Edu/Edu_0018_strict-fully-labeled
```

其 `source_snapshot.json` 记录：85 个严格文件、官方 tar 大小和 SHA-256、`validation=passed`。该提取目录约 586 MB，若能从旧服务器迁移，优先迁移它而不是在新服务器重新下载 7.3 GB tar。

旧 strict v1 已处理：

```text
85 文件 / 620 源事件
5 个源事件隔离
3 个连续源隔离
9 个连续视角隔离
175 个有效候选视角
50 accepted + 98 review + 27 rejected
50 accepted -> 51 条 model-ready Parquet
```

旧路径：

```text
/root/autodl-tmp/soulx-duplug-stage3-cn-replacement/processed/duplexconv_edu_0018_2ch_strict_v1
/root/autodl-tmp/soulx-duplug-stage3-cn-replacement/model_ready/duplexconv_edu_0018_2ch_strict_v1
```

旧 v1 使用官方文本经 fa-zh 强制对齐，再由 Paraformer 做一致性门禁。50/98/27 的划分是旧方法的结果，只作为严格对照证据。新方法必须从 85 文件提取快照重新构造 175 个有效候选，不能直接把旧 review/rejected shard 改名或提升。

`Edu_0018` 的原始 tar、提取目录、processed 和 model-ready **尚未上传百度网盘**。它们可能只存在旧服务器。新服务器开始前必须先确认是否已迁移；若未迁移，需单独确认后再从官方来源重新下载，或者由项目负责人安排旧服务器传输。

### 5.3 暂不使用的数据

- Easy-Turn：片段数据，旧非 idle chunk 时间位置错误，WAIT 也曾错误映射；当前暂缓。
- AISHELL-4/AISHELL-5/AliMeeting/MISP：多人会议主线暂缓。
- DuplexConv `Edu_0045`：保留 strict pilot 和人工听检证据，本轮组合训练不加入，避免混入另一套小样本 override 规则。
- `Edu_0018` 旧 strict v1：不与新伪标签版本同时加入训练，避免同源窗口重复。
- 旧 SmoothConv 八个 utterance 批次：已经废弃，不能恢复为训练输入。

## 6. 官方代码、模型和数值修复

官方训练代码固定提交：

```text
928b06508ed2de1344208d06fb1f6fb2ebfb1df5
```

旧服务器参考路径：

```text
/root/autodl-tmp/soulx-duplug-stage3-cn-replacement/references/
SoulX-Duplug-training-code-928b06508ed2de1344208d06fb1f6fb2ebfb1df5
```

官方模型快照固定修订：

```text
61701c4ab8193cc1ee2220d3848872ac6c720142
```

旧服务器参考路径：

```text
/root/autodl-tmp/soulx-duplug-stage3-cn-replacement/references/
SoulX-Duplug-0.6B-61701c4ab8193cc1ee2220d3848872ac6c720142
```

关键权重：

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `Qwen3-0.6B-expand_vocab_v2/model.safetensors` | 2,384,234,968 | `82f718371044abd7d05e4de2dbfd766938eb9d07a0c7152e8e4bb20cab53b6ec` |
| `SoulX-Duplug/SoulX-Duplug-0.6B-Bilingual.pth` | 3,896,904,668 | `b0703dea0b1dbb1cd51e6e7b6514c60907ea4d4b6752cecc3f71cb6445650dbe` |

model-ready 导出还需要 `glm-4-voice-tokenizer`。新服务器必须取得完整固定模型快照，不能只下载上表两个文件。

### 6.1 官方空状态 head 的 NaN 问题

官方 Stage 3 为 text、EOS 和五状态分别计算带 `ignore_index=-100` 的 mean cross entropy。一个 batch 完全没有某个状态时，该 head 全部是 ignore，官方实现产生 NaN 标量。这是官方数值边界，不是数据要求每条记录同时包含五状态。

项目采用最小修复：

```text
head 有有效目标 -> 完全沿用官方 cross_entropy
head 无有效目标 -> 返回与图相连的 FP32 有限 0，梯度为 0
```

官方参考源码保持不变。训练必须使用由固定官方提交和补丁构造的独立运行副本：

```text
/root/autodl-tmp/soulx-duplug-stage3-cn-replacement/runtimes/
SoulX-Duplug-training-code-928b06508ed2de1344208d06fb1f6fb2ebfb1df5-finite-empty-head-v2
```

项目补丁：

```text
patches/soulx_stage3_finite_empty_head_v2.patch
SHA-256：799a29f204a05378c3b87b45b4572144f1f9484810590f8066846b6cea0f47b8
```

修复后的 `_train_heads.py` SHA-256：

```text
b4c371179bf2ca7cb7c109d5f95a2e373733b761bccdc4850257c862acd89dc3
```

新服务器应从干净官方提交重新构造并校验运行副本：

```bash
cd /root/soulx-duplug-stage3-cn-replacement
python3 scripts/lib/prepare_stage3_runtime.py
python3 scripts/lib/test_stage3_finite_empty_head_patch.py -v
```

只有哈希和 5 项定向测试通过后才可用于训练。不得直接修改官方 reference 仓库。

## 7. Paraformer 伪标签 v1 的精确定义

### 7.1 ASR 与状态完全解耦

新流水线有两条独立监督来源：

```text
目标声道音频 -> Paraformer -> ASR 文本 + token 时间戳 -> chunk_asr_targets
DuplexConv 源 state + speaker.segments -> chunk_state_targets
```

不允许：

- 用 Paraformer 文本猜 complete/incomplete/backchannel；
- 根据最后一个 ASR token 的时间移动 complete/incomplete；
- 因 Paraformer 与官方文本不同而拒绝样本；
- 把官方文本重新喂给 fa-zh；
- 把参考轨道音频或参考轨道文本放进最终 ASR target。

DuplexConv 的状态标注是官方发布的 LLM-assisted 标注，不是人工 gold。产物中必须保留：

```text
state_annotation_quality = official_llm_assisted_not_human_gold
```

### 7.2 Paraformer 文本和时间戳如何进入 160 ms chunk

使用现有本地模型：

```text
iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch
```

旧服务器模型缓存参考路径：

```text
/root/autodl-tmp/soulx-duplug-stage3-cn-replacement/models/modelscope/models/iic/
speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch
```

对 Paraformer 返回的每个 token，验证 `start_ms/end_ms` 后，按现有项目规则只发射一次：

```text
emit_chunk = ceil(end_ms / 160) - 1
```

并限制到 `[0, chunk_count - 1]`。`chunk_asr_targets[t]` 只包含当前 chunk 新增文本，不得放累计全文。

建议复用现有 `tokens_from_result()`、`_validated_timestamp_rows()`、`join_tokens()` 和 `build_chunk_asr_targets()` 的已测试逻辑，但把 Paraformer 自己的识别结果文本作为目标文本，不能再传官方转录。需要新增语义清晰的函数，例如：

```text
build_paraformer_pseudolabel_targets(asr_result, chunk_count, duration_sec, chunk_ms)
```

输出至少包括：

```text
text / asr_target                 = Paraformer 完整识别文本
chunk_asr_targets                 = 160 ms 增量文本数组
token_timestamps                  = Paraformer token/start/end/emit_chunk
asr_alignment_status              = transcribed_paraformer_pseudolabel
asr_supervision_profile           = paraformer-pseudolabel-v1
asr_model                         = 精确模型 ID 或本地快照标识
official_source_transcript        = 仅放 source_fields 供追溯，不作训练目标
```

沿用当前文本规范化和控制 token 检查；首版不要顺手引入新的标点策略。最终 task prefix 仍使用官方 `<|punctuation_off|>`，相关行为必须由现有 model-ready 测试覆盖。

### 7.3 目标说话人和串音边界

Paraformer 只能读取当前 target view 的单声道音频，不能读取参考轨道或双轨混音。DuplexConv 目标轨道存在串音高尾部，因此还要执行一个不依赖第二模型的确定性关系检查：

- 保存每个 ASR token 的 emit chunk；
- 对比现有 `source_fields.target_active_by_chunk`；
- 初始容差固定为前后 2 个 chunk（320 ms）；
- token 若落在距任一目标活动 chunk 超过该容差的位置，记录为 `asr_token_outside_target_activity` 并自动隔离整个记录；
- 该容差、命中数和隔离数必须写入 stats，不能静默放宽；
- 如果该规则造成异常大比例隔离，应停下来汇报，不得自行改阈值。

这只是防止明显把参考说话人串音写成目标 ASR 文本的结构检查，不比较官方文本，也不使用审核模型。

首版训练 WAV 保持原 target mono，不修改、不静音；不要为了生成 ASR 临时音频而覆盖训练音频。以后若要采用“只保留 target activity 的掩码音频做 ASR”，应作为新的 `v2` 决策，不能悄悄加入 v1。

### 7.4 确定性失败条件

以下情况自动隔离，不进入 accepted：

- Paraformer 返回空/无 timestamp，但窗口存在目标语义发声；
- token 与 timestamp 数量不一致；
- 时间戳为负（仅允许复用既有的首 token 小幅原点抖动归一化）、逆序、非单调或超出音频边界；
- ASR 文本含 SoulX 控制 token；
- token 落在目标活动容差之外；
- `chunk_asr_targets`、`chunk_state_targets`、`target_active_by_chunk` 与 `chunk_count` 不等长；
- 音频不是 16 kHz、mono、PCM s16le，或样本数不等于 `chunk_count × 2560`；
- 非法状态 token、状态事件越界、状态与 activity 的既有 schema 约束不成立；
- cache 绑定的模型、代码、音频摘要或 profile 不一致。

隔离项必须保留 sample ID、原因、来源定位和必要证据，但本轮不生成待人工处理的任务，不阻塞其余样本训练。

### 7.5 产物中必须写明的局限

`paraformer-pseudolabel-v1` 没有人工或第二模型审核，无法自动发现所有同音字、漏字、重复字和语言模型式幻觉。其目的只是用真实训练验证该低成本路线是否值得继续。metadata/stats/报告必须明确：

```text
ASR 文本质量 = Paraformer pseudo label, deterministic validation only
状态类别质量 = DuplexConv official LLM-assisted state, not human gold
人工审核 = none
第二审核模型 = none
```

## 8. 需要在项目代码中实现的内容

新 Codex 应先迁移并阅读完整项目树，而不是只凭本文从零重写。项目根目录旧服务器路径为：

```text
/root/soulx-duplug-stage3-cn-replacement
```

该目录不是普通 git 工作区；不能假设可从某个远程 git 仓库恢复所有本地实现。若新服务器没有它，应优先从旧服务器完整迁移项目目录，至少包含 `AGENTS.md`、`scripts/`、`configs/`、`patches/`、`project_plan/` 和必要日志/凭据。`.env` 含密钥，不得打包、打印或传输，除非项目负责人另行明确安排安全迁移。

### 8.1 CLI 和公共对齐模块

建议修改：

```text
scripts/process_one_dataset.sh
scripts/lib/strong_processor_dispatch.py
scripts/lib/stage3_alignment.py
scripts/lib/stage3_runtime.py
```

新增一个明确参数，不要把新语义偷偷塞进现有 `--alignment-backend none`：

```text
--asr-supervision-profile official-transcript-fa-v1|paraformer-pseudolabel-v1
```

兼容要求：

- 默认仍保持历史 `official-transcript-fa-v1`，避免改变冻结 SmoothConv 和旧命令；
- 只有显式选择 `paraformer-pseudolabel-v1` 才绕过 fa-zh 和一致性门禁；
- 伪标签模式应拒绝同时传 `--alignment-backend funasr-fa-zh`、旧 frozen threshold config 或人工 promotion 参数；
- cache signature 必须包含 profile、Paraformer 模型、代码 SHA-256、160 ms 和目标活动容差，不能复用旧 fa-zh/Paraformer cache；
- 支持现有 micro-batch、单样本失败隔离和全局进度。

### 8.2 DuplexConv 适配器

建议修改：

```text
scripts/lib/duplexconv_strong.py
```

要求：

- 继续从 `speaker.segments` 构造目标/参考 activity；
- 继续使用原始 state 生成 `state_events` 和 `chunk_state_targets`；
- 把旧 `timeline["asr_target"]` 的官方文本移入 `source_fields.official_source_transcript`；
- Paraformer 成功后再写训练使用的 `text/asr_target/chunk_asr_targets`；
- `label_source` 仍说明状态来自 DuplexConv，而 ASR provenance 单独说明来自 Paraformer；
- stats 中新增伪标签 profile、Paraformer 成功/失败、活动边界失败和最终 accepted 数；
- 保持数量闭环：`175 = accepted + deterministic_quarantine`；17 项源/连续结构隔离仍单独闭环；
- 不读取或修改旧 v1 的人工工作台决定。

### 8.3 schema 和验证器

建议检查或修改：

```text
scripts/lib/stage3_schema.py
scripts/lib/validate_duplexconv_batch.py
scripts/lib/stage3_model_ready.py
scripts/lib/export_stage3_model_ready.py
scripts/lib/validate_stage3_model_ready.py
```

要求：

- 新 profile 是显式批准的实验 schema，不要伪装成 `aligned_funasr_fa_zh`；
- model-ready exporter 继续要求 ASR/state/audio chunk 数闭合；
- 不因 `terminal_state=null` 拒绝合法连续窗口；
- export metadata 保留 `asr_supervision_profile` 和源 dataset；
- 官方 `index/sequence` 语法、audio token 数、EOS 数、状态数和 1,500-token 上限不变。

### 8.4 测试

至少新增以下离线定向测试：

1. Paraformer 文本和 token 时间戳正确映射到增量 chunk；
2. 同一 token 只发射一次，不生成累计全文；
3. 时间戳负值/逆序/越界/数量不一致被隔离；
4. 控制 token 被拒绝；
5. 目标活动容差内通过、容差外隔离；
6. Paraformer profile 不调用 fa-zh generator；
7. 旧 official-transcript/fa-zh 路径行为不变；
8. cache 不会跨 profile 命中；
9. DuplexConv 175 候选数量闭环；
10. model-ready chunk 语法和官方 loader 继续通过。

修改 Python 后先运行 `py_compile` 和相关单元测试。只有短样本集成通过后，才运行 85 文件正式处理。项目原有规则是单个测试超过 5 分钟或同一问题重复超过 5 轮时停止并报告；长时间正式处理/训练应在另行确认后执行，不冒充短测试。

## 9. 新服务器上的数据构造顺序

### 阶段 A：只读盘点

核对并记录：

- 项目代码树是否完整；
- `Edu_0018_strict-fully-labeled` 是否存在且 `source_snapshot.json` 通过；
- 冻结 SmoothConv model-ready 是否存在或可从百度恢复；
- Paraformer 本地模型是否完整；
- 官方 SoulX 代码固定提交和完整模型快照是否存在；
- GPU 型号、显存、驱动、CUDA、磁盘总量/剩余量；
- Python/torch/torchaudio/transformers/PL/PEFT/FunASR/datasets/pyarrow 版本；
- shell 是否带有指向项目负责人本地电脑的代理。

旧服务器已验证过的环境仅供参考，不应盲目覆盖新服务器环境：

```text
Python 3.10.8
torch 2.1.2+cu118
torchaudio 2.1.2+cu118
transformers 4.51.3
pytorch-lightning 2.5.2
peft 0.19.1
funasr 1.3.9
datasets 3.1.0
pyarrow 18.1.0
```

旧服务器 5 步真实回归在 RTX 4080 SUPER 32 GB 上通过，CUDA peak allocated 约 17.51 GB。官方 `finetune.py` 顶层导入 `wandb`；旧服务器当前没有安装 wandb，因此新服务器开始真实 Lightning 训练前，应在隔离环境中补齐并验证 `wandb`，建议采用官方 requirements 中的 `wandb==0.21.0`，训练时设为 offline，避免依赖外部账号。

### 阶段 B：实现并小样本验证伪标签 profile

- 使用新版本输出目录；
- 先处理极少量候选，仅验证代码；
- 检查 Paraformer 文本、token 时间戳、160 ms ASR 增量、状态数组和目标活动关系；
- 小样本通过后删除的只能是明确由本次命令创建的 debug 输出，正式旧产物不得删除。

### 阶段 C：正式重建 `Edu_0018`

正式输入：85 文件严格提取快照。  
候选上界：175 个有效 target view。  
预期：accepted 应明显多于旧 strict v1 的 50 条，但不预设必须等于 175；任何 Paraformer/结构失败都要自动隔离并统计。

正式输出：

```text
/root/autodl-tmp/soulx-duplug-stage3-cn-replacement/processed/
duplexconv_edu_0018_2ch_paraformer_pseudolabel_v1
```

验收：

- 输入快照哈希通过；
- 17 项源/连续结构隔离与候选闭环可解释；
- `175 = accepted + deterministic_quarantine`；
- accepted shard、metadata、manifest、stats、contract、checksums 全部通过；
- 所有 accepted 音频格式与数组长度通过；
- profile/provenance/局限写入产物；
- 不存在 fa-zh 结果或旧一致性阈值被误当作新 accepted 条件。

### 阶段 D：导出官方 model-ready

输出：

```text
/root/autodl-tmp/soulx-duplug-stage3-cn-replacement/model_ready/
duplexconv_edu_0018_2ch_paraformer_pseudolabel_v1
```

继续使用现有正式入口：

```bash
cd /root/soulx-duplug-stage3-cn-replacement
bash scripts/export_stage3_model_ready.sh \
  --input-dir /root/autodl-tmp/soulx-duplug-stage3-cn-replacement/processed/duplexconv_edu_0018_2ch_paraformer_pseudolabel_v1 \
  --output-dir /root/autodl-tmp/soulx-duplug-stage3-cn-replacement/model_ready/duplexconv_edu_0018_2ch_paraformer_pseudolabel_v1 \
  --device cuda \
  --max-records 0

python3 scripts/lib/validate_stage3_model_ready.py \
  --output-dir /root/autodl-tmp/soulx-duplug-stage3-cn-replacement/model_ready/duplexconv_edu_0018_2ch_paraformer_pseudolabel_v1
```

命令只在相关代码、输入路径和固定模型快照已验证后执行。验收报告 `official_loader_validation.json` 必须为 `passed=true`。

## 10. 构造一个真正给训练器读取的组合目录

首轮训练组合只包含：

```text
冻结 SmoothConv 2ch model-ready（744 行）
+ 新 DuplexConv Edu_0018 Paraformer pseudo model-ready（行数以实际导出为准）
```

明确排除：旧 `Edu_0018 strict v1`、`Edu_0045`、Easy-Turn、AISHELL-4 和 SmoothConv 旧批次。

不要复制 1.80 GB SmoothConv 中间 WAV。组合发生在已经 token 化的 model-ready 层。应新增一个可审计的 model-ready 合并器，而不是手工拼文件：

- 分别验证两个输入目录的 `checksums.sha256`、contract 和官方 loader 报告；
- 读取两边 Parquet 的 `index/sequence`；
- 检查全局 index 唯一，避免同源重复；
- 合并 `export_metadata.jsonl`，增加 `source_model_ready_dir`；
- 写新的 `hf_dataset/train.parquet`、stats、contract、checksums；
- 记录每个来源的行数、chunk 数和五状态计数；
- 对组合目录再次运行未修改官方 loader 验证。

组合输出建议为：

```text
/root/autodl-tmp/soulx-duplug-stage3-cn-replacement/model_ready/
cn_stage3_pilot_smoothconv2ch_duplex0018_paraformer_v1
```

本轮按实际行自然混合，不增加状态过采样、不复制行、不修改 loss weight。这样更容易判断最小可行训练链路。官方 loader 仍按行随机切 5% validation；项目负责人已接受其只作为训练监控，允许与 train 有会话重叠，但报告中必须明确它不是独立泛化测试。

## 11. 真实续训练计划

这里的“续训练”是从已发布的 `SoulX-Duplug-0.6B-Bilingual.pth` 初始化 Stage 3 LoRA/projector 训练，不是从本项目此前产生的 checkpoint 恢复；此前只做过 1 步和 5 步回归，没有保存可继续训练的 checkpoint。

### 11.1 配置原则

在项目仓库新增独立配置，例如：

```text
configs/stage3_train/cn_stage3_pilot_v1.yaml
```

不要改官方 runtime 自带的 `config/train_config.yaml`。配置至少固定：

- `task: state_prediction`；
- 固定 expanded Qwen3、GLM audio tokenizer 和 Bilingual pth 的绝对路径；
- `enable_projector: true`、`freeze_projector: false`；
- `enable_lora: true`、`r=32`、`alpha=64`、`dropout=0.1`；
- `train_data_path` 指向组合目录下的 `hf_dataset`；
- `batch_size=1`、`max_token_length=1500`、`split_size=0.05`；
- 单 GPU 时 `num_gpu_per_node=1`、`num_node=1`、`strategy=auto`、`sync_batchnorm=false`；
- `precision=16-mixed`；
- 继续使用官方 Stage 3 loss weights；
- checkpoint 目录、日志目录和 seed 42；
- 使用 `finite-empty-head-v2` runtime。

建议先做一个会保存 checkpoint 的 300 optimizer-step pilot：

```text
total_steps = 300
warmup_steps = 30
val_check_interval = 100
log_every_n_steps = 5
accumulate_grad_batches = 1
```

这是为了在单张约 32 GB GPU 上尽快获得真实曲线和权重，不是论文级最终训练。若新服务器 GPU/显存不同，新 Codex应先根据一次真实 batch 的峰值显存报告是否需要调整；不得因为 OOM 擅自降低序列上限或改标签。

### 11.2 启动前门禁

必须全部通过：

- 组合 model-ready 官方 loader 验收；
- `finite-empty-head-v2` 来源与 5 项数值测试；
- 用组合数据随机真实记录做 5 步 forward/backward/AdamW 冒烟；
- 所有 loss 有限；空 head loss 可以是有限 0，空 head accuracy 可以是 NaN/N/A；
- 至少一个实际训练参数发生变化；
- checkpoint 路径为空或是本轮明确的新目录；
- GPU/磁盘足够；
- W&B 使用 offline，或经最小代码适配使用 CSV logger，不能因缺外网中断训练。

### 11.3 建议启动方式

在已构造并校验的修复 runtime 中运行，示意命令：

```bash
cd /root/autodl-tmp/soulx-duplug-stage3-cn-replacement/runtimes/SoulX-Duplug-training-code-928b06508ed2de1344208d06fb1f6fb2ebfb1df5-finite-empty-head-v2

CUDA_VISIBLE_DEVICES=0 \
TRANSFORMERS_OFFLINE=1 \
HF_HUB_OFFLINE=1 \
WANDB_MODE=offline \
python3 finetune.py \
  --config_path /root/soulx-duplug-stage3-cn-replacement/configs/stage3_train/cn_stage3_pilot_v1.yaml
```

新服务器实际执行前必须由 Codex核对 `finetune.py`、配置 dataclass 和依赖版本，不能直接照抄而不检查。训练应保存 `last.ckpt`、最佳监控 checkpoint、最终配置、环境版本、GPU 信息和完整日志。

### 11.4 如何判断“性能还行”

首轮 pilot 只能回答“路线是否有训练价值”，不能证明正式泛化性能。至少报告：

- 300 步是否完成、耗时和峰值显存；
- train/validation 总 loss 曲线；
- text、EOS、idle、nonidle、complete、incomplete、backchannel 的 loss/accuracy；
- 每个 head 的有效目标数，避免把空 head 的 NaN accuracy误解释为失败；
- AMP overflow/跳步次数；
- checkpoint 是否能重新加载并在固定 validation 行上复现指标；
- 与 step 100/200/300 相比是否稳定改善，是否出现状态塌缩；
- 明确 validation 是官方行级监控 split，不是独立测试。

如果训练有限、loss 有下降、关键状态没有明显塌缩、checkpoint 可加载，则认为该低成本伪标签路线具备继续扩大/评估的价值。之后再决定：

1. 增加独立审核模型；
2. 处理更多 DuplexConv tar；
3. 建立独立 Testset/evaluation；
4. 进行更长训练和状态池采样。

如果训练不稳定或状态性能明显恶化，先分析伪标签、状态分布、串音和合并比例，不立即扩大数据规模。

## 12. 网络与代理规则

旧服务器的 shell 曾把 HTTP/HTTPS 请求转发到项目负责人本地电脑。新服务器必须先只读检查代理变量，但不得打印任何可能含凭据的完整代理 URL。

原则：

- 禁止使用指向项目负责人本地电脑的代理；
- 百度网盘和国内可直连服务：显式清除 HTTP/HTTPS/ALL_PROXY 等转发变量后直连；
- Hugging Face/GitHub/ModelScope 如确需代理：在同一 shell 命令中先清除旧代理，再使用 AutoDL 官方 `source /etc/network_turbo`；
- 下载完成后不要让代理设置污染后续百度命令；
- 百度网盘只复用当前登录态，命令、日志和文档中禁止出现 BDUSS；
- 不把 `.env`、密钥或带凭据的 shell history 迁移进项目交接包。

示意边界，不是授权立即下载：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
source /etc/network_turbo
# 在同一个 shell 中执行已确认的外网下载
```

新 Codex 应先确认 `/etc/network_turbo` 在新 AutoDL 实例存在，并报告代理出口，不得自动回退到项目负责人的本地转发。

## 13. 迁移优先级

为了最快在新服务器恢复，按以下优先级迁移：

1. 完整项目代码/文档树（不含 `.env`）；
2. `Edu_0018_strict-fully-labeled` 约 586 MB 提取快照；
3. SmoothConv 冻结 model-ready 约 1.59 MB（可从百度下载）；
4. 官方 SoulX 固定代码提交及完整模型快照；
5. Paraformer 本地模型快照；
6. 必要时再迁移 `Edu_0018.tar` 7.3 GB，若第 2 项完整则首轮处理不需要它；
7. 旧 `Edu_0018` 人工审核工作台、约 1.2 GB 试听包、旧 strict processed/model-ready 都不是快速训练的必要输入，可暂不迁移。

所有迁移文件先比较文件数量、字节数和已有 SHA-256，再开始处理。不要为了腾空间删除旧服务器资产；迁移和新服务器训练不授权任何旧服务器清理。

## 14. 当前明确不做的事情

- 不实现或调用第二审核模型；
- 不继续 125 条人工 A/B/C 审核；
- 不用 fa-zh 生成新训练 ASR target；
- 不用官方文本与 Paraformer 的 CER/边界一致率筛样本；
- 不覆盖任何 `strict_v1`、`strict_v2` 或冻结 SmoothConv；
- 不下载新的 DuplexConv tar；
- 不处理 Easy-Turn、AISHELL-4 或其他多人会议集；
- 不做 1.64 TB DuplexConv 全仓下载；
- 不上传或删除新产物，除非训练闭环完成后项目负责人另行确认；
- 不把官方 loader 内部 validation 称为独立泛化测试；
- 不把 Paraformer 伪标签称为人工 gold。

## 15. 新 Codex 的建议执行清单

### 第一次回复项目负责人前

- [ ] 完整阅读本文；
- [ ] 只读列出新服务器已有资产；
- [ ] 核对 GPU、磁盘、依赖和代理边界；
- [ ] 判断是“迁移已有资产”还是“需要重新下载”；
- [ ] 给出实施计划、预计新增空间和每阶段验收；
- [ ] 明确不执行上传、删除、下一个 tar 和审核模型；
- [ ] 等项目负责人确认。

### 代码完成

- [ ] 新增显式 `paraformer-pseudolabel-v1` profile；
- [ ] fa-zh 和旧门禁在该 profile 中不被调用；
- [ ] 状态类别/时刻仍来自源 state/activity；
- [ ] cache 和 provenance 版本隔离；
- [ ] 单元测试和小样本集成通过。

### 数据完成

- [ ] 85 文件源快照通过；
- [ ] 175 候选 accepted/自动隔离闭环；
- [ ] processed 全部校验通过；
- [ ] model-ready 导出和官方 loader 通过；
- [ ] 与 SmoothConv 合并后的官方 loader 再次通过。

### 训练完成

- [ ] 修复 runtime 哈希和测试通过；
- [ ] 组合数据 5 步真实回归通过；
- [ ] 300-step pilot 完成；
- [ ] checkpoint、配置、日志、环境和指标完整保存；
- [ ] 给出有限、准确的性能结论；
- [ ] 再由项目负责人决定是否增加审核模型。

## 16. 关键参考文件

如果完整项目树已迁移，新 Codex 应按顺序阅读：

```text
AGENTS.md
project_plan/new_autodl_paraformer_stage3_handoff.md
project_plan/project_progress.md
project_plan/stage3_conversation_construction_spec.md
project_plan/stage3_dataset_contract.md
project_plan/duplexconv_Edu_0018_processing_report.md
project_plan/smoothconv_2ch_processing_report.md
project_plan/stage3_empty_head_nan_fix_report.md
scripts/lib/stage3_alignment.py
scripts/lib/duplexconv_strong.py
scripts/lib/stage3_model_ready.py
scripts/lib/export_stage3_model_ready.py
scripts/lib/validate_stage3_model_ready.py
```

旧 `project_progress.md` 和 `duplexconv_Edu_0018_processing_report.md` 中的 strict v1 数字仍是历史事实；只有“下一步必须人工审核”的部分已被本文的新决定取代。

## 17. 最终交接结论

现在不是继续追求人工审核覆盖率，而是验证一个更低成本的实验闭环：

```text
真实双人目标声道音频
-> Paraformer 直接生成 ASR 文本/时间戳
-> 源数据 state/activity 独立生成五状态时间线
-> 确定性结构校验
-> 官方 GLM audio token + index/sequence Parquet
-> 与冻结 SmoothConv model-ready 合并
-> finite-empty-head-v2 上真实 300-step Stage 3 续训练
-> 根据训练曲线和 checkpoint 决定是否值得增加审核模型
```

只要新窗口严格区分 ASR 伪标签和状态监督、保持版本隔离、通过官方 loader 和数值修复门禁，就可以在没有旧对话记忆的情况下继续执行。
