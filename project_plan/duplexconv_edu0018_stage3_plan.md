# DuplexConv Edu_0018：SoulX Stage 3 中文训练数据构造与训练计划

更新时间：2026-08-20  
状态：执行中。Gate 0–6、NaN/runtime 验收和真实 5-step 预检已通过；旧 300-step pilot 已撤销，当前先复现论文 benchmark，基线通过前禁止正式续训练。

> 本文件覆盖当前 DuplexConv `Edu_0018` 项目的全部工作，包括工作区整理、旧资产清理、多声道处理、状态补标、Paraformer 伪转录、model-ready 导出、SoulX NaN 修复和真实训练。以后处理另一个完全独立的数据集时，才在 `project_plan/` 下新增另一份计划文件。

## 1. 项目目标

本项目为 SoulX-Duplug Stage 3（Duplex State Prediction Fine-tuning）构造中文训练数据并完成一次真实续训练。它不是 Paraformer 训练项目，也不是普通 ASR 预训练项目。

准确表述为：

> 基于 DuplexConv 官方同步多轨会话数据，使用 Paraformer 生成 ASR 伪标签，使用官方状态、WAIT 映射和 Qwen 补标共同生成状态监督，构造 SoulX Stage 3 中文训练数据。

Paraformer 是本地 ASR 模型，只生成训练 ASR 文本和 token 时间戳；它不是数据集，也不决定用户状态。

本轮训练只使用新构造的 DuplexConv `Edu_0018` 数据，不与 SmoothConv、旧 DuplexConv strict v1/v2 或其他数据集合并。

## 2. SoulX Stage 3 训练格式

正式五状态：

```text
<|user_idle|>
<|user_nonidle|>
<|user_backchannel|>
<|user_complete|>
<|user_incomplete|>
```

每个 160 ms chunk 对应：

```text
2 个 GLM-4-Voice audio token
+ 当前 chunk 新增的 Paraformer 文本
+ <|end_of_sentence|>
+ 1 个用户状态 token
```

完整 sequence：

```text
<|task_duplex_predict|><|punctuation_off|>
(audio × 2 + incremental ASR + EOS + state) × N
```

官方 loader 最终只读取：

```text
index: string
sequence: string
```

WAV、活动关系、状态事件、API 响应和 provenance 都是中间审计层，不作为官方训练表额外字段。

## 3. 当前原始资产与数据事实

正式输入只允许使用：

```text
DuplexConv 官方 Edu/audios/Edu_0018.tar
DuplexConv 官方 Edu/jsons.tar.gz
```

不得读取旧 `duplexconv_2ch_extracted`、旧 processed、旧 model-ready 或旧人工审核决定。

已盘点的 `Edu_0018.tar`：

```text
500 个同步多轨 WAV
495 个双声道文件
5 个三声道文件
1005 个目标声道视角上界
全部目标声道合计约 21.169878 小时
```

官方 metadata 共 8,505 个事件：

```text
<|complete|>       4,570
<|incomplete|>     1,527
<|backchannel|>      798
<|wait|>              11
缺失 state          1,599
```

双声道部分有 1,574 个缺失 state；三声道部分有 25 个缺失 state。

## 4. 固定版本标识

建议固定：

```text
dataset_version = duplexconv_edu0018_stage3_zh_v1
source_view_profile = target-vs-rest-v1
asr_supervision_profile = paraformer-pseudolabel-v1
state_supervision_profile = official-plus-qwen3-235b-2507-v1
wait_policy = wait-to-complete-v1
chunk_profile = 160ms-glm2-v1
```

三条监督必须独立记录：

```text
目标声道音频 -> Paraformer -> ASR 文本和 token 时间戳
官方 DuplexConv state -> 已标事件状态类别
缺失 state 的官方多声道上下文 -> Qwen -> 补充状态类别
```

任何一条监督不得反向修改另一条监督。

## 5. 工作区和目录结构

项目代码放系统盘：

```text
/root/SoulX-stage3-dataset/
├── project_plan/
├── src/
├── scripts/
├── tests/
├── configs/
├── patches/
├── third_party/
│   └── SoulX-Duplug-upstream/
├── runtimes/
├── pretrained_models/
└── dataset -> /root/autodl-tmp/dataset
```

数据盘只建立统一 dataset 根，每个独立数据集一个子目录：

```text
/root/autodl-tmp/dataset/
├── duplexconv/
    ├── raw/
    ├── work/
    ├── cache/
    ├── processed/
    ├── model_ready/
    ├── quarantine/
    └── reports/
└── soulx_duplug_eval/
    ├── raw/
    ├── extracted/
    └── reports/
```

项目中只创建一个数据软链接：

```text
/root/SoulX-stage3-dataset/dataset
  -> /root/autodl-tmp/dataset
```

创建后使用 `readlink -e` 验证，不允许悬空链接或继续跳转到旧项目目录。当前发现的 `/root/miniconda3/envs` 等系统/环境链接不属于本项目，不纳入清理。

## 6. 必需资产迁移和 SoulX 官方代码

必须迁移并验收：

1. `Edu_0018.tar`；
2. `Edu/jsons.tar.gz`；
3. Paraformer 完整固定模型；
4. SoulX 固定模型，包括 expanded Qwen、Bilingual pth 和 GLM tokenizer。

规则：

- 先记录源路径、文件数、字节数和 SHA-256；
- 跨系统盘/数据盘复制后重新计算关键哈希；
- 新副本验收前不删除旧副本；
- 原始 tar/metadata 放 `dataset/duplexconv/raw/`；
- Paraformer 和 SoulX 模型放系统盘 `pretrained_models/`。

SoulX 官方代码从以下仓库重新 clone：

```text
https://github.com/Soul-AILab/SoulX-Duplug.git
```

训练代码固定提交：

```text
928b06508ed2de1344208d06fb1f6fb2ebfb1df5
```

论文模型推理代码使用官方 `main` 的独立 clean worktree，固定提交：

```text
/root/SoulX-stage3-dataset/third_party/SoulX-Duplug-inference-a0b9063
a0b9063843df69619b087b95b74597b2176910b8
```

两份代码用途必须分开：`training-code` 及其 runtime 用于 Stage 3 训练，`main` worktree 用于复现官方流式推理。不得为了让评测通过而直接修改任一 upstream；必要兼容层放本项目 `src/` 或独立 runtime，并记录 diff。

网络命令必须清除现有 HTTP/HTTPS 遗留代理，再按需要使用 AutoDL `/etc/network_turbo`。clone 后记录 commit、tree hash 和 `git status --short`；upstream 必须保持 clean。

## 7. 旧资产删除计划

完成必需资产迁移和验收后，先生成包含路径、大小、删除原因和新位置的 `cleanup_manifest.json`，再删除以下旧派生产物：

```text
/root/autodl-tmp/soulx-duplug-stage3-cn-replacement/datasets/duplexconv_2ch_extracted
/root/autodl-tmp/soulx-duplug-stage3-cn-replacement/processed
/root/autodl-tmp/soulx-duplug-stage3-cn-replacement/model_ready
/root/autodl-tmp/soulx-duplug-stage3-cn-replacement/inspection_samples
/root/autodl-tmp/soulx-duplug-stage3-cn-replacement/calibration
/root/autodl-tmp/soulx-duplug-stage3-cn-replacement/upload_receipts
/root/autodl-tmp/soulx-duplug-stage3-cn-replacement/runtimes
/root/autodl-tmp/smoothconv_min_audit_6WaZkb
```

这组约占 14.4 GB。删除后报告实际释放空间和可恢复性。

以下是与当前项目无关但可能仍有独立保存价值的原始数据，只有项目负责人明确确认“数据盘最终只保留当前 DuplexConv 项目”后才删除：

```text
AISHELL-4                  约 44 GB
SmoothConv                 约 41 GB
Easy-Turn                  约 8.6 GB
Easy-Turn testset          约 141 MB
DuplexConv Edu_0045        约 631 MB
其他 DuplexConv metadata   约 167 MB
```

新系统盘模型/代码哈希通过后，可以删除数据盘中的旧 references、models 和 tools 副本。

禁止使用含未解析变量、宽泛 glob 或工作区根路径的删除命令。所有删除目标必须是上面清单中的精确绝对路径。

## 8. 双声道与三声道处理

### 8.1 最终 sequence 仍是单目标声道

SoulX 的 sequence 只有一条目标用户 audio token 流。因此源双声道文件生成两个视角：

```text
A-as-user：目标音频为 A，B 仅提供离线关系信息
B-as-user：目标音频为 B，A 仅提供离线关系信息
```

不能把 A/B 混音后输入，也不能把两路 audio token 同时放入一个 sequence。

### 8.2 三声道采用 target-vs-rest

三声道不直接丢弃，也不做两两配对：

```text
A-as-user：目标音频 A；参考关系为 B ∪ C
B-as-user：目标音频 B；参考关系为 A ∪ C
C-as-user：目标音频 C；参考关系为 A ∪ B
```

`∪` 是活动区间逻辑合并，不是波形混音。禁止把 A 展开成 A-B、A-C 两份重复目标记录。

多方数据可以补充 Stage 3 状态监督，但它与典型一人一助手双人会话并非完全同分布。三声道必须：

- 标记 `source_ntrack=3`；
- 标记 `conversation_domain=multi_party_supplemental`；
- 不过采样；
- 在 processed、model-ready 和训练报告中单列行数、时长和状态分布；
- 保留 provenance，以便后续做双声道-only 消融。

当前三声道仅 5 文件、15 视角、约 0.398093 小时，不会主导训练。

本版只自动支持 2/3 声道；未来若出现 4 声道以上，必须重新审计多人话轮关系，不自动沿用。

### 8.3 其他人活动信息

每个 target view 按 160 ms 构造：

```text
target_active_by_chunk: bool[]
other_active_by_chunk: bool[]
other_active_count_by_chunk: int[]
overlap_by_chunk: bool[]
```

定义：

```text
other_active = OR(all non-target channel activity)
other_active_count = SUM(all non-target channel activity)
overlap = target_active AND other_active
```

这些信息用于：

- Qwen 缺失状态分类上下文；
- backchannel 证据和校验；
- 串音/活动边界检查；
- 中间 metadata 和统计。

它们不会新增官方格式之外的 input/state token，也不会直接覆盖官方或 Qwen 给出的状态。

## 9. 源数据验证与数量闭环

每个源文件检查：

- tar WAV 与 metadata ID 一一对应；
- WAV channel count 等于 `nTrack`；
- `len(asr)` 等于 `nTrack`；
- sample rate、sample width、frame count 与 metadata 时长一致；
- 事件和 speaker segments 不负、不逆序、不越界；
- 每个 target view ID 稳定唯一；
- activity 数组长度与目标 chunk 数一致。

原始视角上界：

```text
495 × 2 + 5 × 3 = 1005 target views
```

正式统计满足：

```text
1005 = structurally_usable_views + source_quarantined_views
structurally_usable_views
  = exported_or_windowed_views + processing_quarantined_views
```

切窗后的 Parquet 行数可大于 target view 数，但每行必须追溯到唯一 source、target channel 和 chunk 范围。

## 10. 状态标签计划

### 10.1 已有官方状态

已有 `complete/incomplete/backchannel` 的事件直接采用，不调用 Qwen、不改写：

```text
state_label_source = duplexconv_official_llm_assisted
state_label_quality = official_llm_assisted_not_human_gold
```

若官方状态与活动关系看起来反常，保留官方状态并记录 anomaly，不自动修改。

### 10.2 WAIT

固定映射：

```text
<|wait|> -> <|complete|>
```

记录 original/mapped state 和 `wait_policy=wait-to-complete-v1`。WAIT 发声期间为 `user_nonidle`，发声后第一个决策 chunk 为 `user_complete`。

### 10.3 缺失状态的 Qwen 补标

1,599 个缺失 state 使用 OpenRouter 模型：

```text
qwen/qwen3-235b-a22b-2507
```

只允许输出：

```text
complete
incomplete
backchannel
```

Qwen 不允许修改已有官方状态，也不允许输出 WAIT/idle/nonidle。

以一个同步源 WAV 为请求组织单位，提供：

- 源 ID、nTrack 和各 channel；
- 所有声道事件按时间排序的官方文本；
- event ID、channel、start/end；
- 已有官方状态作为上下文；
- 需要补标的 event ID；
- target 前后话轮；
- other active 和三声道的 other active count。

官方文本只用于状态判断和审计，不作为训练 ASR target。

分类定义：

- `complete`：表达在语义和话轮上已完成，可以自然交出话轮；
- `incomplete`：表达尚未完成、被中断或明显需要继续；
- `backchannel`：简短反馈/附和，不意图取得并持续持有主话轮。

不能只凭字数或标点判断。

## 11. OpenRouter API、`.env` 与缓存

endpoint：

```text
https://openrouter.ai/api/v1/chat/completions
```

项目创建：

```text
/root/SoulX-stage3-dataset/.env
/root/SoulX-stage3-dataset/.env.example
```

`.env`：

```dotenv
OPENROUTER_API_KEY=
```

要求：

- `.env` 权限 `0600`；
- `.env` 加入 `.gitignore`；
- key 不进入命令行、日志、异常、缓存、Git 或报告；
- `.env.example` 只含空占位符；
- 不使用遗留本地 HTTP/HTTPS 代理访问 OpenRouter。

使用结构化输出：

```json
{
  "model": "qwen/qwen3-235b-a22b-2507",
  "temperature": 0,
  "provider": {"require_parameters": true},
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "duplexconv_state_labels",
      "strict": true
    }
  }
}
```

每项响应包含稳定 event ID、固定枚举 state、confidence 和简短 reason。请求和响应 event ID 集合必须完全相等，不多、不少、不重复。

官方参考：

- https://openrouter.ai/docs/quickstart
- https://openrouter.ai/docs/guides/features/structured-outputs
- https://openrouter.ai/qwen/qwen3-235b-a22b-2507

缓存签名包含 source metadata SHA-256、source ID、缺失 event 集、prompt/schema 版本与哈希、模型 ID、temperature 和 provider requirements。

缓存保存 request hash、OpenRouter ID、模型/provider、token usage、结构化响应和解析结果，但不保存 Authorization header。

网络超时、429 和可重试 5xx 做有限指数退避；schema/event 集错误使用同一模型修正重试；禁止自动换模型或用默认 complete 填充失败事件。超过重试上限进入 API quarantine。

正式补标前：

1. 从已有官方三状态中分层抽取少量事件，隐藏标签测试 prompt；
2. 不覆盖官方状态；
3. 用一个真正缺失事件测试付费连通；
4. 根据 usage 估算 1,599 个事件总成本和请求数；
5. 冻结 prompt/schema 版本；
6. 获得全量 API 启动确认。

状态事件最终闭环：

```text
8505 = official_state_events
     + deterministic_wait_events
     + qwen_labeled_events
     + state_quarantined_events
```

## 12. 目标声道音频与 Paraformer

每个目标视角：

```text
48 kHz PCM s16le target channel
-> 分离单声道
-> 确定性重采样到 16 kHz
-> 对齐/补齐到 160 ms chunk 边界
```

不得修改原始 tar，不得把参考声道波形混入目标音频。

固定 Paraformer：

```text
iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch
```

对完整目标声道生成：

```text
text
token sequence
token start_ms/end_ms
```

Paraformer 不读取官方文本作为识别目标，不读取参考声道，不决定状态。

每个 token 只发射一次：

```text
emit_chunk = ceil(end_ms / 160) - 1
```

限制到 `[0, chunk_count-1]`。`chunk_asr_targets[t]` 只包含当前 chunk 新增文本，不是累计全文。

必须保存 token、start/end 和 emit chunk。以下情况隔离受影响 target view/窗口：

- 有语义发声但返回空文本/空 timestamp；
- token/timestamp 数量不一致；
- 负时间、逆序、非单调或越界；
- 包含 SoulX 控制 token；
- cache 与音频、模型、代码或 profile 不一致；
- ASR/state/activity 数组与 chunk count 不闭合。

如果 Paraformer 内部 VAD 分段，拼接时间戳必须恢复为源音频全局坐标并通过长音频测试。

## 13. 目标活动和状态时间线

目标活动优先使用官方 `speaker.segments`。若 segments 不合法但官方 event `startInMs/endInMs` 合法，可以使用事件 envelope 作为候选 fallback，但必须：

- 写 `activity_source=event_envelope_fallback`；
- 要求 Paraformer 在该范围内存在语音/token 证据；
- 统计与正常 segments 的差异；
- 无法形成一致时间线时隔离，不猜测状态时刻。

状态映射：

```text
complete/incomplete：
  有效目标发声 chunks -> user_nonidle
  最后发声后的第一个决策 chunk
    -> user_complete / user_incomplete

backchannel：
  有效目标发声 chunks -> user_backchannel

WAIT：
  有效目标发声 chunks -> user_nonidle
  发声后的第一个决策 chunk -> user_complete

其余无目标活动 chunks -> user_idle
```

complete/incomplete 不覆盖仍有目标发声的最后一个 chunk。backchannel 不先标 nonidle 再只在末尾标 backchannel。若音频末尾没有决策 chunk，可只在新增 160 ms 静音 padding 上落 terminal state并记录来源。

初始 target activity 容差为前后各 2 chunk（320 ms）。Paraformer token 若远离所有目标活动范围，记录 `asr_token_outside_target_activity`，在切窗后隔离含异常 token 的窗口。异常率过高时暂停报告，不静默放宽容差或混入参考音频。

训练文本仍严格按 token `end_ms` 发射；活动证据校验使用 token `[start_ms, end_ms]` 区间与目标活动区间是否重叠。原因是 Paraformer 可能把句末字的 `end_ms` 延伸到后续静音，只按发射 chunk 检查会制造假阳性。fallback event 即使同时命中正常活动区间也独立记录证据。两种规则分别记录为 `token_emit_profile` 和 `activity_evidence_profile`，不得混为一谈。

## 14. GLM audio token 与 model-ready 导出

使用固定完整 `glm-4-voice-tokenizer`，token 提取逻辑以固定 SoulX 官方 inference/tokenizer 实现为准。

必须验证：

- 输入 16 kHz mono；
- 每 160 ms 正好 2 个有效 audio token；
- token ID 在官方 audio vocab；
- chunk 顺序、音频长度和末尾 padding 一致。

完整目标声道先构造完整时间线，再按 expanded Qwen tokenizer 的真实长度切窗：

- 每条不超过 1,500 tokens；
- 以完整 chunk group 为最小单位；
- 窗口不重叠、不复制 chunk；
- 不拆 chunk group；
- 每个状态事件 chunk 只出现一次；
- 单个异常超长 chunk 隔离；
- 空窗口不导出。

最终 Parquet 只含 `index/sequence`。独立 metadata 记录 source ID/nTrack、conversation domain、target/reference channels、chunk 范围、音频摘要、ASR/state/WAIT/relation profile、Qwen event IDs、Paraformer/GLM cache signature。

禁止把 API key、Authorization header 或 `.env` 内容写入 metadata。

输出建议：

```text
/root/autodl-tmp/dataset/duplexconv/processed/edu0018_stage3_zh_v1
/root/autodl-tmp/dataset/duplexconv/model_ready/edu0018_stage3_zh_v1
```

正式目录已存在时拒绝覆盖；debug 使用独立目录。

## 15. 数据验收

中间层：

- 1005 target views 数量闭环；
- 8,505 状态事件来源闭环；
- 所有数组长度等于 chunk count；
- WAV 格式、样本数和 padding 通过；
- 双声道/三声道分开统计；
- cache/provenance 可追溯。

model-ready：

- index 全局唯一；
- 每行恰有两个字段；
- tokenized length ≤ 1,500；
- 每 chunk 两 audio token、一 EOS、一合法状态；
- 不含 reference activity 新 token；
- contract、stats、checksums、metadata 完整；
- 未修改 SoulX 官方 loader 能读取并切 train/validation；
- 随机 sequence 回解后与中间层逐 chunk 一致。

## 16. SoulX 空 head NaN 修复

官方 checkout 保持干净，独立 runtime：

```text
/root/SoulX-stage3-dataset/runtimes/
  SoulX-Duplug-928b065-finite-empty-head-v2
```

官方对每个 head 用 `-100` 屏蔽无关位置是正确行为。问题是某 batch 对某 head 全部为 `-100` 时，mean cross-entropy 对零个有效元素求平均而产生 NaN。

最小修复：

```text
若 shifted labels 中有有效目标：
  完全沿用官方 cross_entropy
否则：
  返回与图相连的 FP32 有限 0
```

等价：

```python
return logits.reshape(-1)[0].float() * 0.0
```

不能伪造状态、取消 `-100` 或要求每条记录含五状态。空 head accuracy 可为 NaN/N/A，但不参与 loss。

测试至少包括：

1. 非空 loss/梯度与官方一致；
2. 全 `-100` 返回 FP32 0；
3. 空 head 梯度有限且为 0；
4. AMP 下不出现 `inf*0 -> NaN`；
5. 七 head 总 loss 有限；
6. 真实 batch 可优化；
7. upstream Git 仍 clean。

补丁和修复文件记录 SHA-256。

## 17. Benchmark-first 续训练与评估计划

### 17.1 为什么必须先复现基线

正式续训练前先冻结和复现论文评测场景。否则续训练后即使指标发生变化，也无法区分是模型权重、推理协议、teacher ASR、测试数据版本或端到端系统组件造成的。

评测分为两层：

1. **模型级主门禁：Bilingual Easy Turn**。直接衡量 SoulX 状态预测模块的 Complete/Incomplete 准确率与流式延迟，是续训练 checkpoint 比较的主要依据。
2. **系统级外部验证：Bilingual Full-Duplex-Bench**。将 SoulX 接入论文相同的 Qwen2.5-7B-Instruct 和 IndexTTS-1.5 系统，验证 Pause Handling、Turn Taking、User Backchannel 和 User Interruption。该层会受到 LLM、TTS、ASR 和调度抖动影响，不能代替模型级诊断。

已固定的官方评测资产：

```text
Soul-AILab/SoulX-Duplug-Eval
revision = f6e50e8f07f3d33d8b2e77b14df986d14c817ef2

ASLP-lab/Easy-Turn-Testset
revision = 5812651dbab429b9a4fab293de7127bfb9a56650
```

数据盘位置：

```text
/root/autodl-tmp/dataset/soulx_duplug_eval/
```

当前验收结果：

```text
Easy Turn EN：318 Complete + 299 Incomplete = 617 条
Easy Turn ZH：300 Complete + 300 Incomplete = 600 条
Full-Duplex-Bench ZH：Turn Taking 155、Pause Handling 239、
                      User Backchannel 199、User Interruption 161
```

官方 EN zip SHA-256：

```text
6c045b9543c6f6f5188a5134923b50f96705c609a2c80b3582aafedeb9907387
```

官方 ZH Full-Duplex-Bench zip SHA-256：

```text
8d659ad87dff604da65328d16c14674595fd5f846f30e8121cbf5d04abe5c4cc
```

论文 Easy Turn 目标值：

| 语言 | Complete ACC | Incomplete ACC | Macro Avg. ACC |
| --- | ---: | ---: | ---: |
| EN | 77.67%（约 247/318） | 88.96%（约 266/299） | 83.32% |
| ZH | 89.33%（268/300） | 79.33%（238/300） | 84.33% |

基线门禁先要求准确率复现；论文的 240 ms 是理论算法延迟，部署测量为单张 NVIDIA L20 上 205 ms。本机为 NVIDIA vGPU-32GB，因此延迟只要求测量方法一致并单独报告硬件，不要求数值与 L20 完全相等。

### 17.2 官方 checkpoint 的 step 与学习率语义

本地官方 `SoulX-Duplug-0.6B-Bilingual.pth` 已检查：它是只含 679 个 tensor 的 `OrderedDict`，不含 `global_step`、optimizer、scheduler 或 AMP scaler。因此：

- 不能精确恢复论文训练的 AdamW 动量和 scheduler；
- 本项目属于**从官方权重继续微调**，不是训练状态的 bit-exact resume；
- checkpoint 必须同时记录 `origin_step_estimate` 和 `continuation_optimizer_step`；
- 图表横轴主值使用本地 continuation optimizer step，同时显示估计有效总 step。

官方 2026-07-17 发布的 Stage 3 重实现配置为：

```text
total_steps = 1800
learning_rate = 1e-4
warmup_steps = 200
anneal_steps = 100000
batch_size = 1
accumulate_grad_batches = 72
num_gpu_per_node = 8
```

因此报告暂定：

```text
origin_step_estimate = 1800
estimate_source = official re-implemented training config
estimate_confidence = low
```

这里的 1,800 是公开重实现配置的终止 optimizer step，不是从权重中恢复出的事实，也不保证等于论文内部模型真实 step。按官方 inverse-square-root 规则，若把 step 1,800 当作原训练位置，参考 LR 约为 `3.33e-5`；该值用于设计本地 LR 校准，不能直接当作已恢复 LR。

旧 pilot 的 `learning_rate=1e-4, warmup_steps=30, accumulate_grad_batches=1` 不再作为正式方案。正式配置要在基线复现后，用训练集/独立 validation 做短程 LR 校准；不得用论文 test benchmark 选择 LR。

### 17.3 训练/验证划分和泄漏控制

正式训练只使用：

```text
/root/autodl-tmp/dataset/duplexconv/model_ready/edu0018_stage3_zh_v1
```

双/三声道按自然比例混合，三声道不过采样。不能直接随机切分 2,168 个窗口，因为同一源会话或同一多声道视角的相邻窗口会泄漏到 train/validation。

在训练前生成并冻结 group-aware split manifest：

- 以完整源会话为最小分组；
- 同一个 WAV 的全部目标视角和全部窗口只能落入同一 split；
- validation 目标为约 5%，同时报告会话数、视角数、窗口数、时长、声道数和状态分布；
- Easy Turn 与 Full-Duplex-Bench 永不进入训练或 validation；
- benchmark 结果不能反向修改标注、LR、状态映射或 split。

### 17.4 正式续训练设置和 step 网格

保持官方 Stage 3 架构和 loss weights：projector 可训练，LLM 使用 LoRA `r=32, alpha=64`，GLM speech tokenizer 冻结，`max_token_length=1500`，FP16 初始 scale 为 16,384。

基线通过后先执行不读取 benchmark 的 LR 校准：

```text
候选峰值 LR：1.0e-5 与 3.33e-5
每个候选最多：20 optimizer steps
选择依据：group-aware validation loss、五状态准确率、梯度/AMP 稳定性
```

正式 run 使用冻结后的唯一 LR 方案。单卡环境暂定 `batch_size=1`、`accumulate_grad_batches=72`，每个 step 均指完成一次 optimizer update，而不是一个 micro-batch。若因吞吐调整 accumulation，必须先更新计划、重新计算有效 batch，且 checkpoint 图表不得混用两种 step 定义。

预注册的评测 checkpoint 网格：

```text
0, 5, 10, 20, 30, 45, 60, 90, 120, 180, 240, 300 optimizer steps
```

- 每个点保存只含可训练权重的 evaluation snapshot；
- 另保存最近一次和最佳一次含 optimizer/scheduler/scaler 的 resumable checkpoint；
- 所有 snapshot 记录 SHA-256、LR、累计 micro-batch、样本/音频时长曝光量、epoch-equivalent、耗时、显存和 AMP 跳步；
- 模型级 Easy Turn 在预注册网格上统一批量评测；
- 系统级 Full-Duplex-Bench 至少评测 step 0、最后无明显下降点、首次明显下降点和最终候选点；
- 若出现明显下降，继续到下一个预注册点作一次确认，然后停止继续扩展，避免无意义计算。

### 17.5 “几乎未下降”和“明显下降”的预注册判据

先以官方 step 0 本地复现结果作为配对基线，不直接拿论文四舍五入值计算差值。

主要指标为 EN/ZH 的 Complete、Incomplete 和各语言 macro accuracy。每个 checkpoint 还输出逐样本预测，以便做 paired bootstrap 和 McNemar 检验。

暂定判据：

- **几乎未下降**：EN 与 ZH macro accuracy 相对 step 0 的下降均不超过 1.0 个百分点，且任一单类下降不超过 2.0 个百分点；
- **明显下降**：任一语言 macro accuracy 下降超过 3.0 个百分点，或任一单类下降超过 5.0 个百分点，并在相邻下一个 checkpoint 再次出现；
- 1–3 个百分点之间标记为灰区，结合 95% paired bootstrap CI、McNemar 检验、group-aware validation 和状态预测分布说明，不能武断归类；
- 同时报告中文收益与英文遗忘，不能用中文提升掩盖英文 catastrophic forgetting。

最终文档必须明确给出：最后一个“几乎未下降”的 step 区间、首次“明显下降”的 step、最佳 validation checkpoint、推荐交付 checkpoint，以及这些结论的不确定性。

### 17.6 基线复现门禁

正式续训练前必须全部满足：

1. 推理代码、模型、teacher ASR、测试集 revision 和预处理参数固定；
2. 音频统一按官方协议转单声道、重采样，按 160 ms 模拟在线输入；
3. ZH 使用固定 Paraformer，EN 使用固定 SenseVoice Small；
4. 每条样本保留逐 chunk state、触发时刻、最终分类和耗时；
5. Easy Turn 四个类别准确率与论文对应正确样本数一致；若只能达到四舍五入误差 ±1 条，必须找到并记录协议差异后由项目负责人决定能否放行；
6. 端到端 Full-Duplex-Bench 使用论文相同系统组件和官方评测脚本；随机组件至少重复 3 次并报告均值、标准差和 seed；
7. 基线未通过时不得启动正式续训练，不能通过调 test-set 阈值制造一致结果。

## 18. 实际执行 TODO 与门禁

### Gate 0：计划确认

- [x] 确认三声道采用 target-vs-rest，不直接丢弃。
- [x] 确认 other activity 只作离线关系/标签证据，不新增 sequence token。
- [x] 确认官方状态保留、WAIT 映射 complete、缺失状态用固定 Qwen。
- [x] 确认不合并 SmoothConv。
- [x] 确认旧资产删除范围。

### Phase 1：目录、迁移和官方代码

- [x] 创建新代码/数据目录和 dataset 软链接。
- [x] 迁移并校验原始 tar/metadata。
- [x] 迁移并校验 Paraformer/SoulX/GLM 模型。
- [x] clone SoulX 官方代码并固定 commit。
- [x] 验证 upstream clean 和所有新路径。

### Gate 1

- [x] 文件数、字节、哈希和离线模型加载全部通过。

### Phase 2：旧资产清理

- [x] 生成精确 cleanup manifest。
- [x] 删除已确认旧派生产物并报告释放空间。
- [x] 按确认范围保留独立原始数据，不作越界删除。
- [x] 新副本验收后删除旧模型/代码副本。

### Phase 3：源扫描与 target-vs-rest

- [x] 从官方数据重建 1005 target views。
- [x] 构造 target/other activity、activity count 和 overlap。
- [x] 输出双/三声道 provenance、尾部量化 anomaly 和结构 quarantine。

### Gate 3

- [x] 1005 视角闭环；三声道无 pairwise 重复。

### Phase 4：Qwen 补状态

- [x] 创建 `.env`、`.env.example`、`.gitignore` 和安全权限。
- [x] 实现 OpenRouter client、schema、缓存、重试、usage 和全量确认令牌。
- [x] 已知标签小样本校验 prompt（v2 分层 30 条，20/30 与官方 LLM 辅助标签一致；不覆盖官方标签）。
- [x] 一个缺失事件做付费连通测试。
- [x] 估算 1,599 事件总费用，并核验 API key 每日硬上限为 10 USD。

### Gate 4A：全量 API 确认

- [x] 项目负责人提供 key、固定模型和 10 USD 日预算，并明确授权调用 LLM。

确认后：

- [x] 全量补缺失状态（1,599 事件/404 个源会话请求，accepted-response cost 0.2187791 USD）；
- [x] WAIT 确定性映射（11 条 -> complete）；
- [x] 完成 8,505 事件闭环（6,895 官方 + 11 WAIT + 1,599 Qwen，0 state quarantine）。

### Phase 5：Paraformer 与时间线

- [x] 实现声道分离、16 kHz 规范化和完整声道推理。
- [x] 实现 token 单次发射、activity 和状态时间线。
- [x] 实现缓存、隔离、恢复和测试。
- [x] 双/三声道少量真实样本验收（5 views、87 tokens、0 quarantine、0 activity anomaly）。

### Gate 5：正式 Paraformer 确认

- [x] 小样本 ASR/timestamp/relation 通过；5 views 耗时 7.573 秒，CUDA 峰值 981,647,360 bytes。

确认后：

- [x] 正式处理全部 1,005 target views并闭环（121,183 ASR tokens，0 ASR view quarantine）。

正式时间线结果：476,763 个原始 chunk；为 3 个音频末端 terminal state 各新增 1 个静音决策 chunk，合计 476,766。474,030 个 chunk 可直接使用，2,736 个 chunk 隔离（0.574%），不整条丢弃 248 个受影响 view。隔离来源为 200 个无 Paraformer 证据的 fallback event、280 个远离活动区间的 ASR token 和 15 处 160 ms 状态冲突；范围可重叠。202 个无文本、无 segments 的非声学占位事件只保留审计记录，不伪造活动。

### Phase 6：GLM 与 model-ready

- [x] GLM token、1,500-token 切窗和 Parquet 导出（953,532 audio tokens；2,168 rows）。
- [x] stats/contract/checksums/metadata。
- [x] 未修改官方 loader 和随机回解验收。

### Gate 6

- [x] 官方 loader passed；全部 2,168 条 sequence 语法和长度通过，20 条随机逐 chunk 回解通过。

### Phase 7：NaN runtime

- [x] 构造独立 runtime、应用空 head 与 checkpoint 严格加载补丁，运行定向和真实 batch 测试；upstream 保持 clean。

### Phase 8：真实训练预检

- [x] 5-step 回归、显存、紧凑测试 checkpoint 和重载验收。

5-step 最终 v2：5 个成功 step 均为 102–281 tokens 且包含非 idle 状态；七 head 合计有效目标为 text 26、EOS 167、idle 109、nonidle 38、complete 2、incomplete 2、backchannel 16。AMP 有 2 次可恢复 overflow，scale 从 65,536 降至 16,384 后完成 5 次真实更新；峰值 CUDA memory 8,144,328,192 bytes。可训练参数 13,505,536，projector 跟踪参数 L2 变化 0.0207257。紧凑 checkpoint 含 trainable weights、AdamW、scheduler、GradScaler，162,241,270 bytes，SHA-256 `f12a49a392f7ad319e56c4cb75f21ce8f027f2fb76d9ea88c7799762efa6c4de`，重载通过。

### Phase 9：论文 benchmark 复现

- [x] 核对论文两层评测、指标定义和论文目标值。
- [x] 检查官方权重元数据，确认不存在可恢复的 global step/optimizer/scheduler。
- [x] 固定官方推理 commit `a0b9063` 和训练 commit `928b065`。
- [x] 下载并验收 EN/ZH Easy Turn 与 ZH Full-Duplex-Bench 官方资产。
- [ ] 获取并固定 English Full-Duplex-Bench、Qwen2.5-7B-Instruct、IndexTTS-1.5 及端到端系统依赖。
- [ ] 实现逐样本可审计的 Easy Turn runner、指标汇总和统计检验。
- [ ] 在少量 EN/ZH 样本上核对 160 ms streaming、ASR、state 和 Complete/Incomplete 判定语义。
- [ ] 全量运行官方 checkpoint 的 Easy Turn baseline。
- [ ] 复现官方 checkpoint 的 Full-Duplex-Bench baseline。

### Gate 9：基线一致性

- [ ] Easy Turn 四个类别达到论文对应正确样本数，或差异不超过预定义 ±1 条且原因已完全解释。
- [ ] Full-Duplex-Bench 主要指标达到预定义复现容差，随机运行统计和环境差异完整。
- [ ] 固定 baseline predictions、配置、日志、软件/硬件版本和 checksum。

### Phase 10：正式训练前冻结

- [ ] 生成 source-conversation group-aware train/validation manifest 并做泄漏审计。
- [ ] 只用 train/validation 完成 `1e-5` 与 `3.33e-5` 的 20-step LR 校准。
- [ ] 冻结唯一正式训练配置、step 定义、checkpoint 网格和停止规则。
- [ ] 确认 evaluation snapshot 与 resumable checkpoint 的磁盘预算和重载结果。

### Gate 10：正式续训练确认

- [ ] 项目负责人检查 baseline 报告、LR 校准、有效 batch、FP16、checkpoint 和磁盘方案。

### Phase 11：续训练与固定网格评估

- [ ] 按 `0/5/10/20/30/45/60/90/120/180/240/300` optimizer step 保存评测快照。
- [ ] 汇总训练/validation loss、各 head 指标、梯度、AMP、曝光量和状态分布。
- [ ] 对预注册网格统一运行 Easy Turn，不据中间 test 结果更改超参数。
- [ ] 对关键 checkpoint 运行 Full-Duplex-Bench。
- [ ] 给出最后无明显下降点、首次明显下降点、最佳点和推荐模型。

### Phase 12：会议汇报文档

- [x] 创建模型性能评估文档骨架。
- [ ] 回填官方基线、全部 checkpoint 指标、统计检验和曲线。
- [ ] 回填最终结论、局限、复现命令、模型哈希和推荐 checkpoint。

## 19. 最终完成条件

- 新项目不依赖旧 processed 或失效软链接；
- 1005 target views 和 8,505 事件有完整、可解释闭环；
- 官方、Qwen、WAIT 和 Paraformer provenance 清晰分离；
- 双声道和三声道来源分开统计；
- 官方 loader 通过；
- 空 head 不产生 NaN loss；
- 官方发布模型已在论文 benchmark 上完成可解释的基线复现；
- 正式续训练 checkpoint 已保存并可重载；
- 已确定最后几乎未下降的 step 范围和首次明显下降点；
- 已完成模型级和关键系统级 checkpoint 对比；
- 会议汇报文档包含数据、处理、step/LR、指标、统计不确定性、限制和推荐结论；
- 项目报告不把伪标签或 LLM 标签称为人工 gold。
