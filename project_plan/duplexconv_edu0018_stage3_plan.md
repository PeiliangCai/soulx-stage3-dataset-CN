# DuplexConv Edu_0018：SoulX Stage 3 中文训练数据构造与训练计划

更新时间：2026-08-29
状态：扩展数据构造、Edu_0018–Edu_0045 不可变聚合及百度网盘发布均已完成。Edu_0018 数据构造、官方 Lightning step 0–5 续训练、Table 3 配对评测和正式会议报告已完成；Edu_0019–Edu_0045 已按冻结流程完成至最终 Gate D，Edu_0018 sanitized 版本与其余最终冻结分片已聚合、独立审计、通过未修改官方 loader 全量验证，并完成160条发布记录上传及独立远端验收。存储方案 B 已执行；Edu_0001–0017 已全部完成至最终 Gate D，所有官方 raw 均保持精确闭包；新的不可变 Edu_0001–0045 v2 聚合也已建立、独立审计并通过未修改官方 loader 全量验证。后续基于 v2 的训练 split、正式续训练、checkpoint 评测和新版本上传尚未执行，必须另行披露方案并获得确认。

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
├── .conda-envs/
│   └── soulx-duplug-official -> /root/autodl-tmp/conda_envs/soulx-duplug-official
└── dataset -> /root/autodl-tmp/dataset
```

数据盘存放统一 dataset 根和空间占用较大的 Conda 环境；每个独立数据集仍使用一个子目录：

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
/root/autodl-tmp/conda_envs/
└── soulx-duplug-official/
```

项目代码目录中允许创建两类软链接：统一数据根，以及指向数据盘 Conda 环境的项目内入口：

```text
/root/SoulX-stage3-dataset/dataset
  -> /root/autodl-tmp/dataset
/root/SoulX-stage3-dataset/.conda-envs/soulx-duplug-official
  -> /root/autodl-tmp/conda_envs/soulx-duplug-official
```

创建后使用 `readlink -e` 验证，不允许悬空链接或继续跳转到旧项目目录。不得移动或删除仍在使用的实际 Conda 环境；旧环境只有在确认不被任何项目引用后，才按精确路径单独清理。

Stage 3 和官方 benchmark 共用一个 Python 3.10 Conda 环境。按项目负责人最新决定，不再安装官方快照中的全部 404 个包；只安装两条执行链实际导入的依赖，并以 benchmark 分支的冲突版本为优先。直接依赖固定在：

```text
requirements/soulx_stage3_benchmark_minimal.txt
```

解析后的完整版本清单另行保存，benchmark JSON 同时记录 Python、关键包、CUDA、GPU、requirements 哈希和官方推理代码 commit。`vLLM`、TensorRT、Gradio/Jupyter、ONNX Runtime、DeepSpeed、bitsandbytes、diffusion/TTS 等当前执行链未使用的包不安装；以后某条实际命令出现缺失导入时，再以可验证的最小增量补充。

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

网络路由按实际端点、缓存命中、连通性、稳定性和吞吐综合选择，不按模型国别设置绝对规则。通常国内源先试直连、GitHub/Hugging Face/OpenRouter 等境外端点先试 AutoDL 或已批准代理；若默认路线超时或明显更慢，可切换后重试。大文件下载前优先确认本地缓存并做轻量连通测试，记录最终路由、失败原因、重试次数和文件哈希。clone 后记录 commit、tree hash 和 `git status --short`；upstream 必须保持 clean。

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
- OpenRouter 的网络端点是 `openrouter.ai`，路由选择依据该端点的实际连通性和速度，而不是所调用模型的国别；客户端必须显式记录使用环境代理还是强制直连。

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

评估分为三层：

1. **训练期内部验证**。只使用本项目训练语料中按完整源会话分组留出的 validation，监控 loss、七个 token head 的 accuracy 和学习率，用于 checkpoint 保存、LR 校准及停止判断。
2. **模型级外部测试：Bilingual Easy Turn（论文表 3）**。直接衡量 SoulX 状态预测模块的 Complete/Incomplete 准确率与流式延迟，用于报告固定 checkpoint 的泛化性能和与论文结果对齐。
3. **系统级外部测试：Bilingual Full-Duplex-Bench（论文表 2）**。将 SoulX 接入论文相同的 Qwen2.5-7B-Instruct 和 IndexTTS-1.5 系统，验证 Pause Handling、Turn Taking、User Backchannel 和 User Interruption。该层会受到 LLM、TTS、ASR 和调度抖动影响，不能代替模型级诊断。

三者用途严格分开：Stage 3 训练过程只使用按源会话分组的 train/validation；论文表 3 的 Bilingual Easy Turn 是固定的 Stage 3 checkpoint 外部测试；论文表 2 的 Bilingual Full-Duplex-Bench 只用于完整对话系统验证。任何 benchmark test set 都不参与 LR、训练数据、状态映射或训练超参数选择。Table 3 按预注册网格完整报告，用于事后描述稳定区间、最佳观测值和下降点；若据此推荐 checkpoint，必须明确披露这是 benchmark-informed selection，不能再把同一 Table 3 数值当作完全无偏的最终泛化估计。

官方公开的 Stage 3 **重实现训练代码**没有在 `trainer.fit()` 中调用表 2 或表 3。其 `train_config.yaml` 默认从训练集随机切出 2% 作为 validation、每 1,000 optimizer step 验证一次，并以 `val_acc` 保存 top-2 checkpoint；`val_acc` 是 text、EOS、idle、nonidle、user_complete、user_incomplete、user_backchannel 七个 token head accuracy 的等权平均，`test_step` 为空。由于 README 明确称其为重实现流程，只能据此确认公开代码行为，不能断言论文内部原始训练脚本完全相同。本项目不沿用其样本级随机切分，而使用 17.3 节的会话级分组切分防止相邻窗口泄漏。

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

已生成并冻结 group-aware split manifest：

- 以完整源会话为最小分组；
- 同一个 WAV 的全部目标视角和全部窗口只能落入同一 split；
- validation 目标为约 5%，同时报告会话数、视角数、窗口数、时长、声道数和状态分布；
- Easy Turn 与 Full-Duplex-Bench 永不进入训练或 validation；
- benchmark 结果不能反向修改标注、LR、状态映射或 split。

冻结结果：475 个训练源会话、25 个验证源会话，会话泄漏为 0；训练 2,066 rows/约 19.921 h，验证 102 rows/约 1.147 h。split identity 为 `0f5060afcf27857af17b99755775921851125fdf7b96dbf6e233fbfb967c1f2b`。

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
7. 原机器 gate 未通过时不得静默启动，必须报告差异并由项目负责人明确决定；无论是否放行，都不能通过调 test-set 阈值制造一致结果。

协议选择必须先有外部或实现依据，再运行汇总指标。允许的依据仅包括论文、官方代码/配置、官方数据说明和作者确认；禁止根据“哪种规则更接近 89.33%/79.33%”反向选择门限、尾部静音或状态读出。诊断实验全部保留且明确标为 diagnostic。若官方样本级协议仍不可获得，或按唯一预注册协议运行后仍未达到原机器门禁，则必须向项目负责人报告差异，不以最接近论文的诊断结果替代主结果。项目负责人可在完整证据与限制均披露后决定是否将候选协议作为本项目内部配对 step 0；该决定不把候选协议改称为作者确认的官方样本级协议。

2026-08-20 收到另一台服务器的只读复现包 `soulx-table3-reproduction-bundle-20260820`。包内历史结果与论文接近，但不是由最终 runner 一次性前瞻运行产生：历史轨迹曾比较 first/last/endpoint 等读出规则，随后以最后一个 `speak/wait` 重新聚合；原始聚合前 JSON、Teacher-ASR 缓存和完整日志也未随包提供。因此该包只能作为高价值候选协议和历史参考，不能直接当成本机正式基线或无造假证明。

经项目负责人确认，本机正式运行前冻结候选协议 `frozen-candidate-v1`：直接调用 clean `training-code@928b065` 的 `duplex_predict_160_cascade_asr`；使用官方函数自带 2 秒尾静音；不经过部署 `TurnModel`、不启用 `far_field_threshold`；主规则固定为完整轨迹的最后一个 `speak/wait`。first terminal、离原音频终点最近的 terminal、原音频终点后第一个 terminal 只作为预声明敏感性结果，不得替换主结果。EN 按官方未排序 `os.walk` 顺序，ZH 按发布 `.list` 顺序，四个语言/类别各用新进程、seed 42 和独立空 ASR 文本缓存。由于作者尚未确认最后终态及样本顺序，该结果在获得外部确认前应写作“候选协议独立复现”。

官方 checkpoint 在 clean training-code 中会触发一次 `strict=False` 回退，唯一原因是 `embed_tokens_func.weight` 在类初始化末尾才注册，而导出的 checkpoint 已包含该别名。审计确认它与正式 `llm...embed_tokens.weight`、`lm_head.weight` 共用同一 tensor，且不存在缺失键、形状不匹配或其他多余键。候选 runner 只允许这一项固定别名；任一新增差异立即失败，并把兼容审计写入结果。

2026-08-21 完成修复后的四类全量候选运行 `formal-candidate-v1-ac8fcf1`。主规则结果为 EN Complete 251/318（78.93%）、EN Incomplete 268/299（89.63%）、ZH Complete 263/300（87.67%）、ZH Incomplete 241/300（80.33%）。逐样本轨迹、状态 logits、Teacher-ASR 缓存、模型/数据/代码哈希和日志的严格证据审计通过；但 EN Complete（+1.26 pp）和 ZH Complete（-1.67 pp）超出原机器 gate 的 ±1.0 pp 筛查范围，因此历史 gate JSON 保持 `accuracy_gate_passed=false`，不得篡改。专项反作假审计未发现预测篡改、标签入模、样本排除、分类别调参或事后换主规则。项目负责人随后认定四类结果与论文已基本一致，并明确要求开始续训练；因此该候选运行被固定为本项目内部 step 0 配对基线，同时继续披露 last-terminal 尚缺作者确认、原机器数值 gate 未通过这两个限制。

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
- [x] 实现逐样本可审计、可中断续跑的 Easy Turn runner和指标汇总；结果记录环境、代码、配置、模型与样本身份。
- [x] 实现 checkpoint 配对汇总的 10,000 次 paired bootstrap、exact McNemar 和中文四个固定真人/合成子组统计；正式数值待 checkpoint 推理完成后回填。
- [x] 在少量 EN/ZH 样本上核对 160 ms streaming、ASR、state 输出和精简 Conda 环境运行链；真实 smoke 分别完成 ZH 2 条和 EN 2 条。论文表 3 的样本级读出协议仍待复现。
- [x] 完成 ZH 600 条在线服务语义诊断，确认 Complete 269/300、Incomplete 191/300；该结果不计作官方基线。
- [x] 定向重跑 18 条 `no_decision`：关闭部署 RMS 门限后 14 条为 Incomplete、4 条为 Complete，证明门限只能解释部分差异。
- [x] 只读审计另一台服务器的 Table 3 bundle；确认数据/权重/上游哈希和历史轨迹内部一致，同时记录事后聚合、弱 gate、缺失 ASR 缓存/日志及 `bf16` 字段未实际生效等风险。
- [x] 在本机正式运行前冻结候选协议证据表：training-code 直推理、2 秒尾静音、无部署 RMS 门限、last-terminal 主规则及三项固定敏感性规则。
- [x] 新增独立审计 runner 和严格 gate：全新 per-class ASR cache、禁止 resume、保存 Teacher-ASR 文本/完整 state/状态 logits/日志，并从逐样本证据重新计分；历史 bundle 保持原封不动。
- [x] 建立匹配历史核心版本的独立 Conda 环境；核心包版本、`pip check` 和当前 52 项单元测试通过。
- [x] 完成新 runner 的每语言一条 diagnostic smoke；EN/SenseVoice 与 ZH/Paraformer 均使用本地模型端到端通过，并生成 checkpoint、ASR、state trace 和 logits 审计证据。
- [x] 首次正式候选运行在 ZH Complete 第 120 条暴露 Paraformer 空列表；确认官方 ASR 包装器会记录异常并返回空字符串后，补齐同语义的结构化 fallback。旧 partial 与旧提交下 EN 结果仅保留审计，不跨提交拼接。
- [x] 完成最新 Table 3 runner 反作假审计；未发现预测篡改、标签入模、样本排除、分类别调参或事后换主规则，同时保留 last-terminal 尚未获作者确认的限制。
- [x] 确认中文 600 条是发布包的完整固定 Testset，不是本项目随机抽样；无第二个同分布 600 条池，不伪造重抽实验。
- [x] 删除已废弃的 `TurnModel` Easy Turn runner、单元测试和三个专用配置，代码仓库只保留最新 Table 3 候选实现。
- [ ] 向作者确认或取得 Easy Turn 样本级评测脚本；至少确认 `far_field_threshold`、尾部静音长度、终态读取规则和预切分音频的结束点处理。
- [x] 全量运行官方 checkpoint 的冻结候选 Easy Turn baseline；证据审计通过，数值门禁失败。
- [ ] 复现官方 checkpoint 的 Full-Duplex-Bench baseline。

### Gate 9：候选基线一致性与项目决定

- [ ] Easy Turn 四个类别达到论文对应正确样本数，或差异不超过预定义 ±1 条且原因已完全解释。本轮失败；即使按较宽的 ±1.0 pp 机器筛查，仍有两类超限。
- [x] 使用本机运行前冻结的唯一候选主协议；没有阈值搜索、规则择优、标签条件分支或样本排除，并单独披露其尚缺作者确认。
- [ ] Full-Duplex-Bench 主要指标达到预定义复现容差，随机运行统计和环境差异完整。
- [x] 固定 Easy Turn 候选 baseline predictions、配置、日志、软件/硬件版本和 checksum；产物位于 `dataset/soulx_duplug_eval/table3_audit/formal-candidate-v1-ac8fcf1`。
- [x] 项目负责人在知悉原机器 gate 未通过及 last-terminal 未获作者确认的前提下，接受该运行作为本项目内部配对 step 0 并授权续训练；历史 gate 结果不修改。

### Phase 10：正式训练前冻结

- [x] 生成 source-conversation group-aware train/validation manifest 并做泄漏审计；475/25 源会话、2,066/102 rows、leakage=0。
- [x] 只用 train/validation 完成 `1e-5` 与 `3.33e-5` 的 20-step LR 校准；两档均无 AMP overflow，未读取 Table 3。
- [x] 冻结正式 peak LR=`1e-5`、step 定义、checkpoint 网格和停止规则。两档在 step 20 都触发单 head >5pp guard；`1e-5` 将保护线维持到 step 10，`3.33e-5` 在 step 5 已触发，因此按修正后的 validation-only v2 选择器采用低 LR。错误地从空 eligible set 选高 LR 的 v1 结果已标记为 rejected bug 并保留审计，不用于训练。
- [x] 确认 evaluation snapshot 与 resumable checkpoint 的磁盘预算和重载结果；evaluation 约 54 MB，resume 约 162 MB，端到端 1-step smoke 和 reload 通过。

### Gate 10：正式续训练确认

- [x] 项目负责人已明确要求开始续训练；有效 batch/step 双口径、FP16、checkpoint 和磁盘方案按本计划执行，LR 由 validation-only 校准自动冻结。

### Phase 11：续训练与固定网格评估

- [x] 按 `0/5/10/20/30/45/60/90/120/180/240/300` optimizer step 保存评测快照；正式训练已完成 300 step。
- [x] 汇总训练/validation loss、各 head 指标、AMP、曝光量和状态分布。
- [x] Easy Turn 完整评测 step 0/5/10/20/30；step 10 首次触发类别级明显下降并由 step 20 确认，step 30 继续恶化。项目负责人据此于 2026-08-21 决定提前结束耗时评测，45 及后续点不再执行；该变更按事后决定披露，不改写为预注册方案。
- [x] 本轮不再对续训练 checkpoint 运行 Full-Duplex-Bench：没有续训练点通过模型级稳定性门禁，会议报告的最终结论限于 Table 3 模型级评估。
- [x] 给出退化边界和推荐模型：step 5 已有轻微/不均衡退化，step 10 为首个已确认类别级明显下降点，step 20 为首个 EN/ZH 宏平均均明显下降点；保持通用能力时推荐官方 step 0。

### Phase 12：会议汇报文档

- [x] 创建模型性能评估文档骨架。
- [x] 按最终评测窗口回填官方基线、step 5/10/20/30 指标、配对统计检验和曲线；未测点不插值，step 45 partial 不纳入。
- [x] 回填最终结论、局限、复现信息、模型哈希和推荐 checkpoint，并生成 Markdown 与自包含 HTML 仪表盘。

### Phase 13：官方训练流程纠正

- [x] 确认旧“正式训练”没有运行官方 `finetune.py`/Lightning Trainer，而是项目自定义 optimizer loop；旧结果改按 pilot 方法解释，不再表述为官方训练方法。
- [x] 建立 `agent_governance/EXECUTION_APPROVAL_PROTOCOL.md`；代码修改、训练和评测实行执行前披露与确认。
- [x] 验证 `third_party/SoulX-Duplug-upstream` 保持官方 commit `928b065` 且工作树 clean。
- [x] 从该提交建立独立 `SoulX-Duplug-928b065-official-continual-v1` 运行时，仅保留空 head NaN、严格 checkpoint 加载、冻结会话级 split、scheduler 起点和审计补丁。
- [x] 完成 preflight v1/v2/v3：v1 暴露验证 Callback 读取顺序问题并原样保留；v2 修复指标记录；v3 生成真实54 MB overlay并通过正式 Table 3 加载器。
- [x] 使用官方 `Trainer.fit()`、`training_step()`、`configure_optimizers()`、AdamW、WarmupAnnealSteps、FP16 和 Lightning 梯度累积完成 local step 1/2/3/4/5。
- [x] 正式实际样本数为 `576/576/576/338/576`；step4 是官方 epoch-tail 更新，未跨 epoch 人工拼批。
- [x] 保存 step1/2/3/5 四个 compact overlay 并全部通过118个 trainable tensor 严格加载审计。
- [x] 保存4.005 GB `last.ckpt`；确认 global_step=5、scheduler last_epoch=1805，包含 AdamW、AMP scaler 与 Callback 状态。
- [x] 创建 `evaluation_reports/duplexconv_edu0018_official_continual_training_audit.md`，明确新旧训练方法边界和当前未完成项。
- [x] 经单独确认后补跑官方同口径 group-validation step0；使用完整102行冻结 validation、`Trainer.validate()`，没有 optimizer/backward/训练更新，不能用 sanity check 或旧 runner 指标冒充。
- [x] 经单独确认后对官方流程 step1/2/3/5 执行冻结 Table 3 配对评测；连同 step0 共20项，全部 evidence gate 通过。
- [x] 用纠正后的结果生成新的正式会议汇报 Markdown/HTML/audit JSON；旧自定义结果只保留为 pilot 对照。

### Phase 14：DuplexConv 扩展准备与资源门禁

- [x] 项目负责人确认扩大 DuplexConv 规模是下一研究阶段；扩展仍属于同一个 DuplexConv 项目，不新建另一份独立数据集计划文档。
- [x] 明确当前 Table 3 只证明模型学到了数据中的状态信号，同时已观察到 Complete 提升、Incomplete 下降；不得表述为模型整体性能已确定提高。
- [x] 明确 Table 3、Easy Turn 和 Full-Duplex-Bench 不得进入训练、补标、prompt 示例或数据选择；扩展样本只能依据训练域 metadata 与内部开发集统计选择。
- [x] 建立 `project_state/` 辅助任务账本和上下文恢复规则；当前对话仍是主要依据，磁盘状态用于防止长任务/compact 后方向偏离。
- [x] 冻结资源互斥：Table 3 运行期间只允许 metadata 盘点、官方 shard 清单调查、成本估算和泄漏门禁设计；禁止并行 Paraformer、GLM tokenizer、训练或其他 GPU 工作。
- [x] 完成 22,050 个 Edu metadata 的分 shard 规模、声道、时长、事件、状态缺失、WAIT、LID、存储和 Qwen 成本盘点。
- [x] 从官方发布源固定 45 个 `Edu_0001.tar`–`Edu_0045.tar` 的文件名、精确字节数、Git/Xet 身份；首批候选另固定远端 SHA-256。官方未发布 archive member manifest，因此成员范围在下载前只能标为高置信推断，下载后必须按 tar 成员做精确闭环。
- [x] 冻结 benchmark 跨数据集泄漏 gate 设计；未下载音频时只允许做 ID/metadata 级预检，PCM 与近重复检查必须在音频到位后完成。
- [x] 根据训练域状态分布提出第一批扩展 shard、磁盘、API 费用和运行时间方案；项目负责人于 2026-08-22 确认 `Edu_0019` 首批方案，授权范围不含训练或新的 checkpoint 评测。
- [x] Table 3 manifest 已完成并生成正式 Markdown/HTML/audit JSON；扩展 Paraformer/GLM 的这一前置依赖已经解除，但仍需先补齐英文 Full-Duplex-Bench 音频泄漏 denylist 的下载授权与门禁。
- [x] 项目负责人授权英文 Full-Duplex-Bench 下载；9 个 ZIP（705,372,348 bytes）已完成逐文件 SHA-256 与 ZIP 完整性验收，不保存第二份完整解压副本。
- [x] 数据盘已扩容到 500 GiB；项目负责人确认两阶段扩展方案。第一阶段不删除旧目录，`Edu_0019` 试运行通过后连续扩展到 `Edu_0045`，不含训练和新 checkpoint 评测。
- [x] 只使用 benchmark 自身的同源/异源校准对冻结音频指纹；最终规则为最少64个对齐帧、相似度阈值0.800，且在此之前未下载、查看或打分 `Edu_0019`。首次失败校准及原因完整保留。
- [x] 完成 `Edu_0019` 全链路试运行与 Gate A/B/C/D；全部通过，已满足连续扩展 `Edu_0020`–`Edu_0045` 的前置门禁。

#### Phase 14 盘点结果与待确认首批方案

盘点产物位于：

```text
/root/autodl-tmp/dataset/duplexconv/work/expansion_inventory_v1/
  run_manifest.json
  shards.jsonl
  inventory_summary.md
/root/SoulX-stage3-dataset/project_state/
  duplexconv_edu_official_shards_v1.tsv
```

22,050 个 uppercase `Edu` metadata 合计约 495.918 个去重源会话小时、1,000.479 个 target-view 小时和 397,192 个事件；声道分布为 21,761 个双声道、275 个三声道、14 个四声道。状态分布为 complete 208,394、incomplete 72,511、backchannel 39,032、缺失 76,409、WAIT 846。这里仅覆盖官方 2,000.21 小时全库中的 uppercase `Edu` 子目录，不能把约 496 个去重源小时误写成完整 DuplexConv。

官方发布源固定为 Hugging Face `qualialabsAI/DuplexConv@0bb99da7ab7a2f6f86d6b23df92c9383e711d09a`。官方页面列出 uppercase `Edu/audios` 共 45 个 tar、346 GB。首批建议选择紧邻已处理 `Edu_0018` 的 `Edu_0019`，而不是依据 Table 3 类别结果或论文目标挑选分片：

```text
archive: Edu/audios/Edu_0019.tar
official bytes: 7,612,467,200
official SHA-256: 196df403442b042caf4df430772d68ccb8f8d6229dd7d101b384471debbef852
official Xet hash: 36d8f2fd286b3af17ae8266cace553373c0561e3ce02328603c39e2d0ae78011
inferred source range: Edu--011079 .. Edu--011706
source conversations: 500
tracks: 492 x 2ch, 8 x 3ch
target views: 1,008
source hours: 10.847
target-view hours: 22.024
events: 8,567
official complete/incomplete/backchannel: 4,535 / 1,544 / 852
missing events: 1,620 across 405 source conversations
WAIT: 16 -> complete
Qwen accepted-response cost estimate: 0.219 USD
```

成员范围是按 22,050 metadata 的连续分组推断；它与已下载 `Edu_0018` 的 500/500 成员精确一致，而且 `Edu_0019` 原始 PCM 估算值 7,611,442,470 bytes 与官方 tar 的差额约 1.02 MB，符合 tar header/padding 开销。但这仍不等于官方成员清单：正式下载后必须核验 500 个 WAV 与 metadata ID 完全一致，否则立即停止。

数据盘已于 2026-08-22 扩容为 500 GiB，核验时可用 `398,412,722,176` bytes。`Edu_0019`–`Edu_0045` 预计新增原始 tar 与 16 kHz target-view 音频合计 `272,567,876,528` bytes；连同既有 `Edu_0018`，形成约 303.651 去重源会话小时和 612.680 target-view 小时。第一阶段不删除约 108.2 GB 的旧目录，仍必须为每个 shard 设置 20 GiB 最低空闲门禁并预留临时文件余量。下载直接写唯一 `.part` 后校验并原子改名，禁止产生第二份完整 Hugging Face/Xet cache；全程不解包保存 loose 48 kHz WAV。

现有构造实现包含 `Edu_0018` 专用硬编码，不能直接运行新分片。待确认补丁只把下列内容参数化，并保留原 `Edu_0018` 回归测试：source-scan 预期计数、finalize 的请求/事件闭环、dataset version 和 model-ready index 前缀、网络 route provenance。uppercase `Edu` 全部 metadata 另有 14 个四声道源；通用 profile 必须把支持范围从 2/3 声道扩展到 2/3/4 声道，四声道仍逐个目标声道生成四个 target views，其余三路只聚合为 `other activity`，不混音、不把多路 audio token 塞入一个 sequence。首批 `Edu_0019` 本身只有 492 个双声道和 8 个三声道，不会用“避开四声道”作为丢数据策略。SoulX 官方上游不修改，旧 `Edu_0018` 产物不覆盖。首批单独写入 `source_scan_edu0019_v1`、`state_labels_edu0019_v1`、`target_audio_edu0019_v1`、`paraformer_edu0019_v1`、`timelines_edu0019_v1`、`glm_audio_tokens_edu0019_v1` 和 `model_ready/edu0019_stage3_zh_v1`。

首批执行顺序为：官方 SHA-256/成员闭环 -> benchmark Gate A/B/C -> Qwen 补 1,620 个缺失状态 -> WAIT 映射 -> target audio -> Paraformer -> timeline -> GLM tokenizer -> model-ready 验收 -> Gate D。Qwen 固定 `qwen3-235b-a22b-instruct-2507`，日预算硬上限 10 USD；运行前检查共享 key 当日余额。网络不依赖项目负责人的本地电脑：先对 OpenRouter 直连和 AutoDL 代理做无付费连通性比较，再固定本轮 route 并写入 provenance。

以 `Edu_0018` 实测为依据，首批从下载到 model-ready 预计约 3–5 小时：下载 10–60 分钟，扫描/哈希/泄漏检查 30–75 分钟，Qwen 约 60–90 分钟，target audio 约 10–20 分钟，Paraformer 约 20–30 分钟，GLM/timeline/export/验收约 15–30 分钟。任何异常、重试和网络波动会延长；本估计不含后续训练与 Table 3 再评测。

#### Phase 14.1：Edu_0019 实际执行记录与当前 TODO

截至 2026-08-23，Edu_0019 已完成：

- [x] 官方 `Edu_0019.tar` 下载、7,612,467,200 bytes 和 SHA-256 验收；500 个 WAV 与 metadata 闭合。
- [x] 源扫描：492 个双声道、8 个三声道、1008 target views、22.02385 target-view 小时、8567 个事件。
- [x] benchmark 泄漏筛查 v2.2：保留 v1/v2 失败证据；最终冻结条件为相似度 `0.655`、至少 64 个对齐帧且至少 16 个 LSH votes，Edu_0019 为 0 个 source quarantine。阈值只由 benchmark 校准/验证划分确定，未使用训练候选结果反向调参。
- [x] 状态闭包：6931 个官方三状态、16 个 WAIT 确定性映射、1620 个固定 Qwen 补标，共 8567 个事件；405/405 请求和 1620/1620 event_id 审计通过。
- [x] 目标音频：1008 个 16 kHz/16-bit/mono views，全部与 160 ms chunk 边界闭合，约 2.4 GiB。
- [x] Paraformer 1.2.6 正式运行：1006 个 views 通过、124446 tokens；`Edu--011377/target-ch01` 与 `Edu--011575/target-ch01` 因最后 token `end_ms` 比补齐音频末尾多 20 ms 被严格隔离。
- [x] 使用与 Edu_0018 相同的 FunASR 1.3.9 对上述两条独立复核，仍为 0/2 通过，证明不是 1.2.6 独有差异；不做全量 1.3.9 重跑。
- [x] 项目负责人确认：不裁剪、不夹取、不放宽 Paraformer 时间戳；将两个失败 view 作为 source-view quarantine 原样传播，只处理其余 1006 views。

source-view quarantine 补丁固定闭包：

```text
1008 input views = 1006 processable views + 2 source-view quarantine
8567 input events = 8549 processable-view events + 18 quarantined-view events
input chunks = processable timeline chunks + timeline-level quarantined chunks
             + 980 source-view quarantined chunks
```

补丁只允许修改项目数据构造层的 `render_asr.py`、`timeline.py`、`glm_audio_tokens.py`、`model_ready.py`、`validate_model_ready.py` 及对应测试。它必须保留 quarantine 的 view ID、原因、chunk/event 数量和 provenance；不得修改 Paraformer 输出、官方状态、Qwen 标签、benchmark 阈值或 SoulX 官方上游。

当前 TODO：

- [x] 完成 quarantine 传播补丁的定向测试和全套项目测试；26 项定向测试与 88 项全套 `unittest` 全部通过，Python 语法编译和 `git diff --check` 通过，未安装新依赖。
- [x] 从已冻结的 1.2.6 Paraformer 结果生成独立 `paraformer_rendered_edu0019_v1`；1006 个结果与 2 个 source-view quarantine 分区闭合，checksum 通过。
- [x] 构造 `timelines_edu0019_v1`；1008=1006+2 views、8567=8549+18 events、495985=495005+980 原始 chunks 均闭合。可用 timelines 内另有 2955 个按既有规则隔离的局部 chunks，不与 source-view quarantine 混淆。
- [x] 对 1006 个可用 views 运行固定 GLM-4-Voice tokenizer；1006/1006 通过、0 个新增 GLM quarantine，共 990018 个音频 token。2 个 source-view quarantine 未进入模型推理，其 provenance 与 timeline 文件逐字节一致。
- [x] 导出并验证 `model_ready/edu0019_stage3_zh_v1`：2231 rows、492054 个可用 chunks；官方两列 loader 成功，最大 token 长度 1500，20 个随机窗口逐 chunk 回解、checksum 与 1008-view/495989-chunk 全局闭包全部通过。
- [x] 对最终 model-ready source/view identity 执行 benchmark Gate D：1006 个最终 views 与选择 tar 精确闭合，500 个 source IDs 无 benchmark 身份碰撞；exact、normalized、content exact、window 和 near-duplicate gate hit 均为 0，冻结 v2.2 `gate_passed=true`。
- [x] Edu_0019 全部 gate 通过后，按连续顺序扩展 Edu_0020–Edu_0045；每个 shard 均执行 20 GiB 最低空闲空间门禁，未把 Table 3 结果用于 shard 选择、标签或阈值调整。

#### Phase 14.2：Edu_0020–Edu_0045 连续扩展执行

Edu_0019 已满足全部前置门禁，连续扩展现在按 shard 顺序执行。每个 shard 都使用独立目录、独立 manifest 与 checksum，失败只停止当前 shard，不覆盖前序产物；训练和新 checkpoint 评测仍不在本阶段范围内。

- [x] 冻结 Edu_0020 metadata/官方对象合同：500 sources、488×2ch+12×3ch、1012 views、8786 events、1699 missing states；官方 tar 为 7,966,894,080 bytes，SHA-256=`3c257692...ca135`。
- [x] 断点下载并验收 Edu_0020：7,966,894,080 bytes 与官方 SHA-256 精确一致；500 个 WAV、首尾 source ID 和预冻结 source-ID 哈希闭合，保持 20 GiB 空闲门禁且没有第二份 Hugging Face cache。
- [x] 对 Edu_0020 执行 source contract、冻结 benchmark Gate A/B/C、Qwen 补标、WAIT 映射、target audio、Paraformer、timeline、GLM、model-ready 验证和 Gate D；全部门禁通过。
- [x] Edu_0020 全部闭环后连续扩展至 Edu_0045；每个 shard 的异常和 quarantine 均单独记录，未假设与 Edu_0019 相同。

Edu_0020 当前执行记录：

- [x] source scan 合同通过：500 sources、1012 views、8786 events、0 source quarantine。
- [x] 冻结 v2.2 Gate A/B/C 通过：1512 个原声道/mix 候选，0 泄漏 source。
- [x] OpenRouter 连通、10 USD 日限额和30-event校准门禁通过；fixed Qwen full run 经410-cache恢复后达到411/411 requests、1699/1699 labels，成功响应成本为 `$0.20388199`。
- [x] 状态 finalization 通过：7065 官方三状态、22 WAIT→complete、1699 Qwen，合计8786事件；最终 complete/incomplete/backchannel=`4758/1981/2047`。
- [x] target audio：1012 个 16 kHz/mono/chunk-pad WAV 与 manifest/checksum 全部闭合。
- [x] Paraformer 1.2.6：1012 input views = 1010 passed + 2 strict quarantine；129134 tokens，不裁剪或修改输出。
- [x] timeline：1012=1010+2 views、8786=8728+58 events、519085=515194+3891 原始 chunks 全局闭合；3273 个局部隔离 chunks 与整 view quarantine 分层记录。
- [x] GLM：1012 input views = 1010 eligible + 2 upstream source-view quarantine；1010/1010 全部通过，0 新增 GLM quarantine，生成 1030398 audio tokens；上游隔离 provenance 逐字节一致且 checksum/全局 partition 闭合。
- [x] model-ready：2297 rows、1010 source views、511926 exported chunks；官方未修改 loader 成功读取 train split=2182，20 个确定性随机 roundtrip 通过，checksum 与全局 view/chunk closure 全部通过。
- [x] 对最终 1010 个贡献 views 执行冻结 benchmark Gate D：500 个 source IDs 无身份碰撞；exact、normalized、content exact、window 与 near-duplicate gate hit 均为 0，冻结 v2.2 `gate_passed=true`。

Edu_0021 当前执行记录：

- [x] 预冻结官方对象与 metadata 合同：官方 tar=7,560,140,800 bytes、SHA-256=`239cac1a...a299`；预期500 sources、492×2ch+7×3ch+1×4ch=1009 views、8702 events、1597 missing states，source IDs=`Edu--012313`–`Edu--012925`。
- [x] 通过20 GiB最低空闲空间门禁后断点下载；7,560,140,800 bytes与官方SHA-256精确一致，500个WAV的首尾ID、完整source-ID哈希和唯一性全部闭合，没有第二份Hugging Face cache。
- [x] source contract扫描通过：500 sources、0 source quarantine；492×2ch+7×3ch+1×4ch=1009 views、8702 events全部闭合，6条确定性尾部修复单列记录。
- [x] 冻结v2.2 benchmark Gate A/B/C通过：1509个原声道/mix候选，exact、normalized、content exact、window与near-duplicate gate hit均为0，0泄漏source。
- [x] 为1597个缺失状态准备固定Qwen请求：413 requests、预计1189645 prompt tokens，full确认令牌=`CONFIRM_1597_LABELS`；30-event校准集由22个分组请求构成，本步骤未调用API。
- [x] OpenRouter服务端10 USD/day预算与1-event连通通过；22 requests/30 events校准机械门禁通过，0标签塌缩，1个响应经第2次schema尝试合规；官方LLM辅助标签一致率63.33%仅作诊断，校准成本`$0.00498648`。
- [x] 固定模型/direct route/4 workers full run完成：413/413 requests、1597/1597 labels，0重复/0标签塌缩；386个首轮schema通过、27个第2轮通过。full accepted cost=`$0.18900011`，Edu_0021连通+校准+full唯一成功响应总成本=`$0.19487899`。
- [x] finalization：7085个官方三状态保持不变、20个WAIT→complete、1597个Qwen状态，8702事件1:1闭包；最终complete/incomplete/backchannel=`4830/1900/1972`。
- [x] target audio：1009个16 kHz mono/chunk-pad WAV与manifest/checksum全闭合；984/21/4 views来自2/3/4声道源，四声道`Edu--012645`保留4个独立视图。
- [x] 固定本地Paraformer/FunASR 1.2.6完成：1009=1008 passed+1 strict quarantine，共126864 tokens；`Edu--012727/target-ch00`末token越音频端点20 ms，原样隔离而不裁剪。4个四声道视图全部通过。
- [x] render/timeline：1009=1008+1 views、8702=8684+18 events、492558=491717+841原始chunks全局闭合；4个terminal pad chunks，3298个局部隔离chunks/565条记录与整view隔离分层保留。
- [x] GLM：1009 input=1008 eligible+1 upstream view quarantine；1008/1008通过、0新增GLM quarantine，生成983442 audio tokens，上游隔离provenance逐字节一致。
- [x] model-ready：2265 rows、1008 source views、488423 exported chunks；官方未修改loader读取train split=2151，20个确定性roundtrip、checksum与全局view/chunk closure通过。
- [x] 对最终1008个贡献views执行冻结benchmark Gate D：500个source IDs无身份碰撞；exact、normalized、content exact、window与near-duplicate gate hit均为0，冻结v2.2 `gate_passed=true`。

Edu_0022 当前执行记录：

- [x] 预冻结官方对象与metadata合同：官方tar=8,036,096,000 bytes、SHA-256=`63e00118...f449a`；预期500 sources、496×2ch+4×3ch=1004 views、9175 events、1741 missing states，source IDs=`Edu--012926`–`Edu--013550`。
- [x] 通过20 GiB最低空闲空间门禁后断点下载；8,036,096,000 bytes与官方SHA-256精确一致，500个WAV、首尾ID及完整source-ID哈希全部闭合，没有第二份Hugging Face cache。
- [x] source contract扫描通过：500 sources、0 source quarantine；496×2ch+4×3ch=1004 views、23.249604 target-view hours、9175 events全部闭合，无确定性尾部修复。
- [x] 冻结v2.2 benchmark Gate A/B/C通过：1504个原声道/mix候选，exact、normalized、content exact、window与near-duplicate gate hit均为0，0泄漏source。首次接力因误用base Conda而在评分前失败，修正为既有`soulx-duplug-official`环境后完整重跑；两份失败日志均保留。
- [x] 为1741个缺失状态准备固定Qwen请求：409 requests、预计1252415 prompt tokens，full确认令牌=`CONFIRM_1741_LABELS`；30-event校准集由22个分组请求构成，本步骤未调用API。首次误用默认3条/类生成9-event校准集，发现后在API调用前删除该4.6 MiB无效请求目录并按冻结10条/类重建，错误摘要SHA保留在任务账本。
- [x] OpenRouter服务端10 USD/day预算与1-event连通门禁通过；22 requests/30 events校准机械门禁通过，0标签塌缩、22个响应均首轮schema合规；官方LLM辅助标签一致率63.33%仅作诊断，连通与校准成本分别为`$0.0008206`和`$0.0051004375`。
- [x] 固定模型/direct route/4 workers full run完成：409/409 requests、1741/1741 labels，0重复/0标签塌缩；386个首轮schema通过、23个第2轮通过。full accepted cost=`$0.1968490875`，连通+校准+full唯一成功响应总成本=`$0.202770125`。
- [x] finalization通过：7416个官方三状态保持不变、18个WAIT→complete、1741个Qwen状态，9175事件1:1闭包；最终complete/incomplete/backchannel=`4992/1997/2186`。
- [x] target audio：1004个16 kHz mono/chunk-pad WAV与manifest/checksum全闭合；992/12 views来自2/3声道源。
- [x] 固定本地Paraformer/FunASR 1.2.6完成：1004/1004 views全部通过、0 strict quarantine，共134318 tokens；未裁剪或修改输出。
- [x] 确定性文本渲染完成：1004=1004+0 views分区闭合，token和时间戳保持不变。
- [x] timeline完成：1004 views、9175 events、523576 original chunks全局闭合；1个terminal pad chunk，2903个局部隔离chunks/481条记录，0个整view隔离。
- [x] GLM完成：1004 input/eligible/passed views、0新增GLM quarantine、0 upstream view quarantine，生成1047154 audio tokens，与523577 effective chunks精确对应。
- [x] model-ready完成：2295 rows、1004 source views、520674 exported chunks；官方未修改loader读取train split=2180，20个确定性roundtrip、checksum与523577-chunk全局闭包通过。
- [x] 对最终1004个贡献views执行冻结benchmark Gate D：确定性tar含1004个唯一WAV，与选择列表及model-ready view集合逐项一致；冻结scorer保守产生2008条channel-0/mono-mix记录，500个source IDs无身份碰撞，exact、normalized、content exact、window与near-duplicate gate hit均为0，独立closure审计`gate_passed=true`。

Edu_0023 当前执行记录：

- [x] 预冻结官方对象与metadata合同：官方tar=7,968,808,960 bytes、SHA-256=`e0201c0b...e7a8f`；预期500 sources、489×2ch+11×3ch=1011 views、9084 events、1820 missing states，source IDs=`Edu--013551`–`Edu--014155`。
- [x] 20 GiB最低空闲空间门禁通过，下载前可用345,723,060,224 bytes；已从固定官方revision直接向dataset目录启动可断点下载，不创建第二份Hugging Face cache。
- [x] 下载完成并通过官方bytes/SHA-256；source contract为500 sources、1011 views、9084 events、0结构隔离，3个确定性尾部修复，全部预冻结检查通过。首次post-download watcher因manifest创建竞态退出，第二次因shell转义错误退出，两者都未开始扫描或评分且日志已保留；第三次完成门禁。
- [x] 原始500-source对象按冻结benchmark Gate A/B/C正确失败并在Qwen前暂停：1511候选中四类强证据均为0，但`Edu--014047.wav` ch1对`candor_turn_taking/62/input.wav`产生1个v2.2近重复命中（similarity=0.761276、97 frames、41 votes）。波形相关仅0.0139且候选为低变化指纹，无法确认逐波形复用，也未通过修改阈值将其改判为通过；原始tar、失败输出和诊断均永久保留。
- [x] 经负责人确认，保守隔离整个`Edu--014047`会话而非只删除命中声道；构造确定性的499-source sanitized tar（7,965,470,720 bytes，SHA-256=`382d9202...a0c`），成员集合严格等于官方500项减去该1 source。
- [x] sanitized source contract扫描通过：499 sources、488×2ch+11×3ch=1009 views、23.046735 target-view hours、9081 events、1819 missing states、0结构隔离；原有3条确定性尾部修复照常记录。
- [x] 使用完全不变的冻结v2.2对sanitized对象重跑Gate A/B/C：1508候选，exact、normalized、content exact、window及near-duplicate gate hit全部为0，0泄漏source，`gate_passed=true`。
- [x] 为1819个缺失状态离线冻结400个full请求（预计1,225,270 prompt tokens），固定模型`qwen/qwen3-235b-a22b-2507`、prompt v2、direct route、4 workers和`CONFIRM_1819_LABELS`确认令牌。
- [x] OpenRouter服务端10 USD/day预算门禁和1-event连通通过；26 requests/30 events校准机械门禁通过，三类状态=`9/9/12`、无标签塌缩；与官方LLM辅助标签70%一致仅作为诊断，校准accepted cost=`$0.0059206175`。
- [x] 400 requests/1819 events完整Qwen标注和机械审计通过：事件/签名1:1闭合、0重复、固定模型与direct route一致、0标签塌缩；378/21/1个请求分别在schema attempt 0/1/2合规，状态=`backchannel 1296 / complete 184 / incomplete 339`，accepted cost=`$0.1967818225`。
- [x] finalization通过：7249个官方三状态保持不变、13个WAIT→complete、1819个Qwen状态，9081事件1:1闭包；最终complete/incomplete/backchannel=`4895/1926/2260`，checksums全部通过。
- [x] 从sanitized tar提取1009个16 kHz mono/chunk-pad target views并通过全量checksums：488个双声道source贡献976 views、11个三声道source贡献33 views，输入archive SHA严格指向sanitized对象，隔离source未回流。
- [x] 固定本地Paraformer/FunASR 1.2.6完成：1009/1009 views通过、0 strict quarantine、132709 tokens；没有裁剪或修改输出，耗时994.815491秒。
- [x] 确定性文本渲染完成：1009=1009+0 views分区闭合，token与时间戳保持不变。
- [x] timeline完成：1009 views、9081 events、518985 original chunks全局闭合；3个terminal pad chunks，3354个局部隔离chunks/558条记录，0个整view隔离。
- [x] GLM完成：1009 input/eligible/passed views、0新增GLM quarantine、0 upstream view quarantine，生成1037976 audio tokens，与518988 effective chunks精确对应。
- [x] model-ready完成并验证：2331 rows、1009 source views、515634 exported chunks；官方未修改loader读取train split=2214，20个确定性roundtrip、checksum与518988-chunk全局闭包通过。
- [x] 对最终1009个贡献views执行冻结benchmark Gate D：确定性tar、选择列表和model-ready集合逐项闭合，冻结scorer保守产生2018条channel-0/mono-mix记录；499个source IDs无benchmark身份碰撞，exact、normalized、content exact、window与near-duplicate gate hit均为0，`Edu--014047`在窗口、列表、tar和评分记录中全部缺席，独立closure审计`gate_passed=true`。

Edu_0024 当前执行记录：

- [x] 固定官方revision对象：`Edu_0024.tar`=7,472,896,000 bytes、官方LFS文件SHA-256=`6c1fceb7...65cf6`、Xet存储哈希=`0b21cf99...74aac`；预冻结metadata合同为500 sources、493×2ch+7×3ch=1007 views、8575 events、1660 missing states，source IDs SHA-256=`de615690...fddcb`。
- [x] 20 GiB最低剩余空间门禁通过，下载前可用324,268,965,888 bytes；不创建第二份Hugging Face cache，不在下载阶段调用LLM。
- [x] 可断点下载完成。官方Hugging Face直连超时，AutoDL代理初始吞吐约20 KiB/s；只读线路测试后切换为`hf-mirror`直连。首次低速进程停止时未保留有效partial字节；完整下载后因将Xet hash误当文件SHA而被门禁正确拦截，随后通过Hugging Face官方tree API确认LFS SHA=`6c1f...`、Xet hash=`0b21...`，修正合同并复用完整partial复核成功，未重新下载；首次失败证据永久保留。
- [x] source contract扫描通过：500 sources、0 source quarantine；493×2ch+7×3ch=1007 views、21.619977 target-view hours、8575 events全部闭合，5个确定性尾部修复。
- [x] 完全不变的冻结benchmark Gate A/B/C通过：1507候选，exact、normalized、content exact、window与near-duplicate gate hit均为0，0泄漏source，Qwen前付费调用数为0。
- [x] 离线冻结405个full请求覆盖1660个缺失状态（预计1,172,998 prompt tokens），固定模型`qwen/qwen3-235b-a22b-2507`、prompt v2、direct route、4 workers和`CONFIRM_1660_LABELS`确认令牌。
- [x] 1-event连通性与服务端10 USD/day门禁通过，accepted cost=`$0.00106612`；26 requests/30 events校准机械门禁通过，状态=`backchannel 9 / complete 12 / incomplete 9`、0标签塌缩，accepted cost=`$0.0055021525`；与官方LLM辅助标签53.33%一致仅作诊断。
- [x] 405 requests/1660 events完整Qwen标注及机械审计通过：388/17个请求分别在schema attempt 0/1合规，0重复、0标签塌缩，固定模型/route一致；状态=`backchannel 1232 / complete 179 / incomplete 249`，full accepted cost=`$0.1896586375`，服务端当日累计用量=`$1.082733519 / $10`。
- [x] finalization通过：6899个官方三状态保持不变、16个WAIT→complete、1660个Qwen状态，8575事件1:1闭包；最终complete/incomplete/backchannel=`4708/1846/2021`。
- [x] target audio：1007个16 kHz mono/chunk-pad WAV与manifest/checksum全闭合；986/21 views来自2/3声道源。
- [x] 固定本地Paraformer/FunASR完成：1007输入=1006通过+1 strict quarantine，125644 tokens，checksums通过。`Edu--014193/target-ch00`最后token结束时间比音频边界多20 ms，fresh定向复跑复现；未裁剪时间戳或放宽parser，保守隔离该view。
- [x] 确定性文本渲染完成：1006有效+1传播隔离=1007 views分区闭合，token与时间戳保持不变。
- [x] timeline完成：1007输入views/8575 events/486925 original chunks闭合为1006有效views/8573 events/486396 original chunks + 1隔离view/2 events/529 chunks；另含1个terminal pad chunk和3059个局部隔离chunks/434条记录。
- [x] GLM完成：1006 eligible/passed views、0新增GLM quarantine、1 upstream view quarantine，生成972794 audio tokens，与486397 effective chunks精确对应。
- [x] model-ready完成并验证：2160 rows、1006有效source views、483338 exported chunks；官方未修改loader读取train split=2052，20个确定性roundtrip、checksum与486926输入chunks全局闭包通过。
- [x] 对最终1006个贡献views执行冻结benchmark Gate D：确定性tar、选择列表和model-ready集合逐项闭合，冻结scorer产生2012条channel-0/mono-mix记录；500个source IDs无benchmark身份碰撞，exact、normalized、content exact、window与near-duplicate gate hit均为0，独立closure审计`gate_passed=true`。首次启动因误建空输出目录在评分前退出，失败日志保留；确认目录为空后仅以`rmdir`删除并正确重启。

Edu_0025 当前执行记录：

- [x] 预冻结官方对象与metadata合同：官方tar=7,541,288,960 bytes、官方LFS文件SHA-256=`c8add7e5...91f5`、Xet存储哈希=`59f22464...a170`；预期500 sources、497×2ch+3×3ch=1003 views、8758 events、1669 missing states，source IDs=`Edu--014770`–`Edu--015387`。
- [x] 下载前磁盘门禁通过：可用311,650,254,848 bytes，高于完整archive加20 GiB安全余量所需的29,016,125,440 bytes；下载与后续A/B/C门禁阶段不调用LLM、不训练、不做checkpoint评测。
- [x] 从固定官方revision经已验证的`hf-mirror`直连线路完成可断点下载；单连接先后约60 KiB/s、短暂升至约1.3 MiB/s后回落至约0.3 MiB/s，四线路小范围只读测试确认镜像直连仍最优，保留126,320,640-byte partial并依次切换为8路、16路HTTP Range续传。最终7,541,288,960 bytes与官方LFS SHA-256=`c8add7e5...91f5`精确一致；aria2稀疏逻辑大小未被用作完成依据。
- [x] source contract扫描通过：500 sources、497×2ch+3×3ch=1003 views、21.817886 target-view hours、8758 events、0结构隔离，6条确定性尾部修复；完全不变的冻结benchmark Gate A/B/C对1503候选的五类命中均为0，0泄漏source，Qwen前付费调用数为0。
- [x] 离线冻结415个full请求覆盖1669个缺失事件（预计1,210,710 prompt tokens），固定模型`qwen/qwen3-235b-a22b-2507`、prompt v2、direct route、4 workers和`CONFIRM_1669_LABELS`。首次请求目录仅缺离线token估算且尚未调用API；确认两版语义和签名完全一致后，删除4.5 MB可重建的无估算目录并保留带估算正式版本。
- [x] 1-event连通性及服务端10 USD/day预算门禁通过；22 requests/30 events校准机械门禁通过，三类状态=`backchannel 8 / complete 13 / incomplete 9`、全部首轮schema合规、0标签塌缩，校准成本`$0.0045726325`；与官方LLM辅助标签73.33%一致仅作诊断。
- [x] 使用精确确认令牌完成415 requests/1669 events完整Qwen标注和机械/provenance审计：事件/签名1:1闭合、0重复、固定模型与direct route一致、0标签塌缩；399/16个请求分别在schema attempt 0/1合规，状态=`backchannel 1200 / complete 195 / incomplete 274`，accepted cost=`$0.19622908`，服务端当日累计用量=`$1.291956173 / $10`。
- [x] finalization通过：7071个官方三状态保持不变、18个WAIT→complete、1669个Qwen状态，8758事件1:1闭包；最终complete/incomplete/backchannel=`4781/1906/2071`。
- [x] target audio：1003个16 kHz mono/chunk-pad WAV与manifest/checksum全闭合；994/9 views来自2/3声道source。提取后附加只读hash命令曾误写manifest文件名而非处理失败，纠正文件名后hash通过，无需重跑。
- [x] 固定本地Paraformer/FunASR完成：1003输入=1001通过+2 strict quarantine，共124047 tokens；`Edu--015079/target-ch01`与`Edu--015369/target-ch00`均在存在语义target activity时返回空文本/时间戳，独立fresh-cache复跑逐项复现且隔离清单逐字节一致，未挑选成功输出、伪造时间戳或放宽parser。
- [x] 确定性文本渲染完成：1001有效+2传播隔离=1003 views分区闭合，token与时间戳保持不变。
- [x] timeline完成：1003输入views/8758 events/491392 original chunks闭合为1001有效views/8740 events/490309 original chunks + 2隔离views/18 events/1083 chunks；另含4个terminal pad chunks和2938个局部隔离chunks/519条记录。
- [x] GLM完成：1003 input=1001 eligible/passed+2 upstream source-view quarantine；0新增GLM quarantine，生成980626 audio tokens，与490313 effective chunks精确对应，上游隔离provenance逐字节一致。
- [x] model-ready完成并验证：2195 rows、1001有效source views、487375 exported chunks；官方未修改loader读取train split=2085，20个确定性roundtrip、checksum与491396输入chunks全局闭包通过。
- [x] 对最终1001个贡献views执行完全不变的冻结benchmark Gate D：确定性tar、选择列表和model-ready集合逐项闭合，冻结scorer产生2002条channel-0/mono-mix记录；500个source IDs无benchmark身份碰撞，exact、normalized、content exact、window与near-duplicate gate hit均为0，独立closure审计`gate_passed=true`。

Edu_0026 当前执行记录：

- [x] 从固定官方revision再次核验对象身份：`Edu_0026.tar`=8,177,203,200 bytes、官方LFS文件SHA-256=`460a874c...914b`、Xet存储哈希=`f4bf9af8...6a13`、Git OID=`c9b5d3b...1658`。
- [x] 从权威metadata archive按已由Edu_0018锚定的确定性规则精确复算500个source IDs：`Edu--015388`–`Edu--015998`，完整集合SHA-256=`4c40a8ac...28a9`；预期497×2ch+3×3ch=1003 views、23.657921 target-view hours、9415 events、1811 missing states。
- [x] 下载前20 GiB最低空闲门禁通过：可用298,923,724,800 bytes，高于完整archive加安全余量所需的29,652,039,680 bytes；不会创建第二份Hugging Face cache。
- [x] 固定官方revision的16路HTTP Range可续传下载及whole-file SHA-256核验完成：8,177,203,200 bytes、SHA-256=`460a874c...914b`精确一致，没有第二份cache；此阶段Qwen调用数为0，不训练、不评测checkpoint。
- [x] source成员/统计合同全部通过，但结构门禁曾按规则暂停：`Edu--015901`官方speaker segment结束36.980 s，真实WAV结束36.804 s，176 ms尾差超过既有160 ms可确定性修复上限16 ms，因此整会话进入source quarantine；没有为了通过而放宽阈值。另一个60 ms尾差按原规则正常修复。
- [x] 项目负责人确认保守排除整个`Edu--015901`。官方tar与首次失败证据均保留；已构造499-source sanitized tar，成员集合严格等于官方500项减一，剩余成员大小一致。sanitized tar=8,170,137,600 bytes、SHA-256=`bb4da72b...f6a25`；减少2 views、6 events（其中1个missing state）和0.020447 target-view hours，160 ms修复阈值保持不变。
- [x] 499-source独立source contract通过：499 sources、1001 views、23.637475 target-view hours、9409 events、0结构隔离；完全不变的冻结Gate A/B/C对1500候选五类命中均为0、0泄漏source。首次串联wrapper在已通过source scan后的只读断言因shell转义语法错误退出，失败日志保留；未重跑source scan，改用独立新目录启动同一冻结scorer并通过。
- [x] 离线冻结412个full请求覆盖1810/1810个缺失事件（预计1,284,909 prompt tokens），固定模型`qwen/qwen3-235b-a22b-2507`、prompt v2、direct route、4 workers和`CONFIRM_1810_LABELS`。
- [x] 服务端10 USD/day预算与1-event连通通过；25 requests/30 events校准机械门禁通过，状态=`backchannel 9 / complete 12 / incomplete 9`、0标签塌缩；与官方LLM辅助标签53.33%一致仅作诊断，校准accepted cost=`$0.0044896475`。
- [x] 412 requests/1810 events完整Qwen标注与机械/provenance审计通过：400/12个请求分别在schema attempt 0/1合规，状态=`backchannel 1277 / complete 229 / incomplete 304`，accepted cost=`$0.2117127625`。容器关闭前267 requests/1146 labels已原子缓存；恢复时保留旧running manifest/log，267条仅回读、只调用剩余145条，跨日统一审计闭包通过。
- [x] finalization通过：7575个官方三状态保持不变、24个WAIT→complete、1810个Qwen状态，9409事件1:1闭包；最终complete/incomplete/backchannel=`5192/2015/2202`。
- [x] target audio：从sanitized tar提取1001个16 kHz mono/chunk-pad WAV并通过manifest/checksum；992/9 views来自2/3声道source，被排除会话未回流。
- [x] 固定本地Paraformer完成：1001输入=1000通过+1 strict quarantine，共135188 tokens；`Edu--015989/target-ch01`存在82个target-activity chunks但返回空文本/时间戳，独立fresh-cache单视图复跑逐字节复现。未伪造文本、替换metadata转写、放宽parser或挑选输出。
- [x] 确定性render/timeline完成：1001=1000有效+1传播隔离views、9409=9408+1 events、532290=532184+106 original chunks全局闭合；另有2个terminal pad chunks、3222个局部隔离chunks/581条记录，与整view隔离分层记录。
- [x] GLM完成：1000/1000 eligible views通过、0新增GLM quarantine、1个上游source-view quarantine，生成1064372 audio tokens并与532186 effective chunks精确对应。
- [x] model-ready完成并验证：2364 rows、1000有效source views、528964 exported chunks；官方未修改loader读取train split=2245，20个确定性roundtrip、checksum与532292输入chunks全局闭包通过。
- [x] 对最终1000个贡献views执行完全不变的冻结benchmark Gate D：确定性tar、选择列表和model-ready集合逐项闭合，冻结scorer产生2000条channel-0/mono-mix记录；499个source IDs无benchmark身份碰撞，五类音频命中均为0，`Edu--015901`在最终选择、tar与评分记录中全部缺席，独立closure审计`gate_passed=true`。

最终数据集总结和会议汇报文档必须单列记录`Edu--015901`结构异常、176 ms实际尾差、160 ms既有修复上限、未放宽阈值的实验诚信处理、整会话排除范围及实际规模损失，不能只报告sanitized后的通过结果。

Edu_0027 当前执行记录：

- [x] 从固定官方revision冻结对象身份：`Edu_0027.tar`=8,255,406,080 bytes、官方LFS文件SHA-256=`a8833f8e...e7ccae`、Xet存储哈希=`2d750458...71347`、Git OID=`ae445aa8...af79`。
- [x] 从权威metadata archive按已由Edu_0018锚定的确定性规则复算500个source IDs：`Edu--015999`–`Edu--016632`，完整集合SHA-256=`08ac6859...23c6f`；预期492×2ch+7×3ch+1×4ch=1009 views、23.884200 target-view hours、9652 events、1838 missing states。
- [x] 下载前20 GiB最低空闲门禁通过：可用276,936,830,976 bytes，高于完整archive加安全余量所需的29,730,242,560 bytes；不会创建第二份Hugging Face cache，Qwen调用数为0。
- [x] Gate A下载前metadata预检通过；固定官方revision的16路HTTP Range下载完成，8,255,406,080 bytes与官方LFS SHA-256精确一致，由独立verifier原子发布，没有第二份cache。
- [x] 500-member/source contract复核通过：500 sources、492×2ch+7×3ch+1×4ch=1009 views、23.884200 target-view hours、9652 events、0结构隔离；4条确定性尾部修复均沿用既有160 ms规则。
- [x] 完全不变的冻结benchmark Gate B/C通过：1509候选的exact、normalized、content exact、window与near-duplicate gate hit全部为0，0泄漏source，`gate_passed=true`；至此Qwen调用数为0。
- [x] 离线冻结408个full请求覆盖1838/1838个缺失事件（预计1,284,145 prompt tokens），固定模型`qwen/qwen3-235b-a22b-2507`、prompt v2、direct route、4 workers和`CONFIRM_1838_LABELS`；30-event校准集由20个请求构成。
- [x] 服务端10 USD/day预算门禁与1-event连通通过；20 requests/30 events校准机械门禁通过，0重复、0标签塌缩，19/1个请求分别在schema attempt 0/1合规；状态=`backchannel 10 / complete 6 / incomplete 14`，与官方LLM辅助标签60%一致仅作诊断，连通+校准成本=`$0.00705447`。
- [x] 固定模型/direct route/4 workers完成408 requests/1838 events full labeling并通过机械/provenance审计：384/24个请求分别在schema attempt 0/1合规，状态=`backchannel 1268 / complete 188 / incomplete 382`；full accepted cost=`$0.208000545`，连通+校准+full总成本=`$0.215055015`，服务端当日累计=`$0.306820164 / $10`。
- [x] finalization通过：7785个官方三状态保持不变、29个WAIT→complete、1838个Qwen状态，9652事件1:1闭包；最终complete/incomplete/backchannel=`5259/2178/2215`。
- [x] target audio：1009个16 kHz mono/chunk-pad WAV及全量checksum闭合；984/21/4 views来自2/3/4声道source。全量校验后附加只读hash命令曾误写manifest文件名，纠正为`audio_manifest.jsonl`后hash通过，无需重跑。
- [x] 固定本地Paraformer/FunASR 1.2.6完成：1009/1009 views通过、0 strict quarantine、137842 tokens；未裁剪或替换任何输出，耗时1067.315247秒。
- [x] Paraformer文本render与状态/activity timeline闭包通过：1009 views、9652 events、537831 original chunks + 2 terminal-silence padding chunks；3753个局部冲突chunks按既有规则隔离，0 source-view quarantine。
- [x] 固定本地GLM-4-Voice tokenizer完成：1009/1009 eligible views通过、0新增或上游source-view quarantine，生成1075666 audio tokens，与537833 effective chunks精确对应。首次启动因误指向仅含`model/`的inference副本而在模型加载前失败且无缓存/输出；保留日志后改用前批次相同的冻结官方`models/`树重启，核心tokenizer实现SHA-256字节一致，未改代码、数据或规则。
- [x] model-ready导出与验证通过：2392 rows、1009 source views、534080 exported chunks；内部checksum、20个确定性roundtrip及537833输入chunks全局闭包通过，官方未修改loader读取train split=2272。
- [x] 对最终1009个贡献views执行完全不变的冻结benchmark Gate D：选择集合、确定性tar与model-ready 1:1闭合，冻结scorer对每个mono view保守评分两次共2018条记录；500个source IDs无身份碰撞，五类音频命中全部为0，独立closure审计`gate_passed=true`。

Edu_0028 当前执行记录：

- [x] 固定官方对象身份与预冻结metadata合同：`Edu_0028.tar`=7,947,059,200 bytes、官方LFS文件SHA-256=`f5c59284...977ccd`、Xet存储哈希=`ffdfc05a...db6cd`；预期500 sources、492×2ch+7×3ch+1×4ch=1009 views、9093 events、1657 missing states，Gate A的source ID碰撞为0。
- [x] 下载前空间门禁通过；16路断点下载中依据8 MiB只读实测从直连切到更快的AutoDL代理并保留全部partial区间。完整传输后7,947,059,200 bytes与官方LFS SHA-256精确一致，独立verifier原子发布；单路TLS关闭由aria2隔离后其余15路继续，未导致整任务失败。
- [x] source contract通过：500 sources、0结构隔离、1009 views、22.991977 target-view hours、9093 events；2条确定性尾部修复沿用既有160 ms规则。
- [x] 完全不变的冻结benchmark Gate B/C通过：1509候选的exact、normalized、content exact、window与near-duplicate gate hit全部为0，0泄漏source；至此付费Qwen调用数为0。
- [x] 离线冻结404个full请求覆盖1657/1657个缺失事件（预计1,231,051 prompt tokens），固定模型`qwen/qwen3-235b-a22b-2507`、prompt v2、direct route、4 workers和`CONFIRM_1657_LABELS`；30-event校准集由26个请求组成。
- [x] 服务端10 USD/day预算门禁与1-event连通通过；26 requests/30 events校准机械门禁通过，26个请求全部在schema attempt 0合规，0重复、0标签塌缩；状态=`backchannel 13 / complete 9 / incomplete 8`，与官方LLM辅助标签46.67%一致仅作诊断。连通+校准accepted cost=`$0.00733271`，full前服务端剩余约`$9.6920`。
- [x] 固定模型/direct route/4 workers完成404 requests/1657 events full labeling并通过机械/provenance审计：393/11个请求分别在schema attempt 0/1合规，0重复、0标签塌缩，状态=`backchannel 1174 / complete 189 / incomplete 294`；full accepted cost=`$0.197410685`，连通+校准+full总accepted cost=`$0.204743395`，服务端当日累计=`$0.517051546 / $10`。
- [x] finalization通过：7407个官方三状态保持不变、29个WAIT→complete、1657个Qwen状态，9093事件1:1闭包；最终complete/incomplete/backchannel=`5048/2011/2034`。
- [x] target audio完成：1009个16 kHz mono/chunk-pad WAV及全量checksum闭合；984/21/4 views来自2/3/4声道source。
- [x] 固定本地Paraformer/FunASR 1.2.6完成：1009输入=1008通过+1 strict quarantine，共131617 tokens，耗时1023.899681秒。`Edu--016994/target-ch02`属于4声道会话，232.16秒音频内有115个语义target-activity chunks且均值约-27.5 dB，但模型返回空文本/时间戳；独立fresh-cache单视图复跑逐字节复现隔离。未伪造文本、替换metadata转写、放宽parser、裁剪音频或挑选输出。
- [x] 确定性文本render完成：1008有效+1传播隔离=1009 views分区闭合，131617个token与时间戳保持不变。
- [x] timeline完成：1009输入views/9093 events/517780 original chunks闭合为1008有效views/9092 events/516329 original chunks + 1隔离view/1 event/1451 chunks；另有1个terminal-silence padding chunk和3137个局部隔离chunks/664条记录。
- [x] 固定本地GLM-4-Voice tokenizer完成：1008/1008 eligible views通过、0新增GLM quarantine、1个上游source-view quarantine；生成1032660个audio tokens，与516330个effective chunks精确对应。
- [x] model-ready导出与验证通过：2349 rows、1008有效source views、513193 exported chunks；内部checksum、20个确定性roundtrip及517781输入chunks全局闭包通过，官方未修改loader读取train split=2231。
- [x] 对最终1008个贡献views执行完全不变的冻结benchmark Gate D：选择集合、确定性tar与model-ready 1:1闭合，冻结scorer对每个mono view保守评分两次共2016条记录；500个source IDs无身份碰撞，exact、normalized、content exact、window与near-duplicate gate hit全部为0，独立closure审计`gate_passed=true`。

Edu_0029 当前执行记录：

- [x] 从固定官方revision/API及本地冻结官方清单交叉确认对象身份：`Edu_0029.tar`=8,074,332,160 bytes、官方LFS文件SHA-256=`fe2238a4...bdab7b`、Xet存储哈希=`5501f117...b3774`、Git OID=`39fc41f7...e45e`。
- [x] 从权威metadata archive按Edu_0018锚定的确定性规则复算500个source IDs：`Edu--017239`–`Edu--017861`，完整集合SHA-256=`27919058...d68de2`；预期498×2ch+2×3ch=1002 views、23.360253 target-view hours、9274 events、1722 missing states。下载前Gate A身份碰撞为0。
- [x] 下载前20 GiB最低空闲门禁通过：可用249,587,937,280 bytes，高于完整archive加安全余量所需的29,549,168,640 bytes；目前Qwen调用数为0，不训练、不评测checkpoint。
- [x] 固定官方revision的可断点下载完成：继承代理线路先升至约5 MiB/s后降至1.4–1.6 MiB/s；默认代理新连接超时，而4路只读8 MiB强制直连探针聚合约4.0 MiB/s。优雅停止并保留约1.9 GiB断点/aria2控制状态后，改为仅aria2阶段关闭代理，16路直连升至约16 MiB/s；最终8,074,332,160 bytes与官方LFS SHA-256精确一致，独立verifier发布，没有第二份cache。
- [x] source contract通过：500 sources、0结构隔离、498×2ch+2×3ch=1002 views、23.360253 target-view hours、9274 events；4条确定性尾部修复沿用既有160 ms规则。
- [x] 完全不变的冻结benchmark Gate B/C通过：1502候选的exact、normalized、content exact、window与near-duplicate gate hit全部为0，0泄漏source；至此付费Qwen调用数为0。
- [x] 离线冻结415个full请求覆盖1722/1722个缺失事件（预计1,260,813 prompt tokens），固定模型`qwen/qwen3-235b-a22b-2507`、prompt v2、direct route、4 workers和`CONFIRM_1722_LABELS`。首次纯离线构造遗漏显式`--calibration-per-state 10`而产生9-event校准集；确认尚未调用API后删除该可重建目录并从同一source scan正确重建，full请求语义/数量不变，不混用两版校准。
- [x] 服务端10 USD/day预算门禁与1-event连通通过，成本`$0.00037378`；23 requests/30 events校准机械门禁通过，21/2个请求在schema attempt 0/1合规，状态=`backchannel 10 / complete 12 / incomplete 8`、0重复、0标签塌缩；与官方LLM辅助标签70%一致仅作诊断，校准成本=`$0.00680213`。
- [x] 使用精确确认令牌完成415 requests/1722 events完整Qwen标注并通过机械/provenance审计：365/49/1个请求分别在schema attempt 0/1/2合规，0重复、0标签塌缩，状态=`backchannel 1235 / complete 191 / incomplete 296`；full成本=`$0.2044284525`，连通+校准+full总accepted cost=`$0.2116043625`，服务端当日累计=`$0.755915693 / $10`。
- [x] finalization通过：7531个官方三状态保持不变、21个WAIT→complete、1722个Qwen状态，9274事件1:1闭包；最终complete/incomplete/backchannel=`5118/1999/2157`。
- [x] target audio完成：1002个16 kHz mono/chunk-pad WAV及全量checksum闭合；996/6 views来自2/3声道source。
- [x] 固定本地Paraformer/FunASR 1.2.6完成：1002/1002 views通过、0 strict quarantine、135581 tokens；未裁剪或替换任何输出，耗时1140.391057秒。
- [x] Paraformer文本render与状态/activity timeline闭包通过：1002 views、9274 events、526049 original chunks + 4 terminal-silence padding chunks；3620个局部冲突chunks/632条记录按既有规则隔离，0 source-view quarantine。
- [x] 固定本地GLM-4-Voice tokenizer完成：1002/1002 eligible views通过、0新增或上游source-view quarantine；生成1052106个audio tokens，与526053个effective chunks精确对应。
- [x] model-ready导出与验证通过：2347 rows、1002 source views、522433 exported chunks；内部checksum、20个确定性roundtrip及526053输入chunks全局闭包通过，官方未修改loader读取train split=2229。
- [x] 对最终1002个贡献views执行完全不变的冻结benchmark Gate D：选择集合、确定性tar与model-ready 1:1闭合，冻结scorer对每个mono view保守评分两次共2004条记录；500个source IDs无身份碰撞，exact、normalized、content exact、window与near-duplicate gate hit全部为0，独立closure审计`gate_passed=true`。

Edu_0030 当前执行记录：

- [x] 从固定官方revision/API确认对象身份：`Edu_0030.tar`=7,972,321,280 bytes、官方LFS文件SHA-256=`2bb6fa17...171f7`、Xet存储哈希=`da45c880...34a38`、Git OID=`8e77e70a...deac`。
- [x] 从权威metadata archive按Edu_0018锚定的确定性规则复算500个source IDs：`Edu--017862`–`Edu--018482`，完整集合SHA-256=`dfd80727...c1935`；预期494×2ch+6×3ch=1006 views、23.065087 target-view hours、8995 events、1729 missing states。下载前Gate A身份碰撞为0。
- [x] 下载前20 GiB最低空闲门禁通过：可用235,955,290,112 bytes，高于完整archive加安全余量所需的29,447,157,760 bytes；目前该分片Qwen调用数为0，不训练、不评测checkpoint。
- [x] 使用在Edu_0029验证过的16路直连线路完成可断点下载；7,972,321,280 bytes与官方LFS SHA-256精确一致，独立verifier原子发布，没有第二份cache。
- [x] 原始source成员/统计合同全部通过，但结构门禁按规则暂停：`Edu--018200`官方最后一个speaker segment结束46.304 s，真实WAV结束46.084 s，220 ms尾差超过既有160 ms可确定性修复上限60 ms；未放宽阈值，整个双声道会话进入source quarantine。
- [x] 沿用Edu_0026已批准的保守规则，构造确定性的499-source sanitized tar（7,963,473,920 bytes，SHA-256=`dfa04ec5...a6c3b`）；成员集合严格等于官方500项减去`Edu--018200.wav`，其余成员大小逐项不变，原始官方tar与失败scan证据均保留。
- [x] sanitized source contract通过：499 sources、493×2ch+6×3ch=1004 views、23.039485 target-view hours、8981 events、1726 missing states、0结构隔离；原始分片其余3条160 ms范围内的确定性尾部修复照常记录。
- [x] 对sanitized tar运行完全不变的冻结benchmark Gate B/C：1503候选的exact、normalized、content exact、window与near-duplicate gate hit均为0，0身份碰撞、0泄漏source；独立复算五类计数一致。门禁前Qwen调用数为0。
- [x] 离线冻结399个full请求覆盖1726/1726个缺失事件（预计1,219,229 prompt tokens），固定模型`qwen/qwen3-235b-a22b-2507`、prompt v2、direct route、4 workers和`CONFIRM_1726_LABELS`；30-event校准集由27个请求组成，尚未调用API。
- [x] 服务端10 USD/day预算门禁与1-event连通通过，成本`$0.00081675`；27 requests/30 events校准机械门禁通过，25/2个请求在schema attempt 0/1合规，状态=`backchannel 11 / complete 9 / incomplete 10`、0重复、0标签塌缩；与官方LLM辅助标签60%一致仅作诊断，校准成本=`$0.0052655`，full前服务端剩余约`$9.2379`。
- [x] full首轮完成并持久化398/399个签名缓存后，唯一`Edu--018035`请求收到截断的provider JSON，触发现有客户端未捕获的传输层`JSONDecodeError`。失败manifest/log原样归档；恢复运行复用398个缓存且只请求缺失项，模型、路由、prompt、请求文件和确认令牌均不变。首次误恢复因正式manifest路径仍存在而在API前安全退出，该临时日志不冒充模型请求证据。
- [x] 恢复后399 requests/1726 events完整Qwen标注及机械/provenance审计通过：374/24/1个请求分别在schema attempt 0/1/2合规，0重复、0标签塌缩，状态=`backchannel 1204 / complete 206 / incomplete 316`；accepted-response full成本=`$0.198805505`，连通+校准+full accepted总成本=`$0.204887755`。服务端当日累计=`$0.98043282 / $10`，full阶段服务端增量比accepted ledger多`$0.019544749`，可能包含截断响应或被拒重试，两个口径均保留。
- [x] finalization通过：7235个官方三状态保持不变、20个WAIT→complete、1726个Qwen状态，8981事件1:1闭包；最终complete/incomplete/backchannel=`4971/1967/2043`。
- [x] target audio完成：1004个16 kHz mono/chunk-pad WAV及全量checksum闭合；986/18 views来自2/3声道source。
- [x] 固定本地Paraformer/FunASR 1.2.6完成：1004/1004 views通过、0 strict quarantine、132048 tokens；未裁剪、替换或挑选任何输出，耗时1191.299342秒。
- [x] Paraformer文本render与状态/activity timeline闭包通过：1004 views、8981 events、518842 original chunks + 2 terminal-silence padding chunks；3448个局部冲突chunks/621条记录按既有规则隔离，0 source-view quarantine。
- [x] 固定本地GLM-4-Voice tokenizer完成：1004/1004 eligible views通过、0新增或上游source-view quarantine；生成1,037,688个audio tokens，与518,844个effective chunks精确对应。
- [x] model-ready导出与验证通过：2348 rows、1004 source views、515396 exported chunks；内部checksum、20个确定性roundtrip及518844输入chunks全局闭包通过，官方未修改loader读取train split=2230。
- [x] 对最终1004个贡献views执行完全不变的冻结benchmark Gate D：选择集合、确定性tar与model-ready 1:1闭合，冻结scorer对每个mono view保守评分两次共2008条记录；499个source IDs无身份碰撞，exact、normalized、content exact、window与near-duplicate gate hit全部为0；`Edu--018200`在最终选择、tar与评分记录中均不存在，独立closure审计`gate_passed=true`。

最终数据集总结和会议汇报文档必须单列记录`Edu--018200`结构异常、220 ms实际尾差、160 ms既有修复上限、未放宽阈值的实验诚信处理、整会话排除范围，以及Qwen full首轮截断响应/原子缓存恢复和服务端/accepted成本口径差异，不能只报告sanitized后的通过结果。

Edu_0031 当前执行记录：

- [x] 从固定官方revision/API确认对象身份：`Edu_0031.tar`=7,819,653,120 bytes、官方LFS文件SHA-256=`50312ad9...6a4fa1`、Xet存储哈希=`d8152c60...f98449`、Git OID=`0fb15b06...64557`。
- [x] 从权威metadata archive按Edu_0018锚定的确定性规则复算500个source IDs：`Edu--018483`–`Edu--019126`，完整集合SHA-256=`977654bc...bf33`；预期494×2ch+6×3ch=1006 views、22.623341 target-view hours、9040 events、1688 missing states。下载前Gate A身份碰撞为0。
- [x] 下载前20 GiB最低空闲门禁通过：可用214,521,458,688 bytes，高于完整archive加安全余量所需的29,294,489,600 bytes；目前该分片Qwen调用数为0，不训练、不评测checkpoint。
- [x] 16路直连hf-mirror可断点下载完成；7,819,653,120 bytes与官方LFS SHA-256精确一致，独立verifier原子发布，没有第二份cache。
- [x] source contract通过：500 sources、0结构隔离、494×2ch+6×3ch=1006 views、22.623341 target-view hours、9040 events；9条确定性尾部修复均沿用既有160 ms规则。
- [x] 完全不变的冻结benchmark Gate B/C通过：1506候选的exact、normalized、content exact、window与near-duplicate gate hit全部为0，0泄漏source；至此付费Qwen调用数为0。
- [x] 离线冻结398个full请求覆盖1688/1688个缺失事件（预计1,223,790 prompt tokens），固定模型`qwen/qwen3-235b-a22b-2507`、prompt v2、direct route、4 workers和`CONFIRM_1688_LABELS`；30-event校准集由24个分组请求组成，尚未调用API。
- [x] 服务端10 USD/day预算门禁与1-event连通通过，连通成本=`$0.00139755`；24 requests/30 events校准机械门禁通过，24个请求全部在schema attempt 0合规，状态=`backchannel 9 / complete 13 / incomplete 8`、0重复、0标签塌缩；与官方LLM辅助标签73.33%一致仅作诊断，校准成本=`$0.005010355`，full前服务端剩余约`$9.0178`。
- [x] 固定模型/direct route/4 workers完成398 requests/1688 events full labeling并通过机械/provenance审计：381/17个请求分别在schema attempt 0/1合规，0重复、0标签塌缩，状态=`backchannel 1145 / complete 228 / incomplete 315`；full accepted cost=`$0.20188136`，连通+校准+full总accepted cost=`$0.208289265`。服务端累计=`$1.199821451 / $10`，full阶段服务端增量比accepted ledger多`$0.015753401`，可能包含被拒schema重试，两个口径均保留。
- [x] finalization通过：7329个官方三状态保持不变、23个WAIT→complete、1688个Qwen状态，9040事件1:1闭包；最终complete/incomplete/backchannel=`5072/1933/2035`。finalization后的只读hash命令曾误写不存在的`events.jsonl`文件名，纠正为实际`events_with_final_state.jsonl`后hash通过，数据未重跑或修改。
- [x] target audio完成：1006个16 kHz mono/chunk-pad WAV及全量checksum闭合；988/18 views来自2/3声道source。
- [x] 固定本地Paraformer/FunASR 1.2.6完成：1006输入=1005通过+1 strict quarantine，共130676 tokens，耗时1059.155112秒。`Edu--018729/target-ch00`有227/437个语义target-activity chunks、活动区RMS约−27.85 dBFS，但模型返回空文本/时间戳；独立fresh-cache单视图复跑逐字节复现隔离。未伪造文本、替换metadata转写、放宽parser、裁剪音频或挑选输出。
- [x] 确定性文本render与timeline完成：1006输入=1005有效+1传播隔离views，9040=9033+7 events，509456=509019+437 original chunks全局闭合；另有6个terminal pad chunks、3166个局部隔离chunks/583条记录。
- [x] 固定本地GLM-4-Voice tokenizer完成：1005/1005 eligible views通过、0新增GLM quarantine、1个上游source-view quarantine；生成1018050个audio tokens，与509025个effective chunks精确对应。
- [x] model-ready导出与验证通过：2291 rows、1005有效source views、505859 exported chunks；内部checksum、20个确定性roundtrip及509462输入chunks全局闭包通过，官方未修改loader读取train split=2176。
- [x] 对最终1005个贡献views执行完全不变的冻结benchmark Gate D：选择集合、确定性tar与model-ready 1:1闭合，冻结scorer对每个mono view保守评分两次共2010条记录；500个source IDs无身份碰撞，exact、normalized、content exact、window与near-duplicate gate hit全部为0，独立closure审计`gate_passed=true`。

最终数据集总结和会议汇报文档必须单列记录`Edu--018729/target-ch00`的Paraformer空输出、227个语义target-activity chunks、活动区约−27.85 dBFS、fresh-cache逐字节复现、整视图隔离范围（7 events/437 chunks/69.92 s）及实际规模损失，不能把它描述为静音样本或用metadata文本补齐。

Edu_0032 当前执行记录：

- [x] 从固定官方revision/API确认对象身份：`Edu_0032.tar`=7,693,424,640 bytes、官方LFS文件SHA-256=`86838aa8...742a30`、Xet存储哈希=`16916837...1a060f`、Git OID=`c7e62754...643e7`。
- [x] 从权威metadata archive按Edu_0018锚定的确定性规则复算500个source IDs：`Edu--019127`–`Edu--019749`，完整集合SHA-256=`3932fee7...7b10b`；预期493×2ch+7×3ch=1007 views、22.258097 target-view hours、8564 events、1672 missing states。下载前Gate A身份碰撞为0。
- [x] 下载前20 GiB最低空闲门禁通过：可用201,320,181,760 bytes，高于完整archive加安全余量所需的29,168,261,120 bytes；目前该分片Qwen调用数为0，不训练、不评测checkpoint。
- [x] 16路直连hf-mirror可断点下载完成；7,693,424,640 bytes与官方LFS SHA-256精确一致，没有第二份cache。
- [x] source contract通过：500 sources、0结构隔离、493×2ch+7×3ch=1007 views、22.258097 target-view hours、8564 events；7条确定性尾部修复均沿用既有160 ms规则。
- [x] 第一次冻结benchmark Gate B/C scorer在容器/进程被外部终止时以return code -9结束，只生成原子`.partial`文件，不构成数据门禁失败，且当时Qwen调用数为0。中断目录已原样归档为`leakage_gate_v2_2_interrupted_20260824`；2026-08-25使用字节级一致的三份冻结实现、benchmark清单、配置、archive和阈值完成恢复运行。
- [x] 恢复后的原始500-source冻结Gate B/C正式完成并诚实失败：`Edu--019578.wav` channel 1对`candor_turn_taking/62/input.wav`产生1个near-duplicate gate hit（similarity=0.753533、aligned frames=115、LSH votes=70，分别超过冻结0.655/64/16门槛）。候选metadata为中文、benchmark为英文，16 kHz重采样后的最大绝对滑窗波形相关约0.051607，故指纹碰撞是合理可能性；但该只读诊断不用于推翻冻结门禁，也不改阈值。
- [x] 按保守整会话隔离生成499-source sanitized tar（7,690,178,560 bytes，SHA-256=`457f2795...ad66f3`）；成员集合严格等于官方500项减去`Edu--019578.wav`，其余成员大小逐项不变。官方tar、失败门禁、诊断与中断证据全部保留。sanitized source contract通过：499 sources、492×2ch+7×3ch=1005 views、22.248702 target-view hours、8561 events、1671 missing states；完全不变的冻结Gate B/C对1504候选复测五类命中均为0，独立计数一致。
- [x] 离线冻结412个full请求覆盖1671/1671个缺失事件（预计1,181,247 prompt tokens），固定模型`qwen/qwen3-235b-a22b-2507`、prompt v2、direct route、4 workers和`CONFIRM_1671_LABELS`；22-request/30-event校准机械门禁通过，状态=`backchannel 9 / complete 14 / incomplete 7`、0重复、0标签塌缩，连通与校准accepted cost分别为`$0.000631645`和`$0.0046256725`。与官方LLM辅助标签56.67%一致仅作诊断。
- [x] 412 requests/1671 events完整Qwen标注及机械/provenance审计通过：390/22个请求分别在schema attempt 0/1合规，0重复、0标签塌缩，状态=`backchannel 1181 / complete 206 / incomplete 284`。首轮在411/412个签名缓存后，唯一`Edu--019462`收到OpenRouter HTTP 429；失败manifest/log原样归档，恢复运行复用411个缓存且只实际请求缺失1条，模型、路由、prompt、请求文件和确认令牌均不变。full accepted cost=`$0.1991017025`，连通+校准+full accepted总成本=`$0.20435902`；服务端累计=`$0.218366018 / $10`，比accepted ledger多`$0.014006998`，两个口径均保留。
- [x] finalization通过：6871个官方三状态保持不变、19个WAIT→complete、1671个Qwen状态，8561事件1:1闭包；最终complete/incomplete/backchannel=`4636/1853/2072`。
- [x] target audio完成：1005个16 kHz mono/chunk-pad WAV及全量checksum闭合；984/21 views来自2/3声道source。
- [x] 固定本地Paraformer/FunASR 1.2.6完成：1005输入=1003通过+2 strict quarantine，共126275 tokens，耗时1060.492574秒。`Edu--019535/target-ch01`与`Edu--019576/target-ch00`的末端token时间戳均比音频边界多20 ms；独立fresh-cache复跑复现相同view、cache signature、越界端点与失败，未裁剪时间戳、放宽parser或挑选成功输出。第一条复现记录逐字节一致；第二条仅报告token序号变化，异常端点与签名不变。
- [x] 确定性文本render与timeline完成：1005输入=1003有效+2传播隔离views，8561=8542+19 events，501038=499896+1142 original chunks全局闭合；另有7个terminal pad chunks、2923个局部隔离chunks/520条记录。
- [x] 固定本地GLM-4-Voice tokenizer完成：1003/1003 eligible views通过、0新增GLM quarantine、2个上游source-view quarantine逐字节传播；生成999806个audio tokens，与499903个effective chunks精确对应。
- [x] model-ready导出与验证通过：2229 rows、1003有效source views、496980 exported chunks；内部checksum、20个确定性roundtrip及501045输入chunks全局闭包通过，官方未修改loader读取train split=2117。
- [x] 对最终1003个贡献views执行完全不变的冻结benchmark Gate D：选择集合、确定性tar与model-ready 1:1闭合，冻结scorer对每个mono view保守评分两次共2006条记录；499个source IDs无benchmark身份碰撞，exact、normalized、content exact、window与near-duplicate gate hit全部为0；`Edu--019578`在最终选择、tar与评分记录中不存在，独立closure审计`gate_passed=true`。closure首次只读调用误写不存在的`manifest.jsonl`，在读取前安全失败且未产生输出；纠正为真实`audio_manifest.jsonl`后通过，scorer未重跑或修改。

最终数据集总结和会议汇报文档必须单列记录：原始Gate B/C对`Edu--019578.wav`与CANDOR样本的冻结近重复命中及整会话保守排除；只读波形相关诊断不能作为推翻门禁依据；Qwen full首轮HTTP 429后的411-cache原子恢复与成本双口径；两条Paraformer末端时间戳超出音频边界20 ms、fresh-cache复现和整视图隔离范围（19 events/1142 chunks/182.72 s）。

Edu_0033 当前执行记录：

- [x] 从固定官方revision/API确认对象身份：`Edu_0033.tar`=7,609,466,880 bytes、官方LFS文件SHA-256=`1a102214...0df5d4`、Xet存储哈希=`1604e760...8f5250`、Git OID=`104027fd...2df0`。无代理API探针超时、服务器现有代理成功；仅作为API路由证据，大文件线路独立选择。
- [x] 从权威metadata archive按Edu_0018锚定的确定性规则复算500个source IDs：`Edu--019750`–`Edu--020361`，完整集合SHA-256=`07705a4e...8761`；预期493×2ch+7×3ch=1007 views、22.015169 target-view hours、8786 events、1713 missing states。下载前Gate A对3119条benchmark记录/7721个标识串的身份碰撞为0。
- [x] 下载前20 GiB最低空闲门禁通过：可用180,600,848,384 bytes，高于完整archive加安全余量所需的29,084,303,360 bytes；目前该分片Qwen调用数为0，不训练、不评测checkpoint。
- [x] 使用近期已验证的16路直连hf-mirror线路完成可断点下载；7,609,466,880 bytes与官方LFS SHA-256精确一致，没有第二份cache。
- [x] source contract通过：500 sources、0结构隔离、493×2ch+7×3ch=1007 views、22.015169 target-view hours、8786 events；5条确定性尾部修复均沿用既有160 ms规则。
- [x] 完全不变的冻结benchmark Gate B/C通过：1507候选的exact、normalized、content exact、window与near-duplicate gate hit全部为0，0泄漏source；5195个LSH near candidates全部`is_gate_hit=false`。至此付费Qwen调用数为0。
- [x] 离线冻结411个full请求覆盖1713/1713个缺失事件（预计1,206,143 prompt tokens），411个source IDs/请求签名和1713个事件ID均唯一，集合差为0；固定模型`qwen/qwen3-235b-a22b-2507`、prompt v2、direct route、4 workers和`CONFIRM_1713_LABELS`。26-request/30-event校准集按三状态各10条平衡，尚未调用API。
- [x] 服务端10 USD/day预算与1-event连通通过：请求前当日用量`$0.218679468`、剩余`$9.781320532`，固定模型/direct route/schema attempt 0正常，本次accepted cost=`$0.0006512`。
- [x] 26 requests/30 events校准机械/provenance门禁通过：24/2个请求在schema attempt 0/1合规，状态=`backchannel 10 / complete 10 / incomplete 10`、0重复、0标签塌缩，accepted cost=`$0.0061126675`；与官方LLM辅助标签66.67%一致仅作诊断。校准后服务端当日用量=`$0.225764128 / $10`。
- [x] 411 requests/1713 events完整Qwen标注及机械/provenance审计通过：404/7个请求分别在schema attempt 0/1合规，0重复、0标签塌缩，状态=`backchannel 1192 / complete 207 / incomplete 314`。首轮在410/411个签名缓存后，唯一`Edu--020275`的OpenRouter响应解析为`JSONDecodeError`（返回JSON截断或不完整）；失败manifest/log原样归档，恢复运行复用410个缓存且只实际请求缺失1条，模型、路由、prompt、请求文件和确认令牌均不变。full accepted cost=`$0.1987352775`，连通+校准+full accepted总成本=`$0.205499145`；服务端当日累计=`$0.430207704 / $10`，与本分片accepted ledger口径差=`$0.006029091`，两个口径都保留。
- [x] 状态finalization通过：7056个官方三状态保持不变、17个WAIT→complete、1713个Qwen状态，8786事件1:1闭包；最终complete/incomplete/backchannel=`4889/1875/2022`。
- [x] target audio完成：1007个16 kHz mono/chunk-pad WAV与全量checksum闭合；986/21 views来自2/3声道source。
- [x] 固定本地Paraformer/FunASR 1.2.6完成：1007输入=1007严格通过+0隔离，共124432 tokens，耗时1097.13046秒；没有裁剪时间戳、放宽parser或挑选输出。
- [x] Paraformer文本渲染通过：1007 rendered + 0 quarantine = 1007输入，124432 tokens保持不变，分区闭包。
- [x] timeline完成：1007 views、8786 events、495810原始chunks全部进入处理；补3个终端静音chunk后为495813 effective chunks，0整视图隔离，3227个局部chunk严格隔离，全局分区闭包。
- [x] 固定GLM-4-Voice tokenizer完成：1007/1007合格views通过，0新隔离、0上游整视图隔离；495813 effective chunks精确对应991626个80 ms audio tokens（每160 ms chunk两个），分区闭包。
- [x] model-ready导出完成：2269 rows、1007个有效source views、492586 exported chunks；3227个局部隔离chunks未进入Parquet，0整视图隔离，495813输入chunks全局闭包。
- [x] model-ready验证通过：0 checksum失败、492586导出chunks全部可解析、20个确定性roundtrip与全局视图/chunk闭包通过；官方未修改loader成功读取train split=2155。
- [x] 对最终1007个贡献views执行完全不变的冻结benchmark Gate D：selection、确定性tar与model-ready 1:1闭匈，冻结scorer对每个mono view保守评分两次共2014条记录；500个source IDs无benchmark身份碰撞，exact、normalized、content exact、window与near-duplicate gate hit全部为0，独立closure审计`gate_passed=true`。

Edu_0033 分片已完整通过source contract、冻结Gate A/B/C、Qwen标注机械审计、Paraformer/GLM严格处理、官方loader验证与最终Gate D。最终汇报必须保留Qwen首轮单条截断JSON失败、410-cache原子恢复、成本双口径、3227个局部chunk隔离与0整视图隔离的事实。

Edu_0034 当前执行记录：

- [x] 固定官方revision对象身份：`Edu_0034.tar`=8,369,879,040 bytes，官方LFS文件SHA-256=`53357750...f00605e`，Xet存储哈希=`d0836f5e...1b430f`，Git OID=`fb1088ea...819f4`。
- [x] 从权威metadata archive确定性复算第34分片500个source IDs：`Edu--020362`–`Edu--020976`，完整集合SHA-256=`89c04b42...29c6d`；预期495×2ch+4×3ch+1×4ch=1006 views、24.215406 target-view hours、9486 events、1838 missing states。四声道会话保留4个独立target views，不丢弃或混音。
- [x] 下载前Gate A对3119条benchmark记录/7721个标识串的身份碰撞为0；20 GiB最低空闲门禁通过，可用167,752,212,480 bytes，高于完整archive+安全余量所需的29,844,715,520 bytes。
- [x] 使用可断点直连hf-mirror 16路完成下载；8,369,879,040 bytes与官方LFS SHA-256精确一致，`.part/.aria2`已消失、没有第二份cache，可用空闲约159.38 GB。
- [x] source contract通过：500 sources、0结构隔离、495×2ch+4×3ch+1×4ch=1006 views、24.215406 target-view hours、9486 events；2条确定性尾部修复均沿用既有≤160 ms规则。
- [x] 完全不变的冻结benchmark Gate B/C通过：1506候选的exact、normalized、content exact、window与near-duplicate gate hit全部为0，0泄漏source；5310个LSH near candidates全部`is_gate_hit=false`。至此付费Qwen调用数为0。
- [x] 离线冻结406个full请求覆盖1838/1838个缺失事件（预计1,278,486 prompt tokens），406个source IDs/请求签名和1838个事件ID均唯一，集合差为0；固定模型`qwen/qwen3-235b-a22b-2507`、prompt v2、direct route、4 workers和`CONFIRM_1838_LABELS`。25-request/30-event校准集按三状态各10条平衡，尚未调用API。
- [x] 服务端10 USD/day预算与1-event连通通过：请求前当日用量`$0.430599454`、剩余`$9.569400546`，固定模型/direct route/schema attempt 0正常，本次accepted cost=`$0.00020925`。
- [x] 25 requests/30 events校准机械/provenance门禁通过：23/2个请求在schema attempt 0/1合规，状态=`backchannel 9 / complete 13 / incomplete 8`、0重复、0标签塌缩，accepted cost=`$0.006414935`；与官方LLM辅助标签73.33%一致仅作诊断。校准后服务端当日用量=`$0.438282956 / $10`。
- [x] 406 requests/1838 events完整Qwen标注及机械/provenance审计通过：393/12/1个请求分别在schema attempt 0/1/2合规，0重复、0标签塌缩，状态=`backchannel 1265 / complete 189 / incomplete 384`。首轮在405/406个完整请求签名缓存后，唯一`Edu--020902`（14 events）的OpenRouter响应解析为`JSONDecodeError`；失败manifest/log原样归档，恢复运行复用405个缓存并只实际请求缺失1条，所有调用条件不变。full结果账本cost=`$0.21022784`；连通响应签名恰与一个full请求重合并已包含在full账本中，因此唯一accepted-response总成本为校准+full=`$0.216642775`；若再加连通项会重复计算`$0.00020925`，简单分阶段加总=`$0.216852025`仅作对账说明。服务端当日累计=`$0.662785353 / $10`。
- [x] 状态闭环通过：9486/9486 events均有最终状态；7621个官方标签保持不变、27个WAIT确定性映射为complete、1838个缺失标签来自已审计Qwen结果；最终分布=`backchannel 2198 / complete 5204 / incomplete 2084`。
- [x] 提取全部1006个确定性目标声道单声道WAV且1006/1006校验和通过；视图来源=`2ch 990 / 3ch 12 / 4ch 4`，多声道会话仍按每个目标声道独立建view，未混音、未丢弃。
- [x] 固定本地Paraformer/FunASR 1.2.6完成：1006输入=1006严格通过+0隔离，共140224 tokens，耗时1187.121725秒；没有裁剪时间戳、放宽parser或挑选输出。
- [x] Paraformer文本渲染通过：1006 rendered + 0 quarantine = 1006输入，140224 tokens保持不变，分区闭包。
- [x] timeline完成：1006 views、9486 events、545322原始chunks全部进入处理；补5个终端静音chunk后为545327 effective chunks，0整视图隔离，3765个局部chunk严格隔离，全局分区闭包。
- [x] 固定GLM-4-Voice tokenizer完成：1006/1006合格views通过，0新隔离、0上游整视图隔离；545327 effective chunks精确对应1090654个80 ms audio tokens（每160 ms chunk两个），分区闭包。
- [x] 官方两列Stage 3 Parquet导出并验证通过：2415 rows覆盖1006 views，541562导出chunks+3765局部隔离chunks=545327 effective chunks；checksum失败0，20条随机roundtrip通过，未修改SoulX官方loader可读取train split 2294行。
- [x] 最终冻结benchmark Gate D通过：model-ready selection/tar均为1006 views/500 sources，2012 scorer/identity records对每个单声道view保守计分两次；selection缺失/额外均0，source标识碰撞及exact/normalized/content exact/window/near-duplicate命中均为0，闭包通过。Edu_0034完成。

Edu_0035 当前执行记录：

- [x] 固定官方revision对象身份：`Edu_0035.tar`=7,440,353,280 bytes，官方LFS文件SHA-256=`4eea63b0...117981`，Xet存储哈希=`bb4aef2b...09bc6`，Git OID=`c409e6f4...6ce6e`。
- [x] 从权威metadata archive确定性复算第35分片500个source IDs：`Edu--020978`–`Edu--021596`，完整集合SHA-256=`34b2b2cf...18fbd`；预期494×2ch+6×3ch=1006 views、21.525815 target-view hours、8697 events、1736 missing states。
- [x] 下载前Gate A对3119条benchmark记录/7721个标识串的身份碰撞为0；20 GiB最低空闲门禁通过，可用153,621,180,416 bytes，高于完整archive+安全余量所需的28,915,189,760 bytes。
- [x] 使用可断点直连hf-mirror 16路完成下载；独立verifier确认7,440,353,280 bytes与官方LFS SHA-256=`4eea63b0...117981`完全一致后才原子发布，未生成第二份下载cache。
- [x] 原始500-source source contract通过：494×2ch+6×3ch=1006 views、21.525815 target-view hours、8697 events、1736 missing states、0结构隔离。
- [x] 原始500-source冻结Gate B/C诚实失败：`Edu--021051.wav` channel 1对`candor_turn_taking/62/input.wav`产生1个near-duplicate gate hit（similarity=0.677300、aligned frames=144、LSH votes=26，分别超过冻结0.655/64/16门槛）；exact、normalized、content exact和window命中均为0。未修改阈值、scorer或benchmark，门禁在Qwen调用数为0时阻断后续。
- [x] 经负责人确认，保守隔离整个`Edu--021051.wav`会话，生成确定性的499-source sanitized tar（7,436,410,880 bytes，SHA-256=`3a187226...a659a`）；成员集合严格等于官方500项减去该source，剩余成员元数据逐项一致，官方tar与失败证据均保留。
- [x] sanitized source contract通过：499 sources、493×2ch+6×3ch=1004 views、21.514428 target-view hours、8692 events、1734 missing states、0结构隔离。完全不变的冻结Gate B/C对1503候选五类命中均为0、0隔离source、`gate_passed=true`；独立复算4985个near candidates中gate hit为0，被排除source未回流。
- [x] 为sanitized数据离线冻结408个full请求覆盖1734/1734个缺失事件（预计1,194,362 prompt tokens），固定模型`qwen/qwen3-235b-a22b-2507`（Qwen3-235B-A22B-Instruct-2507）、prompt v2、direct route、4 workers和`CONFIRM_1734_LABELS`；请求签名、source与event集合独立闭包。1-event连通和26-request/30-event校准通过，校准状态=`backchannel 10 / complete 10 / incomplete 10`、0标签塌缩，与官方LLM辅助标签80%一致仅作诊断。
- [x] 408 requests/1734 events完整Qwen标注及机械/provenance审计通过：373/34/1个请求分别在schema attempt 0/1/2合规，0重复、0标签塌缩，状态=`backchannel 1225 / complete 194 / incomplete 315`。full accepted cost=`$0.2023435425`，连通+校准+full唯一accepted-response总成本=`$0.207138605`；服务端当日累计=`$0.889928677 / $10`。
- [x] 状态闭环通过：8692/8692 events均有最终状态；6937个官方标签保持不变、21个WAIT确定性映射为complete、1734个缺失标签来自已审计Qwen结果；最终complete/incomplete/backchannel=`4776/1916/2000`。
- [x] 提取1004个16 kHz mono/chunk-pad目标声道WAV并通过全量checksum：986/18 views来自2/3声道source；固定本地Paraformer/FunASR 1.2.6对1004/1004严格通过、0隔离，共124598 tokens，耗时908.185833秒。
- [x] 文本render与timeline闭包通过：1004 views、8692 events、484509 original/effective chunks，0末端补块、0整视图隔离；3353个局部chunks/532条记录按既有规则隔离。
- [x] 固定GLM-4-Voice tokenizer对1004/1004视图通过、0新增隔离；484509 effective chunks精确对应969018个80 ms audio tokens。官方两列Stage 3 Parquet导出及loader验证通过：2224 rows、481156 exported chunks、checksum失败0、20条随机roundtrip通过、官方train split 2112行。
- [x] 最终冻结Gate D通过：model-ready selection/tar均为1004 views/499 sources，2008 scorer/identity records对每个mono view保守计分两次；selection缺失/额外均0，被排除`Edu--021051`未回流，源标识碰撞及exact/normalized/content exact/window/near-duplicate命中均为0。Edu_0035全链路完成。

Edu_0036 当前执行记录：

- [x] 从固定官方revision/API确认对象身份：`Edu_0036.tar`=7,846,338,560 bytes、官方LFS文件SHA-256=`c69e0c9c...9c5643`、Xet存储哈希=`03f5cf11...d624a`、Git OID=`6766447c...d7312`。API直连探针超时后由服务器现有代理成功获取权威元数据；大文件下载线路单独采用近期已验证的hf-mirror直连方案。
- [x] 从权威metadata archive确定性复算第36分片500个source IDs：`Edu--021597`–`Edu--022236`，完整集合SHA-256=`9f9a3e08...b7fbc`；预期487×2ch+12×3ch+1×4ch=1014 views、22.700556 target-view hours、8965 events、1812 missing states。多声道会话仍按每个目标声道建立独立view，不混音、不直接丢弃。
- [x] 下载前Gate A对3119条benchmark记录/7721个标识串的身份碰撞为0；20 GiB最低空闲门禁通过，可用133,588,697,088 bytes，高于完整archive+安全余量所需的29,321,175,040 bytes。本阶段Qwen调用数为0，不训练、不评测checkpoint。
- [x] 16路可续传hf-mirror直连下载完成；独立verifier确认7,846,338,560 bytes与官方LFS SHA-256=`c69e0c9c...9c5643`完全一致后才原子发布，没有第二份下载cache。
- [x] source contract通过：500 sources、0结构隔离、487×2ch+12×3ch+1×4ch=1014 views、22.700556 target-view hours、8965 events；5条确定性尾部修复均沿用既有规则。
- [x] 完全不变的冻结benchmark Gate B/C通过：1514个候选声道中，exact、normalized、content exact、window与near-duplicate gate hit全部为0，0泄漏source；独立复算5165个near candidates中gate hit为0。至此该分片Qwen调用数为0。
- [x] 离线冻结408个full请求覆盖1812/1812个缺状态事件（预计1,212,455 prompt tokens），固定模型`qwen/qwen3-235b-a22b-2507`、prompt v2、direct route、4 workers和`CONFIRM_1812_LABELS`。独立复算408个请求签名/source均唯一，事件漏项/额外/重复均0；22-request/30-event校准集按三状态各10条平衡。尚未调用本分片API。
- [x] 服务端10 USD/day预算门禁与1-event连通通过：调用前当日用量`$0.889928677`、剩余`$9.110071323`，固定模型/direct route/schema正常，accepted cost=`$0.00021586`。
- [x] 22 requests/30 events校准机械/provenance门禁通过：14/8个请求在schema attempt 0/1合规，状态=`backchannel 9 / complete 8 / incomplete 13`、0重复、0标签塌缩，accepted cost=`$0.008049275`；与官方LLM辅助标签50%一致仅作诊断，未据此改变prompt、标签或数据。校准后服务端当日剩余`$9.100569263`。
- [x] full首轮完成407/408个请求签名的原子缓存后，唯一`Edu--021946`（5 events）的provider响应在传输层解析为截断/不完整JSON，触发`JSONDecodeError`；wrapper按失败关闭，未生成正式full结果或审计。失败manifest/log已原样归档；恢复复用407个有效缓存并只调用唯一缺失签名，所有冻结条件不变。
- [x] 恢复后408 requests/1812 events完整Qwen标注及机械/provenance审计通过：386/22个请求分别在schema attempt 0/1合规，0重复、0标签塌缩，状态=`backchannel 1318 / complete 165 / incomplete 329`；full accepted cost=`$0.19892091`，连通+校准+full唯一accepted总成本=`$0.207186045`。服务端当日累计=`$1.111748925 / $10`；相对本分片调用前增量比accepted ledger多`$0.014634203`，可能包含截断响应或被拒schema重试，两个口径均保留。
- [x] 状态finalization通过：7139个官方三状态保持不变、14个WAIT→complete、1812个缺失状态来自已审计Qwen结果，8965事件1:1闭包；最终状态分布=`backchannel 2225 / complete 4802 / incomplete 1938`，全量checksum通过。
- [x] 提取1014个独立target-vs-rest 16 kHz mono视图并通过全量checksum：双/三/四声道来源分别贡献974/36/4 views；多声道会话未混音、未丢弃。
- [x] 固定本地Paraformer/FunASR 1.2.6完成：1014输入=1012通过+2 strict quarantine，共129820 tokens、耗时944.334869秒。`Edu--021678/target-ch00`最后token结束159.540 s、比159.520 s音频边界超20 ms；fresh-cache以相同signature/端点复现（仅token序号69→68）。`Edu--021847/target-ch00`有19个语义活动chunks、活动区约−34.09 dBFS但返回空文本/时间戳，fresh-cache以相同signature复现。未裁剪时间戳、放宽parser、用metadata补文或挑选输出；两条整视图隔离共20 events/1455 chunks/232.8 s。
- [x] 确定性渲染1012个有效Paraformer文本并逐字节传播2条隔离，1012+2=1014输入分区闭包；129820个token不变，隔离SHA与上游一致。
- [x] timeline全局闭包通过：1014=1012有效+2隔离views，8965=8945+20 events，511219=509764+1455 original chunks；另补6个terminal silence chunks形成509770 effective chunks，3013个局部chunks/540条记录严格隔离。内部checksum已通过；之后只读hash命令误写不存在的`quarantine.jsonl`并安全失败，纠正为`timeline_quarantine.jsonl`后通过，产物未修改。
- [x] 固定本地GLM-4-Voice tokenizer以batch size 16完成：1012/1012 eligible views通过、0新增隔离、2条上游整视图隔离逐字节传播；509770 effective chunks精确对应1019540个80 ms audio tokens。内部checksum已通过；之后只读统计/哈希命令误写`quarantine.jsonl`，纠正为`glm_quarantine.jsonl`后通过，产物未修改。
- [x] 官方两列Stage 3 Parquet导出与验证通过：2281 rows、1012个有效source views、506757 exported chunks；3013个局部隔离+1455个整视图隔离与511225输入chunks全局闭包。checksum失败0、20条确定性roundtrip通过，未修改官方loader读取train split=2166行。验证后额外只读hash命令遗漏`data/`和`metadata/`子目录而返回1，纠正路径后通过，未重导出或修改结果。
- [x] 最终冻结benchmark Gate D通过：selection/tar均为1012个model-ready views、覆盖500 sources，冻结scorer生成2024条candidate/identity records且每个mono view恰好评分两次；selection缺失/额外均0，source身份碰撞及exact/normalized/content exact/window/near-duplicate命中均为0，独立closure`gate_passed=true`。Edu_0036全链路完成。

Edu_0036 最终总结必须单列保留：Qwen full首轮唯一`Edu--021946`截断JSON失败、407-cache精确恢复及服务端/accepted成本双口径；`Edu--021678/target-ch00`末端时间戳超界20 ms和`Edu--021847/target-ch00`有语义活动却空输出的fresh-cache复现；两条整视图隔离造成20 events/1455 chunks/232.8 s规模损失；timeline/GLM/model-ready完成后只读postcheck三次文件名/子目录误写均发生在内部验证通过之后、未修改产物。

Edu_0037 当前执行记录：

- [x] 固定官方revision/API确认对象身份：`Edu_0037.tar`=7,735,255,040 bytes、官方LFS文件SHA-256=`f9c22a8e...01d0e9`、Xet存储哈希=`ebb7dfab...17c70`、Git OID=`553dbaeb...b5ee`；API元数据使用服务器现有代理，大文件下载使用既有hf-mirror直连方案。
- [x] 从权威metadata archive确定性复算第37分片500个source IDs：`Edu--022237`–`Edu--022839`，完整集合SHA-256=`eb24ae32...998ff1`；预期493×2ch+7×3ch=1007 views、22.379126 target-view hours、9188 events、1668 missing states。多声道会话仍逐目标声道建立独立view。
- [x] 下载前Gate A对3119条benchmark记录/7721个标识串的身份碰撞为0；20 GiB最低空闲门禁通过，可用120,348,286,976 bytes，高于完整archive+安全余量所需的29,210,091,520 bytes。本阶段Qwen调用数为0，不训练、不评测checkpoint。
- [x] 16路可续传hf-mirror直连下载完成；独立verifier确认7,735,255,040 bytes与官方LFS SHA-256=`f9c22a8e...01d0e9`完全一致后才原子发布，没有第二份下载cache。下载后可用112,612,913,152 bytes。
- [x] 原始500-source contract通过：493×2ch+7×3ch=1007 views、22.379126 target-view hours、9188 events、0结构隔离、2条确定性尾部修复。
- [x] 原始对象按完全不变的冻结Gate B/C正确失败并在Qwen前停止：1507 candidates中exact、normalized、content exact和window命中均为0，但`Edu--022345.wav` ch1与`Edu--022764.wav` ch1均对`candor_turn_taking/62/input.wav`产生near-duplicate gate hit；前者similarity=0.665179、77 frames、23 votes，后者similarity=0.781710、68 frames、17 votes。原始archive、1507条identity/match记录及失败manifest全部保留，Qwen调用数为0。
- [x] 只读诊断未据此修改冻结门禁：两条候选与benchmark的最大原始波形相关分别仅0.0271/0.0212，compact-active相关仅0.0349/0.0199，candidate官方LID均为中文；`Edu--022764`的68帧指纹仅22个唯一值、相邻帧相等率56.72%，提示低变化指纹碰撞，但这不能作为事后忽略门禁的依据。
- [x] 经项目负责人确认后保留原始archive与失败证据，另建sanitized对象并整源排除`Edu--022345`和`Edu--022764`：498 sources、1003 views、22.363669 target-view hours、9181 events、1667 missing states；成员闭包缺失/额外均0，剩余成员metadata不一致0，sanitized tar SHA-256=`7f8336d9...260d3`。
- [x] sanitized source contract全项通过，完全不变的冻结Gate B/C对1501 candidates重跑通过；独立复算498个唯一source members、5125 near candidates，exact/normalized/content exact/window/near-duplicate gate hit均为0，被排除source未回流。通过前Qwen调用数为0。
- [x] 离线冻结385个full请求覆盖1667/1667个缺状态事件（预计1,198,004 prompt tokens），固定模型`qwen/qwen3-235b-a22b-2507`、prompt v2、direct route、4 workers和`CONFIRM_1667_LABELS`。独立复算385个请求签名/source均唯一，事件漏项/额外/重复均0；被排除source未回流；24-request/30-event校准集按三状态各10条平衡。
- [x] 服务端10 USD/day预算门禁与1-event连通测试通过，accepted成本0.00049095 USD；30-event校准的模型、路由、事件/签名闭包、schema和label-collapse机械门禁通过，accepted成本0.0061404425 USD，校准后服务端当日余额8.886756818 USD。隐藏官方标签一致率15/30=50%仅作诊断，未据此修改标签、prompt、请求或数据。
- [x] 385-request/1667-event完整Qwen标注与机械/provenance审计通过：请求/签名/事件闭包均完整，0重复，固定模型与固定route全通过；分布backchannel=1167、complete=186、incomplete=314，schema attempt 0/1=363/22。full accepted成本0.1908832375 USD，本分片连通+校准+full保留响应合计0.19751463 USD；服务端当日累计从1.112752232增至1.320831156 USD、余额8.679168844 USD，差额0.010564294 USD按拒绝schema尝试等未保留响应如实记录。
- [x] 合并9181个最终状态并执行sanitized目标音频、Paraformer、确定性渲染、timeline、GLM audio token、官方两列Stage 3 Parquet和最终Gate D全链路；任何新增严格隔离均已逐条记录并向后传播。
- [x] 合并9181个最终状态并通过checksum：官方辅助7490、WAIT→complete 24、Qwen 1667；最终backchannel=2141、complete=5048、incomplete=1992。目标音频从sanitized tar抽取1003 views并全checksum通过。
- [x] Paraformer完成1003=1001通过+2严格隔离，126193 tokens；`Edu--022269/target-ch01`有142个target-active chunks和约-28.877 dBFS active RMS却无文本，`Edu--022304/target-ch00`末token超20.800 s边界20 ms。fresh-cache以相同signature复现两条；后者前一token起点有60 ms浮动但末端越界与结论不变。两条共8 events、350 chunks、55.824 s整view向后传播。
- [x] 渲染、timeline与GLM全局闭包通过：1003=1001有效+2整view隔离，9181=9173+8 events，503614=503264+350 original chunks，另补1 terminal-silence chunk形成503265 effective chunks；GLM新增隔离0并精确生成1006530个80 ms audio tokens。
- [x] 官方两列Stage 3 Parquet导出和未修改SoulX loader验证通过：2255 rows、1001 views、500226 exported chunks，3039局部+350整view隔离与503615输入chunks闭包；20条roundtrip和train split=2142通过。
- [x] 最终冻结Gate D通过：1001个model-ready views覆盖498个sanitized sources，2002条identity records且每个mono view评分两次；selection缺失/额外0，source身份碰撞及exact/normalized/content/window/near命中均0。独立确认`Edu--022345`和`Edu--022764`在selection、tar与identity中均未回流。Edu_0037全链路完成。

Edu_0038 当前执行记录：

- [x] 固定官方对象身份：`Edu_0038.tar`=8,289,576,960 bytes、LFS文件SHA-256=`4ef813ba...8fcb3`、Xet哈希=`aadd39a9...c20c`、Git OID=`0b6a3305...7827`，三类身份不混用。
- [x] 权威metadata确定性推导500 sources（`Edu--022840`–`Edu--023461`，集合SHA-256=`ea7e82fd...e7c5`），493×2ch+6×3ch+1×4ch=1008 views、23.983054 target-view hours、9547 events、1840 missing states、408个缺状态sources。
- [x] Gate A对3119条benchmark记录/7721个标识串碰撞0；下载前空闲99,528,380,416 bytes，高于archive+20 GiB保底所需29,764,413,440 bytes。Qwen调用0，不训练、不评测checkpoint。
- [x] 可续传下载与独立官方字节/SHA验证通过；500-source contract全部通过（1008 views、9547 events、0结构隔离、0尾部修复）。冻结Gate B/C对1508 records重跑通过，5390 near candidates但gate hit=0，exact/normalized/content/window命中均0，Qwen调用0。
- [x] 冻结1840个缺状态事件请求并完成独立闭包、服务端预算门禁、连通与30-event校准；连接调用固定模型/路由且成本$0.000264，30-event校准机械门禁通过（30/30有效、无标签坍缩、成本$0.006150，服务端当日余额$8.669701）；已准入full标注。
- [x] 离线冻结408个full请求覆盖1840个缺状态事件（预计1,283,765 prompt tokens），固定Qwen模型、prompt v2、direct route、4 workers和`CONFIRM_1840_LABELS`；正确独立复算确认408 source/signature唯一、事件漏项/额外/重复0，校准三状态各10条。
- [x] 独立复算前两次只读统计曾误用`state`字段而把全部9547 events计为missing；实际source-scan字段是`official_state`。两次错误均未修改产物、未调用API，第三次按`official_state is null`复算得到1840 events/408 sources并通过全部断言。
- [x] 固定模型、direct route、4 workers完成408个full请求/1840 events；结果、签名、事件、模型、路由和预算审计全部通过，重复0、标签坍缩0。full accepted-response成本$0.216417，整分片保留accepted响应成本$0.222832；服务端当日计数从$1.322393增至$1.564656，差额包含被拒绝的schema尝试。
- [x] 状态定稿闭包通过：9547 events=官方7688+WAIT→complete 19+Qwen 1840；最终backchannel/complete/incomplete=2172/5262/2113。定稿后的只读哈希命令一度误用`events.jsonl`，实际文件为`events_with_final_state.jsonl`，按真实文件名复核全部通过，未影响产物。
- [x] 目标音频抽取完成且所有checksum通过：1008 views=986个2-track目标视图+18个3-track目标视图+4个4-track目标视图，1,381,423,936 resampled frames、1,173,184 padding samples。首次只读后检误把WAV当作根目录文件并假设summary有`status`字段；递归按真实schema复核得到1008 summary=1008 manifest=1008 WAV，未影响产物。
- [x] Paraformer曾在远端tmux执行；保守接力按预声明规则在出现严格隔离时安全停止，随后完成fresh-cache复核并继续下游，没有绕过门禁。
- [x] 远端Paraformer完成1008=1007通过+1严格隔离；接力按设计安全停止。`Edu--023048/target-ch01`有104个target-active chunks（16.64秒），active RMS约-28.9665 dBFS、非零样本88.24%，却无文本/时间戳；全新缓存以同一signature复现，故整view隔离而不伪造文本。
- [x] render和timeline闭包通过：1008=1007+1 views，9547=9543+4 events，540077=539780+297 original chunks；隔离原音频47.376秒/网格47.52秒，另补1 terminal-silence chunk得到539781 effective chunks。GLM已启动。
- [x] GLM闭包通过：1008输入=1007 eligible/passed+1上游整view隔离，新增隔离0；539781 effective chunks精确生成1079562个80 ms tokens。
- [x] 官方两列Stage 3 Parquet及未修改SoulX loader验证通过：2342 rows、1007 views/500 sources、536532 exported chunks；3249局部+297整view隔离与540078输入chunks闭包，checksum失败0、20条roundtrip、train split=2224。
- [x] 最终冻结Gate D通过：1007 model-ready views/500 sources，2014 identity和match records且每个mono view保守评分两次；selection/tar分区闭包，source标识碰撞及exact/normalized/content/window/near命中均0。独立确认`Edu--023048/target-ch01`在selection、tar、identity、matches均未回流；Edu_0038全链路完成。
- [x] 为提高吞吐并行预下载Edu_0042：下载前空闲61,801,226,240 bytes，高于archive+20 GiB保底所需28,888,156,160 bytes；固定官方对象7,413,319,680 bytes、SHA-256=`dd4c5d51...aaadb`，v3直连16路下载和全文件校验均已完成。完成后按计划暂停继续下载并复核真实空间。
- [x] 为缩短串行等待，在Edu_0038标注/处理期间预下载Edu_0039–Edu_0041官方archive；每个对象独立做官方bytes/SHA-256及20 GiB保底磁盘门禁，不提前解包、不调用下一分片LLM、不训练/评测。Edu_0041之后已暂停预下载，给Edu_0038处理产物预留空间。
- [x] 预下载初次启动发现v2路由名虽写`direct-aria2c-16x`，实现实际为单连接curl且继承代理，实测约67 KiB/s；保留约9 MiB断点后改为v3，显式清除代理并使用16路aria2c，速度稳定约11–13 MiB/s。同步修复分段稀疏文件不能用逻辑大小代表下载进度的断点判断，最终仍以官方bytes+全文件SHA-256为准。
- [x] Edu_0039预下载完成：8,089,958,400 bytes，整文件SHA-256=`2244f2b5...a369`与固定官方清单完全一致；完成后自动接续Edu_0040，尚未对Edu_0039解包、调用LLM或改变其正式处理门禁状态。
- [x] Edu_0040预下载完成：7,522,283,520 bytes，整文件SHA-256=`d75dc91e...0c11`与固定官方清单完全一致；完成后空闲75,582,021,632 bytes。
- [x] Edu_0041下载前空闲75,582,038,016 bytes，高于archive+20 GiB保底所需29,595,627,520 bytes；v3直连16路下载完成，8,120,791,040 bytes、整文件SHA-256=`b959d580...3a75`与固定官方清单一致，完成后空闲67,460,603,904 bytes并停止继续预下载。

Edu_0039 当前执行记录：

- [x] 固定官方对象身份：`Edu_0039.tar`=8,089,958,400 bytes、LFS文件SHA-256=`2244f2b5...a369`、Xet哈希=`e0789909...692f4`、Git OID=`ab9e8211...604c`；本地整文件字节数/SHA已与官方清单一致。
- [x] 官方tar的500项成员与权威metadata第39分片严格相等：`Edu--023462`–`Edu--024072`，source集合SHA-256=`404c58c7...d2dae`；491×2ch+9×3ch=1009 views、23.405457 target-view hours、9286 events、1808 missing states。
- [x] metadata-only Gate A对3119条benchmark记录/7721个标识串碰撞0；本阶段Qwen调用0，不训练、不评测checkpoint。
- [x] 原始source contract所有预期计数通过，但诚实隔离`Edu--023487`：其`ch01/event0015`含越过精确213.604秒WAV边界的official speaker interval；因此原始500-source串联门禁按设计停在冻结Gate B/C之前，没有改变160 ms尾部修复容差、泄漏阈值或数据。
- [x] 按既有保守策略保留官方tar与失败证据，整源移除`Edu--023487.wav`并验证成员集合严格等于parent minus exclusion、其余成员字节不变；sanitized对象为8,048,936,960 bytes、SHA-256=`a7782e4e...ecdb`，499 sources、490×2ch+9×3ch=1007 views、23.286788 target-view hours、9244 events、1795 missing states。
- [x] sanitized tar构造及成员/大小闭包通过。首次sanitized source scan在发布产物前安全失败，原因是合同中的source-ID集合SHA误从JSONL错误字段位置计算；现已按scanner实际的“排序source ID+换行”口径由tar basename独立复算并更正。失败日志保留；数据、160 ms容差和所有门禁阈值均未改变。
- [x] sanitized source contract全项通过：499 sources、1007 views、9244 events、1795 missing states、0结构隔离，4条既有容差内尾部修复照常记录。
- [x] 完全不变的冻结benchmark Gate B/C通过：1506 candidate records覆盖499个唯一source members；exact、normalized、content exact、window及near-duplicate gate hit均为0，0泄漏source，`gate_passed=true`。独立复算1506 identity/match行、1035条含near candidates记录且`gate_hit=true`为0；通过前Qwen调用数为0。
- [ ] 正在冻结1795个缺状态事件的正式Qwen请求。首次命令使用了不存在的本地tokenizer目录，Transformers在生成输出/API调用前安全失败；随后按Edu_0038 summary恢复正确本地路径。第二次离线输出完整但脚本默认只抽3×3=9-event校准，已保留为`calibration3_not_used`且不会调用；现显式使用10×3=30-event校准重建，固定模型、prompt、direct route、4 workers和$10/day上限不变。
- [x] 正式离线冻结403个full请求精确覆盖1795/1795缺状态事件，403个source/signature唯一、隔离source未回流；23个校准请求覆盖30个官方已标事件且三状态各10。1-event连通通过，accepted成本`$0.00070982`；30-event机械/provenance门禁通过，0重复/0标签坍缩、状态=`backchannel 9 / complete 11 / incomplete 10`、accepted成本`$0.005773445`，与官方LLM辅助标签76.67%一致仅作诊断。
- [ ] 已以`CONFIRM_1795_LABELS`启动403-request full标注：固定`qwen/qwen3-235b-a22b-2507`、direct route、4 workers及服务端$10/day cap；完成后仍须做signature/event/model/route/schema/成本审计，未通过不得进入状态finalization。
- [ ] full首轮在402/403个签名已原子缓存时，唯一`Edu--023894`（4 events）的HTTP响应解析为截断/不完整JSON并触发`JSONDecodeError`；wrapper按失败关闭、未发布full结果。失败log/manifest已归档，独立闭包确认仅缺签名`7c214a8c...60cd1`；现以完全相同条件恢复，复用402个full缓存并只补唯一缺失请求。
- [x] 恢复只新增1个cache后403 requests/1795 events完整结果发布并通过机械/provenance审计：0重复、0标签坍缩，schema attempt 0/1/2=`382/20/1`，状态=`backchannel 1220 / complete 210 / incomplete 365`。full accepted成本=`$0.2093372125`，连通+校准+full保留accepted合计=`$0.2158204775`；服务端当日计数=`$0.238362396 / $10`，差额如实归因于被拒schema尝试/截断响应等未进入accepted ledger的调用。
- [x] 状态finalization通过：9244 events=官方7430+WAIT→complete 19+Qwen 1795，最终`backchannel/complete/incomplete=2137/5083/2024`，全量checksums通过。
- [x] Edu_0039 sanitized对象的1007个target-vs-rest 16 kHz mono视图提取与全量checksum闭包通过：980个视图来自双声道来源、27个来自三声道来源，共1,341,319,040 resampled frames和1,234,560 padding samples；现以固定本地模型在独立tmux执行Paraformer。
- [x] Edu_0039 Paraformer与Edu_0040 full Qwen并行执行且资源未冲突。Edu_0040的407 requests/1649 events完整标注与独立机械/provenance审计通过：0重复/0标签坍缩，固定模型/route成立，schema attempt 0/1/2=`383/22/2`，分布`backchannel 1151 / complete 181 / incomplete 317`；full accepted成本`$0.196317`，连通+校准+full保留accepted合计`$0.201632`，服务端当日累计`$0.456360/$10`，与accepted口径差额如实归因于被拒schema尝试。状态定稿通过：8564=官方6894+WAIT→complete 21+Qwen 1649，最终状态分布2041/4688/1835。
- [x] Edu_0039固定本地Paraformer完成1007=1006通过+1严格隔离、130965 tokens、耗时1054.314秒；`Edu--023854/target-ch00`有320个target-active chunks但返回空文本/时间戳，全新cache以相同signature和原因复现。未修改parser或伪造文本，整view隔离6 events/531 chunks/84.912秒原音频（84.96秒chunk网格）。render与timeline闭包通过：9244=9238+6 events、524435=523904+531 original chunks，另补2个terminal silence形成523906 effective chunks；固定本地GLM tokenizer随后完成。
- [x] Edu_0039 GLM与官方两列Stage 3 Parquet完成并闭包：1006 eligible views、0新增隔离、1047812个80 ms audio tokens；model-ready为2367 rows/1006 views/499 sources、520657 exported chunks，3249局部+531整view隔离与524437输入chunks闭包。checksum失败0、20条roundtrip、未修改SoulX loader读取train split=2248；Gate D待Edu_0043唯一CPU scorer退出后启动，避免同时运行两个大音频scorer。
- [x] Edu_0039最终Gate D通过：1006个model-ready views/499 sources，冻结scorer生成2012 identity/match records且每个mono view保守评分两次；selection缺失/额外0，`Edu--023487`未回流，source标识及exact/normalized/content/window/near命中全0。首次接力错误地预建了scorer输出目录，冻结脚本在评分前安全拒绝覆盖；确认空目录后`rmdir`并复用完全相同、已闭包的2.683 GB输入tar重试，未重建输入或改阈值。Edu_0039全链路完成。
- [x] Edu_0042与上述CPU/磁盘门禁并行下载完成：7,413,319,680 bytes、整文件SHA-256=`dd4c5d51...aaadb`与固定官方值一致；500项tar成员与权威metadata第42切片逐项相等（`Edu--025322`–`Edu--025946`，集合SHA=`3fd5835a...daac5`），source contract已冻结。完成后数据盘可用46,288,281,600 bytes；因后续需保留20 GiB安全余量，Edu_0039真实处理占用复核前不启动Edu_0043。
- [x] 利用Edu_0039冻结查重与Edu_0042下载的等待时间，只读预检已下载的Edu_0040/0041：两者官方tar均为500项且与权威metadata第40/41切片逐项完全一致；集合SHA分别为`6ed50351...e3b51`与`4a3a034a...fb4a`，source contracts已提前冻结。首次临时预检误用了相对Edu_0018的切片偏移并在`cmp`处安全失败，随即按inventory代码实际的全局`(shard-1)*500`规则纠正；未写数据产物或门禁结论。
- [x] Edu_0040–0042的metadata-only Gate A均已对3119条benchmark记录/7721个标识串复算，source-ID碰撞全为0；在Edu_0039 full Qwen仅占网络等待时，并行启动Edu_0040无API source scan。不会同时启动多个大音频scorer，避免磁盘争用。
- [x] Edu_0040 source contract全项通过：500 sources、495×2ch+5×3ch=1005 views、21.762900 target-view hours、8564 events、1649 missing states、0结构隔离；6条容差内尾部修复照常记录。现与Edu_0039 full Qwen并行运行Edu_0040完全不变的冻结Gate B/C，不并行启动第二个付费标注。
- [x] Edu_0040冻结Gate B/C通过：1505 identity/match records覆盖500个唯一source members，exact、normalized、content exact、window及near-duplicate gate hit全为0，0隔离source；独立行数/非空数组/gate-hit复算一致。CPU scorer退出后接力Edu_0041 source scan，仍与Edu_0039唯一full Qwen任务并行。
- [x] Edu_0041 source contract全项通过：500 sources、492×2ch+8×3ch=1008 views、23.494676 target-view hours、9694 events、1856 missing states、0结构隔离；4条容差内尾部修复照常记录。现接力运行Edu_0041冻结Gate B/C，Edu_0039仍是唯一付费Qwen任务。
- [x] Edu_0041冻结Gate B/C通过并独立复核：1508 identity/match records、500唯一source members，exact、normalized、content exact、window及near-duplicate gate hit全为0。scorer退出后接力Edu_0042 Gate B/C，同时只离线准备Edu_0041请求，不发起付费调用。
- [x] Edu_0041离线请求冻结及独立闭包完成：408 requests/signatures/sources精确覆盖1856/1856缺状态事件，25个校准请求覆盖三状态各10的30 events，确认串`CONFIRM_1856_LABELS`；固定模型/prompt成立且`full_api_gate=not_authorized`，没有并行付费调用。
- [x] Edu_0040离线请求冻结及独立闭包完成：407 requests/signatures/sources精确覆盖1649/1649缺状态事件，18个校准请求覆盖三状态各10的30 events，确认串`CONFIRM_1649_LABELS`；固定模型/prompt成立且`full_api_gate=not_authorized`，没有并行发起第二个付费任务。
- [x] 系统load降至可接受范围后，利用空闲CPU完成Edu_0042 source scan：500 sources、494×2ch+4×3ch+2×4ch=1008 views、21.447588 target-view hours、8580 events、1743 missing states、0结构隔离，2条容差内尾部修复照常记录。Edu_0042 Gate B/C等待Edu_0041 scorer退出后接力，仍不叠加同类重任务。
- [x] Edu_0042冻结Gate B/C通过并独立复核：1508 identity/match records、500唯一source members，exact、normalized、content exact、window及near-duplicate gate hit全为0。已启动30-event校准配置的离线请求冻结，不发起第二个付费Qwen任务。
- [x] Edu_0042离线请求冻结及独立闭包完成：411 requests/signatures/sources精确覆盖1743/1743缺状态事件，22个校准请求覆盖三状态各10的30 events，确认串`CONFIRM_1743_LABELS`；固定模型/prompt成立且`full_api_gate=not_authorized`，没有并行付费调用。
- [x] Edu_0039 target audio全部落盘后，按真实可用43,449,974,784 bytes重新执行20 GiB空间门禁；Edu_0043以16路直连完成7,444,357,120 bytes下载，整文件SHA-256=`1806e902...68fc`与固定官方值一致，下载后可用35,994,611,712 bytes。它与Edu_0039 GPU识别、Edu_0040 API标注资源正交；为给剩余物化产物留空间，不并行启动Edu_0044。
- [x] 利用下载等待时间完成Edu_0043 metadata-only Gate A：全局第43个500-source切片为`Edu--025947`–`Edu--026565`、集合SHA=`f7bd13c0...e760`，与冻结3119-record/7721-identifier benchmark清单的直接标识碰撞为0；该结论明确不代替下载后的tar成员闭包与音频Gate B/C。首次只读临时提取错误要求成员名带目录前缀，0条结果被数量断言拦截并删除，纠正为tar根目录成员后得到500条；失败尝试未生成门禁产物或下游结论。
- [x] Edu_0043 post-download接力在官方bytes/SHA通过后完成source contract scan：tar 500成员与metadata切片精确闭包，495×2ch+5×3ch=1005 views、21.537404 target-view hours、8508 events、1641 missing states、0结构隔离、2条容差内尾部修复。原始对象的完全不变冻结Gate B/C按设计失败并在Qwen前停止：1505 candidates中唯一`Edu--026481.wav` ch1对`candor_turn_taking/62/input.wav`触发near gate hit，similarity=0.655014（阈值0.655）、aligned frames=354（阈值64）、LSH votes=89（阈值16）；没有事后放宽规则，原始证据全量保留，原始对象Qwen调用0。
- [x] Edu_0043保留官方parent与失败证据，另建sanitized对象并整源排除`Edu--026481.wav`；第一次tmux启动因日志重定向位置错误立即退出，确认无partial/final且parent未改后用绝对路径重新启动并完成。sanitized tar为7,432,970,240 bytes、SHA=`4cb6b77d...ed60`，499成员精确等于parent minus exclusion，剩余成员size metadata逐项不变；sanitized contract通过：499 sources、494×2ch+5×3ch=1003 views、21.504471 hours、8493 events、1633 missing、0结构隔离；完全相同冻结Gate B/C复跑通过。
- [x] Edu_0043 sanitized冻结Gate B/C通过：1502 records/499 sources，exact/normalized/content/window/near gate hit全0且排除源未回流。离线冻结393 requests精确覆盖1633 missing events；独立只读闭包前两次分别误用`json.load(Path)`及误判answer-key为`events`列表，均在结论前失败且无写入/API，第三次按真实`event_id→state`映射通过全部断言。1-event连通和25-request/30-event校准通过，固定模型/route、0重复/0坍缩、输出8/12/10、隐藏官方一致率80%仅作诊断。
- [x] Edu_0043 full首轮在392/393个签名已原子缓存后因唯一`Edu--026422`（2 events）连续返回错误顶层JSON结构而失败关闭，没有发布不完整结果；零资源watcher按设计退出且未自动重试。原失败manifest保留，随后在相同模型、prompt、route、4 workers、确认串和$10/day门禁下，以新manifest恢复：复用392个cache并只新增1个cache，最终393 requests/1633 events完整发布。
- [x] Edu_0043 full机械/provenance审计通过：signature/event/model/route全闭包、重复0、标签坍缩0，schema attempt 0/1=`365/28`，状态=`backchannel 1152 / complete 171 / incomplete 310`，accepted-response成本`$0.1891192825`；恢复前预算预检记录当日服务端用量`$0.666663316 / $10`。状态finalization通过：8493 events=官方6846+WAIT→complete 14+Qwen 1633，最终状态分布`1995/4556/1942`。
- [x] Edu_0040固定本地Paraformer完成1005=1004通过+1严格隔离、126064 tokens；`Edu--024152/target-ch01`无文本/时间戳，全新独立cache以相同signature和原因复现，因此没有伪造文本或放宽门禁。render/timeline闭包通过：8564=8563+1 events、490110=490018+92 original chunks，另补6个terminal silence形成490024 effective chunks。
- [x] Edu_0040 GLM、官方两列Stage 3 Parquet及未修改SoulX loader验证通过：1004 eligible/passed views、0新增隔离、980048个80 ms audio tokens；model-ready为2236 rows/1004 views/500 sources、486890 exported chunks，3134局部+92整view隔离与490116输入chunks闭包。checksum失败0、20条roundtrip、official loader train split=2124。
- [x] Edu_0040 Gate D首次因空间门禁未启动：可用23,088,947,200 bytes，以2,510,328,115-byte候选参考计算至少短缺896,217,395 bytes；没有越过20 GiB保底。负责人随后明确批准只删除Edu_0019–0039的21个已通过Gate D、可由保留输入重建的临时候选tar；删除前逐分片确认`gate_passed=true` closure存在，实际删除55,131,822,080 bytes。所有selection、manifest、哈希、Gate D报告、target audio及model-ready输入保留；tar本身不能直接恢复但可确定性重建。
- [x] 清理后Edu_0040最终Gate D按未修改冻结v2.2配置通过：1004个model-ready views/500 sources，selection/tar无缺失或额外，冻结scorer生成2008 identity/match记录且每个mono view保守评分两次；source标识碰撞及exact/normalized/content/window/near命中均为0。候选tar为2,509,926,400 bytes、SHA=`55da6316...5c12`，独立closure SHA=`167c965f...e3e3`；Edu_0040全链路完成。
- [x] Edu_0043 sanitized目标音频及模型输入全链路完成：1003个16 kHz mono视图（988来自双声道来源、15来自三声道来源）全部checksum通过；固定本地Paraformer为1003/1003通过、0严格隔离、124646 tokens；timeline为8493 events/484309 original chunks并补2个terminal-silence chunks，3259局部chunk隔离均显式记账。GLM为1003/1003、0新增隔离、968622 audio tokens；model-ready为2249 rows/1003 views/499 sources、481052 exported chunks，checksum失败0、20条roundtrip、未修改官方loader train split=2136。
- [x] Edu_0043 sanitized最终Gate D通过：1003 views/499 sources、2006 identity/match且每个mono view保守评分两次；selection/tar分区闭包，source标识及exact/normalized/content/window/near命中全0。被原始Gate B/C命中的`Edu--026481`在selection、tar、identity和matches均未回流；候选tar SHA=`7b3edda4...555a`，独立closure SHA=`5483c7dc...fc81`。Edu_0043 sanitized全链路完成。
- [x] 第二层零资源watcher在Edu_0043上游泄漏门禁blocked后按设计退出，未生成请求、未访问OpenRouter；Edu_0040始终是唯一付费标注任务。

Edu_0041 最终执行记录：

- [x] 固定Qwen模型、prompt v2、direct route、4 workers及服务端`$10/day`门禁完成1-event连通、25-request/30-event校准和408-request/1856-event full；full独立审计确认signature/event/model/route闭包、重复0、标签坍缩0。full accepted-response成本`$0.24580177`，分布`backchannel/complete/incomplete=1282/212/362`。
- [x] 状态finalization通过：9694 events=官方7820+WAIT→complete 18+Qwen 1856，最终分布`2262/5265/2167`。1008个target audio视图及全量checksum通过，来源为984个2-track视图和24个3-track视图。
- [x] 固定本地Paraformer首轮1008=1007通过+1严格隔离、136789 tokens；`Edu--024703/target-ch00`存在语义target activity但返回空文本/时间戳，全新独立cache复现同一失败，因此未伪造文本、放宽parser或挑选输出。timeline闭包为1007有效+1整视图隔离，隔离3 events/86 chunks/13.76 s；GLM对1007个有效视图全部通过并生成1058008个80 ms audio tokens。
- [x] 官方两列Stage 3 Parquet与未修改SoulX loader验证通过：2376 rows、1007 views、525415 exported chunks，20条roundtrip、checksum失败0、official loader train split=2257。首次验证因编排未等待导出原子重命名而过早看到目录不存在；导出进程一直健康且未重启，完成后单独验证通过，数据产物未受影响。
- [x] 最终冻结Gate D通过：1007 model-ready views/500 sources，2014 identity records且每个mono view评分两次；selection缺失/额外0，source标识碰撞及exact/normalized/content/window/near命中全0。closure SHA-256=`ddd2f5e0...29e3`，Edu_0041全链路完成。

Edu_0042 最终执行记录：

- [x] 固定Qwen模型、prompt v2、direct route、4 workers及服务端`$10/day`门禁完成1-event连通、22-request/30-event校准和411-request/1743-event full；full独立审计确认全部闭包、重复0、标签坍缩0。full accepted-response成本`$0.22276274`，分布`1237/182/324`；完成后服务端当日累计`$1.184955256/$10`。
- [x] 状态finalization通过：8580 events=官方6813+WAIT→complete 24+Qwen 1743，最终分布`2030/4701/1849`。1008个target audio视图及全量checksum通过，来源为988个2-track、12个3-track和8个4-track视图。
- [x] 固定本地Paraformer首轮1008=1005通过+3严格隔离、120880 tokens。`Edu--025454/target-ch00`和`Edu--025755/target-ch00`为空文本且fresh-cache复现；`Edu--025697/target-ch00`首轮末token比181.120 s音频边界超20 ms，但唯一一次fresh-cache复现通过。因事前未冻结“复现通过即可回收”规则，为避免后验挑选成功输出，仍保留首次隔离且不再重复运行。三条整视图隔离共25 events/1547 chunks/247.52 s，最终报告必须单列这一不稳定边界和实际规模损失。
- [x] timeline闭合为1005有效+3整视图隔离；GLM对1005个有效视图全部通过、无新增隔离并生成962946个80 ms tokens。官方两列Parquet和未修改SoulX loader验证通过：2218 rows、1005 views、478322 exported chunks，20条roundtrip、checksum失败0、official loader train split=2107。
- [x] 最终冻结Gate D通过：1005 model-ready views/500 sources，2010 identity records且每个mono view评分两次；selection缺失/额外0，source标识碰撞及exact/normalized/content/window/near命中全0。closure SHA-256=`288bf4eb...a1ac`，Edu_0042全链路完成。

Edu_0044 最终执行记录：

- [x] 官方对象下载与源合同闭包通过：500 sources、1006 target views，原始 tar 为 7,812,597,760 bytes，SHA-256=`9a00f428919ced3c54739a7e52402abde1019d7f6905e3dc8a5c2ede5dfea09d`；冻结 Gate A/B/C 均通过。
- [x] 固定 `qwen/qwen3-235b-a22b-2507`、prompt v2、direct route、4 workers 与服务端 `$10/day` 上限。full 首轮在 392/401 requests 已原子缓存后因唯一截断 OpenRouter JSON 失败关闭，未发布不完整结果；随后使用完全相同的模型、prompt、route、请求集和缓存安全恢复。最终 401 requests/1638 events 完整通过独立机械/provenance 审计，重复 0、标签坍缩 0，Qwen 分布=`1148/194/296`，accepted-response 成本 `$0.23874963`。
- [x] 状态定稿闭包：8808 events=官方 7156+WAIT→complete 14+Qwen 1638，最终 `backchannel/complete/incomplete=2057/4885/1866`。1006 个 target views 全量 checksum 通过，其中 988 个来自双轨源、18 个来自三轨源。
- [x] 固定本地 Paraformer 完成 1006/1006、整视图隔离 0。timeline 闭包为 8808 events、509003 original chunks+1 terminal padding=509004 effective chunks，3044 个局部 chunk 严格隔离但无整视图隔离；GLM 完成 1006/1006、1018008 个 80 ms audio tokens。
- [x] 官方两列 Stage 3 Parquet 与未修改 SoulX loader 验证通过：2292 rows、1006 views、505960 exported chunks，checksum 失败 0、20 条 roundtrip、全局 view/chunk 闭包通过，official loader train split=2177。
- [x] 最终冻结 Gate D 通过：1006 views/500 sources，2012 identity/match records 且每个 mono view 恰好评分两次；selection 缺失/额外 0，source 标识碰撞及 exact/normalized/content exact/window/near-duplicate 命中全 0。closure SHA-256=`cc57e8474bc4d2151c6f3586cac40417cf75f968497c56d5c48bc940ba485aca`。

Edu_0045 最终执行记录：

- [x] 官方对象下载与源合同闭包通过：50 sources、101 target views，原始 tar 为 660,613,120 bytes，SHA-256=`e23ddd68a3091123aa80749c9b6a03f1133b19f4415dba664cf5c519f1c219f7`；冻结 Gate A/B/C 均通过。
- [x] 原冻结每状态 10-event 校准在任何 API 调用前安全失败：12 个不含缺标事件的独立源中仅有 9 个合格 backchannel 校准事件。负责人明确批准平衡 9/9/9 机械校准例外；正式 127-event 请求、模型、prompt、route、状态集和泄漏门禁均未改变。27-event 校准与 38-request/127-event full 均通过独立审计，重复 0、坍缩 0，Qwen 分布=`93/15/19`，accepted-response 成本 `$0.0150515875`。
- [x] 状态定稿闭包：739 events=官方 610+WAIT→complete 2+Qwen 127，最终 `177/423/139`。101 个 target views 全量 checksum 通过，98 个来自双轨源、3 个来自三轨源。
- [x] Edu_0045 仅在 Edu_0044 最终 Gate D 闭包且 tmux 退出后才启动 GPU，严格串行成立。Paraformer 为 101/101、整视图隔离 0；timeline 为 739 events、43045 original chunks+1 terminal padding=43046 effective chunks，267 个局部 chunk 隔离、整视图隔离 0；GLM 为 101/101、86092 audio tokens。
- [x] model-ready 为 196 rows、101 views、42779 exported chunks。未修改官方 loader 验证中 checksum 失败 0、20 条 roundtrip、全局闭包通过，official loader train split=186。
- [x] 最终冻结 Gate D 通过：101 views/50 sources，202 identity/match records 且每个 mono view 恰好评分两次；selection 缺失/额外 0，source 标识碰撞及五类音频命中全 0。closure SHA-256=`f8815742b54a8b24136ca2938e15aadcef7ee2f62f305211a25c250e736b781a`。
- [x] 完成后独立重跑 Edu_0044/Edu_0045 的 state、target audio、Paraformer、render、timeline、GLM 和 model-ready 共 14 组 `sha256sum -c --quiet`，全部通过；tmux 与相关进程均已正常退出。本阶段未聚合、未训练、未评测新 checkpoint。

## 19. Edu_0018–Edu_0045 可追加聚合阶段（2026-08-27）

- [x] 负责人批准：先补做冻结 Edu_0018 的未修改官方 loader validation 与完全不变的 v2.2 Gate D；仅在两者均通过时，才创建 Edu_0018–Edu_0045 的不可变多 Parquet 聚合。该阶段不训练、不评测 checkpoint、不创建 train/validation split、不安装或调用百度网盘工具。
- [x] 已实现并通过 91 项项目测试的聚合基础设施：每个冻结分片保留独立 Parquet/metadata/quarantine shard；聚合版本只用同文件系统 hardlink、版本化 registry/manifest/checksum，不重写或覆盖已有分片。后续扩展必须创建新的不可变聚合版本，不能原地追加修改旧版本。
- [x] Edu_0018 未修改官方 loader validation 通过：2168 rows、1005 source views、474030 exported chunks、2736 local quarantined chunks；checksum、20 条确定性 roundtrip 和全局 view/chunk 闭包通过，官方 train split 可读取 2059 行。报告：`/root/autodl-tmp/dataset/duplexconv/reports/expansion_v1/Edu_0018/model_ready_validation_v1.json`。
- [x] Gate D 启动前空间门禁通过：可用 45,707,399,168 bytes；候选 target audio 为 2,441,991,431 bytes，加 20 GiB 保底后要求 23,916,827,911 bytes，余量 21,790,571,257 bytes。
- [x] 完全不变的冻结 v2.2 Gate D 已执行并诚实失败：1005 个最终 model-ready views/500 sources 与 selection/tar 完全闭包，冻结 mono scorer 生成 2010 条证据。exact、normalized、content exact 和 window 命中均为 0，但 `Edu--010902/target-ch01` 对 `candor_turn_taking/62/input.wav` 达到 similarity=`0.6772959184`、aligned frames=`147`、LSH votes=`28`，超过冻结门槛 `0.655/64/16`。报告中的 2 条 near-duplicate hit 是同一 mono WAV 的 channel 0 与 mono mix 双份保守证据，只对应 1 个唯一 source member，不应错误表述为两个独立音频泄漏源。
- [x] 按批准的失败策略在 Gate D 后停止：没有调阈值、改 scorer、选择性忽略结果、自动 sanitized、创建聚合、创建 split、训练、checkpoint 评测或上传。失败 closure：`/root/autodl-tmp/dataset/duplexconv/reports/expansion_v1/Edu_0018/gate_d_closure_v1.json`（SHA-256=`337ddbd68fd3ce30c1d956c2cce96e3bd62e898a3aa621ef26d1aee9b05ef344`）。
- [x] 负责人选择保留 Edu_0018、仅整会话排除 `Edu--010902`，明确否决“因一条重复排除整个 Edu_0018”。只读诊断显示候选为中文物理题语音，候选与英文CANDOR样本的最大原始/活动压缩波形滑窗Pearson相关分别仅约`0.01564/0.01565`；同一个`candor_turn_taking/62/input.wav`在既有冻结v2.2门禁中还与另外6个互不相同的中文音频成员命中，因此指纹碰撞是合理解释，但该诊断未用于推翻、修改或绕过冻结门禁。第一次诊断把目标WAV路径少写`audio/`层级，在实际读取前安全失败且未生成文件；纠正后全程纯内存只读。
- [x] 以冻结config执行完整整源排除：只修改项目自己的model-ready导出器，增加可审计的`--excluded-source-id`，未修改SoulX官方上游。新输出严格等于parent model-ready减去完整`Edu--010902`会话的2 views/2 rows/262 chunks/6 events；保留499 sources、1003 views、2166 rows、473768 exported chunks和2736局部隔离chunks。92项项目测试通过，独立parent-minus-source审计与全量checksum通过。
- [x] sanitized未修改官方loader验证通过：2166 rows、473768 chunks全量解析，20条确定性roundtrip、checksum与全局view/chunk闭包均通过，official train split=2057。原始Edu_0018、原model-ready、原失败Gate D证据及target audio全部保留且未覆盖。
- [x] sanitized最终冻结Gate D通过：selection/tar均为1003 views/499 sources，冻结scorer生成2006 identity/match records并对每个mono view恰好评分两次；selection缺失/额外均0，`Edu--010902`在metadata、quarantine、selection、tar、identity和matches中均未回流，source标识碰撞以及exact/normalized/content exact/window/near-duplicate五类命中全部为0。closure SHA-256=`cec46c67aeb3d741989722471064640550a3e023853b63118762aaf8574ac218`。
- [x] 不可变多 Parquet 聚合已完成并冻结：输入为 sanitized Edu_0018 加 Edu_0019–0045 各自最终 Gate-D-passed 版本，共28分片、13540个source conversations、27247个source views、61980 rows、13689991个exported chunks。跨分片61980个index全部唯一，24个整view隔离、14818个整view隔离chunks、209个整view隔离events和87124个局部隔离chunks均完成闭包；所有28个Parquet均为原分片hardlink，未复制payload。
- [x] 聚合独立审计通过：所有输入Gate D closure均通过、Gate D证据传递并集闭包、aggregate checksum全通过、metadata/index/row均为61980且`Edu--010902`未回流。audit=`/root/autodl-tmp/dataset/duplexconv/reports/aggregate_edu0018_0045_v1/aggregate_audit.json`（SHA-256=`5681c87361c978f520a10bfa39af34e7992a6e7e789521a53f3bd6939c7a5fa7`）。
- [x] 未修改SoulX官方loader全量验证通过：28个Parquet、61980 rows和13689991 chunks全部解析一致，checksum失败0、100条随机roundtrip通过、全局view/chunk闭包通过，官方train split可读取58881行。报告=`/root/autodl-tmp/dataset/duplexconv/reports/aggregate_edu0018_0045_v1/model_ready_validation.json`（SHA-256=`acacea5ee888d81d5913d6232c228bbb2057c035f66890f829e8ef742ddd32fa`）。官方上游保持固定提交`928b06508ed2de1344208d06fb1f6fb2ebfb1df5`且工作树无改动。
- [x] 聚合目录为`/root/autodl-tmp/dataset/duplexconv/aggregates/edu0018_0045_stage3_zh_v1`；manifest SHA-256=`095e18de040f812aa305b06513356470727fad9f18d579081d04335f67819bdb`、stats SHA-256=`17187ca65e7b94b927d4b2b5c54fe163ac4b6f691b8af68884ec2089777d44b0`、checksums manifest SHA-256=`0c246ead053da895fb9e72ee94d3bc29201f22b216b20c104f711bedfdee2b1b`。该v1不得原地修改；以后扩展数据必须新建不可变v2并复跑同等门禁。本阶段未创建train/validation split、未训练、未评测checkpoint、未上传。

## 20. 百度网盘数据发布阶段（2026-08-27）

- [x] 负责人批准发布范围：上传28个官方Edu_0018–0045原始tar和metadata、最终不可变训练就绪聚合、聚合审计及发布收据；不上传68 GiB处理中间层、cache、work、失败重试、诊断目录、Conda环境、模型权重、`.env`或任何凭据。
- [x] 生成冻结的本地文件/字节/checksum清单并全部复核：core manifest 157条、211,968,657,404 bytes；28个原始tar均与官方冻结身份一致，凭据路径0、远端冲突0。
- [x] 安装并固定BaiduPCS-Go v4.0.1，二进制SHA-256=`56cb5457...b715`；复用现有受保护登录配置且全程无需负责人重新登录，凭据没有进入日志或发布物。
- [x] 在tmux中以同大小跳过、逐项可恢复策略上传到`/soulx-stage3-dataset-CN/datasets/duplexconv_edu0018_0045_stage3_zh_v1`。12个合法0-byte quarantine视图文件因客户端限制，由负责人批准的确定性tar、manifest与幂等恢复脚本表示；160/160条记录完成，payload为211,968,676,948 bytes，覆盖0、删除0。
- [x] 独立后置验收重新查询148个直接存储的远端文件，148/148精确字节一致；12个0-byte路径的兼容表示、archive SHA和恢复回归均通过，失败报告未生成。最终12文件回执包另上传到`receipts/final/`并逐项核验，共703,088 bytes；最终receipt SHA-256=`7bb21715...17eb`。

## 21. Edu_0001–Edu_0017 第二轮扩展（2026-08-27 批准）

- [x] 负责人批准扩展边界：补齐官方 `Edu_0001.tar`–`Edu_0017.tar`，不重新处理已冻结的 Edu_0018–0045；完成后新建不可变 `Edu_0001–0045 v2`，不原地修改 v1。本阶段只构造和验证数据，不训练、不创建新 checkpoint、不运行 Table 3。
- [x] 官方对象与容量盘点完成：17 个 tar 合计 134,040,770,560 bytes（124.835 GiB），预计 8,500 个 source conversations、192.267 source hours、387.799 target-view hours、154,159 events、29,818 个缺失状态、313 个 WAIT。补齐后 Edu_0001–0045 预计约 495.918 unique source hours。
- [x] 状态规则保持冻结：官方三状态原样保留，WAIT 确定性映射为 complete，缺失状态固定使用 OpenRouter `qwen/qwen3-235b-a22b-2507`（项目名称 `qwen3-235b-a22b-instruct-2507`）、prompt v2 和既有机械/provenance 审计。客户端与服务端每日预算上限均为 10 USD；17 分片预计成本约 3.7685 USD。不得根据 benchmark 结果改标签。
- [x] 执行顺序冻结：先用 Edu_0001 做完整 pilot；仅在官方 SHA/member closure、source contract、冻结 Gate A/B/C、Qwen 审计、target audio、Paraformer、timeline、GLM、model-ready、未修改官方 loader 和最终 Gate D 全部通过后，才按相同流程继续 Edu_0002–0017。
- [x] 并发与资源边界冻结：最多 3 个纯下载任务并行；下载与轻量 CPU 核验可并行；Qwen 每次只处理一个分片、内部最多 4 workers；Paraformer 和 GLM 严格单 GPU 串行。环境固定为数据盘 Conda `soulx-duplug-official`，模型使用已有本地 Paraformer、GLM tokenizer 和 SoulX 资产，不在系统盘新增大文件。
- [x] 失败策略冻结：任一分片发生结构异常、泄漏门禁命中、API 机械审计失败、时间戳/loader/Gate D 闭包失败时，停止该分片并保留证据；不得自动调阈值、挑选输出、删除会话、生成 sanitized 版本或继续付费/GPU 阶段。任何例外都重新提交负责人确认。
- [x] Edu_0001 sanitized pilot全链路完成。自官方tar下载启动至最终Gate D闭包约11,043秒（3小时4分3秒）；选定的raw/sanitized/work/processed/reports资产表观合计20,948,221,118 bytes，其中包含可重建的sanitized与Gate-D工作tar重复副本。API accepted-response成本按connectivity+calibration+full合计`$0.20682442`；完成full后共享key平台当日累计`$0.247618161/$10`，服务端计数仍是预算权威口径。
- [x] Edu_0001 官方tar及source contract通过后，原始冻结v2.2 Gate B/C在`Edu--000241.wav` channel 1对`candor_turn_taking/62/input.wav`产生唯一near hit：similarity=`0.6553030303`、aligned frames=`231`、LSH votes=`41`，冻结阈值=`0.655/64/16`；其余四类命中均为0。流水线按设计在Qwen/GPU前停止，API调用和费用均为0，原始失败证据完整保留。
- [x] 只读诊断显示候选为33.232秒中文物理辅导对话、benchmark为94秒英文CANDOR；原始/活动压缩波形最大绝对Pearson仅`0.0221712/0.0220492`，且同一benchmark member已在冻结v2.2/Gate D中命中8个互不相同source。该结果强烈支持指纹碰撞，但未用于推翻或修改冻结门禁。
- [x] 按负责人批准的恢复方案，仅整会话排除`Edu--000241`，构造并独立审计499-source sanitized Edu_0001；其余499个WAV的tar元数据与逐文件payload SHA完全相同。精确source contract与完全不变的冻结v2.2 Gate B/C均通过：1,502保守评分视图、5,210 near candidates，五类gate hit全部为0。原始失败证据未覆盖，阈值/scorer未修改。
- [x] Edu_0001 sanitized Qwen补标完成并通过独立机械/provenance审计：397/397 requests、1,725/1,725 events闭合，固定`qwen/qwen3-235b-a22b-2507`与`direct-no-proxy-v2`路由均通过，重复0、标签坍缩0；full accepted-response成本`$0.20119368`，完成后平台当日累计`$0.247618161/$10`。状态定稿为8,943 events=官方7,202+WAIT→complete 16+Qwen 1,725，最终`backchannel/complete/incomplete=2139/4887/1917`，全量checksum通过。
- [x] 从sanitized 499-source tar确定性提取1,003个16 kHz mono/chunk-pad target views（988个来自双轨源、15个来自三轨源），全部WAV checksum通过；输入archive SHA-256=`b707026ca261f1cefdfc4748e8be88ca1436468773cd5c156f1a6312d94ed23a`，未读取或回流被排除的`Edu--000241`。
- [x] 固定本地Paraformer/FunASR 1.2.6单GPU完成：1,003/1,003 views严格通过、0整视图隔离、131,626 tokens，耗时1,017.060674秒，分区与checksum闭包通过；未裁剪时间戳、用metadata补文、放宽parser或挑选输出。
- [x] 确定性render/timeline闭包通过：8,943 events、507,725 original chunks+2 terminal padding=507,727 effective chunks；2,714个局部chunks按既有规则隔离，0整视图隔离。固定GLM单GPU完成1,003/1,003 views、0新增隔离、1,015,454个80 ms audio tokens，checksum通过。
- [x] 官方两列Stage 3 model-ready导出与未修改SoulX loader验证通过：2,276 rows、1,003 views、505,013 exported chunks，20条roundtrip、checksum与全局闭包通过，official train split=2,162。Parquet SHA-256=`96fa57dc20fdc25646b557732bcf5b6463839fb034ad8e42774ab73c8f276e65`，validation SHA-256=`e6877c657d5b2bbb11e1eb31638e68f032324f69f8472f2d625e9dbe2fc56677`。
- [x] 最终完全不变冻结Gate D通过：1,003 model-ready views/499 sources，selection/tar分区闭合，冻结mono scorer产生2,006条记录且每个view恰好评分两次；`Edu--000241`未回流，source标识碰撞及exact/normalized/content exact/window/near-duplicate五类命中全0。closure SHA-256=`dea05f7e80d7ec06244d3330e1df253bea889413d818987bd16dc2dfbe788f28`。
- [x] 一次附加的只读摘要哈希误用了旧文件名`dataset.parquet/metadata.jsonl/summary.json`而返回不存在；model-ready自身checksum此前已通过，随后按实际文件名`data/train-00000-of-00001.parquet/metadata/windows.jsonl/stats.json`补验成功，未重跑或修改导出与loader验证。
- [x] 为隐藏下载等待时间，在不改变处理顺序的前提下启用最多三路原始 tar 预取：Edu_0001–0003 已于 2026-08-27 启动并各自使用可断点 `.part`、精确 bytes/SHA-256 和 20 GiB 空闲门禁。Edu_0002/0003 仅下载，不提前调用 Qwen、占用 GPU或进入下游；Edu_0001 最终 Gate D 通过前不处理 Edu_0002。
- [x] Edu_0001–0006官方raw均已完成精确bytes/SHA-256闭包；释放的下载槽按批准范围补入Edu_0007–0009，当前仍恰好三路raw-only预取。Edu_0002–0009均未因预取而提前调用Qwen或GPU。
- [x] Edu_0002已完成完全不变的冻结流水线并通过最终Gate D：500 sources、1,008 views、22.608096 target-view hours、1,691个缺失状态，Qwen full accepted-response成本`$0.2052010575`；model-ready为2,336 rows/505,889 exported chunks。最终closure SHA-256=`343f85a526389c092e3b2bfcc499e939f8ff53b6466e2f0f571e64203f09b81d`，完成时间`2026-08-27T16:53:43Z`。
- [x] 为Edu_0003–0006建立逐分片精确source contract和单一编排队列；合同SHA依次为`3d0c72ee…257da`、`f69110d1…d5ea7`、`3545b8e7…780de`、`d9c15d6e…36fc8`。队列只按Edu_0003→0006串行调用既有冻结runner；任一非零退出、API审计、时间戳、loader或Gate D失败均阻止下一分片，禁止自动重试、挑选输出、自动排除或sanitized。
- [x] 启动时Edu_0003已通过Gate A和source scan，正在运行完全不变的Gate B/C；该记录点Qwen调用0、GPU尚未启动。队列tmux=`soulx_edu0003_0006_pipeline_queue`，实时来源为`/root/autodl-tmp/dataset/duplexconv/reports/expansion_v1/edu0003_0006_queue_manifest.json`及当前分片pipeline manifest。
- [x] Edu_0001–0012官方raw均已完成精确bytes/SHA-256闭包。新一轮预取使用最多三槽：Edu_0013/0014/0015当前并行，完成槽位后分别接力Edu_0016/0017；只下载raw，不提前处理、调用Qwen或占用GPU。预计新增raw约36.7 GiB，处理Edu_0003–0006另需约20–24 GiB，启动前数据盘可用108 GiB。
- [x] 新增合同与队列编排代码后，用固定项目Conda Python运行`unittest discover`，92项全部通过。此前base与项目Conda中的两次`python -m pytest`均因未安装pytest而在收集前退出，只是测试运行器不可用，不产生或修改任何数据产物。
- [x] Edu_0003、0004、0005均由同一冻结runner完成至最终Gate D并通过，closure SHA依次为`1da875e1…359e`、`69ba107a…c77f`、`66a70cc0…939a`；合计69.290062 target-view hours、7,100 model-ready rows、1,549,844 exported chunks，full Qwen accepted-response成本`$0.630893225`。没有并发Qwen或GPU任务。
- [x] Edu_0013–0017三槽raw预取与接力全部完成，五个官方tar均通过精确bytes/SHA-256闭包；至此Edu_0001–0017全部17个raw已下载完成。下载tmux均正常退出，当前无相关运行进程；数据盘可用约56 GiB。
- [x] Edu_0006原始Gate A与全部预期source contract计数通过，但source scan诚实隔离唯一会话`Edu--003492`：WAV精确结束于207.780 s，末事件12 ms与倒数第二segment 100 ms尾部越界已由既有规则裁剪，但最后segment `207.980–208.080 s`完全位于音频结束之后、没有对应音频。原队列按设计在Qwen/GPU前停止，调用与费用均为0；未忽略异常、改160 ms容差、自动排除或继续Edu_0007。
- [x] 只读审计全部官方metadata后，这一异常类别全库仅2个source；另一个是此前Edu_0039已按负责人批准整源排除并最终Gate D通过的`Edu--023487`。若负责人批准对Edu_0006采用相同保守策略，预计仅排除1/500会话及2/1010 views，sanitized规模为499 sources、1008 views、9109 events、1778 missing states、23.12830184027778 target-view hours；原始tar和失败证据必须保留。
- [x] 负责人批准Edu_0006单会话sanitized恢复及独立variant编排补丁；默认普通分片命名/流程保持不变，新增5项编排回归测试后项目97项unittest全部通过。sanitized tar为7,994,173,440 bytes、SHA-256=`45c08f36…23a3d`；parent-minus-one逐成员元数据与payload SHA审计通过，499个保留成员不一致数0。sanitized Gate A/source contract及完全不变Gate B/C均通过，结构隔离与泄漏门禁命中均为0。
- [x] Edu_0006固定Qwen连通性与30-event校准通过机械门禁；407-request/1778-event full在服务端已产生406个签名一致的原子缓存后，唯一外层OpenRouter HTTP JSON响应体在7,854 bytes处截断并触发`JSONDecodeError`。失败不是标签schema错误；缺失请求仅`Edu--003305`的43 events。流水线未发布不完整`full_results.jsonl`，未进入full audit、状态定稿或GPU；自动重试0。406个full缓存成本`$0.2133895425`，失败后服务端当日总使用`$0.239153515/$10`。
- [x] 负责人批准一次完全相同的cache-safe恢复以及仅编排层的Qwen后断点续跑保护。恢复前保存失败pipeline不可变快照（SHA-256=`8df4eeb9…0890`），新增恢复入口只接受完全一致的失败快照、合同/request SHA、407/407结果闭包和结果文件SHA；模型、prompt、route、407条请求集合与OpenRouter客户端重试逻辑均未修改。新增边界测试后项目100项unittest全部通过。
- [x] 该一次性恢复确实复用了406个完整请求缓存，并只尝试唯一缺失的逻辑请求`Edu--003305`（43 events），但约5分5秒后再次收到在约7.8 KiB处截断的外层HTTP JSON（`JSONDecodeError`：line 1427/char 7843）。缓存仍为433 total/406 full，未发布`full_results.jsonl`，未调用新增断点续跑入口，也未启动full audit、状态定稿、Paraformer、GLM、Gate D或Edu_0007。恢复manifest SHA-256=`3308d10d…a5df`，日志SHA-256=`63056fbf…dd4`；按批准边界没有进行第二次恢复，等待重新讨论方案。
- [x] 负责人批准proxy recovery v2：仅把唯一缺失的`Edu--003305`逻辑请求改经当前可达的本地国外IP代理`127.0.0.1:17890`，模型/prompt/407请求集合/1778事件和406个缓存均不变。新增严格混合路由合同只接受`direct-no-proxy-v2=406`与`environment-proxy-aware-v2=1`，旧单一路由审计行为不变；105项unittest通过。代理请求成功，新增结果费用`$0.0033565`；全量结果407/407 requests、1778/1778 events闭合，模型/签名/事件/标签/路由机械门禁全部通过，full accepted-response成本`$0.2167460425`，审计后平台当日累计`$0.244763305/$10`。
- [x] Edu_0006从Qwen后断点恢复并完整通过下游：target audio为1008 views；Paraformer严格通过1007/1008，`Edu--003598/target-ch02`因“semantic target activity returned no text/timestamps”整视图隔离；timeline闭合为1007 views/9108 events，GLM通过1007 views且0新增隔离。model-ready为2345 rows/1007 views/517424 chunks，未修改官方loader验证通过（train split=2227），全量checksum通过。
- [x] Edu_0006最终完全不变冻结Gate D通过：1007 views/499 sources，2014条mono保守证据，selection/tar闭包，`Edu--003492`未回流，source标识碰撞与exact/normalized/content exact/window/near-duplicate五类命中均为0；closure SHA-256=`2866ad93…0f87`。
- [x] 按既有批准执行Edu_0003–0006滚动清理：删除4个已通过Gate D的candidate tar和1个可由官方Edu_0006 raw确定性重建的sanitized tar，共5个目标、18,651,586,560 apparent bytes；文件系统可用字节增加11,186,417,664。删除前逐文件SHA复核、进程/软链接/训练引用检查均通过；raw、model-ready、target audio、cache、selection、manifest、checksum和报告全部保留。审计位于`project_state/storage_cleanup_edu0003_0006_20260828/`。
- [x] Edu_0007–0017共11份source contract已创建并独立交叉核对：官方冻结TSV、完成下载清单、本地tar实际500成员与metadata inventory四方在bytes/SHA/member集合/轨数/时长/事件/状态统计上完全闭合。合计5500 sources、11065 target views、250.0926575289352 target-view hours、99408 events、19148 missing states和193 WAIT；尚未启动Edu_0007。
- [x] 负责人确认Edu_0007–0017队列实现：runner新增显式可选Qwen路由且旧默认仍为`direct-no-proxy-v2`；本队列固定采用已解决7.8 KiB截断的`environment-proxy-aware-v2`；每个分片最终Gate D通过、closure/candidate SHA一致且重建输入存在后，只滚动删除该candidate tar。新增队列清理边界测试后109项unittest全部通过。
- [x] Edu_0007–0017严格串行proxy队列启动前门禁通过：本地代理与OpenRouter key status正常、服务端当日`$0.244763305/$10`、`.env`权限600、11个pipeline及queue/log路径均为fresh、无并发Qwen/GPU、数据盘约60G可用。历史队列tmux=`soulx_edu0007_0017_proxy_queue_v1`按fail-closed策略启动；后续每次结构、泄漏、API或网络异常均停止并保留证据，没有自动sanitized、拆请求或改阈值。
- [x] 经过各次异常的独立诊断、逐项负责人批准和全量不变门禁重跑，Edu_0007–0017已全部完成最终Gate D；每个分片的正式model-ready、官方loader验证、Gate D closure、Qwen/ASR/GLM审计与清理回执均保存在`reports/expansion_v1/Edu_*`及`project_state/duplexconv_expansion_edu0001_0017_v2_run_manifest.json`。
- [x] Edu_0017原始冻结Gate B/C在`Edu--009870.wav/ch0`对`candor_turn_taking/62/input.wav`产生唯一near hit（similarity=`0.6888786765`、68 frames、32 votes；冻结阈值`0.655/64/16`），四类精确/内容命中全0并在Qwen/GPU前停止。独立逐采样lag波形诊断的完整21.584秒最大绝对Pearson仅`0.04713923`、1秒最低重叠最大值`0.10223170`；同一benchmark已对11个不同来源产生Gate hit，强烈支持反复指纹误碰撞，但原始门禁失败保持权威且未被覆盖。
- [x] 负责人批准只排除完整`Edu--009870`会话。独立499-source sanitized tar审计确认parent-minus-one成员集合、tar字段和逐WAV payload SHA完全一致，剩余成员不一致数0；独立source scan与正式Gate A/B/C全部通过，正式Gate B/C共1499 candidate views、零quarantine，阈值/scorer未修改。
- [x] Edu_0017固定Qwen流程通过：connectivity/calibration/full合计费用`$0.21990488`，其中full为408/408 requests、1881/1881 events、`qwen/qwen3-235b-a22b-2507`与`direct-no-proxy-v2`各408条、费用`$0.2119686725`，机械/provenance审计通过。Paraformer与GLM均为1000/1000 views、0整视图隔离；最终model-ready为2356 rows、1000 views、530297 chunks，未修改官方loader train split=2238。
- [x] Edu_0017最终冻结Gate D通过：1000 views/499 sources、2000条mono保守记录，每个view恰好评分两次；selection/tar分区闭合，`Edu--009870`未回流，source标识与exact/normalized/content exact/window/near五类命中全0。closure SHA-256=`67a53f5d7be0fdd5342112962ab9ea3802cb4d78955a8d7c5023f5a378225c9d`。七组processed/model-ready checksums已独立复核；sanitized与Gate-D candidate两个可重建tar在现时SHA匹配闭包后删除，共10,932,070,400 bytes，官方raw/最终产物/cache/合同/报告均保留。
- [x] 新的不可变 Edu_0001–0045 v2 聚合已建立于`/root/autodl-tmp/dataset/duplexconv/aggregates/edu0001_0045_stage3_zh_v2`。冻结配置45/45输入预检通过，Edu_0018–0045的28项输入与v1配置逐项完全一致；109项全量unittest通过。聚合包含45个Parquet、22,032个source conversations、44,332个最终views、101,395 rows和22,352,510个exported 160 ms chunks（约993.444889训练有效小时），输入总计22,515,635 chunks（约1,000.694889小时）。跨分片index/view/source重复及exported/quarantined view重叠均为0；35个整view隔离、20,743个整view隔离chunks、300个整view隔离events及142,382个局部隔离chunks均完成闭包。
- [x] v2独立审计通过：101,395个Parquet index与101,395个metadata index一一对应且全局唯一，45个数据文件全部为原分片同盘hardlink，全量checksum及Gate D传递并集闭包通过。aggregate manifest SHA-256=`f6838ad77b4ea533366f7bbf079814f0db16488424e940159a5e3f11ccbe8e72`，stats SHA-256=`7401b25e2757bbd1767982e1c8f9f28e71ee35b549e3f7c84ad5d66ded189763`，checksums SHA-256=`4d3f0745056398d98058377d2368d243e6d350ba7884cd48f09bb63396681ad6`，独立audit SHA-256=`1fe0397e8be7312b47fb18efb7420a4f47f8a8d16fca86f0af98f37e3642d5aa`。
- [x] 未修改SoulX官方loader全量验证通过：固定上游提交`928b06508ed2de1344208d06fb1f6fb2ebfb1df5`且工作树干净；45个Parquet、101,395 rows与22,352,510 chunks全部重新tokenize/解析，checksum失败0、100条固定随机roundtrip通过、全局view/chunk闭包通过，官方train split长度96,325。验证报告SHA-256=`7dbf2d65620378d09a5b823b938c7a8f98199734951e09c9d934c7b290aa1ae1`。既有v1全量checksum复核仍通过，manifest/checksums SHA保持`095e18de…9bdb`/`0c246ead…b1b`，没有原地追加或覆盖。本阶段未创建训练split、未训练、未运行Table 3、未上传。

## 22. 存储方案 B 与 `work/` 依赖边界（2026-08-27）

- [x] 删除前逐文件固定目标、大小和 SHA-256，并检查进程、软链接、挂载和当前项目引用。审计记录：`project_state/storage_cleanup_b_20260827/storage_cleanup_b_predelete.json`。
- [x] 删除废弃旧根目录 `/root/autodl-tmp/soulx-duplug-stage3-cn-replacement`（106,745,069,447 apparent bytes）及 `work/` 内精确列出的 15 个可重建中间 tar（80,093,716,480 bytes）；未删除任何原始官方 tar、model-ready Parquet、聚合文件、状态/ASR/GLM 缓存、selection、manifest、checksum、门禁报告或训练/评测产物。
- [x] 数据盘可用空间增加 188,281,516,032 bytes，清理后可用 228,196,986,880 bytes；完成记录：`project_state/storage_cleanup_b_20260827/storage_cleanup_b_completion.json`。
- [x] 清理后独立执行现有 Edu_0018–0045 聚合 `sha256sum -c`，全部通过。后续官方训练读取 model-ready/聚合 Parquet 与冻结 split，不读取已删除的 15 个 tar。
- `work/` 整体仍与项目有关：其中保留的清单、哈希、API 请求/回执、选择集、门禁与闭包报告用于构造、复现和审计。已删除 tar 只是在重新运行相应泄漏/结构门禁时需要由保留的官方 raw、排除配置、model-ready selection 和冻结 Gate-D 输入重新生成，不影响当前训练输入或 v2 聚合输入。
- 废弃旧根目录不是当前项目输入，恢复需重新下载；15 个中间 tar 可确定性重建。清理没有覆盖或修改现有不可变 v1 发布。

## 23. Edu_0001–Edu_0045 官方流程续训练与评测（2026-08-29）

- [x] 负责人批准上传与训练并行、训练与GPU评测串行；训练必须使用官方commit `928b06508...`的既有补丁副本和官方Lightning `Trainer.fit()`调用链，禁止使用旧自定义optimizer loop。
- [x] 从不可变v2聚合创建seed=42、完整source-conversation分组的98/2冻结split；实际train/validation为99,334/2,061 rows、21,591/441 sources、source leakage=0、identity=`a2599191...dc6fb`。
- [x] 使用官方发布权重完成30个local optimizer step的weight-only continuation；官方起点仅按公开`total_steps=1800`作低置信度估计，新AdamW moments如实记录。第一次运行在step 10 validation后因Lightning/fsspec原生checkpoint事务临时文件写系统`/tmp`而ENOSPC，未产生可精确恢复的optimizer checkpoint；失败证据归档后仅将`TMPDIR/TEMP/TMP`改到数据盘，从相同发布权重以完全相同配置重跑并完成step 30。
- [x] 单卡microbatch=1、梯度累积=576，使一个本地optimizer step保持约576个样本的官方全局有效batch尺度；30 step实际暴露17,280个不重复首epoch样本。
- [x] 紧凑checkpoint网格固定为local step `1/2/3/5/10/20/30`；另保留最终完整Lightning checkpoint。Table 3结果不得反向改变该网格、数据、LR或规则。
- [x] 训练后已对step 0及全部紧凑checkpoint执行冻结内部validation，并只按内部validation的有限性、五状态不下降超过5pp、objective及早step tie-break确定候选；在查看Table 3汇总结果前选择local step 1，`table3_used_for_selection=false`。
- [ ] Table 3最初批准为7点顺序全测；因单点完整评测耗时较长，负责人在step 1四类汇总结果生成前批准改为预注册`endpoint-first-coarse-to-fine-v1`：固定顺序`1→30→10→5→20`，且仅当step 5的冻结`obvious_decline_trigger=true`时补测`2→3`。训练及内部validation的7点网格不变；Table 3只描述外部性能、不参与checkpoint选择。本调整不假设性能单调，也不修改EN/ZH Complete/Incomplete样本、`last-terminal-v1`、ASR、尾静音、阈值、指标或配对统计。最终报告必须披露条件省略点及可能漏掉窄幅非单调变化的限制。
- [ ] 2026-08-30经负责人确认，将Table 3执行资源层改为可配置checkpoint GPU池：`--devices 0`为单卡兼容模式，`--devices 0,1`为当前双卡模式，`--devices auto`自动使用可见GPU；每GPU同时至多一个checkpoint，单checkpoint内四类仍串行，只有主线程写全局manifest/index。该改动不改变冻结顺序、触发条件、checkpoint、样本、规则、阈值、指标、ASR或推理runtime。此前step10 ZH Incomplete因继承的本地代理`127.0.0.1:17890`不可达而在ModelScope辅助模型初始化失败，原失败证据已进入`failure_history`；恢复仅清除worker代理环境并走已验证可达的ModelScope直连，已有完整产物按身份校验复用。双GPU初始任务为GPU1恢复step10剩余类别、GPU0执行step5，空闲GPU再领取step20；若step5触发，则step2/3并行。
- [ ] 新建Edu_0001–0045专属MD/HTML/audit JSON报告，包含数据规模与处理、标签来源与映射、split、起点/LR/optimizer、实际样本暴露、内部validation、Table 3曲线、统计不确定性、限制、旧Edu_0018对比和全部证据SHA。

## 24. 本阶段并行与资源边界

- 百度上传/远端验证仅使用网络与数据盘顺序读取，可与GPU训练并行；Table 3和训练不得同时占用同一GPU。
- split、训练checkpoint、W&B offline与Table 3结果均写入数据盘；系统盘只保存配置、报告、状态清单及软链接。预计新增5–7 GiB，API费用为0。
- 任何split身份漂移、上游/评测runtime变脏、NaN/Inf、optimizer step跳跃、CUDA OOM、磁盘不足或方法变化均须fail-closed并重新报告。

## 25. 最终完成条件

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

## 26. Complete/Incomplete 平衡消融与 Table 2 后续计划（2026-09-12）

状态：**Table 2 暂缓；平衡数据与 B/C/D 配置构造已于 2026-09-12 完成，训练和新评测尚未授权。** 后续训练、评测、依赖/模型下载或方法变化仍须按`agent_governance/EXECUTION_APPROVAL_PROTOCOL.md`重新披露并获得负责人明确确认。

### 26.1 问题与已确认事实

Edu_0001–Edu_0045 首轮官方 Lightning 续训练的 Table 3 显示：随着 local step 增加，Complete 准确率快速上升，而 Incomplete 明显下降。该现象至少有两个可检验假设：

1. 训练数据中的 Complete/Incomplete 监督暴露不平衡；
2. 官方公开配置中的 Complete/Incomplete loss rate 进一步偏向 Complete。

当前冻结数据的只读统计为：

```text
全量终态事件：
  Complete   217,725
  Incomplete  86,452
  比例约 2.52:1

冻结 train Parquet（99,334 rows）：
  Complete token   210,359
  Incomplete token  78,037
  比例约 2.70:1

含 Complete 的窗口   78,559（79.09%）
含 Incomplete 的窗口 47,182（47.50%）
head 激活窗口比例约 1.67:1
```

训练窗口不是单标签样本，其组合分布为：

```text
Complete + Incomplete  40,261
Complete-only           38,298
Incomplete-only          6,921
两者均不含              13,854
合计                    99,334
```

官方公开配置及本次首轮配置实际使用：

```yaml
user_complete_loss_rate: 0.24
user_incomplete_loss_rate: 0.13
```

99,334 个训练 index 全部以`duplexconv`开头，没有任何一条以`fe`开头，因此`*_loss_rate_switched`在本数据上没有生效。由于官方实现对每个状态 head 在单行中的有效 token 先取 mean cross entropy，训练平衡应优先以“每个 head 被激活的窗口暴露次数”为主口径，同时附带报告 token 数分布。将窗口暴露与 loss rate 简单相乘得到的 Complete/Incomplete 影响比例约为 3.07:1，但该值只是诊断性近似，不能表述为实际梯度范数或因果证明。

### 26.2 实验目的和诚信边界

本轮目标是研究 Complete/Incomplete 平衡对论文指标的影响，并寻找更高的 Table 3 四类宏平均结果，不以工程偏好（例如宁可多回答）作为选择标准。

由于首轮 Table 3 结果已经被查看，并直接用于提出采样和权重调整假设，从本节开始：

- Table 3 必须标记为**开发/调参指标**或**事后假设驱动消融**；
- 可以使用 Table 3 比较和平衡方案，但不得再把新结果表述为完全独立、无偏的最终测试；
- 不得修改 Table 3 样本、标签、ASR、解码规则、尾静音、阈值、指标或失败样本处理；
- 不得只保留高分 checkpoint，所有预注册点、失败和异常均需保存；
- Table 2 在候选模型运行前保持封存；如果其协议先用官方模型完成复现并冻结，之后只评测预先选定的候选且不根据 Table 2 回调参数，则可作为一次确认性系统级外部评测；一旦根据 Table 2 结果继续调参，Table 2 也必须降级为开发集。

### 26.3 平衡训练视图

不修改、覆盖或重新标注现有 99,334-row train Parquet，也不改变现有 2,061-row validation。新建独立、不可变的 balanced-CI train artifact，按状态 head 是否在窗口中出现来实现 1:1：

1. 保留全部 40,261 条 Complete+Incomplete 窗口；
2. 保留全部 6,921 条 Incomplete-only 窗口；
3. 从 38,298 条 Complete-only 中用固定 seed=42 确定性抽取 6,921 条；
4. 保留全部 13,854 条不含 Complete/Incomplete 的窗口，以维持 Idle、Non-idle、Backchannel 和文本监督；
5. 最终得到 67,957 条唯一窗口，其中 Complete-head-active 与 Incomplete-head-active 均为 47,182，严格 1:1；
6. 不做有放回过采样，不人为复制 Incomplete-only 行，不从 validation 回流任何 source/row；
7. Complete-only 抽取按 shard、`source_ntrack`及是否含 Backchannel 分层，并用 largest-remainder 方式分配精确配额，避免单一 shard 或声道类型被非比例删除；
8. 新 manifest 必须记录输入/输出 Parquet SHA-256、row identity、source identity、全五状态 token 数和 head-active 行数、每 shard/ntrack 分布、抽样 seed/算法/代码哈希，以及 train/validation source leakage=0；
9. 使用未修改的官方 Dataset、collator、DataLoader 和`shuffle=True`。因总训练只读取首 epoch 的一部分，运行审计还必须记录实际消耗窗口中五状态的 head-active 次数和 token 数，不能只报告完整 balanced artifact 的理论分布。

计划输出位置：

```text
/root/autodl-tmp/dataset/duplexconv/splits/
  edu0001_0045_stage3_zh_v2_seed42_group98_2_balanced_ci_v1/
```

现有原始 split 和 validation 保持字节不变。若上述分层抽样无法同时满足 67,957 rows、47,182/47,182 head-active 闭包或 source leakage=0，必须停止并重新讨论，不能静默改用过采样。

### 26.4 Equal-loss 配置

官方训练实现已经支持从 YAML 读取每个状态的 loss rate，因此**不需要修改`state_prediction_model.py`或官方训练代码**。为了只改变 Complete/Incomplete 的相对权重，同时保持两者总权重`0.24+0.13=0.37`不变，equal-loss 实验固定为：

```yaml
user_complete_loss_rate: 0.185
user_incomplete_loss_rate: 0.185
```

其余 text/EOS/Idle/Non-idle/Backchannel 权重、LR、scheduler、LoRA、有效 batch、seed 和精度均不变。不得改成`0.24/0.24`或`0.13/0.13`，因为那会同时改变终态任务相对其他 head 的总强度，无法形成单变量对照。`enable_switch_loss_rate`保持首轮配置不变；所有 DuplexConv index 必须再次确认不会触发`fe` switched 分支。

### 26.5 2×2 消融设计

| 组别 | Train 数据 | Complete/Incomplete 权重 | 状态 |
|---|---|---:|---|
| A | 原始冻结 train | 0.24 / 0.13 | 已完成的首轮基线，只读复用 |
| B | 原始冻结 train | 0.185 / 0.185 | 30-step训练及0/5/10部分内部validation已完成 |
| C | balanced-CI train | 0.24 / 0.13 | 30-step训练及0/5/10部分内部validation已完成 |
| D | balanced-CI train | 0.185 / 0.185 | 30-step训练及0/5/10部分内部validation已完成 |

B/C/D 必须各自从同一个官方发布 checkpoint 开始，不得从 A 或其他候选 checkpoint 接续。共同固定：

```text
官方上游基准 commit：928b06508ed2de1344208d06fb1f6fb2ebfb1df5
训练调用链：finetune.py -> Lightning Trainer.fit() -> training_step() -> configure_optimizers()
已批准 runtime：SoulX-Duplug-928b065-official-continual-v1
local batch：1
accumulate_grad_batches：576
有效 batch：576
seed：42
local optimizer steps：30
checkpoint：1 / 2 / 3 / 5 / 10 / 20 / 30
起点估计：1800（低置信度，不宣称精确恢复 optimizer 现场）
```

NaN 空 head、严格 checkpoint 加载、冻结 split 审计、scheduler 起点和 compact checkpoint 补丁保持不变。不得使用旧自定义 optimizer loop。A/B/C/D 均使用独立配置、run ID、checkpoint、日志、validation 和评测目录，不覆盖首轮产物。

候选配置与产物的建议命名：

```text
configs/duplexconv_edu0001_0045_stage3_official_continual_weight_equal_v1.yaml
configs/duplexconv_edu0001_0045_stage3_official_continual_sample_balanced_v1.yaml
configs/duplexconv_edu0001_0045_stage3_official_continual_joint_balanced_v1.yaml

/root/autodl-tmp/dataset/duplexconv/training/
  edu0001_0045_official_continual_weight_equal_v1/
  edu0001_0045_official_continual_sample_balanced_v1/
  edu0001_0045_official_continual_joint_balanced_v1/
```

### 26.6 Checkpoint 选择和 Table 3

1. 原始计划是 B/C/D 训练后对 step 0及`1/2/3/5/10/20/30`运行同一冻结内部 validation；负责人在执行前将本阶段外部精确内部-validation范围收缩为`0/5/10`，实际已经按该变更完成。validation 保持原始自然分布，未制作 balanced validation；本结果只能作为部分轨迹。
2. Table 3 使用现有已审计 inference runtime 和完全相同的四类数据，明确标记为开发/调参评测。
3. 为完整比较非单调轨迹，B/C/D 原则上评测全部七个 checkpoint；若为了节约时间希望改用 coarse-to-fine，必须在任何候选 Table 3 结果产生前另行冻结统一规则，并对三组完全一致执行。
4. 开发集主选择指标预先固定为四类等权宏平均：`mean(EN Complete, EN Incomplete, ZH Complete, ZH Incomplete)`；同时完整报告每类准确率、EN/ZH macro、相对 step 0 的 paired change 和置信区间。
5. 不允许通过修改 Complete/Incomplete 决策阈值、终态选择规则或样本排除来提高结果。

### 26.7 Table 2 仓库审计与复现门禁

第三方评测实现：

```text
https://github.com/time-northern/soulx-duplug-eval.git
计划固定 commit：ce779713513f049b020019c8eab7d71f1080131d
```

服务器当前没有可用于该仓库的 GitHub SSH key；仓库可通过 HTTPS 读取，因此后续 clone 使用 HTTPS。代码放系统盘独立`third_party`目录，只读固定；若必须补丁，另建 runtime 副本并记录完整 diff，禁止直接修改 clone。

第一道门禁只 clone 和静态审计，不安装依赖、不下载模型、不运行 GPU：

- 对照 SoulX 论文 Table 2、官方 Full-Duplex-Bench 和数据目录检查样本集合、场景、时序及指标；
- 检查是否存在按模型名分支、读取答案、paper-target 调参、失败样本过滤、选择性重试、手工覆盖预测、改变音频或时序等问题；
- 核对 Pause Handling TOR、Turn Taking TOR/RL、User Backchannel RsR、User Interruption TOR/RpR/SL/RL、Overall Score/Latency 的公式和方向；
- 核对论文系统链：SoulX-Duplug + Qwen2.5-7B-Instruct + IndexTTS-1.5，英文 SenseVoice、中文 Paraformer，模拟在线流式推理；
- 生成依赖、模型、显存、磁盘、网络和预计耗时清单；
- 输出独立代码审计报告；发现可疑实现时停止，不运行任何分数。

当前本地 Table 2 数据事实：

```text
英文 Full-Duplex-Bench v1/v1.5：9 个 ZIP，705,372,348 bytes，SHA-256 与 ZIP 完整性复核通过
中文 Full-Duplex-Bench：已解压，约 1.2 GiB
```

静态审计通过后，第二道门禁是先运行官方发布 SoulX 权重的 Table 2 基线。复现容差、随机种子、失败重试、超时、样本分母和指标聚合必须在运行前冻结；若官方基线与论文不能在预注册容差内对应，先诊断评测实现或环境，不评测 B/C/D。

Table 2 是含 LLM/TTS 的系统级评测，不能替代 Table 3 对状态预测模型的直接诊断。候选选择完成后，只将预先选定的一个模型运行一次冻结 Table 2，并禁止根据其结果返回修改采样、权重、checkpoint 或推理规则；失败或下降同样如实报告。

### 26.8 执行顺序和逐阶段审批

后续严格按以下顺序，每一项开始前分别披露并确认：

1. **Table 2 repo 静态审计**：clone固定commit、只读审计、写审计报告；
2. **Table 2 官方基线复现**：根据审计结果建立独立Conda环境、下载所需模型、冻结协议并运行官方权重；
3. **平衡数据与三份配置构造**：创建balanced-CI artifact、B/C/D YAML和机械审计，不训练；
4. **B/C/D官方流程训练**：三组独立30-step训练及内部validation；
5. **Table 3开发评测**：按预先确认的完整网格或统一coarse-to-fine规则运行；
6. **一次性Table 2候选确认**：按冻结选择规则选定一个候选并运行，不回调参数；
7. **最终报告**：生成MD、HTML和audit JSON，旧报告保持不变。

任何阶段发现依赖、方法、数据、耗时或资源与批准方案显著不同，必须停止并重新确认，不能自动进入下一阶段。

### 26.9 当前资源估算与存储边界

2026-09-12 只读快照：

```text
GPU：1 × NVIDIA GeForce RTX 4080 SUPER 32 GiB
GPU占用：空闲
系统盘可用：约 4.5 GiB
数据盘可用：约 101 GiB
```

- 三个30-step训练按首轮实际耗时估计，单GPU串行约4小时；内部validation合计约1.5小时。
- 首轮Table 3单checkpoint四类约需2小时GPU时间；若B/C/D全部评测七点，最坏约42 GPU小时。两卡checkpoint池可把墙钟时间大致减半，但开始前必须按实际GPU数量重新估算。
- 三组训练/checkpoint/日志预计新增约15–21 GiB；不得写系统盘，系统盘只放小型代码、配置、计划和报告。
- Table 2所需Qwen2.5-7B、IndexTTS和环境的新增磁盘量、显存可行性及总耗时在repo静态审计后确定，未审计前不承诺可在当前32 GiB单卡直接运行。
- 所有Conda环境、大模型、cache和运行产物放数据盘，并以项目内软链接引用；不得覆盖现有环境或模型。
- 预计付费API费用为0；若仓库实现要求任何外部付费API，必须停止并另行披露费用与预算，不能自动调用。

### 26.10 计划产物与最终报告要求

建议新增但不得覆盖首轮正式报告：

```text
evaluation_reports/soulx_table2_eval_repo_audit_v1.md
evaluation_reports/duplexconv_edu0001_0045_ci_balance_ablation_v1.md
evaluation_reports/duplexconv_edu0001_0045_ci_balance_ablation_v1.html
evaluation_reports/duplexconv_edu0001_0045_ci_balance_ablation_v1_audit.json
evaluation_reports/soulx_table2_reproduction_and_candidate_v1.md
evaluation_reports/soulx_table2_reproduction_and_candidate_v1.html
evaluation_reports/soulx_table2_reproduction_and_candidate_v1_audit.json
```

最终会议材料必须包含：原始与balanced数据的完整分布、抽样定义和实际训练暴露、所有loss rate、A/B/C/D单变量归因、全部step轨迹、Table 3已作为开发集的披露、Table 2官方基线复现门禁、候选一次性Table 2结果、失败/限制、运行时间/显存/磁盘、代码/数据/模型/结果SHA-256，以及“不根据Table 2结果回调参数”的确认。

### 26.11 平衡数据与配置构造执行记录（2026-09-12）

- [x] 负责人决定暂缓 Table 2 实现，批准先执行第 3 阶段“平衡数据与三份配置构造”；该决定不代表 Table 2 门禁通过。
- [x] 新增`src/duplexconv_stage3/ci_balance.py`、`scripts/create_ci_balanced_training_view.py`和`tests/test_ci_balance.py`。选择算法按`shard × source_ntrack × Backchannel presence`分层，以最大余数法分配 Complete-only 配额，层内用`SHA256(seed + NUL + index)`稳定排序；不依赖 Python RNG 版本。
- [x] 4 个新采样测试和 6 个既有 continual-training 测试通过。构造前验证父 train/validation Parquet、父 manifest 和 45 份 metadata 哈希；临时目录内完成闭包后才原子发布，没有覆盖父 split。
- [x] balanced-CI train 实际为 67,957 rows、15,335,148 个 160 ms chunk、681.562133 h；Complete-head-active/Incomplete-head-active 均为 47,182。状态 token 为 Idle 9,673,940、Non-idle 5,062,896、Backchannel 391,372、Complete 128,903、Incomplete 78,037；Complete/Incomplete token 仍约 1.652:1，不能表述为 token 1:1。
- [x] 保留 40,261 条 Complete+Incomplete、6,921 条 Incomplete-only、13,854 条 neither；从 38,298 条 Complete-only 中选择 6,921 条并删除 31,377 条。最终 67,957 个 index 唯一，均来自父 train。
- [x] validation 保持 2,061 rows，字节级 SHA-256 与父产物一致；balanced split 实际 train/validation source leakage=0，split identity=`c63fbb08e71b6eba68a22a57a3ee22cadb38611d63f69bb29b0f3c6b98314177`。
- [x] 新增 B（原数据+0.185/0.185）、C（balanced数据+0.24/0.13）、D（balanced数据+0.185/0.185）三份 YAML。自动字段差异门禁确认：除批准的数据路径、两项权重和独立实验/输出标识外，所有字段与 A 相同；没有修改官方训练 runtime。
- [x] B/C/D 分别通过补丁版官方`State_Prediction_DataModule.setup("fit")`：B 为 99,334/2,061 rows，C/D 为 67,957/2,061 rows；三组 manifest/hash/row identity/source leakage 门禁通过，`fe` switched 分支触发行数均为 0。
- [x] 首次 loader 门禁发现 Hugging Face 临时 Arrow cache 已达到约 1.3 GiB，明显超过原先只按最终数据估计的 100–150 MiB，因此主动终止并清理。负责人重新批准峰值 3 GiB 后重跑，观测最大约 1.4 GiB；门禁结束后临时缓存自动清理，无残留。
- [x] 本阶段未加载模型、未创建 optimizer/Trainer、未使用 GPU、未调用付费 API、未启动训练或 Table 3。最终 balanced artifact 约 96 MiB；执行后数据盘仍约 101 GiB 可用。
- [x] 创建持续更新的会议总结`evaluation_reports/duplexconv_edu0001_0045_abcd_experiment_summary.md`。A 的既有结果、B/C/D 冻结设计、数据/配置身份和诚信边界已落盘；后续每阶段完成后更新，结果齐备再生成 HTML。

关键 SHA-256：

```text
balanced train     cf8e4a59d2aef793e81bac0cf734b4dd75a8f59371457d3838903f656675c432
balanced validation 6bc82ad10fd7c4e4baa0b0902faec589029e6673e70cb802ca699680fbd017fb
balanced manifest  b7b162df52ea1ebce29c129d1b7a131163c4f657254eb562627ee89669e8ca73
B config           6c659bbf97e860cd668eb5ed79e77e9160e27f562fee4a17a723c4e20dd5a569
C config           b14e91e7bbe3ad986641c60b7c781cedf83a57899f76ea4a0724237824613557
D config           513a439e51717e10c286b9438ca3e8d671e30075234334d04e39f2dd53dc64b9
```

### 26.12 B/C/D 官方训练与部分内部 validation（2026-09-13，已完成）

- [x] 负责人批准单 GPU 严格按 B→C→D 执行；每组从相同官方发布权重独立运行官方 Lightning 30 local optimizer step，禁止从 A 或另一组 checkpoint 接续。
- [x] 负责人将本阶段外部精确内部-validation网格调整为`0/5/10`。训练配置仍保存 compact checkpoint`1/2/3/5/10/20/30`，官方训练内置 validation 仍在 step`10/20/30`运行；本阶段不额外评测`1/2/3/20/30`，也不得把两点 continuation 结果写成完整七点选择结论。
- [x] 新增纯调度/审计辅助`src/duplexconv_stage3/official_ablation_queue.py`、`scripts/run_official_ablation_training_queue.py`和`tests/test_official_ablation_queue.py`；它们不实现 forward/backward/optimizer/scheduler/AMP/梯度累积，只对子进程调用现有补丁版官方`finetune.py`并校验证据。
- [x] 15 项相关测试通过。只读 preflight 验证官方基准 commit、A 训练时 tracked diff与逐文件哈希、基础权重、A/B/C/D机械差异、两个 split、输出路径和空间门禁；启动前数据盘可用 100.43 GiB。
- [x] 2026-09-13 启动`tmux=soulx_abcd_train_bcd_v1`；实时来源为`/root/autodl-tmp/dataset/duplexconv/training/edu0001_0045_abcd_queue_v1/orchestration_manifest.json`，启动阶段为 B step 0 validation。
- [x] B：step 0 → 30-step训练 → step 5/10 → 部分内部-validation索引；30 optimizer updates、17,280 sample exposure、七个 compact checkpoint tensor audit 全通过。step 5/10 accuracy=`0.8056413159/0.7341992514`，Complete=`0.5721542415/0.6716693641`，Incomplete=`0.3680297416/0.4876084286`。
- [x] C：step 0 → 30-step训练 → step 5/10 → 部分内部-validation索引；30 optimizer updates、17,280 sample exposure、七个 compact checkpoint tensor audit 全通过。step 5/10 accuracy=`0.8050862322/0.7325262507`，Complete=`0.5795428354/0.7007619522`，Incomplete=`0.3537794318/0.4479553923`。
- [x] D：step 0 → 30-step训练 → step 5/10 → 部分内部-validation索引；30 optimizer updates、17,280 sample exposure、七个 compact checkpoint tensor audit 全通过。step 5/10 accuracy=`0.8054233509/0.7319944168`，Complete=`0.5301316139/0.5529900757`，Incomplete=`0.4250309811/0.6226765823`。
- [x] 三组 step 5/10 都因至少一个状态 head 相对 step 0 下降超过 5pp 而未通过保护门禁；部分索引均返回 step 0。最早且最大的共同问题是 step 5 Non-idle 已下降约 16.25–16.37pp。该结果不得写成完整七点选择结论，也不代表 Table 3 结果。
- [x] 队列从`2026-09-12T16:10:09Z`运行至`2026-09-12T21:13:02Z`，约5小时2分53秒；B/C/D 峰值CUDA allocated约17.91GiB。队列、tmux及相关进程均已退出，GPU空闲；Table 3 与 Table 2 均未运行，付费API费用为0。
- [x] 持续会议总结已更新：`evaluation_reports/duplexconv_edu0001_0045_abcd_experiment_summary.md`。后续是否运行开发性 Table 3、checkpoint范围和统一顺序须另行披露并确认。

失败策略：任一身份、NaN/Inf、optimizer step、有效 batch、checkpoint、数据泄漏或磁盘门禁失败即停止整个队列；保留 live manifest、子任务日志与已产生证据，不自动重试、恢复、删除、覆盖或进入下一组。

### 26.13 B/C/D Step 5/10 开发性 Table 3（2026-09-13，已完成）

- [x] 负责人批准 B/C/D 只评测 step 5、10；当前实际有三张RTX 4080 SUPER，批准改为三卡并行。
- [x] 冻结三卡波次：`baseline@GPU0+B5@GPU1+C5@GPU2`；基线精确门禁通过后运行`D5@GPU0+B10@GPU1+C10@GPU2`；最后`D10@GPU0`。四类在单个checkpoint内串行。
- [x] 新增只负责调度、身份和失败门禁的`scripts/run_abcd_table3_step5_10.py`及实验配置；没有改动隔离Table 3 runtime、官方upstream、推理、ASR、终态规则、阈值或指标。5项新测试及4项既有GPU池测试通过。
- [x] 第一次启动在模型初始化和样本推理前被英文数据身份门禁拦截。617个WAV的名称、大小和SHA-256与冻结基线全部一致；原因是目录重建改变了未排序`os.walk`的目录项顺序。原始ZIP完整性通过，临时解压同时精确复现 Complete identity=`4a8a2823...ab4f5e`及Incomplete identity=`e61a1351...a9010339`。
- [x] 以按ZIP顺序验证通过的目录接替评测路径，旧目录保留为70MiB可恢复备份；首次失败目录与日志归档。该修复没有改动任何音频字节或样本集合。
- [x] 正式tmux任务于`2026-09-13T02:29:31Z`启动，当前首轮为baseline/B5/C5三卡并行。live manifest：`/root/autodl-tmp/dataset/soulx_duplug_eval/table3_abcd_step5_10_v1/orchestration_manifest.json`。
- [x] 新鲜step0基线精确匹配`251/318、268/299、263/300、241/300`并通过证据审计。
- [x] B/C/D step 5、10全部四类评测与continuation evidence gate完成。
- [x] 更新持续会议总结，并生成最终MD、HTML和audit JSON；完整披露本轮属于事后假设驱动的开发性Table 3。

中断恢复补充（2026-09-13）：新鲜基线已精确匹配冻结A并通过证据审计，B5/C5也已完成。服务器在第二轮D5/B10/C10的EN Complete中途关闭；三份`status=running`半成品均未作为结果。恢复时发现控制器把JSON重载后的字符串step键与内存整数键直接比较的缺陷；补丁仅规范化JSON身份并允许已知旧/新控制器哈希迁移，11项相关测试通过，迁移写入live manifest。三份半成品已分别留证归档并从类别头部重跑，首轮完整输出按身份校验复用。

第二次服务器中断补充：停止监控最后一次仍观测到D5/B10/C10三项同时活跃，因此不是单GPU停止条件。中断前三组前三类均完整，只有ZH Incomplete处于未完成状态（D5 0条、B10 59条、C10 72条）；再次恢复只归档并重跑这三份半成品，其他类别全部按身份复用。替代监控`stop_at_one_active_receipt_resume2.json`继续在连续三次观测到仅一个active assignment时停止主任务。

单GPU恢复补充：替代监控已按要求在控制器进入D10单任务时停止，D10持久记录为0。负责人随后将容器缩容至单张RTX 4080 SUPER并明确要求继续；`2026-09-13T06:55:44Z`以同一冻结身份、`--devices 0 --resume`恢复，旧单任务停止监控不再启用。B5/B10/C5/C10/D5全部复用，仅运行D10四类及证据门禁，完成后生成最终MD/HTML/audit。

单GPU停止结果：D5/B10/C10完成四类与证据门禁后，调度进入D10单任务；监控于`2026-09-13T06:32:57Z`按负责人指令停止主tmux，D10持久记录为0。当前B5/B10/C5/C10/D5完整，D10未评测，无tmux/评测进程且GPU空闲。四类宏平均分别为基线84.14%、B5 81.19%、B10 71.69%、C5 80.10%、C10 70.50%、D5 81.74%。保持暂停，未获新指令不得恢复D10，完整网格未齐前不生成最终MD/HTML/audit。

最终完成记录：负责人将容器缩容为单GPU并明确恢复后，D10于`2026-09-13T08:40:48Z`完成四类、证据门禁和最终索引，控制器正常退出。D10为EN Complete 90.88%、EN Incomplete 54.85%、ZH Complete 82.00%、ZH Incomplete 75.00%，四类宏平均75.68%。完整消融中D5为最佳新候选（81.74%，相对基线-2.41pp）；联合平衡在step5/10均优于A/B/C同step，但没有新候选超过官方step0基线。最终产物为`evaluation_reports/duplexconv_edu0001_0045_abcd_experiment_summary.md/.html/_audit.json`，audit status=`passed`；Table 2未执行，训练与评测付费API费用为0。
