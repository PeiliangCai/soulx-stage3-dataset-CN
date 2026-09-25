# DuplexConv expansion benchmark leakage guard

状态：英文 benchmark 官方资产已获授权、下载并完成逐 ZIP 校验；音频指纹实现和数值阈值已在未下载、未查看、未打分 `Edu_0019` 前完成最终冻结。

## 1. 禁入范围

以下资产及其任何派生文本、标签、状态、伪标签或音频片段不得进入训练、内部验证、Qwen prompt 示例或 shard 选择：

- English Easy Turn Testset（Table 3）；
- Chinese Easy Turn Testset（Table 3）；
- Full-Duplex-Bench（Table 2）；
- 上述 benchmark 的 ASR cache、state trace、logits、预测和 gate 结果。

Table 3 结果只用于当前冻结模型的最终评估，不得用于反向选择某个 DuplexConv shard、标签映射、LR 或处理规则。扩展候选只能依据 DuplexConv metadata、内部 train/validation 和另建开发集统计。

## 2. 分阶段 gate

### Gate A：下载前 metadata 预检

- 固定官方 tar 文件名、字节数、校验值和成员范围；
- 检查 source ID、文件名、公开来源说明和已知 provenance 是否与 benchmark 重合；
- 只允许把候选标为 `metadata_clear`，不得在没有音频时宣称无泄漏。

### Gate B：下载后精确音频检查

- 对原始多声道音频和每个 target view 生成统一 16 kHz PCM identity；
- 与 benchmark 每条音频的统一 PCM identity 做精确比较；
- 对固定窗口生成可抵抗容器、采样率和声道差异的滑窗音频 fingerprint；
- 任一精确或局部窗口命中都直接 quarantine，禁止进入后续流程。

### Gate C：近重复筛查

- 使用冻结的音频指纹/embedding 方法做候选检索，阈值在查看匹配结果前冻结；
- 使用规范化 transcript 高重合只作为人工审计线索，不能单独证明音频重复；
- 保存所有候选对、分数、判定和理由，不得只保存被排除结果。

### Gate D：model-ready 闭环

- 最终 train/validation 的 source IDs、PCM identities 和窗口 fingerprints 必须再次与 benchmark denylist 比较；
- 输出 `source_leakage_count=0`、`exact_audio_match_count=0` 和 `window_match_count=0` 才能进入训练；
- gate 使用的 benchmark labels 和模型预测不得传入数据构造代码。

## 3. 审计产物

音频到位后的正式实现至少要保存：

- benchmark identity manifest；
- 每个候选 tar/source/target view 的 identity manifest；
- exact/window/near-duplicate 比对结果；
- quarantine 清单；
- gate 配置、代码 SHA-256、输入清单 SHA-256 和最终结论。

任何 gate 规则修改必须在查看新候选匹配结果前披露并重新确认。

### 3.1 首次校准失败与修订候选（尚待确认）

首次校准在未下载、未查看、未打分 `Edu_0019` 的前提下完成。实现采用最少 24 个 Chromaprint 对齐帧，并把增益、12 kHz 重采样、首尾裁剪和 30 dB 加噪均作为硬正控制；同时错误地把 benchmark 内“数字样本 ID 不同”的最高匹配当作负控制。结果为：

```text
positive retrieval          192/192
positive minimum            0.6776785714
apparent unrelated maximum  0.9915865385
recommended threshold       1.015
calibration_passed          false
```

失败原因不是据 `Edu_0019` 结果调阈值。Full-Duplex-Bench 的多个合成场景会跨不同数字 ID 复用 context、clean input 和其他音频成分，这些高分是应当被 denylist 捕获的真实局部复用，不能当作负样本；24 帧短对齐也会放大偶然高分。部分短/静音占比较高的语音在叠加随机噪声后 Chromaprint 平均位分数明显下降，而当前门禁承诺的是容器、采样率、声道、增益与裁剪鲁棒性，不承诺任意加噪不变性。

在不修改代码、不写入阈值、不接触 `Edu_0019` 的只读预检中，修订候选为：

- 最少对齐帧从 24 提高到 64（约 9 秒）；
- 硬正控制只使用增益、12 kHz 重采样和首尾裁剪；30 dB 加噪仅保留为诊断；
- 负控制固定为 `candor_pause_handling/input.wav` 内数字 ID 不同、统一 PCM 身份不同的自然片段；不把合成场景的跨 ID 复用误称负样本；
- 阈值公式保持 `ceil(max(0.80, max_negative+0.02)*200)/200`。

只读预检得到 144/144 正控制召回、正控制最低 `0.9555921053`；2,472 个负控制最高 `0.6376953125`，候选冻结阈值为 `0.800`，最低正控制安全间隔为 `0.1555921053`。该修订必须由项目负责人确认后才能写入正式冻结配置、重新运行校准或打分 `Edu_0019`。

项目负责人理解查重目的后于 2026-08-22 明确要求继续执行。最终代码重新校准结果通过并冻结为：

```text
minimum_aligned_frames                 64
near_duplicate_similarity_threshold    0.800
same-source controls                   144/144 retrieved
same-source minimum similarity         0.9555921053
different-source control count         2472
different-source maximum similarity    0.6376953125
positive p01 minus threshold margin    0.1718675492
```

冻结身份：

```text
benchmark identity manifest sha256  a27380aff2d625b24c441a9fc47bc2a2285993b60d230eb6f9a6fd7fede9ec7d
final calibration result sha256     cdd3f00e4ce9d07a1d0c13145e494eb792c28c468389a292e1e175a8628d5032
implementation sha256               e5b36c76a6e5fdaf801c609df809855c53874af17c1475d2ba49d9d8b140c536
frozen config                       configs/benchmark_audio_leakage_frozen_v1.json
```

评分入口会同时校验 benchmark manifest、实现代码和全部核心参数；任一漂移直接失败。首次失败校准、修订中间校准和最终校准均保留，未覆盖历史证据。

## 4. 英文 Full-Duplex-Bench 固定资产（2026-08-22）

英文 Full-Duplex-Bench 官方仓库固定到 `DanielLin94144/Full-Duplex-Bench@3e799c45a045256f47d5f1c9cda90157e2d2ec9e`。仓库只保存说明并链接 Google Drive；音频不在 Git tree 中：

- v1.0：5个 ZIP、727个样本目录；
- v1.5：4个 ZIP、499个样本目录；
- 合计：9个 ZIP、1,226个样本目录、705,372,348 bytes（约0.657 GiB）；
- 官方说明：https://github.com/DanielLin94144/Full-Duplex-Bench/tree/3e799c45a045256f47d5f1c9cda90157e2d2ec9e/v1_v1.5/dataset

项目负责人已授权下载用于泄漏检查。实际下载于 `2026-08-22T10:33:51Z` 开始、`2026-08-22T10:36:44Z` 完成，唯一原始副本位于：

```text
/root/autodl-tmp/dataset/soulx_duplug_eval/raw/full_duplex_bench_en_v1_v1_5/
```

验收结果：

- 9/9 ZIP，总字节数 `705372348`；
- `checksums.sha256` 逐文件复算全部通过；
- 9/9 `unzip -tq` 全部通过；
- `source_manifest.tsv` 与系统盘固定清单内容及 SHA-256 一致；
- 下载使用服务器 AutoDL 代理，不依赖项目负责人的本地电脑；
- 未保存完整解压副本，后续从 ZIP 流式读取 WAV 构建 denylist。

Google Drive 内容不具备 Git blob 身份，因此以下实际 SHA-256 是本项目固定身份；任一内容漂移都必须停止门禁：

```text
7e994d4792a668fee89480d6cd06c606ddce157041c2b1b60ed2e63bf2628969  v1.0/candor_pause_handling.zip
a0764ebb98584d07876a194167be48a4f9aa7d309f454c822fe63e4599207ee2  v1.0/candor_turn_taking.zip
e4ce7903fbc18c0abc49a262d07df2fbd2fd17dc8bf7ad39f1f591fcd6d8b26f  v1.0/icc_backchannel.zip
593859a99fe27455caeb6e5de0cce8d392cf48aa4a62f9c05e66d67ab8d34797  v1.0/synthetic_pause_handling.zip
279d258e16c5b09a4ed1abd19ecf0c992935968a55d732d151218b82c35480de  v1.0/synthetic_user_interruption.zip
4dd2c23b3e03fe01ae36e2572bff1bc2679469f419741fc259f99fbc32d0a1d1  v1.5/background_speech.zip
2029a0687190a3bc9a9ec65ec620f484d831cb9dcdacb025a46e16a39cf62bc1  v1.5/talking_to_other.zip
abe3f1765b542e1b3c47b90f2d18a24d5203e392a78f2b7c90474fac06a69bb9  v1.5/user_backchannel.zip
0560418f1d02e9d73a6136b24efac803daa63a2056d3f0d539a901fe9a70912e  v1.5/user_interruption.zip
```
