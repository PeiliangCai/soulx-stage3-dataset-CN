# SoulX Stage 3 与 benchmark 运行环境

更新时间：2026-08-21
状态：已完成部署诊断环境和独立 Table 3 审计环境；已完成全量候选协议基线，证据审计通过但数值门禁失败。

## 1. 决策与用途

本项目保留现有共享 Python 3.10 Conda 环境用于 Stage 3 和部署诊断；Table 3 候选复现另建独立环境，避免为匹配论文复现依赖而污染已验收环境。用途包括：

1. SoulX 官方推理服务与 Bilingual Easy Turn benchmark；
2. SoulX Stage 3 状态预测续训练；
3. 中文 Paraformer 与英文 SenseVoice 级联 ASR。

不再逐项安装官方环境快照中的全部 404 个包。当前执行路径未使用的 vLLM、TensorRT、Gradio/Jupyter、ONNX Runtime、DeepSpeed、bitsandbytes、diffusion/TTS 等依赖不安装；实际运行遇到新的必要导入时，再做最小增量补充。

## 2. 存储布局

```text
实际环境：
/root/autodl-tmp/conda_envs/soulx-duplug-official
/root/autodl-tmp/conda_envs/soulx-table3-audit

项目内软链接：
/root/SoulX-stage3-dataset/.conda-envs/soulx-duplug-official
  -> /root/autodl-tmp/conda_envs/soulx-duplug-official
/root/SoulX-stage3-dataset/.conda-envs/soulx-table3-audit
  -> /root/autodl-tmp/conda_envs/soulx-table3-audit
```

环境位于数据盘，项目代码仍位于系统盘。当前环境约占 7.2 GiB。

## 3. 可复现依赖

- 直接依赖：`requirements/soulx_stage3_benchmark_minimal.txt`
- 当前完整解析快照：`requirements/soulx_stage3_benchmark_resolved.txt`
- Python：3.10.20
- PyTorch：2.6.0，CUDA runtime 12.4
- Transformers：4.52.1
- PEFT：0.16.0
- PyTorch Lightning：2.5.2
- ModelScope：1.28.2
- FunASR：1.2.6

重建时先创建 Conda 环境，再从最小清单安装。只有需要逐包复现本次环境时才使用完整解析快照：

```bash
/root/miniconda3/bin/conda create \
  --prefix /root/autodl-tmp/conda_envs/soulx-duplug-official \
  python=3.10.20 pip

/root/autodl-tmp/conda_envs/soulx-duplug-official/bin/python \
  -m pip install \
  -r /root/SoulX-stage3-dataset/requirements/soulx_stage3_benchmark_minimal.txt
```

网络路由按目标端点、本地缓存、实时连通性、稳定性和吞吐综合选择。国内源通常先试直连，GitHub、Hugging Face、OpenRouter 等境外端点通常先试 AutoDL 或已批准代理；任一路线超时或明显更慢时允许切换。下载前做轻量探测，执行记录保留最终路由、失败原因、重试次数和文件哈希。不得将 OpenRouter API key 写入依赖清单、日志或 benchmark 结果。

## 4. 系统依赖和硬件

```text
ffmpeg 4.4.2
SoX 14.4.2
GPU NVIDIA vGPU-32GB
显存 32760 MiB
驱动 560.35.03
```

安装音频系统包时 `ldconfig` 对 AutoDL 预置的部分空 NVIDIA 占位库给出警告；安装后已重新验证 `nvidia-smi` 和 `torch.cuda.is_available()`，GPU 功能正常，因此不修改这些平台文件。

## 5. 验收结果

- `pip check`：通过，无 broken requirements；
- 官方 benchmark 入口：`TurnModel`、`ParaformerASR`、`SensevoiceASR` 导入通过；
- 官方 Stage 3 入口：`finetune`、DataModule、Dataset、Model 导入通过；
- 项目测试：63/63 通过；
- CUDA：可用，PyTorch 能识别 `NVIDIA vGPU-32GB`。

真实执行链 smoke：

| 语言 | 样本 | 结果 | 用途 |
|---|---:|---:|---|
| 中文 Paraformer | complete/incomplete 各 1 条 | 1/2，macro accuracy 0.5 | 环境和端到端链路验收 |
| 英文 SenseVoice | complete/incomplete 各 1 条 | 2/2，macro accuracy 1.0 | 环境和端到端链路验收 |

smoke 中发现并补齐了 ModelScope 运行时实际需要、但其包元数据未完整声明的 `addict==2.4.0`、`simplejson==3.20.1` 和 `sortedcontainers==2.4.0`；同时按 SoulX 官方快照将 `setuptools` 从 83.0.0 固定为 78.1.1，以保留 ModelScope 仍调用的 `pkg_resources`。这些是已由真实执行证明必要的依赖，不是从 404 项快照中照搬的无关包。

benchmark runner 会在每个新结果中记录 Python、关键包版本、CUDA/GPU 信息、最小依赖清单哈希，以及官方推理仓库 commit/tree/dirty 状态。旧结果不含这些字段，不应与新结果混为同一批正式实验。

## 6. Table 3 独立审计环境

另一台服务器的已知核心版本为 Python 3.10、torch/torchaudio 2.6.0、transformers 4.55.0、pytorch-lightning 2.5.2、funasr 1.2.6、modelscope 1.28.2、numpy 1.24.4、omegaconf 2.3.0、soundfile 0.12.1、soxr 0.5.0.post1。现有环境的 transformers 4.52.1 和 numpy 1.26.4 不满足该身份，因此不得用于正式 Table 3 结果。

新环境放在数据盘 `/root/autodl-tmp/conda_envs/soulx-table3-audit`，并从项目 `.conda-envs/soulx-table3-audit` 软链接进入。审计 runner 会在加载模型前硬检查上述版本；YAML 中的 `precision: bf16` 只是上游未消费的配置字段，正式结果以模型参数实际 dtype 和 runner 是否启用 autocast 为准。

2026-08-20 已从现有环境克隆该独立环境，并只按官方 training-code 清单将 transformers 固定到 4.55.0、numpy 固定到 1.24.4；其余核心包已匹配。安装使用阿里云 PyPI 镜像，`pip check` 无冲突，项目测试 63/63 通过。环境约 7.2 GiB，数据盘剩余约 24 GiB。

候选 Table 3 runner 已分别完成 EN Complete 1 条和 ZH Complete 1 条 diagnostic smoke。两条均端到端成功，英文使用本地 SenseVoice Small，中文使用本地 Paraformer，未发生运行时下载。smoke 保存了每次 teacher-ASR 文本、完整状态轨迹、五个状态 token logits、初始化日志及全部输入/模型哈希。

官方 checkpoint 加载时会报告唯一多余键 `embed_tokens_func.weight` 并由上游回退到 `strict=False`。源码和 tensor 审计确认：该键是在 checkpoint 加载后才注册的嵌入层别名，与 checkpoint 中正式 embedding 和 LM head 共用同一存储；679 个最终模型键全部闭合、无缺失键和形状差异。runner/gate 只对白名单中的这一固定别名放行，任何其他差异都会失败。

首次正式候选运行在 ZH Complete 第 120 条遇到 Paraformer 返回空列表。官方 training-code 的 `ParaformerASR.recognize` 会捕获该异常、打印日志并返回空字符串；本地严格模型路径包装器已补齐相同行为，并额外把异常类型和信息写入 cache/逐样本证据。旧 partial 不续跑，旧提交下已完成的英文结果也不与新提交结果混用。

修复后的正式运行固定在项目 commit `ac8fcf1`，run ID 为 `formal-candidate-v1-ac8fcf1`。EN Complete、EN Incomplete、ZH Complete、ZH Incomplete 分别为 251/318、268/299、263/300、241/300。严格 gate 对 1,217 个样本的轨迹、state logits、ASR 调用/缓存、输入哈希、模型清单、代码身份和日志全部重算通过，但 EN Complete 和 ZH Complete 超出 ±1.0 pp 数值范围，故 `continued_training_authorized=false`。完整产物在数据盘 `dataset/soulx_duplug_eval/table3_audit/formal-candidate-v1-ac8fcf1`，共 2,447 个文件、约 57 MiB；gate 报告 SHA-256 为 `e013f7ee866498a7797f5226cb0558be42c41675cc76ef8732461877d87336c9`。
