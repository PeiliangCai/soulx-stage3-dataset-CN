# SoulX Stage 3 与 benchmark 运行环境

更新时间：2026-08-20
状态：已完成精简环境安装及基础验收；正式 benchmark 基线尚未完成。

## 1. 决策与用途

本项目使用一个共享的 Python 3.10 Conda 环境，优先覆盖：

1. SoulX 官方推理服务与 Bilingual Easy Turn benchmark；
2. SoulX Stage 3 状态预测续训练；
3. 中文 Paraformer 与英文 SenseVoice 级联 ASR。

不再逐项安装官方环境快照中的全部 404 个包。当前执行路径未使用的 vLLM、TensorRT、Gradio/Jupyter、ONNX Runtime、DeepSpeed、bitsandbytes、diffusion/TTS 等依赖不安装；实际运行遇到新的必要导入时，再做最小增量补充。

## 2. 存储布局

```text
实际环境：
/root/autodl-tmp/conda_envs/soulx-duplug-official

项目内软链接：
/root/SoulX-stage3-dataset/.conda-envs/soulx-duplug-official
  -> /root/autodl-tmp/conda_envs/soulx-duplug-official
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

国内 Python/模型源下载前清除遗留代理；只有访问 GitHub、Hugging Face 等国外资源时才启用 AutoDL 代理。不得将 OpenRouter API key 写入依赖清单、日志或 benchmark 结果。

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
- 项目测试：48/48 通过；
- CUDA：可用，PyTorch 能识别 `NVIDIA vGPU-32GB`。

真实执行链 smoke：

| 语言 | 样本 | 结果 | 用途 |
|---|---:|---:|---|
| 中文 Paraformer | complete/incomplete 各 1 条 | 1/2，macro accuracy 0.5 | 环境和端到端链路验收 |
| 英文 SenseVoice | complete/incomplete 各 1 条 | 2/2，macro accuracy 1.0 | 环境和端到端链路验收 |

smoke 中发现并补齐了 ModelScope 运行时实际需要、但其包元数据未完整声明的 `addict==2.4.0`、`simplejson==3.20.1` 和 `sortedcontainers==2.4.0`；同时按 SoulX 官方快照将 `setuptools` 从 83.0.0 固定为 78.1.1，以保留 ModelScope 仍调用的 `pkg_resources`。这些是已由真实执行证明必要的依赖，不是从 404 项快照中照搬的无关包。

benchmark runner 会在每个新结果中记录 Python、关键包版本、CUDA/GPU 信息、最小依赖清单哈希，以及官方推理仓库 commit/tree/dirty 状态。旧结果不含这些字段，不应与新结果混为同一批正式实验。
