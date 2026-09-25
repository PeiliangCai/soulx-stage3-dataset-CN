# SoulX Stage 3 Official-Trainer Continuation Patch

## 基准与隔离

- 官方仓库：`third_party/SoulX-Duplug-upstream`
- 官方基准提交：`928b06508ed2de1344208d06fb1f6fb2ebfb1df5`
- 本目录是独立 clone，不是软链接、硬链接或官方目录上的直接修改。
- 正式训练结束后再次检查，官方仓库 Git 工作树为空。

本派生运行时的目标是保留官方 Lightning 训练语义，只修复已知错误并增加续训练所需的数据、scheduler 和审计边界。正式训练调用链为：

```text
finetune.py::train
  -> pytorch_lightning.Trainer.fit
  -> State_Prediction_Model.training_step
  -> State_Prediction_Model.configure_optimizers
  -> official AdamW + WarmupAnnealSteps
```

没有自行实现 `backward()`、`optimizer.step()`、GradScaler、梯度累积或数据 IndexStream。

## 补丁清单

### `models/_train_heads.py`

当一个状态 head 的 shifted labels 全部是 `-100` 时，PyTorch cross entropy 会返回 NaN。补丁返回与计算图连接的有限 FP32 零，保持官方七 head 加权求和语义且不产生梯度。

### `models/state_prediction_model.py`

1. 先创建 LoRA/PEFT 与 `embed_tokens_func` 别名，再加载发布 checkpoint；
2. 使用 `strict=True`，禁止官方实现中的失败后静默 `strict=False`；
3. 记录每个 head 的 loss、accuracy 和 valid-target count；
4. 当 `origin_step_estimate>0` 时，将同一个官方 `WarmupAnnealSteps` 公式定位到估计起点。发布权重没有 optimizer state，因此 AdamW moments 仍从空状态开始。

### `models/state_prediction_data.py`

可选读取已经冻结的 train/validation Parquet，保留官方 Dataset、tokenizer、mask、collator 和 DataLoader。启动时校验：

- split manifest SHA-256；
- 两个 Parquet 的 SHA-256；
- 2,066/102 行数；
- row identity；
- source-conversation leakage 为 0。

未配置独立 validation artifact 时仍保留官方随机行切分行为。

### `finetune.py`

仍由官方 `train()` 构造 `Trainer` 并调用 `Trainer.fit()`。仅增加审计 Callback，并让非 EMA `ModelCheckpoint` 的 `save_top_k/save_last` 可由配置指定，以便只保留一个完整恢复 checkpoint。

### `config/config.py`

增加冻结 validation、split manifest、起始 step 估计、审计目录、紧凑 checkpoint 网格和原生 checkpoint 保存策略字段。默认值保持官方行为。

### `utils/continual_audit.py`

只观察 Lightning 已执行的动作：记录实际 optimizer step、实际 microbatch/sample 数、使用的 LR、validation metrics、环境和文件哈希，并保存只含118个可训练 tensor 的评测 overlay。它不控制训练计算。

## 正式运行身份

- Run ID：`duplexconv-edu0018-official-lightning-v1`
- 配置：`configs/duplexconv_edu0018_stage3_official_continual_v1.yaml`
- 训练入口：本目录 `finetune.py`
- 官方发布权重 SHA-256：`b0703dea0b1dbb1cd51e6e7b6514c60907ea4d4b6752cecc3f71cb6445650dbe`
- 派生运行时 tracked diff SHA-256：`b1d55ba7530771cd3df6674456df498e1d7444dc93b55a1405f5557cc618f8da`
- Split identity：`0f5060afcf27857af17b99755775921851125fdf7b96dbf6e233fbfb967c1f2b`
- 起始 step：依据公开 `total_steps=1800` 估计，低置信度；不是发布权重中的元数据。

各关键文件的 SHA-256 保存在正式 `run_manifest.json` 的 `runtime.audited_files` 中。

