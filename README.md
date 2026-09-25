# SoulX Stage 3 中文 DuplexConv 数据与续训练实验

> 交接状态：**项目已停止继续开发，当前成果已冻结并整理用于交接**（2026-09-25）。当前没有运行中的训练、评测或数据处理任务。

本项目把 DuplexConv 中文教育会话转换为 SoulX-Duplug Stage 3 状态预测训练数据，并完成了官方训练流程续训练、Table 3 Easy Turn 复现及 Complete/Incomplete 平衡消融。仓库保存代码、配置、审计证据和会议报告；原始音频、最终数据、模型、checkpoint、Conda 环境及密钥不进入 Git。

## 最终结论

### 数据集

最终冻结版本为 `duplexconv_edu0001_0045_stage3_zh`：

| 项目 | 数量 |
|---|---:|
| DuplexConv shard | 45（Edu_0001–Edu_0045） |
| 源会话 | 22,032 |
| 目标说话人视角 | 44,332 |
| Stage 3 训练窗口 | 101,395 |
| 有效 160 ms chunk | 22,352,510 |
| 训练暴露时长 | 约 993.445 小时 |

“训练暴露时长”按目标说话人视角统计，同一源会话的不同目标声道会重复贡献时长，不能解释为去重后的原始录音小时数。2/3/4 轨训练窗口分别为 99,021 / 2,206 / 168；超过两轨的会话没有丢弃，而是按“目标声道对其余声道联合关系”展开。

状态事件共 396,950 条：官方标签 319,767 条、`WAIT → complete` 846 条、Qwen 补标 76,337 条。Qwen 固定为项目名 `qwen3-235b-a22b-instruct-2507`，OpenRouter 模型 ID 为 `qwen/qwen3-235b-a22b-2507`；这些结果属于 LLM 辅助标签，不是人工 gold。

### 训练与 Table 3

正式续训练基于 SoulX 官方 `training-code` 提交 `928b06508ed2de1344208d06fb1f6fb2ebfb1df5`，沿用 Lightning `Trainer.fit()`、官方 optimizer 和 scheduler。补丁只修复/增加：空状态 head 的 NaN、严格权重加载、冻结会话级 split、scheduler 起点、审计与紧凑 checkpoint；没有使用早期自定义 optimizer loop。

官方权重的本机 Table 3 四类等权宏平均为 **84.14%**。原始数据续训练后，Complete 上升但 Incomplete 明显下降。2×2 消融结果中，“平衡采样 + Complete/Incomplete 等权 loss”的 D5 是六个新候选中最好的一项，宏平均 **81.74%**，仍低于官方基线 **2.41 个百分点**。因此，本项目没有得到可以无条件替代官方基线的续训练 checkpoint。

完整结果见：

- [A/B/C/D 最终实验报告](evaluation_reports/duplexconv_edu0001_0045_abcd_experiment_summary.md)（会议汇报主报告，另有 HTML 可视化和 audit JSON）
- [原始 A 组官方流程续训练报告](evaluation_reports/duplexconv_edu0001_0045_official_continual_final.md)
- [Table 3 反作弊审计](evaluation_reports/soulx_table3_anti_cheating_audit.md)

Table 2 Full-Duplex-Bench **没有执行**。A 的结果在提出 B/C/D 假设前已经被查看，所以 B/C/D 的 Table 3 属于事后假设驱动的开发性评测，不应表述为独立最终测试。

## 资产在哪里

| 位置 | 内容 | 是否在 GitHub |
|---|---|---|
| 本仓库 | `src/`、`scripts/`、`tests/`、配置、计划、报告、状态与发布回执 | 是 |
| 百度网盘 `/soulx-stage3-dataset-CN/datasets/duplexconv_edu0001_0045_stage3_zh` | 45 个原始 shard、metadata、最终 `model_ready` 聚合、证据与回执 | 否 |
| 本地 `dataset/` | 数据盘目录的软链接 | 否 |
| `pretrained_models/`、`checkpoints/` | 官方模型与实验 checkpoint | 否 |
| `runtimes/`、`third_party/`、`.conda-envs/` | 派生运行时、官方仓库副本与 Conda 环境 | 否 |
| `.env` | OpenRouter API key | 否 |

百度网盘发布已经独立验收通过：242 条逻辑记录，其中 221 个直接远端文件、21 个零字节兼容表示，直接远端文件合计 346,099,226,326 bytes，覆盖和删除均为 0。详见 [v2 发布说明](release/duplexconv_edu0001_0045_stage3_zh/README.md) 与 [远端验收回执](release/duplexconv_edu0001_0045_stage3_zh/remote_verification.json)。

发布包不保存可重建的 `processed/` 中间层、cache、work、失败重试目录或运行环境；它保存原始数据和可直接供官方 loader 使用的最终 `model_ready` 数据。旧 v1 已退役，恢复时以 v2 为准。

## 建议阅读顺序

1. 本 README；
2. [当前任务注册表](project_state/ACTIVE_TASKS.md)——所有任务均已结束，用于理解产物之间的依赖；
3. [完整项目计划](project_plan/duplexconv_edu0018_stage3_plan.md)——虽以 Edu_0018 命名，但记录了 Edu_0001–Edu_0045 的完整演进；
4. [A/B/C/D 最终实验报告](evaluation_reports/duplexconv_edu0001_0045_abcd_experiment_summary.md)；
5. [v2 数据发布说明](release/duplexconv_edu0001_0045_stage3_zh/README.md)；
6. [执行审批规范](agent_governance/EXECUTION_APPROVAL_PROTOCOL.md)。

`new_autodl_paraformer_stage3_handoff.md` 是 2026-08-20 的历史迁移快照，不能覆盖上述最终状态。

## 数据处理契约

流水线的主要阶段为：

```text
源 tar/metadata
  → source scan 与多轨目标视角展开
  → benchmark 音频泄漏门禁
  → 官方状态保留 + WAIT 映射 + Qwen 缺失状态补标
  → 目标声道音频导出
  → Paraformer 中文伪转录与 token 时间戳
  → 160 ms timeline 与状态对齐
  → GLM-4-Voice audio token
  → index/sequence Parquet
  → 未修改官方 loader 验收与 Gate D
  → 45-shard 冻结聚合
```

每条训练序列只包含一个目标声道的 audio token。其他声道不混音、不同时塞进 sequence，只用于离线判断 overlap、对方是否说话、backchannel 等关系。若某一源会话被认定泄漏或损坏，只排除该完整源会话及其全部视角，不因一条问题记录删除整个 Edu shard。

关键实现位于 `src/duplexconv_stage3/`，命令行入口位于 `scripts/`。各 shard 的冻结输入契约在 `configs/*source_contract.json`；聚合配置为 `configs/duplexconv_edu0001_0045_aggregate_v2.json`。

## 环境与测试

推荐使用 Python 3.10 Conda 环境。经过验证的最小依赖和完整解析快照分别在：

- `requirements/soulx_stage3_benchmark_minimal.txt`
- `requirements/soulx_stage3_benchmark_resolved.txt`
- `requirements/soulx_table3_audit_core.txt`

示例：

```bash
conda create -n soulx-stage3-handoff python=3.10 -y
conda activate soulx-stage3-handoff
python -m pip install -r requirements/soulx_stage3_benchmark_minimal.txt
PYTHONPATH=src python -m unittest discover -s tests -v
```

交接前最后一次完整运行结果：**146 tests passed**。部分集成测试会读取本机已经存在且被 Git 忽略的官方仓库/运行时；若这些资产不存在，应先按下一节重建。

## 重建官方补丁运行时

官方干净仓库和补丁运行时不直接提交。可从固定提交重建：

```bash
git clone https://github.com/Soul-AILab/SoulX-Duplug.git third_party/SoulX-Duplug-upstream
git -C third_party/SoulX-Duplug-upstream checkout --detach 928b06508ed2de1344208d06fb1f6fb2ebfb1df5
bash scripts/prepare_official_continual_runtime.sh
```

实际补丁和新增审计模块已保存在 [补丁包](patches/soulx_stage3_official_continual_v1/README.md)。构建脚本拒绝覆盖已有目录，并验证 commit 和文件哈希。正式训练配置为 `configs/duplexconv_edu0001_0045_stage3_official_continual_v1.yaml`；B/C/D 消融配置也在 `configs/` 中。

官方发布权重不包含 optimizer、scheduler、GradScaler 和精确 `global_step`，所以公开配置中的 step 1800 只被用作低置信度起点估计，不是精确断点恢复。

## API key 与网络

仓库只提供空模板 `.env.example`。如需重新补标：

```bash
cp .env.example .env
chmod 600 .env
# 在 .env 中本地填写 OPENROUTER_API_KEY；不要提交该文件
```

客户端会检查服务端每日限额，项目冻结上限为 10 USD/day。网络路由支持继承当前环境代理或显式直连，应按实际连通性和下载速度选择，不依赖项目负责人的本地电脑。任何响应缓存、登录信息、百度网盘 cookie/token 或代理凭据都不得提交。

## 已知限制与后续接手注意事项

- 数据、模型和 checkpoint 均不在 GitHub；仅克隆本仓库不能直接训练或复现实验数值。
- Table 3 官方权重基线与论文四类结果接近，但原 ±1 pp 门禁并非四类全部通过；报告中已如实保留。
- Table 2 尚未实现正式复现，审计记录见 `evaluation_reports/soulx_table2_eval_repo_audit_v1.md`。
- D5 只是在新候选中最好，并未超过官方基线，也没有通过全部内部状态 head 的 5 pp 保护门禁。
- `project_state/active_tasks.json` 是长任务的完整机器可读历史，文件较大；恢复进度应同时核对最终报告、manifest 和实际文件哈希。
- 绝对路径来自原 AutoDL 服务器。迁移到新机器时应通过配置或软链接适配，不要盲目复用旧路径。
- 新实验必须使用新 run ID 和输出目录，不覆盖冻结的 v2 数据、checkpoint、评测结果或回执。

## 仓库安全边界

`.gitignore` 排除了 `.env`、凭据文件、音频/Parquet、大模型、checkpoint、缓存、日志、Conda 环境和第三方仓库。提交前至少执行：

```bash
git status --short
git diff --cached --check
git ls-files | grep -E '(^|/)(\.env|.*\.(pem|key|p12|pfx))$' && echo 'STOP: credential-like file tracked'
```

若接手者需要继续项目，应先只读复核数据盘、百度网盘回执、模型哈希、官方 commit 和磁盘容量，再按 [执行审批规范](agent_governance/EXECUTION_APPROVAL_PROTOCOL.md) 提交新方案；不要仅依据历史对话或旧运行中的 manifest 启动任务。
