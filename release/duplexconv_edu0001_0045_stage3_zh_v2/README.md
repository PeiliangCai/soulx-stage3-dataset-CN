# DuplexConv Edu_0001–Edu_0045 Stage 3 中文数据发布包

状态：上传、服务器端 raw 迁移和独立远端验收均已完成（`remote_verification.json: status=passed`）。

本发布包是已经退役的 `duplexconv_edu0018_0045_stage3_zh_v1` 的完整后继版本，包含：

1. `raw/`：45 个官方 `Edu_0001.tar`–`Edu_0045.tar` 及权威 metadata archive；
2. `model_ready/`：冻结的 Edu_0001–0045 Stage 3 多 Parquet 聚合；
3. `evidence/` 与 `receipts/`：配置、Gate D、独立审计、未修改官方 loader 验证、发布清单和远端回执。

为了避免重复占用约 211.83 GB，Edu_0018–0045 与 metadata 的既有远端 `raw/` 已通过百度网盘服务器端移动进入本版本；本地只增量上传 Edu_0001–0017。新版本全部远端验收通过后，旧版本的过期 `model_ready/evidence/receipts` 已按冻结方案退役，旧根目录保留迁移说明。

明确不包含：`processed/` 中间层、`cache/`、`work/`、失败重试与诊断目录、Conda 环境、模型权重、API 密钥或 `.env`。

## 冻结数据身份

- 数据集版本：`duplexconv_edu0001_0045_stage3_zh_v2`
- 分片：45
- source conversations：22,032
- 最终 source views：44,332
- Stage 3 rows：101,395
- exported 160 ms chunks：22,352,510（约 993.444889 训练有效小时）
- aggregate manifest SHA-256：`f6838ad77b4ea533366f7bbf079814f0db16488424e940159a5e3f11ccbe8e72`
- aggregate stats SHA-256：`7401b25e2757bbd1767982e1c8f9f28e71ee35b549e3f7c84ad5d66ded189763`
- aggregate checksums SHA-256：`4d3f0745056398d98058377d2368d243e6d350ba7884cd48f09bb63396681ad6`
- SoulX 官方上游提交：`928b06508ed2de1344208d06fb1f6fb2ebfb1df5`

最终远端路径：

```text
/soulx-stage3-dataset-CN/datasets/duplexconv_edu0001_0045_stage3_zh_v2
```

最终验收覆盖 242 条逻辑记录：221 个直接远端文件和 21 个零字节兼容表示，直接远端文件合计
346,099,226,326 bytes；覆盖数和删除数均为 0。以 `remote_verification.json`、
`upload_completion.json` 和 `raw_migration_receipt.json` 为最终状态依据。`run_manifest.json`
保留上传过程中的历史阶段值 `upload_running`，不得用它覆盖已经完成的独立验收结论。
