# DuplexConv Edu_0018–Edu_0045 Stage 3 中文数据发布包

状态：准备上传。

本发布包只覆盖项目负责人在 2026-08-27 确认的三类内容：

1. `raw/`：官方 `Edu_0018.tar`–`Edu_0045.tar` 及权威 metadata archive；
2. `model_ready/`：已经冻结、通过独立审计和未修改 SoulX 官方 loader 全量验证的不可变聚合；
3. `evidence/` 与 `receipts/`：聚合配置、验证报告、计划说明、发布清单和远端验收记录。

明确不包含：`processed/` 中的 68 GiB 中间层、`cache/`、`work/`、失败重试目录、诊断输出、Conda 环境、模型权重、API 密钥和 `.env`。

## 冻结数据身份

- 数据集版本：`duplexconv_edu0018_0045_stage3_zh_v1`
- 分片范围：`Edu_0018`–`Edu_0045`，共 28 个分片
- Edu_0018：使用仅完整排除 `Edu--010902` 的 sanitized 版本
- source conversations：13,540
- source views：27,247
- Stage 3 rows：61,980
- exported chunks：13,689,991
- aggregate manifest SHA-256：`095e18de040f812aa305b06513356470727fad9f18d579081d04335f67819bdb`
- aggregate stats SHA-256：`17187ca65e7b94b927d4b2b5c54fe163ac4b6f691b8af68884ec2089777d44b0`
- aggregate checksums manifest SHA-256：`0c246ead053da895fb9e72ee94d3bc29201f22b216b20c104f711bedfdee2b1b`
- SoulX 官方上游提交：`928b06508ed2de1344208d06fb1f6fb2ebfb1df5`

## 本地来源与预计远端位置

```text
/root/autodl-tmp/dataset/duplexconv/raw
  -> /soulx-stage3-dataset-CN/datasets/duplexconv_edu0018_0045_stage3_zh_v1/raw

/root/autodl-tmp/dataset/duplexconv/aggregates/edu0018_0045_stage3_zh_v1
  -> /soulx-stage3-dataset-CN/datasets/duplexconv_edu0018_0045_stage3_zh_v1/model_ready/edu0018_0045_stage3_zh_v1
```

上传前、上传中和上传后的权威状态以同目录 `run_manifest.json`、`release_spec.json`、checksum 文件和最终 receipt 为准。

