# DuplexConv Edu_0001–Edu_0045 raw 数据删除回执

- 执行时间（UTC）：2026-09-07T12:56:51Z
- 用户授权：已明确确认删除已上传百度网盘、当前训练与评测不再需要的 raw 数据
- 执行状态：成功

## 删除范围

- `/root/autodl-tmp/dataset/duplexconv/raw/audio/Edu_0001.tar` 至 `Edu_0045.tar`
- `/root/autodl-tmp/dataset/duplexconv/raw/metadata/jsons.tar.gz`
- 共 46 个普通文件；删除后移除了空的 `audio`、`metadata` 和 `raw` 目录
- 文件逻辑总量：345,867,800,157 bytes（345.868 GB / 322.114 GiB）
- 实际释放文件系统块：345,866,534,912 bytes

删除目标严格来自发布清单：

- `release/duplexconv_edu0001_0045_stage3_zh_v2/file_manifest.jsonl`
- 清单 SHA-256：`42382210089f784408cad8a50caf73bf96b153bef489e3bf553496ff3d6041af`

## 删除前门禁

- 46/46 个本地文件完成全量 SHA-256 重新计算，均与发布清单一致。
- 46/46 个本地文件大小和 `mtime_ns` 与发布清单一致。
- 46/46 个本地文件硬链接数均为 1。
- 没有进程打开 raw 目录内文件。
- 没有项目软链接指向 raw 目录。
- 2026-09-07 删除前重新连接百度网盘，对 46/46 个 raw 远端文件执行 `meta` 查询；全部存在且远端字节数精确一致。
- 百度 API 不提供远端内容哈希，因此远端复核依据是逐文件精确字节数；本地内容哈希和上传时清单/进度链保持完整。

既有独立上传验收报告：

- `release/duplexconv_edu0001_0045_stage3_zh_v2/remote_verification.json`
- 报告 SHA-256：`92445c5a65b00fea3b1ddff1fc9a40cbcfd537c3fc5bfc68b4237a221c41bf7f`
- 远端恢复目录：`/soulx-stage3-dataset-CN/datasets/duplexconv_edu0001_0045_stage3_zh_v2/raw`

## 空间变化

| 状态 | Used | Available | Use% |
|---|---:|---:|---:|
| 删除前 | 501,185,495,040 bytes | 35,685,416,960 bytes | 94% |
| 删除后 | 155,318,960,128 bytes | 381,551,951,872 bytes | 29% |

## 明确保留

- `/root/autodl-tmp/dataset/duplexconv/aggregates/edu0001_0045_stage3_zh_v2`
- `/root/autodl-tmp/dataset/duplexconv/splits/edu0001_0045_stage3_zh_v2_seed42_group98_2`
- `/root/autodl-tmp/dataset/duplexconv/training/edu0001_0045_official_continual_v1`
- `/root/autodl-tmp/dataset/soulx_duplug_eval`
- `processed/` 中间产物、模型、Conda 环境、checkpoint、评测结果和发布证据

## 恢复说明

本次删除在本机不可直接撤销。如需重新构造或扩展数据集，应从上述百度网盘 v2 raw 目录重新下载，并以发布清单核验 SHA-256。
