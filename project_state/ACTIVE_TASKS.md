# Active task registry

更新时间：2026-09-13。

本目录用于辅助长任务开展和上下文恢复。当前对话与项目负责人的最新指令始终是主要依据；这里的状态不能替代对话，也不能覆盖更新指令。若两者冲突，必须暂停并确认。

每次连接中断、上下文 compact 或新会话开始后，先执行：

```bash
python scripts/show_active_tasks.py
```

然后读取 `active_tasks.json` 中引用的实时 manifest。运行进度以 manifest 与实际 `tmux`/进程为准，不以本文件中的时间点快照为准。

| Task ID | 当前阶段 | 实时来源 | 资源/依赖 | 下一动作 |
|---|---|---|---|---|
| `table3_official_lightning_v1` | 已完成并冻结 | `table3_official_continual_sweep/official_lightning_v1_b17bcf/orchestration_manifest.json` | 20/20；四个 continuation evidence gate 均通过 | 不改变协议、样本、规则或结果；只读引用 |
| `official_continual_final_report_v1` | 已完成并复核 | 正式 MD/HTML/audit JSON | audit=`passed`；旧自定义报告仅为 pilot | 作为会议汇报与后续研究决策的正式记录 |
| `duplexconv_expansion_v1` | Edu_0019–Edu_0045 扩展构造已完成至最终 Gate D | Edu_0044/0045 loader validation、Gate D closure、阶段 checksum 与审计 JSON | Edu_0044 截断 JSON 安全恢复；Edu_0045 已批准 9/9/9 校准例外；GPU 严格串行；所有 tmux 已退出 | 保持分片产物冻结；Edu_0018–0045 聚合 v1 已由独立任务完成；训练或 checkpoint 评测仍须另行提交并确认方案 |
| `duplexconv_aggregate_edu0018_0045_v1` | 已完成并冻结 | `project_state/duplexconv_aggregate_edu0018_0045_v1_run_manifest.json` | 28 分片；61,980 rows；13,689,991 chunks；hardlink、跨分片闭包、独立审计和未修改官方 loader 均通过 | 保持 v1 不可变；后续扩展新建 v2，不原地追加；本阶段未训练、未创建 split、未上传 |
| `duplexconv_baidu_release_edu0018_0045_v1` | 已完成并独立远端验收 | `release/duplexconv_edu0018_0045_stage3_zh_v1/run_manifest.json` | 160/160；148个实体文件精确字节复核；12个0-byte路径兼容表示；211,968,676,948 bytes；覆盖/删除0；最终回执12/12 | 保持发布不可变；另行批准的Edu_0001–0017扩展由新任务跟踪 |
| `duplexconv_expansion_edu0001_0017_v2` | Edu_0001–0017 已全部完成最终 Gate D；Edu_0017 经负责人批准只排除完整 `Edu--009870` 会话，499-source sanitized 恢复及独立闭包通过 | Edu_0017 `sanitized_recovery_supervisor_v1.json`、`full_pipeline_manifest_sanitized_v1.json`、`gate_d_closure_sanitized_v1.json` | 无 tmux/处理进程；499 sources、1000 views、23.715031h、2356 rows、530297 chunks；Qwen累计$0.21990488（全量$0.2119686725）；官方loader train=2238；Gate D 2000记录五类命中全0；两个可重建tar精确清理完成 | 保持官方 tar、原始失败、诊断、sanitized审计和最终产物不可变；新 v2 聚合已由独立任务完成 |
| `duplexconv_aggregate_edu0001_0045_v2` | 已完成并冻结 | `project_state/duplexconv_aggregate_edu0001_0045_v2_run_manifest.json` | 45分片；22,032 sources；44,332 views；101,395 rows；22,352,510 chunks≈993.445h；hardlink、全局闭包、全量checksum、独立审计及未修改官方loader均通过；official train=96,325 | 保持v2不可变；训练split、正式续训练、checkpoint/Table 3评测或上传均须另行披露方案并确认 |
| `duplexconv_baidu_release_edu0001_0045_v2` | 上传及独立远端验证已完成 | `release/duplexconv_edu0001_0045_stage3_zh_v2/remote_verification.json` | 242条逻辑记录；221个直接远端文件；21个零文件兼容表示；346,099,226,326 bytes；覆盖/删除0；tmux已退出 | 保持远端v2发布证据；过期v1退役状态继续以发布收据为准 |
| `duplexconv_edu0001_0045_official_continual_v1` | 已完成并冻结 | 正式 MD/HTML/audit 与 Table 3 sweep index | 官方 Lightning 30步、七点内部 validation、Table 3 和最终报告均已完成；当前无 tmux/进程 | 只读作为 A 组对照，不改变训练或评测证据 |
| `duplexconv_edu0001_0045_abcd_official_training_v1` | 已完成：B/C/D 官方30-step训练及0/5/10部分内部validation | live=`/root/autodl-tmp/dataset/duplexconv/training/edu0001_0045_abcd_queue_v1/orchestration_manifest.json` | 三组各30 updates、17,280 samples、七个checkpoint审计通过；step5/10均未通过5pp全状态保护门禁；无tmux/进程；Table 2/3未运行 | 保持训练证据不变；任何B/C/D Table 3前须另行确认统一checkpoint范围与顺序 |
| `duplexconv_edu0001_0045_abcd_table3_step5_10_v1` | 已完成并审计冻结 | audit=`evaluation_reports/duplexconv_edu0001_0045_abcd_experiment_summary_audit.json` | B/C/D step5/10共24个候选类别+4个新鲜基线类别完整，全部证据门禁通过；D5为最佳新候选，四类宏平均81.74%，仍比基线低2.41pp；MD/HTML/audit已生成；无tmux/进程，GPU空闲 | 保持证据只读；Table 2仍暂缓，需另行确认协议方可启动 |

硬性依赖：

```text
Table 3 complete + final evidence/report frozen
  └─ satisfied

English Full-Duplex-Bench download authorized + downloaded + ZIP integrity verified
  └─ fingerprint calibration/freeze (must precede any Edu_0019 score)
      └─ Edu_0019 pilot gates
          └─ contiguous Edu_0020–Edu_0045 expansion if all gates pass
```
