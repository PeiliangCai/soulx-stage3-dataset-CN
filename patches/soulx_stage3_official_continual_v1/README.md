# SoulX Stage 3 官方续训练补丁包

本目录把正式实验使用的派生运行时还原为可提交的小型补丁包；不包含官方仓库、模型、Conda 环境或 checkpoint。

- 官方仓库：`https://github.com/Soul-AILab/SoulX-Duplug.git`
- 固定提交：`928b06508ed2de1344208d06fb1f6fb2ebfb1df5`
- tracked diff：`tracked_changes.patch`
- 新增审计模块：`continual_audit.py`，目标路径 `utils/continual_audit.py`
- 运行时说明：`OFFICIAL_CONTINUAL_PATCH.md`，目标路径为运行时根目录

文件 SHA-256：

```text
b1d55ba7530771cd3df6674456df498e1d7444dc93b55a1405f5557cc618f8da  tracked_changes.patch
d5cfa8b4c90a57db22e45f2e70b0c3d92e2dcf6ea9aa017528459e2ee33f16c4  continual_audit.py
f8f8571f957a8e9dae3f4950997945fabd6118e0de806e8000407332bcf76e9d  OFFICIAL_CONTINUAL_PATCH.md
```

从项目根目录执行：

```bash
git clone https://github.com/Soul-AILab/SoulX-Duplug.git third_party/SoulX-Duplug-upstream
git -C third_party/SoulX-Duplug-upstream checkout --detach 928b06508ed2de1344208d06fb1f6fb2ebfb1df5
bash scripts/prepare_official_continual_runtime.sh
```

构建脚本会拒绝覆盖已有目标目录，并在结束前验证官方 commit、tracked diff 与两个新增文件的哈希。
