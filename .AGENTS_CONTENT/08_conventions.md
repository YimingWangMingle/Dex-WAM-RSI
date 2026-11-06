# 代码风格、测试策略、Git 卫生与工作总结流程

## 代码风格与约定

- 纯 Python（未配置 formatter/linter）；遵循所在文件现有风格（新文件用类型标注，不统一）。
- 入口脚本/工具把仓库根加入 `sys.path`，跨顶层包（`data_prepare`、`models`、`utils`、`wan`、`train`）导入；保持从仓库根的绝对导入。
- `wan/` 是 **vendor 第三方代码**（阿里 Wan，Apache 版权头）——除非必要不修改；`libs/motus-data-factory/` 是 **git submodule**，改动走其自身仓库流程，本仓库侧数据逻辑放 `data_prepare/`。Motus 专有代码在 `models/`、`data_prepare/`、`train/`、`utils/`。
- `models/old.py` 是遗留代码；`models/ubt_wm.py` 是当前在用的模型。
- checkpoint 完整性用 `.motus_checkpoint_complete.json` 标记（见 `train/train.py`）；改动 checkpoint 代码遵守该约定。
- 日志用 `logging`；指标用 TensorBoard（`--report_to tensorboard`），可选 wandb。
- **训练 seed 缺省非 0（20260818 起）**：数据采样链（跨源调度/组内洗牌/窗口起点）是 (seed, epoch, index) 的确定性函数，seed=0 会让所有运行看到完全相同的训练数据。新建训练入口不得硬编码 `seed=0`，统一从 `config.training.seed` 读取（缺省 `20260818`；写法见 `train/train.py` `create_dataloaders`）；调试/冒烟脚本例外，可显式传 0。

## 测试策略

无单元测试框架。工作流：(1) 数据通路改动先跑 `scripts/smoke_indexed_mix_cpu.py`（本机 micro 8 源 / H200 集群 10 源，纯 CPU）；(2) `bash train_micro.sh` 本机训练验证（40 步，CUDA）；(3) 小 `MAX_STEPS` 短训练（如弹性重启验证 `MAX_STEPS=30 SAVE_INTERVAL=10 ELASTIC_SIMULATE_FAILURE_STEP=10`）；(4) `scripts/eval_video_ckpt.py` 可视化检查。manifest 转换改动用 `scripts/validate_motus_manifest.py` 校验；`libs/motus-data-factory` 自带 pytest（`python -m pytest libs/motus-data-factory/tests`）。

## Git 卫生与安全

- 不要纳入 Git（已在 `.gitignore`）：checkpoint、`outputs/`、日志、计时目录、`*.pt/*.pth/*.safetensors/*.tar*`、`__pycache__`、`.vscode/`、`micro_data/`、`pretrained_models/`。
- YAML 配置和 manifest 含**指向共享集群存储的绝对路径**（`/data/.../manifests/`、预训练模型目录）——与环境相关；未确认目标机器目录结构前不要"修正"。
- `H-Server-ssh/` 已纳入 Git（原 `chen-liao-script/`），含个人 SSH 跳板机信息（内网地址/账号），对外分享时注意。
- 绝不要提交凭据、集群端点或生成的调试归档（上条是有意的例外，不新增此类内容）。
- shell 长路径用变量，避免意外换行（见 README）。
- **Git SSH 免密（2026-08 配置）**：远端 `gitlab.rd.ubtrobot.com` 的 SSH 需要密码，本仓库已设 `core.sshCommand=SSH_ASKPASS=$HOME/.ssh/git_askpass.sh SSH_ASKPASS_REQUIRE=force ssh`，askpass 脚本读取 `~/.ssh/git_pass`（权限 600，用户手填，**绝不入库、不经对话记录**）。密码变更后需用户手动更新该文件；其他仓库如需免密再设全局 `core.sshCommand`。

## 工作总结流程

用户要求总结某段时间修改时：

1. **收集提交**：`git log --since=<起始时间> --pretty=format:"%h %ad %s" --date=format:"%m-%d %H:%M"` 列范围提交，再 `git log --stat` 看涉及文件与改动规模，据此归纳主题。会话开始时间可能陈旧，需要时用 `date` 取当前真实日期。
2. **归纳要点**：按主题合并为少数几点（用户指定点数），每点说明**做法、作用、效果**。默认分点+小标题；用户要求时改每点一整段。引用提交用短 hash（如 `893ccca`），代码位置用 `path/to/file.py`。
3. **保存归档**：用户要求保存时写入 `/save/`（工作目录外），文件名 `<YYYYMMDD>_<主题名称>.md`，带标题和日期范围。
