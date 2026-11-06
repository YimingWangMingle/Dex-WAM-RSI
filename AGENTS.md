# AGENTS.md

**ThinkerWM Pretrain**：世界模型风格视频/动作模型（"Motus"）的分布式预训练代码库。详细说明按主题拆分在 `.AGENTS_CONTENT/` 下；**做任何修改前，先阅读本文件，并按修改主题阅读下方对应文件**。

## 索引

| 文件 | 内容 |
|---|---|
| [01_overview.md](.AGENTS_CONTENT/01_overview.md) | 项目概览、仓库结构、运行环境（本地/生产、H 集群同步与 rsync 处理） |
| [02_commands.md](.AGENTS_CONTENT/02_commands.md) | 冒烟脚本、训练启动、micro debug、环境变量、输出根约定、续训、离线评估、manifest 工具 |
| [03_config.md](.AGENTS_CONTENT/03_config.md) | 配置 YAML、dataset schema、pool/sampling_ranges 机制、重要约束 |
| [04_data_model.md](.AGENTS_CONTENT/04_data_model.md) | batch 契约、Episode Catalog + SamplePool、action/state 归一化 |
| [05_runtime_safety.md](.AGENTS_CONTENT/05_runtime_safety.md) | 训练器安全环境变量、诊断输出 |
| [06_periodic_eval.md](.AGENTS_CONTENT/06_periodic_eval.md) | 周期性定集评测 |
| [07_monitor.md](.AGENTS_CONTENT/07_monitor.md) | 实时监控库 |
| [08_conventions.md](.AGENTS_CONTENT/08_conventions.md) | 代码风格、测试策略、Git 卫生、工作总结流程 |
| [09_weekly_report.md](.AGENTS_CONTENT/09_weekly_report.md) | 周报撰写流程（示例格式、git+monitor 数据收集、耗时对比、写作约定） |
| [feishu/](.AGENTS_CONTENT/feishu/README.md) | 飞书集成库（libs/motus-feishu）：安装与凭据、文档读写/插图/插视频/插表格示例、编辑安全红线、可访问文档索引、常用文档（含临时中转文档）、自动周报工作流、预训练基准文档维护 |

## 绝对必要的注意事项

- **禁令：未经用户明确允许，严禁向 H 集群同步任何代码或运行会改动其文件系统的命令**（含 rsync/scp/tar-over-ssh 推送）；需要时先在对话中征得同意。
- **修改前必读**：数据管线改动读 `03_config.md`/`04_data_model.md`；训练/启动改动读 `02_commands.md`/`05_runtime_safety.md`；监控/评测改动读 `06_periodic_eval.md`/`07_monitor.md`；代码与流程约定见 `08_conventions.md`；周报/工作总结撰写读 `09_weekly_report.md`（示例在 `/save/每周报告/`，git + monitor 数据源见文档）。
- **架构红线**：数据索引选择策略归 `data_prepare/` 所有，motus_data 库只负责统一格式/加载/打包/传输；`wan/` 是 vendor 代码、`libs/motus-data-factory/` 是 git submodule，除非必要不修改。
- **Git 红线**：不提交凭据、集群端点、checkpoint/outputs/日志等生成物；YAML 中的共享存储绝对路径与环境相关，不要擅自"修正"（详见 `08_conventions.md`）。
- **配置命名红线**：训练配置/启动脚本名必须与当前 dev 分支一致（`dev/YYYYMMDD` → `stage2_YYYYMMDD_v1`，同周期内 `_v2`… 迭代）；**配置名称必须询问用户拍板，新建还是修改原配置由用户决定**（详见 `03_config.md`）。
- **训练 seed 红线（20260818 起）**：新建配置/启动脚本必须显式设置新的非零 `training.seed`（建议日期值，禁止 0/复用旧值），否则不同运行会采样到完全相同的训练数据；代码兜底默认 `20260818`（train.py/micro_train.py），详见 `03_config.md`。
- **飞书安全**：用 `libs/motus-feishu` 操作飞书文档时，编辑前自动快照不可绕过、优先追加而非改写、删除/覆盖必须用户明确确认；飞书凭据只在环境变量或 `~/.config/motus-feishu/credentials.env`，绝不入库（详见 `.AGENTS_CONTENT/feishu/`）。
