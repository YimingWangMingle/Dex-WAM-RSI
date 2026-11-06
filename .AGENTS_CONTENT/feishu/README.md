# 飞书集成（libs/motus-feishu）

`libs/motus-feishu` 是 git submodule（url 暂为本地路径 `/home/fengodchen/workspace/motus-feishu`，将来推送远端后用 `git submodule set-url` 切换），提供飞书（国内版 `open.feishu.cn`）云文档读写、媒体上传与训练数据只读汇总能力。库的完整说明（自建应用创建、权限开通、凭据配置）见其 `README.md`；本文件夹是 agent 操作手册与安全红线（2026-08-17 起从库内 `.AGENTS_CONTENT` 统一归拢到本仓库，库内不再单独维护）。

## 索引

| 文件 | 内容 |
|---|---|
| [01_doc_ops.md](01_doc_ops.md) | 文档读写 skill：读文档、追加文本/标题、插表格、插图/视频/文件（Python + CLI） |
| [02_safety.md](02_safety.md) | 安全红线：写前快照、追加优先、破坏性操作确认、只读开关、凭据与 H 集群约束 |
| [03_doc_index.md](03_doc_index.md) | 可访问文档索引的自动注册约定（register/list/unregister） |
| [04_weekly_report.md](04_weekly_report.md) | 自动周报工作流：数据收集 → 汇总 → 追加到周报文档；大组周报模板与结果先行写作结构；具身X「每周重点进展」文档写作要求（主干/Infra 结构、指标归属、图表红线） |
| [05_benchmark_doc.md](05_benchmark_doc.md) | 预训练基准文档维护：文档结构、数据口径（去重/取舍规则）、曲线图约定、更新工作流与脚本、飞书 API 实测经验（插图三步法等） |

**修改本文件夹前**：内容应与库的实际行为一致——修改库代码导致行为变化时，
同步更新对应 skill 文件；凭据与私密信息绝不写入本文件夹（见 02_safety.md）。

## 常用文档

- **预训练基准文档**：用户说"基准文档"时直接用该文档，不再要链接。
  - 链接：<https://my.feishu.cn/docx/WxKadSPApokyv0xTpmGcVhRTnjh>（已 register，
    note=`ThinkerWM 预训练基准（EWMBench/WorldModelBench 结果+版本变更+指标含义）`）。
  - 维护口径与工作流见 [05_benchmark_doc.md](05_benchmark_doc.md)。
- **临时中转文档**：用户说"放到临时中转文档"时直接用该文档，不再要链接。
  - 链接：<https://my.feishu.cn/wiki/S7LKwjh0Ii9kuJkpGqQcJrZTnQh>
    （docx token 与 wiki token 相同，可直接用：`https://my.feishu.cn/docx/S7LKwjh0Ii9kuJkpGqQcJrZTnQh`；
    已 register，本地索引 note=`临时中转文档`）。
  - 注意：库接口只认 docx token。一般 wiki 链接需先用 wiki API
    （`client.wiki.v2.space.get_node`）解析出 obj_token 再读写（需应用开通
    `wiki:node:read` 等 scope），参考 `tmp/resolve_wiki_20260817.py`；
    若应用未开 wiki scope，可先尝试把 wiki token 直接当 docx token 用
    （如本文档即可）。

## 安装与凭据

- 安装：`pip install -e libs/motus-feishu`（依赖官方 `lark-oapi` SDK）。
- 凭据：环境变量 `FEISHU_APP_ID` / `FEISHU_APP_SECRET`，或
  `~/.config/motus-feishu/credentials.env`（chmod 600）。**凭据绝不入库、
  绝不写进任何代码/配置/文档**。
- 验证：`motus-feishu-cli auth-check`（或 `python -m tools.feishu_cli auth-check`）。
- 目标文档需先把应用加为协作者（文档 → 分享 → 添加应用 → 可编辑）。

CLI：`motus-feishu-cli {auth-check, doc dump, doc append, doc register, doc list, doc unregister, doc snapshot, doc restore, upload}`，写操作支持 `--dry-run`。

## 读写、索引、安全与周报

- 文档读写/插图/插视频/插表格示例 → [01_doc_ops.md](01_doc_ops.md)
- 安全红线（写前快照、追加优先、破坏性操作确认、只读开关、H 集群只读）→ [02_safety.md](02_safety.md)
- 可访问文档索引的自动注册约定（用户提供链接时 agent 自动 register）→ [03_doc_index.md](03_doc_index.md)
- 自动周报工作流与大组周报模板 → [04_weekly_report.md](04_weekly_report.md)
