# Skill 02：安全红线（勿绕过）

本库操作的是用户的真实飞书文档与 H 集群数据，以下红线任何情况下不得绕过。

## 写操作

- **编辑前自动快照**：所有写操作（append/update/delete）默认先把整篇文档
  备份到 `~/.local/share/motus-feishu/backups/<doc_id>/<UTC时间戳>.json`。
  不要用 `auto_snapshot=False` 绕过，除非用户明确给出理由。
- **优先追加，避免改写**：写周报等场景一律用 `append_text` / `insert_table` /
  `insert_image`；`update_text_block` / `delete_root_children` /
  `restore_document` 等破坏性操作必须用户明确确认（代码 `confirm=True`，
  CLI `--confirm`）。
- **预演**：不确定的写操作先 `dry_run=True` / CLI `--dry-run`。
- **全局只读开关**：`MOTUS_FEISHU_READONLY=1` 时库拒绝一切写操作。
- **误改恢复**：优先 `motus-feishu-cli doc restore <doc> --confirm`
  （本地快照），其次飞书网页端版本历史（文档 → ⋯ → 历史记录）。

## 凭据与私密信息

- app_id/app_secret 只放环境变量 `FEISHU_APP_ID` / `FEISHU_APP_SECRET` 或
  `~/.config/motus-feishu/credentials.env`（chmod 600）。
- **凭据绝不入库**、绝不写进任何代码/配置/文档（含本 `.AGENTS_CONTENT`）。
- 文档索引、快照等用户级数据放 `~/.local/share/motus-feishu/`，不入 git。

## H 集群只读

- `data_readers` 只允许白名单命令
  （ls/cat/tail/head/grep/du/df/find，find 禁 -delete/-exec 等动作参数）
  和 scp pull-only（远端→本地）。
- 绝不执行任何写/删命令，绝不向 H 集群推送任何内容。
