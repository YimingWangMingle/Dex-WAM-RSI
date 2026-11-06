# 飞书集成（libs/motus-feishu）

`libs/motus-feishu` 是 git submodule（url 暂为本地路径 `/home/fengodchen/workspace/motus-feishu`，将来推送远端后用 `git submodule set-url` 切换），提供飞书（国内版 `open.feishu.cn`）云文档读写、媒体上传与训练数据只读汇总能力。库的完整说明（自建应用创建、权限开通、凭据配置）见其 `README.md`；本文档是本仓库侧的操作手册与安全红线。库内另带 `.AGENTS_CONTENT/`（agent skill 集：文档操作/安全红线/索引约定/周报工作流），内容与本文件一致、随库 git 管理，更新前需征得用户同意（见该文件夹 README 的更新约定）。

## 安装与凭据

- 安装：`pip install -e libs/motus-feishu`（依赖官方 `lark-oapi` SDK）。
- 凭据：环境变量 `FEISHU_APP_ID` / `FEISHU_APP_SECRET`，或
  `~/.config/motus-feishu/credentials.env`（chmod 600）。**凭据绝不入库、
  绝不写进任何代码/配置/文档**。
- 验证：`motus-feishu-cli auth-check`（或 `python -m tools.feishu_cli auth-check`）。
- 目标文档需先把应用加为协作者（文档 → 分享 → 添加应用 → 可编辑）。

## 操作 skill 示例

```python
from motus_feishu import auth, docx, media, snapshot, data_readers as dr

client = auth.get_client()
doc_id = docx.parse_doc_id("https://xxx.feishu.cn/docx/<token>?...")

# 读文档
meta_and_blocks = docx.dump_document(client, doc_id)   # 元信息 + 全量 block
plain = docx.document_raw_text(client, doc_id)         # 纯文本

# 写周报文档（追加为主）
docx.append_text(client, doc_id, "本周进展（2026-08-12）", heading_level=1)
docx.append_text(client, doc_id, "loss 从 0.5 降到 0.3 ...")
docx.insert_table(client, doc_id, [
    ["数据集", "样本数", "占比"],
    ["agibot_beta", "12000", "40%"],
])

# 插图片 / 插视频（视频无独立 block，用 file block 插入 mp4）
media.insert_image(client, doc_id, "/tmp/loss_curve.png")
media.insert_video(client, doc_id, "/tmp/rollout.mp4")

# 快照与恢复
snapshot.snapshot_document(client, doc_id)
snapshot.restore_document(client, doc_id, confirm=True)   # 默认恢复最新快照
```

CLI：`motus-feishu-cli {auth-check, doc dump, doc append, doc register, doc list, doc unregister, doc snapshot, doc restore, upload}`，写操作支持 `--dry-run`。

## 可访问文档索引（自动注册约定）

应用无法枚举用户个人云盘的文档，可访问文档只能靠显式链接维护。库在本地维护
一份"已确认可访问"的文档清单（`~/.local/share/motus-feishu/docs_index/index.json`，
用户级目录，不入任何 git 仓库）：

- **用户提供飞书文档链接时，agent 应自动执行**
  `motus-feishu-cli doc register <链接> [--note 用途]`，把该文档记入索引；
  若用户同时说明了用途（如"周报文档"），用 `--note` 记录。
- register 会先校验可访问性（调 `get_document_meta`）：应用无权限/文档不存在时
  报错且不写入，此时向用户反馈"应用尚无该文档权限，需在文档分享中添加应用为协作者"。
- **用户询问"有哪些可访问文档 / 你记得哪些文档"时**，执行
  `motus-feishu-cli doc list` 展示清单（含标题、备注、注册时间）。
- 移除记录用 `motus-feishu-cli doc unregister <链接>`（只删本地记录，不动飞书文档）；
  误删无恢复，谨慎使用。

## 安全红线（勿绕过）

- **编辑前自动快照**：库对所有写操作默认先落本地快照
  （`~/.local/share/motus-feishu/backups/<doc_id>/<UTC时间戳>.json`），
  不要用 `auto_snapshot=False` 绕过，除非有明确理由。
- **优先追加，避免改写**：写周报等场景一律用 append；`update_text_block` /
  `delete_root_children` 等破坏性操作必须用户明确确认（代码里 `confirm=True`，
  CLI `--confirm`）。
- **误改恢复流程**：优先 `motus-feishu-cli doc restore <doc> --confirm`
  （本地快照），其次飞书自带版本历史（文档 → ⋯ → 历史记录）网页端回滚。
- **全局只读开关**：`MOTUS_FEISHU_READONLY=1` 时库拒绝一切写操作；
  不确定时先用 `--dry-run` 预演。
- **H 集群只读**：`data_readers` 只允许白名单命令
  （ls/cat/tail/head/grep/du/df/find，find 禁 -delete/-exec 等动作参数）
  和 scp pull-only（远端→本地）；绝不执行任何写/删命令，绝不向 H 集群推送。

## 典型工作流：自动周报

1. 本地：`dr.read_jsonl_metrics(<monitor run_dir>)` 读指标、
   `dr.tail_log(<train.log>)` 看训练状态；
2. H 集群（如需要）：`dr.run_h_command("tail -n 50 <log>")`、
   `dr.fetch_h_file(<remote>, <local>)` 拉取只读数据；
3. 汇总生成周报内容（写作格式见 `09_weekly_report.md`）；
4. `docx.append_text` / `insert_table` / `media.insert_image` 追加到周报文档；
5. 周报示例：`tmp/dev_20260807_prompt_guidance_report.md` 等生成物可
   `media.insert_file` 附件化。
