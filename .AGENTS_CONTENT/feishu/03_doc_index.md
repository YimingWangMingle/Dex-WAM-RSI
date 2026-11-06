# Skill 03：可访问文档索引（自动注册约定）

应用无法枚举用户个人云盘的文档，可访问文档只能靠显式链接维护。库在本地维护
一份"已确认可访问"的清单（`~/.local/share/motus-feishu/docs_index/index.json`，
用户级目录，不入任何 git 仓库）。

## agent 行为约定

- **用户提供飞书文档链接时**，自动执行
  `motus-feishu-cli doc register <链接> [--note 用途]`，把该文档记入索引；
  用户同时说明了用途（如"周报文档"）时用 `--note` 记录。
- register 会先调 `get_document_meta` 校验可访问性：应用无权限/文档不存在
  时报错且不写入。此时向用户反馈"应用尚无该文档权限，需在文档分享中添加
  应用为协作者（可编辑）"。
- **用户询问"有哪些可访问文档 / 你记得哪些文档"时**，执行
  `motus-feishu-cli doc list` 展示清单（标题、备注、注册时间，倒序）。
- 移除记录用 `motus-feishu-cli doc unregister <链接>`（只删本地记录，不动
  飞书文档）；误删无恢复，谨慎使用。

## Python 接口

```python
from motus_feishu import doc_index

doc_index.register_document(url, note="周报文档")   # 重复注册会更新标题/备注
docs = doc_index.list_documents()
doc_index.remove_document(url)
```
