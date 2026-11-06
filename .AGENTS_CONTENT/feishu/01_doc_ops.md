# Skill 01：文档读写操作

飞书 docx 文档的读、追加、表格与媒体插入。所有写操作默认先自动快照
（见 02_safety.md），目标文档需已把应用加为协作者（可编辑）。

## Python 接口

```python
from motus_feishu import auth, docx, media, snapshot

client = auth.get_client()
doc_id = docx.parse_doc_id("https://xxx.feishu.cn/docx/<token>?...")

# 读文档
meta_and_blocks = docx.dump_document(client, doc_id)   # 元信息 + 全量 block
plain = docx.document_raw_text(client, doc_id)         # 纯文本

# 追加文本 / 标题（写文档一律优先追加，不改写已有内容）
docx.append_text(client, doc_id, "本周进展（2026-08-13）", heading_level=1)
docx.append_text(client, doc_id, "loss 从 0.5 降到 0.3 ...")

# 插表格（header 行 + 数据行）
docx.insert_table(client, doc_id, [
    ["数据集", "样本数", "占比"],
    ["agibot_beta", "12000", "40%"],
])

# 插图 / 插视频 / 插文件 —— 一律三步法（2026-08-21 实测；media.insert_image/
# insert_video/insert_file 的"带 token 直接建块"已被 API 拒绝：1770001 invalid param）：
#   ① children.create 建空媒体块  ② upload_media 绑到媒体块 id  ③ patch replace 回填
from lark_oapi.api.docx.v1.model import (
    PatchDocumentBlockRequest, ReplaceFileRequest, ReplaceImageRequest, UpdateBlockRequest,
)
from motus_feishu.docx import Block

def _create_empty(client, doc_id, parent_id, blocks, index=None):
    body = docx.CreateDocumentBlockChildrenRequestBody.builder().children(blocks)
    if index is not None:
        body = body.index(index)
    req = (docx.CreateDocumentBlockChildrenRequest.builder().document_id(doc_id)
           .block_id(parent_id).request_body(body.build()).build())
    resp = docx._check_response(client.docx.v1.document_block_children.create(req), "建空媒体块")
    return [b.block_id for b in (resp.data.children or [])]

def _patch_replace(client, doc_id, block_id, kind, token):
    b = UpdateBlockRequest.builder()
    b = (b.replace_image if kind == "image" else b.replace_file)(
        (ReplaceImageRequest if kind == "image" else ReplaceFileRequest)
        .builder().token(token).build())
    req = (PatchDocumentBlockRequest.builder().document_id(doc_id)
           .block_id(block_id).request_body(b.build()).build())
    resp = client.docx.v1.document_block.patch(req)
    assert resp.success(), f"replace_{kind}: {resp.code} {resp.msg}"

def insert_image3(client, doc_id, path, width=None, height=None, parent=None, index=None):
    """图片三步法；parent 缺省为文档根，也可传表格 cell_id（配 index=0 插到最前）。"""
    payload = {"width": width, "height": height} if width else {}
    bid = _create_empty(client, doc_id, parent or doc_id,
                        [Block({"block_type": 27, "image": payload})], index)[0]
    token = media.upload_media(client, path, bid, parent_type="docx_image")  # ② 绑图片块 id
    _patch_replace(client, doc_id, bid, "image", token)                      # ③ 打到图片块 id
    return bid

def insert_video3(client, doc_id, path, parent=None, index=None):
    """视频/文件三步法。空 file 块会被 API 自动包一层 View(33)，必须取内层 File(23) 块操作。"""
    view_id = _create_empty(client, doc_id, parent or doc_id,
                            [Block({"block_type": 23, "file": {"view_type": 1}})], index)[0]
    req = (docx.GetDocumentBlockChildrenRequest.builder()
           .document_id(doc_id).block_id(view_id).page_size(50).build())
    resp = docx._check_response(client.docx.v1.document_block_children.get(req), "取内层 File 块")
    file_bid = resp.data.items[0].block_id
    token = media.upload_media(client, path, file_bid, parent_type="docx_file")  # ② 绑内层块 id
    _patch_replace(client, doc_id, file_bid, "file", token)                      # ③ 打到内层块 id
    return view_id

# 富文本与公式（2026-08-20 起，blocks.py）
from motus_feishu import blocks
docx.append_blocks(client, doc_id, [
    # 混合段：str 自动转普通文本；text_element 支持 bold/italic/inline_code/link；
    # equation_element 为行内 LaTeX 公式（KaTeX 语法）
    blocks.text_block([
        blocks.text_element("Overall", bold=True), " 为等权求和 ",
        blocks.equation_element(r"O=\mathrm{SceneC}+\mathrm{HSD}+\cdots"),
    ]),
    blocks.bullet_block(["均值 ", blocks.equation_element(r"\bar{x}=\frac{1}{n}\sum_i x_i")]),
    # 行间公式：行内公式元素独占一个 text block，align=2 居中
    # （块级公式 equation_block 创建会被 API 拒绝，见注意事项）
    blocks.text_block([blocks.equation_element(r"O=\sum_i w_i m_i")], align=2),
])

# 手动快照 / 恢复
snapshot.snapshot_document(client, doc_id)
snapshot.restore_document(client, doc_id, confirm=True)  # 破坏性，需 confirm
```

## CLI 等价操作

```bash
motus-feishu-cli auth-check                        # 验证凭据
motus-feishu-cli doc dump <链接> --format text      # 读
motus-feishu-cli doc append <链接> --text "..." --dry-run   # 预演（不触网）
motus-feishu-cli doc append <链接> --text "..."     # 追加
motus-feishu-cli doc snapshot <链接>
motus-feishu-cli doc restore <链接> --confirm
motus-feishu-cli upload <链接> /tmp/fig.png
```

## 注意事项

- 写操作先 `--dry-run` / `dry_run=True` 预演，确认无误再真写。
- 媒体上传到某文档后只能用于该文档（token 作用域）。
- **媒体必须三步法**（2026-08-21 实测，上文 `insert_image3`/`insert_video3`）：带 token
  直接建 image/file 块报 `1770001 invalid param`；replace 时 upload 的 `parent_node`
  绑错对象（如绑文档 id 而非媒体块 id）报 `1770013`；`replace_file`/`replace_image`
  打错块类型（如打到 file 块外层的 View(33) 而非内层 File(23)）报
  `1770025 operation and block not match`。表格 cell 内插媒体：建表时 cell 只放文本，
  建表后 list 表格子块拿 cell_id（row-major），再 `_create_empty(..., parent=cell_id,
  index=0)` 走同一三步法。
- 视频用 file block 插入，网页端可预览播放；图片 `upload_all` 上限 20MB，
  更大文件自动分片。
- 表格等嵌套容器在 restore 时会跳过（保真度限制），重要表格恢复需配合
  快照 JSON 手动重建或用飞书版本历史。
- 公式：`equation_element`（行内 KaTeX）实测可用；`equation_block`（块级公式，
  block_type 34）创建会被 API 拒绝（1770001 invalid param，2026-08-20 实测）——统一用
  行内公式元素，长公式独占一个 text block。
- **行间公式默认居中**：独占一个 text block 的公式默认带 `align=2`
  （`blocks.text_block([blocks.equation_element(...)], align=2)`；用户 2026-08-20 拍板，
  是默认样式而非硬性要求，特殊排版需求可例外）。
  对已有文档批量修正用 `UpdateTextStyleRequest(style=TextStyle(align=2), fields=[1])` patch。
- 大批量 `append_blocks` 需分批（单次 ~78 块报 99992402 field validation failed，
  实测 20 块/批正常）。
- 文字颜色（2026-08-20 实测）：`text_element_style` 的 `text_color` /
  `background_color` 枚举 1~7（粉红/橙/黄/绿/蓝/紫/灰），raw dict 直写即可、
  读回持久化；TextElementStyle 无字号字段（bold/italic/strikethrough/underline/
  inline_code/background_color/text_color/link），"更小字体"只能靠浅色替代。
- 表格单元格多分点：table_cell 的 children 放多个 text block（每段一个）创建
  正常（2026-08-20 实测）；`equation_element` 在表格单元格内同样可用。
