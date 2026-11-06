# Skill 04：自动周报工作流

用本库把训练数据汇总成周报并追加到周报文档的典型流程。周报的写作格式与
数据源细节由本仓库的文档定义（见
`.AGENTS_CONTENT/09_weekly_report.md`），本文件只描述库侧的数据收集与写入
步骤。

## 1. 收集数据（只读）

```python
from motus_feishu import data_readers as dr

# 本地 monitor JSONL 指标
rows = dr.read_jsonl_metrics("/save/outputs/<run>/monitor", last_n=100)
# 训练日志 tail
lines = dr.tail_log("/save/outputs/<run>/train.log", n=200)
# tensorboard 标量（需 tbparse 或 tensorboard）
scalars = dr.read_tensorboard_scalars("/save/outputs/<run>/tb")

# H 集群（白名单只读命令 + scp pull，见 02_safety.md）
r = dr.run_h_command("tail -n 50 /save/outputs/<run>/train.log")
dr.fetch_h_file("/save/outputs/<run>/metrics.jsonl", "/tmp/metrics.jsonl")
```

## 2. 汇总生成内容

按本仓库的周报格式约定整理（`09_weekly_report.md`）：本周进展、关键指标对比
（含耗时/性能变化）、问题与修复、下周计划。图表先生成为本地图片文件。

## 3. 追加到周报文档

```python
from motus_feishu import auth, docx, media

client = auth.get_client()
doc_id = docx.parse_doc_id("<周报文档链接>")   # 未注册过会先 register（见 03）

docx.append_text(client, doc_id, "本周进展（2026-08-13）", heading_level=1)
docx.append_text(client, doc_id, "...")
docx.insert_table(client, doc_id, [["指标", "上周", "本周"], ["loss", "0.5", "0.3"]])
media.insert_image(client, doc_id, "/tmp/loss_curve.png")
media.insert_file(client, doc_id, "/tmp/详细报告.md")   # 长报告附件化
```

## 4. 注意

- 一律**追加**，不改写历史周报内容（02_safety.md）。
- 首次写入前可先 `doc dump` 确认文档结构；写操作支持 `--dry-run` 预演。
- 若周报文档链接未在本地索引，按 03_doc_index.md 的约定先 register。

## 5. 大组周报场景（thinkerwm-pretrain）

「大组周报」文档已 register（note=周报文档，链接用 `motus-feishu-cli doc list`
查看）。数据收集见 §1，本节是该文档的模板、写作结构与 block 级写入方法
（2026-08-13 首次实战后沉淀，2026-08-20 模板定稿）。

### 文档模板（每周一节，追加在文档末尾；2026-08-20 用户四轮拍板定稿）

- H1：`YYYY-MM-DD | YYYYWww`（周五日期，如 `2026-08-21 | 2026W34`）。
- H3 `上周计划`：照抄上一节的"下周计划"原文。
- H3 `本周总结`：加粗小标题 `已完成` / `进行中` / `风险同步` / `附录（AI生成）`。
  - `已完成`：**整节由成员本人撰写，只建加粗标题**——不写内容，也不代列
    概括性标题；`进行中`同样只留标题。
  - `风险同步`：无要求填「无」，不转述无法核实的旧风险。
  - `附录（AI生成）`：位于风险同步之后、下周计划之前，与风险同步同级
    （加粗小标题），约定见下节。
- H3 `下周计划`：只建 `【任务一】/【任务二】` 空占位，内容由成员本人填写，
  agent 禁止代写。
- 历史周次的【任务】双栏 grid 布局不必复刻，纯文本段落即可。

### 附录（AI生成）约定（2026-08-20 定稿）

- 数据范围：本周廖琛在主仓 + motus-data-factory / motus-feishu 子模块的
  commit（motus-monitor 是主仓内库、非 submodule；子模块署名差异见
  `09_weekly_report.md` §1）；开头一行灰色注记写明统计范围。
- 结构：拆两个字表，各配一个加粗蓝色（text_color=5）分组小标题——
  「训练版本（按版本演进排列）」与「通用工程与库（非版本绑定）」。
- 表格：两列「版本/模块 | 实现功能」，**不要 commit 列**；单元格内多段落，
  每段一处功能改进，统一为「功能标题：<原因><实现内容><实现效果（若有）>」，
  自然语言表述（杜绝「源数 14→15」式电报体），必要时用行内 KaTeX 公式
  （equation_element 在表格单元格内实测可用）；效果只写有依据的
  （commit / monitor 实测数字）。
- 排版：分点小标题加粗紫色（text_color=6）；版本号非粗体、括注灰色（7）；
  注记灰色（7）。飞书 docx 的 TextElementStyle 无字号字段，"更小字体"
  只能以浅色代替。

### 写作结构（用户明确要求，违反即返工）

（本节适用于成员要求 agent 代写「已完成」内容的场景；2026-08-20 起本周总结的
已完成/进行中只建标题，附录写作以「附录（AI生成）约定」一节为准。）

1. **结果先行**：第一条 bullet 是"本周核心结果"——实现了什么效果 + 实测数字
   （step、指标趋势、吞吐）；带防误读注释（如"归一化已改、指标绝对值与旧版不可比"）。
2. **路径随后**：之后每条 bullet 对应一个效果，说清"该效果通过什么实现"
   （诊断结论 → 落地手段）。
3. **实验经验单列**：一条"本周实验经验"bullet，列可复用结论（指标刻度误读、
   配置陷阱、被实验反证无效的方向等）。**实验结果与实验经验是周报最被看重的
   内容，其余一切围绕实验组织。**
4. **杂项收尾**：训练稳定性修复、monitor/环境/库等工程项合并为最后一条"杂项"。
5. **禁内部编号**：不写"修复轮 #1-#4"、"对齐 B1/B2/B3/D1"这类只有报告作者看得懂
   的引用；必须回到原始报告（问题报告/对照报告的清单表）展开为具体内容。
6. 简洁、说结论不说过程。

### block 级写入方法

- 组装：`block_lib.heading_block / text_block / bullet_block`，一次
  `append_blocks` 全量提交（只落一个快照）。
- 加粗小标题：`Block({"block_type": 2, "text": {"elements": [text_element(x, bold=True)], "style": {}}})`。
- 富文本样式（2026-08-20 实测）：text_element 的 style 支持
  `text_color` / `background_color`（枚举 1~7：粉红/橙/黄/绿/蓝/紫/灰），
  直接写 raw dict 即可；TextElementStyle 无字号字段。多分点单元格：
  table_cell 的 children 放多个 text block（每段一个），创建正常；
  `equation_element` 在表格单元格内可用。附录的 `styled()` / `point()` /
  `rich_table_blocks()` 助手见 `tmp/append_weekly_20260821.py`。
- 定位改写：`docx.list_root_children` 按根索引定位（`block_type` 31=表格、
  12=bullet、5=heading3、3=heading1），删除用
  `delete_root_children(start, end, confirm=True)`（区间 [start, end)，
  自动快照），然后在原 index 重新 `append_blocks` / `append_descendants`；
  用户在对话中明确要求的改写即视为确认；改前先 `--dry-run` 预演，
  改写仅限本周新增内容，不动历史周次。
- 生成脚本留在本仓库 `tmp/append_weekly_<date>.py` 备下周复用；若后续改写了
  文档，把脚本内容同步成最终版。

## 6. 具身X项目「每周重点进展」文档场景（2026-08-21 用户三轮拍板定稿）

「具身X项目每周重点进展」文档（docx token `MUbvdyxUko6csUxLZrocBACznTb`）
按周更新，分「训练主干」「Infra」两节。以下要求均为用户明确拍板，违反即返工。

### 写作要求

1. **训练版本与重点 commit**：与大组周会附录同一口径——每项用自然语言
   （必要时行内 KaTeX 公式）说清 **修改原因/动机 → 修改内容 → 修改后结果
   （若有，只写有依据的）**；大组周会内容有机融入，不两份各说各话。
2. **数据源**：monitor 本地数据（`~/.local/share/motus_monitor_data`，见
   `09_weekly_report.md` §2）+ 预训练主干 Benchmark 评测数据（口径见
   `05_benchmark_doc.md`）；系统性展示本周成效，附折线图、视频等多媒体。
3. **主干不看 monitor 评估指标**：MSE / PSNR / action_mse 等 monitor 周期
   评测指标一律不进主干（用户原话"去除评估的 mse 数据"）——主干「评测结果」
   只保留 Benchmark（一句话总结 + EWMBench/WMB/WAB 表 + 图 + 注）；monitor
   周期评测内容（评测表 / psnr 图 / 版本注记）整体归到 **Infra 节**。
   Qualitative 观察格也不写 mse 数字。
4. **Infra 节**：按 git log + monitor 数据填；monitor 数据要给足支撑并
   **绘制各类图**（评测覆盖、数据规模等）；**每一项都必须写 修改原因 /
   修改方式 / 修改成效**，成效用 monitor 实测或 commit 实测数字量化。

### 文档结构（W34 定稿版式）

- **训练主干 @廖琛**：训练记录表（含公式）→ loss 曲线图 → 单步耗时图
  （caption + 链式耗时/稳态分解）→ commit 两表（训练版本表按版本演进 +
  通用工程表）→ 评测结果（仅 Benchmark）→ Qualitative 表（视频 + 曲线
  在 cell 内）。
- **Infra @廖琛**：一句话总结 bullet（monitor 量化：指标条数 / 评测轮次·
  样本 / 媒体数 / 落盘量 / 吞吐变化）→ Infra 重点表（每行 原因/方式/成效
  + 数字）→「monitor 数据支撑」（评测覆盖图 + 数据规模图）→「monitor
  周期评测产出」（评测表 + psnr 折线图 + 版本注记）。
- **下周计划 / 需决策**：只留空占位，成员本人填写（同
  `09_weekly_report.md` 硬性约束）。

### 流程与质量红线

- 图：生成后必须验证非空白、无 CJK 豆腐块（**图内文字一律英文**）；无法
  真看图时用尺寸/数据/字体告警做间接验证，并在汇报中如实说明。
- 数据采集快照留存 `tmp/`（如 `infra_monitor_summary.json`），数字可溯源。
- 主脚本 `tmp/rewrite_weekly_<date>.py` 每次修订后同步为最终态并
  `--dry-run` 验证；改文档前先 dump、改后再 dump 验证结构，快照备份。
- 文档中非自己创建的块（如他人实时编辑产生的空块）不动；训练状态定格时点
  与评测数据刷新口径在汇报中向用户说明。
- 媒体三步法（图/视频/文件）与表格行列删除 API 见 `01_doc_ops.md`。
