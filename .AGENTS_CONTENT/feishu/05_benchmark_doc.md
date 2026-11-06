# Skill 05：预训练基准文档维护（EWMBench / WorldModelBench）

基准文档（链接：<https://my.feishu.cn/docx/WxKadSPApokyv0xTpmGcVhRTnjh>，已 register）
汇总 ThinkerWM 各训练版本在两个基准上的评测结果。2026-08-17 已按用户要求原地重排并
确立下列维护口径；**每次新增评测数据或调整表格/曲线时都按本文件执行**。

## 文档当前结构

- H1「基准」+ 口径说明段
- H2「EWMBench」：H3 外部基线（11 行，取自 Qwen-RobotWorld 论文 Table 2）→
  H3 ThinkerWM 各版本结果（大表）→ 注释段 → H3 曲线图（2 张：Overall + 8 子指标小倍数图）
- H2「WorldModelBench」：同构（基线取自论文 Table 5；曲线图 2 张：Total + 10 子指标小倍数图）
- H2「WorldArenaBench」（2026-08-20 新增）：简介段 → H3 ThinkerWM 各版本结果
  （版本|Step|样本数|15 指标，无外部基线）→ 注释段 → H3 曲线图（15 指标小倍数图）。
  指标为 CSV 原名；motion_smoothness（缺 mamba_ssm 未算）/action_following（未产出）/
  EWMScore_P（pending formula）不列入表格，表头暂不加 ↑（方向/量纲未完全统一）。
- H2「版本变更记录」：stage0/stage1、0721、0731、0802、0803、0805、0807、0812、0815、0818_v1、
  0818_v2 各一节（H3）。2026-08-21 按用户拍板改为**散文分点格式**（不用表格）：每节内每段一处
  功能改进，统一为「功能标题：<原因><实现内容><实现效果（若有）>」自然语言表述（杜绝
  「源数 14→15」式电报体），与周报附录「实现功能」内容形式一致；分点小标题加粗紫色
  （text_color=6）、版本 H3 的括注灰色（7）、节下注记灰色；版本名统一 `stage2_YYYYMMDD_vN`。
  脚本 `tmp/benchmark_reorg_20260817/reformat_version_log_20260821.py`（先插后删、锚点校验）。
- H2「指标含义」：H3 EWMBench 指标（9 个分点）+ H3 WorldModelBench 指标（11 个分点）+
  H3 WorldArenaBench 指标（18 个分点：15 表内指标 + motion_smoothness/action_following 未产出说明
  + EWMScore 聚合）。2026-08-20 起改为逐指标分点详解（粗体指标名 bullet + 含义/计算方式 +
  LaTeX 公式），原两张三列速查表已删；指标定义来源论文 arXiv:2505.09694（EWM）/
  2502.20694（WMB）/2602.08971（WorldArena），调研稿含逐字原文佐证，存
  `tmp/benchmark_reorg_20260817/glossary_{ewm,wmb,wab}.md`。

## 数据口径（用户 2026-08-17 拍板，后续更新必须遵守）

- **表格按基准区分**；ThinkerWM 表列 = 版本 | Step（| 样本数，仅 WMB）| 各指标，
  **不放评测备注列**。版本名统一 `stage0_20260706` / `stage1_20260720` / `stage2_YYYYMMDD_vN`
  （对应 `configs/stage2_*.yaml` 与 checkpoints 目录）。
- **每个「版本 × Step」只保留一行**，取舍规则：
  1. 统一取 **V100 (FP16+math-SDPA)** 环境的评测（A100/A6000 行不列入）；
  2. 同 step 有 **adjust prompt** 变体时取 adjust prompt；
  3. WMB 同 step 多样本量时取**样本量大者**（350 sample，与外部基线 350 实例口径一致）。
- **新评测数据存放约定（用户 2026-08-20 拍板）**：新数据统一保存在对应 ckpt 目录内——
  `…/checkpoints/<版本>/<任务名>/checkpoint_step_<N>/<版本>_checkpoint_step_<N>/eval_results/`
  （含 `manifest.json` 记录 train_run/step/样本数/来源 run 路径，及
  `EWMBench/ewm_all_metrics.csv`、`WorldModelBench/wmb_all_metrics.csv` 等各基准子目录）。
  更新文档时从该路径拉取（如 `/save/<版本>_checkpoint_step_<N>/`），不再到处找散置结果。
- **曲线图颜色 + marker 形状双重区分版本**：tab10 配色 + 10 种 marker
  （o s ^ D v P X < > h）按版本序循环；**x 轴不按 step 数值、按表格行先后顺序等距排布**
  （用户 2026-08-20 拍板；tick 标签为 step），同版本各点实线连接、相邻两版本之间用
  后一版本颜色的虚线连接；**所有曲线图都必须加 Qwen-RobotWorld 基线**（2026-08-21 用户拍板：
  全部指标都加，不只是总分）——Overall/Total 单图加 `axhline`，子指标小倍数图每个子图加
  该指标自己的基线（灰色点线 + 右端 `QRW x.xx` 标注），数值从 content.json 的
  `ewm_baselines` / `wmb_baselines` 读取（`make_charts.py` 的 `qwen_baselines` /
  `add_qwen_baseline`），禁止硬编码。
  图内标签全英文（matplotlib 无 CJK 字体）。

## 维护工作流（脚本在 `tmp/benchmark_reorg_20260817/`）

1. `extract_data.py`：`dump_document`（或读最新快照）解析全部表格 → `data.json`。
   版本归属规则：日期小表看所在 H1 标题；后期行从 Version 单元格的 ckpt 路径正则
   （`checkpoints/(stage2_\d{8}_v\d+)/.*checkpoint_step_(\d+)`）解析。
2. `content_build.py`：组装 `content.json`（基线表 + ThinkerWM 行 + 指标表），
   内含 `dedup_pick` 实现上述取舍规则；有内置断言抽查关键数值，改动后必须全绿。
3. `make_charts.py`：用系统 python（motus311，有 matplotlib；feishu_venv 没有）生成 3 张 PNG。
4. 写入文档，二选一：
   - **增量更新**（`apply_update.py`，首选）：`insert_table(..., index=i)` 在同位插新表 →
     `delete_root_children(i+1, i+2, confirm=True)` 删旧表（先换靠后的表避免索引位移）；
     图片用 replace_image 原位替换；注释文本用 `update_text_block(..., confirm=True)`。
   - **整体重建**（`rebuild_doc.py`，结构大改时才用）：先快照 →
     `delete_root_children(0, N, confirm=True)` 清空 → 按结构顺序 append/insert。
5. 验证：重新 dump，逐格比对两张 ThinkerWM 表与 content.json（必须 0 误差）、
   检查表尺寸/图片 token/注释文本。

若 `tmp/` 被清理，按本文件第 1–5 步重写脚本即可；数据真源是文档自身快照
（`~/.local/share/motus-feishu/backups/<doc_id>/`）与本文档。

## 飞书 API 经验（2026-08-17 实测）

- `insert_table` 单次可建 30×12（721 个 block）的表，无需分块；不必实现 insert_table_row 拼接。
- **块级公式（block_type 34）创建被拒**（1770001 invalid param，2026-08-20 实测）——公式一律用
  行内公式元素（`blocks.equation_element`，可混入 text/bullet 等块，长公式独占一个 text block）。
- **`append_blocks` 单次 ~78 块会报 99992402 field validation failed**（2026-08-20 实测；逐块均合法），
  分批 20 块/次正常——大批量追加务必分块。
- **插图必须三步法**：① children.create 建**空** image block（`{"block_type":27,"image":{}}`）→
  ② 上传媒体时 `parent_node` 传**图片块 id**（`parent_type="docx_image"`）→
  ③ `UpdateBlockRequest.replace_image(token)` 回填。
  库 `media.insert_image` 直接带 token 建块会被拒（1770001 invalid param）；
  replace_image 报 1770013 relation mismatch = 上传时 parent_node 传成了文档 id。
- wiki 链接当 docx token 直接读写通常可行，但媒体上传对 wiki 文档会报
  1061044 parent node not exist（临时中转文档即如此；基准文档是原生 docx，正常）。

## 数据注意点

- WMB 前 2 行（Step 4000/16000）为早期记录，原表未标注训练版本/ckpt，版本列保留
  「—（未记录）」；原有 3 行（两个 16000），按去重规则舍 1 行并在注释中说明。
- fast 模式行只有 SceneC/HSD/Dyn/nDTW/Diversity 五项，BLEU/CLIP/Logics/Overall 留空
  （如 0803_v1 step 2000 行只有 SceneC）。
- 版本变更记录来源：0721/0731/0803/0805 = 文档原版本段落；0807/0812 =
  `tmp/update_log_dump_20260816.txt`（预训练主干更新日志 dump）；0815 = git log
  （92c79e5 / 465fa9a / 3b51f09）；0815 补充（egoscale 坐标系/7 节点）、0818_v1 全节与
  0818_v2 = 2026-08-21 周报附录已审定内容（`tmp/append_weekly_20260821.py`）。
  新版本补记时优先查这三类来源 + 当周周报附录。
- 外部基线数值来源：Qwen-RobotWorld 论文（arXiv 2606.17030）Table 2（EWMBench）/
  Table 5（WorldModelBench），存档 `tmp/papers/2606.17030_qwen_robotworld.txt`。
- 2026-08-20 新增 stage2_20260818_v1 step 4000 行（EWM standard V100 + WMB 56 sample，
  脚本 `update_content_20260820.py` / `apply_update_20260820.py`；评测原始结果已拉至本地
  `/save/stage2_20260818_v1_checkpoint_step_4000/`，其中还含 WorldArenaBench 结果，
  该基准暂未纳入本文档结构）。
