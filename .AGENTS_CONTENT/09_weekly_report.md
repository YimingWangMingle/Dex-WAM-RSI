# 周报撰写流程（每周报告 skill）

为 `/save/每周报告/`（工作目录外）下的周报编写某成员/某主题内容时的完整流程。示例团队报告格式见 `/save/每周报告/` 下的 `* M8W1.md`（月+周命名）。基础版 git 总结流程见 `08_conventions.md`「工作总结流程」，本文件是其周报场景的完整扩展。

## 触发与格式

- 触发：用户要求"参考 `/save/每周报告` 中的 md 示例，基于 git 记录 + monitor 统计，为某人编写周报内容"。
- 格式样板（对齐团队报告）：标题 `# <姓名> M<月>W<周>（MMDD-MMDD）周报`；顶部 `**一句话总结**`；正文按团队报告的**对应小节**组织（如 `- **世界模型**` 下的 `1. **AI Infra** @人@人`、`2. **训练主干** @人@人`），每节先 `**一句话总结**`，再列要点；数字尽量进表格。
- **不要用反斜杠转义**（示例文件里的 `0801\-0807` 是历史产物，不要模仿）。
- 输出到 `/save/每周报告/<姓名> M<月>W<周>.md`，独立成文件；不修改团队总报告，除非用户明确要求插入。

## 1. 收集 git 提交

- 主仓：`git log --author=<姓名> --since=<起始> --until=<结束> --format='%h|%ad|%s' --date=format:'%m-%d %H:%M' --no-merges` 列范围提交；`--shortstat` 看改动规模；`git log --author=<姓名> --format='%ad|%s' --date=format:'%Y-%m-%d'` 看全量时间分布。
- motus_data 子模块（`libs/motus-data-factory/`）：**作者署名可能与主仓不同**（例：主仓 `廖琛`，子模块 `Liao, Chen`），先用 `git log --format='%an|%ae' | sort | uniq -c` 确认署名再按作者过滤；子模块合并动作由主仓 `chore(submodule): update ...` 提交体现。
- 周范围：会话起始日期可能陈旧，用 `date` 取真实日期；周报通常按周一~周日。本周无提交但有合并/评审动作时，以"合并/推进"口径计入。

## 2. 拉取 monitor 统计

数据目录默认 `~/.local/share/motus_monitor_data/`（或 `MOTUS_MONITOR_DIR`、仓库上一级 `motus_monitor_data/`）——本地已含集群同步数据，**无需连集群**。用 registry 读，不要手工 parse 目录：

```python
import sys; sys.path.insert(0, "libs/motus-monitor")  # 从仓库根运行时
from motus_monitor.registry import Registry
r = Registry(os.path.expanduser("~/.local/share/motus_monitor_data"))
runs = r.list_runs()            # 全部 run，跳过 demo-* 前缀
r.get_run(run_id)               # run 详情（status/ckpt/parent 谱系）
r.list_annotations()            # benchmark 标注（EWMbench 各指标 + Overall + note）
```

- run 元数据：`runs/<id>/run.json`（配置快照）、`status.json`（`state`/`last_step`/`last_total_loss`/`last_checkpoint_step`/`max_steps`/`world_size`/`parent`（finetune/resume 来源，构成 ckpt 谱系））。`status.ts` 是最后同步时点；step 记录 `ts` 是 unix 秒。
- loss 轨迹：`runs/<id>/metrics/*.jsonl`（分块轮转、逐行 JSON），`type=='step'` 记录含 `step`/`ts`/`total_loss`（个别在 `metrics.total_loss`）；按目标 step 采样（500/1000/2000/...）画收敛。
- ckpt 事件：`runs/<id>/events/*.jsonl` 的 `type=='checkpoint'`（step+路径+耗时）。
- 周期评测：`runs/<id>/eval/*.jsonl` 的 `type=='eval_round'`（step/num_samples/ready_ranks/world_size/按源分布 sources/平均指标 video_mse/video_psnr/action_mse/seconds）。
- 注意：`status.json` 里 finetune 的 `parent.step` 可能与路径 `checkpoint_step_N` 不一致，**以配置/k8s 脚本里的路径为准**（monitor 解析字段可能滞后）。

## 3. 耗时对比（AI Infra 等效率小节）

- 单步耗时：`step_time_seconds`（或 `train_step_total_seconds`）取**中位数**，跳过前 ~50 步预热。
- 每 1k 步墙钟：用 step 记录 `ts` 在**稳定段**（如 step 1000→2000、3000→4000）的差值，不要用 run 总时长（含初始化）。
- 关键耗时字段：`model_vae_encode_video_seconds`、`model_world_model_forward_seconds`、`backward_seconds`、`dataloader_next_seconds`、`ready_queue_get_wait_seconds`、`periodic_eval_seconds`、`train_steps_per_vae_encode`/`vae_encode_repeat_index`（VAE 复用：encode 字段只在 repeat_index==0 的步出现，摊薄 = 单次耗时 / N）。
- GPU：`runs/<id>/system/node<N>/*.jsonl` 的 `gpus[]` 列表（键 `util_percent`/`mem_used_bytes`/`power_watts`；注意早期记录有脏值如 `7403840%`，用中位数/p50/p95，mean 会被污染）。
- **对比口径**：同节点数、同代码分支才有可比性；**batch size 单列一行指标**，不要写进表头（如 `0803_v2（bs 12）`）；吞吐 = bs × 步速，用墙钟保守估计（如 `(16/12)×(7.2h/5.65h) ≈ 1.7×`）。

## 4. 组织与写作

- 按团队报告对应小节归类：效率/构建/部署类进 **AI Infra**；模型/训练/评测类进 **训练主干**；跨他人小节的工作（如数据侧）单列"数据支撑（配合）"并标注。
- 语气自然专业、第一人称；loss/step/评测分等事实数据如实呈现，**不做无依据归因**（如评测分异常只列数据点 + 注明评测环境，如"wan2.2 语义对齐后首评"）。
- 数字精度：耗时取 ~0.1s/0.1h、指标取 2-3 位有效数字；来源标注"（monitor 实测）"或"（来自 monitor）"。

## 5. 硬性约束（违反即为事故）

- **禁止决定"下周计划"**：不替成员编写"下周计划 / 下一步 / 展望 / TODO"等内容——那是成员本人填写的内容，需求里没有就一律不写；用户明确要求时才写，且只能基于事实转述，不得自己规划方向。
- **每句话必须有依据**：正文每一个论断都必须能溯源到 git 提交（hash）、monitor 数据（run_id/字段）、配置文件或示例文档；**禁止离开数据和代码私自编造**——不编数字、不编进度、不编结论、不脑补动机。
- **归因克制**：只陈述数据直接支持的结论（如"VAE 编码耗时从每步 4.05s 摊薄至 ~0.54s/步"），不写无法验证的因果判断（如"是耗时下降的主要来源"）；拿不准就只列数据点或省略。
- **宁缺毋滥**：拉不到的数据不写，不要用推断填充；涉及多人/他人小节的内容标注"（配合）"，不替他人下结论。
