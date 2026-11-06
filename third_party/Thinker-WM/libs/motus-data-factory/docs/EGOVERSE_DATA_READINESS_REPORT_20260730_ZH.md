# EgoVerse 数据可用性评估报告

日期：2026-07-30
状态：Pre-production，尚未发布 EgoVerse production Catalog/release

## 1. 评估范围

本报告覆盖集群只读数据源 `/data/EgoVerse`，本地只读映射为
`/home/ubt/egoverse`。所有审计、Catalog、日志和临时产物只能写入
`/data/jiyuan.wang` 对应的项目或数据资产目录。

当前 `processed_v3` 下共有 109,292 个 `.zarr` 候选目录：

| 数据族 | 候选 Episode 数 |
|---|---:|
| Aria | 2,501 |
| EVA | 3,058 |
| Scale | 17,235 |
| Mecka freeform | 50,772 |
| Mecka flagship | 35,726 |

每个合法 Zarr store 对应一个完整 Episode。`root.total_frames` 是唯一有效
时间轴边界；物理数组可能因 shard padding 更长。主视角统一为
`images.front_1`，文本为英文 episode description 和/或英文 annotation span。

根目录确实还包含 `processed_v2`，但当前挂载内容只有 4 个测试 Episode：
`test_aria` 2 个、`test_eva` 2 个。它是旧版/测试转换产物，不是另一套完整训练
数据。本轮接入以完整的 `processed_v3` 为唯一主源，不将 v2 与 v3 混用，避免
协议差异、测试数据误入和重复采样。

## 2. 状态定义

- **A：可进入 production 构建**
  Reader、真实 payload、SamplePool、batch 和 queue 小规模闭环已经通过。
  仍需全量 Catalog 校验、SQLite index、release 注册后才能正式长训。
- **B：有价值，但需修复或冻结策略**
  数据主体可访问，但当前存在明确的数据质量或语义问题，不能直接发布。
- **C：当前不可使用**
  数据不存在、源文件损坏、协议尚未实现或已明确过期。

## 3. 数据可用性结论

| child/profile | 规模 | 状态 | 当前结论 |
|---|---:|:---:|---|
| `aria_bimanual` | Aria 总量内 | A | 真实 JPEG、14D state/action、batch、queue 通过 |
| `aria_left_arm` | Aria 总量内 | A | 真实 JPEG、7D state/action、batch、queue 通过 |
| `aria_right_arm` | Aria 总量内 | A | 真实 JPEG、7D state/action、batch、queue 通过 |
| `eva_bimanual` | 2,970 | A | 真实 JPEG、16D observed state/commanded action、batch、queue 通过 |
| `eva_right_arm` | 88 | C/B | 当前 8D contract 不可用；视频和 pose 理论上可回收 |
| `eva_left_arm` | 0 | C | 数据根目录中不存在该 profile，不能注册空 child |
| `scale_bimanual` | 17,235 | A | 真实 JPEG、14D state/action、batch、queue 通过 |
| `mecka_freeform_bimanual` | 50,772 | A | 真实 JPEG、14D state/action、batch、queue 通过 |
| `mecka_flagship_bimanual` | 35,726 候选 | B | 主体可用，但需先处理 store 和文本范围异常 |
| Lightwheel | 独立数据族 | C | 不是 Zarr；专用 reader/Catalog 尚未实现 |
| `processed_v3/scale_old` | 旧副本 | C | 已过期，必须排除，避免重复和版本混用 |

这里的 A 不代表已经能够直接加入正式长训。目前任何 EgoVerse child 都还没有
production Catalog、SQLite index 和 immutable release。

## 4. 当前可继续投入的部分

### 4.1 Aria 三个 profile

三个 profile 的真实数据读取闭环均已通过。视频为 480x640，state/action 使用
模型侧分别提供的原始索引。

需要注意：Aria 的 action 并不是真实机器人控制指令，而是 future observed
end-effector pose。它适合：

- Stage1 文本条件视频预测；
- 明确定义为 observed trajectory prediction 的任务。

它不能在没有语义说明的情况下当作真实 control action 用于 Stage2。

### 4.2 EVA bimanual

EVA 共 3,058 个 Episode，其中 88 个是 right-arm、left-arm 为 0，因此当前
bimanual 为 2,970 个。该 profile 的 state 是 observed pose+gripper，action
是 commanded pose+gripper，是当前 EgoVerse 中最适合后续真实 action
conditioned 训练的 child。

### 4.3 Scale bimanual

真实闭环通过。视频为 480x640，14D state/action 当前都保留 raw observed
end-effector pose。Stage2 使用边界与 Aria 相同：它不是控制命令。

### 4.4 Mecka freeform bimanual

真实闭环通过。视频为 360x640，默认使用等比例缩放和居中补零进入模型分辨率，
不拉伸画面。14D state/action 是 raw observed pose，不是真实控制命令。

## 5. 当前不能直接使用的部分

### 5.1 EVA right-arm

全量审计覆盖 88/88 个 Episode、114,378 个有效 timestep：

- `right.obs_ee_pose`：全部有限；
- `right.cmd_ee_pose`：全部有限；
- `right.obs_gripper`：114,378/114,378 均为 NaN/Inf；
- `right.cmd_gripper`：114,378/114,378 均为 NaN/Inf。

官方转换代码会对旧数据执行 `(x-min)/(max-min)`，但没有处理恒定 gripper
导致的零分母，符合本次异常形态。当前处理原则：

- 不关闭 finite validation；
- 不在数据工厂里静默补零；
- 不把该数据伪装成合法 8D state/action。

推荐恢复顺序：

1. 数据所有者修复并重转 gripper，恢复完整 8D contract；
2. 若确认该本体本来就没有有效 gripper，另建语义明确的 7D pose-only profile；
3. 若仅希望用于 Stage1，需新增显式 video/text-only 或可选模态 contract。
   在该 contract 实现前，不通过全局关闭数值检查来绕过问题。

### 5.2 EVA left-arm

完整扫描结果为 0 个 Episode。这不是 reader bug，而是数据不存在。保持 profile
代码模板可以，但不能生成或注册 production child。

### 5.3 Mecka flagship

全量 annotation 审计结果：

- `.zarr` 候选目录：35,726；
- 合法根 Zarr store：35,725；
- 不完整转换残留：1；
- annotation：77,464 条；
- 存在 annotation 范围异常的 Episode：1,974（5.53%）；
- 部分可达但尾部超过 `total_frames`：1,974 条；
- 完全不与有效帧相交：8 条；
- 最大尾部超量：2,186 帧。

不完整目录
`692ea8fb727c13b350cb7cc6.zarr` 缺少根 `zarr.json`，内部只有部分
`left.obs_keypoints` 数据，不能视为 Episode。

文本覆盖方面：

- 27,029 个 Episode 没有 span annotation；
- 另有 27,029 个 Episode 没有 episode description；
- 首轮审计未记录两个集合的交集，暂不能断言有多少 Episode 两种文本都没有。

因此 flagship 当前不能直接发布。可恢复方案：

1. 不完整 store 进入显式 quarantine 报告，不写入 Catalog；
2. 对 `start < total_frames < end` 的部分可达 span，将有效范围裁为
   `[start, total_frames)`，并在 Catalog metadata 保留原始 `[start,end)`；
3. 对 `start >= total_frames` 的完全无交集 span，不构造伪对齐；
4. Catalog 保留无文本 Episode 的真实状态并标记 `missing_instruction`；
5. 文本条件训练的 SamplePool 只选择有有效文本映射的窗口；
6. 重新运行旗舰 profile 的真实 JPEG、numeric、batch 和 queue smoke。

### 5.4 Lightwheel

Lightwheel 使用双目 MP4、`pose.json`、`annotation.json`、`metadata.json`、
MCAP、音频和点云，不属于 Zarr family。当前 Zarr reader 不能读取它。需要单独
实现完整 Episode 枚举、MP4/JSON reader、Catalog builder 和真实 queue 闭环。

## 6. Stage1 video-only 的建议数据池

第一批 production 候选：

- Aria bimanual/left/right；
- EVA bimanual；
- Scale bimanual；
- Mecka freeform bimanual。

Mecka flagship 在完成范围规范化和重新 smoke 后加入，但文本条件 SamplePool
默认只使用具有有效 annotation 或 episode description 的窗口。无文本 Episode
只有在训练策略明确支持 empty prompt/unconditional video prediction 时才进入。

EVA right-arm 暂不加入。它的 RGB/文本可能有价值，但必须通过单独降级 profile
或源数据修复来使用，不能污染其他 child 的 state/action contract。

采样权重应按最终 SamplePool 的可用窗口数计算，而不是直接按原始 Episode
目录数计算。不同 profile 必须作为独立 child 统计和采样。

## 7. Stage2/Stage3 action-state 使用边界

| 数据源 | state 语义 | action 语义 | 建议 |
|---|---|---|---|
| Aria | observed pose | future observed pose | 只能作为 observed trajectory 目标 |
| Scale | observed pose | future observed pose | 不能等同真实控制命令 |
| Mecka | observed pose | future observed pose | 不能等同真实控制命令 |
| EVA bimanual | observed pose+gripper | commanded pose+gripper | 可作为真实 action-conditioned 候选 |
| EVA right-arm | pose 有效、gripper 全坏 | pose 有效、gripper 全坏 | 修复或显式降级前禁用 |

模型端 SamplePool 继续为 video/state/action/text 分别提供索引。数据工厂不根据
video index 推导 action，也不隐式构造 action horizon。

## 8. 后续实施计划

1. 在 Catalog builder 中实现可审计的 incomplete-store quarantine。
2. 实现 annotation 尾部裁剪并保留原始范围；完全无交集数据 fail-closed。
3. 补齐 flagship 四类文本来源交叉统计：
   annotation+description、annotation-only、description-only、no-text。
4. 重跑 flagship 小规模真实 payload/queue smoke。
5. 对 A 类 child 做全量 finite、长度、JPEG、文本覆盖率 Catalog 扫描。
6. 为每个 profile 分别生成 split-neutral Catalog 和 SQLite offset index。
7. 由模型侧生成正式 SamplePool，完成 weighted mix、batch、queue 和多 rank
   互斥验证。
8. 验收后注册 immutable release；此前不得接入正式长训。

## 9. 当前最终判断

- **优先可推进**：Aria 三个 profile、EVA bimanual、Scale、Mecka freeform。
- **修复后可利用**：Mecka flagship；EVA right-arm 的视频/pose 部分。
- **当前不可利用**：EVA right-arm 的 gripper contract、空的 EVA left-arm、
  不完整 flagship store、Lightwheel、`scale_old`。
- **尚未完成**：任何 EgoVerse production Catalog/release 和正式长训准入。

## 10. 证据位置

- 初始结构报告：
  `outputs/egoverse_intake_20260729/EGOVERSE_INTAKE_REPORT.md`
- 真实 smoke：
  `outputs/egoverse_image_smoke_20260730/`
- EVA right 全量数值审计：
  `eva_right_numeric_audit.json`
- Mecka flagship 全量文本范围审计：
  `mecka_flagship_annotation_audit.json`
