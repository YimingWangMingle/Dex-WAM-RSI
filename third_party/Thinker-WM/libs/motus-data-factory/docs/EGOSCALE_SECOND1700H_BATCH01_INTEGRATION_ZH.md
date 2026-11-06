# EGOSCALE SECOND-1700H batch_01 接入说明

## 1. 版本与权限

本次开发分支：

```text
dev/20260820-egoscale-second-1700h-batch01
```

基线为训练负责人最新合并版本：

```text
main@6210dcf42cf2aa231488dd56f3277211f51d93e2
```


## 2. 全量目录与发布方式

第二批盘点结果：

| 项目 | 数量 |
|---|---:|
| Episode | 100,448 |
| source group | 58,756 |
| 带源帧范围的切段 Episode | 71,124 |
| 未切段 Episode | 29,324 |
| 单 source group 最大 Episode 数 | 7 |

与 FIRST-500H 的 48,691 Episodes/29,314 source groups 对比：

- Episode 目录名交集为 0；
- `source_group_key` 交集为 0；


后续每个独立到货批次各自发布一个新的完整 Catalog/SQLite，不向旧 Catalog 追加，
也不覆盖已被 SamplePool 绑定的产物。各批次都是 `egoscale` family 下的独立 child。

## 3. Episode 目录和模态

第二批和 FIRST-500H 的必需目录完全一致。每个顶层目录对应一个 Episode：

```text
session_..._lerobot_v3/
├── source_manifest.json
├── calibration/
│   ├── undistorted_intrinsics.json
│   └── stereo_calibration.json
├── meta/
│   ├── tasks.parquet
│   ├── episodes.parquet
│   ├── custom_metadata.csv
│   ├── custom_annotation.json
│   └── info.json
├── data/chunk-000/episode_000000.parquet
└── videos/
    ├── base_0_camera/rgb/image/chunk-000/episode_000000.mp4
    └── base_1_camera/rgb/image/chunk-000/episode_000000.mp4
```

已抽检确认：

- `codebase_version=v3.0`、`robot_type=orb_stereo`、30 FPS；
- Parquet 行数等于 manifest `frame_count`；
- `frame_index` 严格为 `[0,T)`；
- `timestamp=frame_index/30`，head/左右手 timestamp 单调；
- 左目、右目和双目水平拼接均可真实解码，左右目内容不相同；
- camera/world 两种 pose profile 均输出有限 24D state/action；
- source-native tracks、eepose 和 MANO 字段可按显式索引读取。

标准输出协议与 FIRST-500H 不变：

```text
[head_pose(7), head_valid(1),
 left_hand_pose(7), left_hand_valid(1),
 right_hand_pose(7), right_hand_valid(1)]
```

支持 `base_0`、`base_1`、`stereo_horizontal`，以及
`camera_pose_with_validity`、`world_pose_with_validity`。数据工厂不决定模型使用
哪些模态或 loss。

## 4. 本批确认的 schema 差异

主 Parquet 仍包含同一组 31 个字段，但部分 Episode 的物理列顺序发生调整。
100,448-Episode 全量门禁中：

- 97,677 条保持 canonical 顺序；
- 2,771 条字段集合完全相同、仅物理顺序不同；
- 0 条缺列或增列。

Reader 一直按字段名读取，因此重排不会影响 Dataset/DataLoader/batch。旧 Catalog
builder 使用 `tuple(schema.names)` 强制顺序相等，会错误拒绝这些 Episode。本分支
将校验改为：

1. 字段数量和字段名集合必须与 31 个 canonical 字段完全一致；
2. 缺列、增列或重复导致的集合/数量变化继续 fail-closed；
3. 物理顺序不同允许构建，但在 Episode quality 和 Catalog metadata 中记录
   order fingerprint 和 warning count；
4. `raw_parquet_columns` 保存该 Episode 的真实物理顺序。

部分整手缺失 Episode 会把对应 Arrow 物理类型推断为 `list<null>`，有值的动态
list 也可能是 `list<double>`，而不是 FIRST-500H 常见的 fixed-size float list。
字段名、逐行 shape 和 validity 语义不变，现有 Reader 已能读取并稳定转换为
float32 tensor。

抽样还发现少量帧存在 `MANO valid=true` 但 eepose/world_eepose 含非有限值。
Reader 会再次执行 shape/finite 检查，将该 pose 置零并把模型输出 validity 改为
0，因此异常数值不会进入模型 batch；该现象仍作为数据质量信息保留，不把 raw
MANO valid 静默改写。

## 5. 文本协议

第二批继续提供：

- `task_name_en`；
- `task_description_en`；
- `task_detail_en`；
- `task_label`；
- 带 `[start_frame,end_frame)` 的 `fine_action_en`；
- fine action 的 action label、hand、scene semantic 和 camera motion。

结构与 FIRST-500H 一致，可直接使用 `main@6210dcf` 新增的 EGOSCALE
per-source text composer。无需为第二批引入新的文本调用接口。

## 6. 发布工具兼容改动（数据工厂离线构建数据集与维护工具，训练端可无视）

`tools/build_egoscale_release.py` 新增向后兼容参数：

```text
--dataset-name
```

默认仍为 `egoscale_first500h`，所以已有 FIRST-500H 命令和文件名不变。第二批
正式构建使用：

```text
dataset name: egoscale_second1700h_batch01
partition:    full
data version: 20260820_full_v1
shard count:  1
```

其中 `shard count: 1` 是构建工具表示“选择全部 Episode 并输出单个 Catalog”的
兼容参数，不表示本次数据被切成了一个分片体系。

计划输出根目录：

```text
/data/jiyuan.wang/EpisodeCatalogs/_20260820/
  egoscale_second1700h_batch01_full_v1/
```

全量命令模板：

```bash
python tools/build_egoscale_release.py \
  --data-root /data/EGOSCALE-SECOND-1700H-batch_01 \
  --storage-root /data/EGOSCALE-SECOND-1700H-batch_01 \
  --output-root /data/jiyuan.wang/EpisodeCatalogs/_20260820/egoscale_second1700h_batch01_full_v1 \
  --dataset-name egoscale_second1700h_batch01 \
  --partition-name full \
  --data-version 20260820_full_v1 \
  --shard-count 1 \
  --shard-index 0 \
  --workers 64 \
  --resume \
  --checkpoint-interval-episodes 500
```

正式构建前不得使用 `--force` 覆盖已发布 Catalog。工具生成 Catalog JSONL、
metadata、SQLite index 和 report，模型端为该完整 Catalog 生成独立且绑定 Catalog
ID/SHA 的 SamplePool。

`--resume` 使用稳定 `.partial` 和 `.checkpoint.json` 文件。checkpoint 每 500 个
Episode 执行一次 flush/fsync 并原子更新，绑定 Catalog ID、data version、数据根、
storage root、canonical schema 和完整 Episode 清单指纹。恢复时校验这些身份信息，
按已提交字节和 SHA256 验证 partial，并截断 checkpoint 之后未提交的尾部。输入清单
变化或 partial 被修改时 fail-closed，不会拼接错误数据。正式 Catalog 仍只在全量
完成后原子发布。若目录中已有断点但重启命令遗漏 `--resume`，工具也会拒绝另起
构建，必须显式恢复或使用 `--force` 放弃旧断点。

日志默认每 5 秒输出 Catalog 百分比、已完成/总 Episode、累计帧数、速率和 ETA；
SQLite 阶段输出已扫描字节百分比和记录数。Catalog/meta 已完成而 SQLite 中断时，
重启会复用 Catalog 并只重建 SQLite。

## 7. 当前验证状态

- Parquet 重排 synthetic 回归通过；缺列继续被拒绝；
- 真实 canonical、重排列、双手整段缺失和单手缺失 Episode 均可读取；
- 左目、右目、双目、camera/world pose、source-native 字段均已真实抽检；
- EGOSCALE 聚焦测试 14/14；
- 全库回归 133/133；
- 100,448-Episode 完整 builder-compatible 审计全部通过，错误为 0；
- 11 个必需文件在所有 Episode 中均存在且非空；
- 全量包含 126,606,454 帧、2,313,458 条文本和 2,013,604 条细粒度动作文本；
- 100,448 条均为 `teleop`，任务可用标记均为 true；
- 审计报告位于
  `outputs/egoscale_second_1700h_batch01_audit_20260820/06_full_delivery_audit_postfix.json`；
- 正式 full Catalog、metadata、SQLite index 和 report 已完成发布，未修改任何
  FIRST-500H 或其他已发布产物。

全量 schema 只有两个列顺序 fingerprint。除上述 2,771 条顺序变体外，没有第三种
未知字段集合。全量 Arrow null-value column 统计中，left/right tracks、eepose、
world pose 整列 null 分别涉及 194/132 个 Episode；left/right MANO 相关整列 null
分别涉及 1,010/991 个 Episode。Reader 继续使用显式 validity 和 finite 门禁处理，
不会把非有限 pose 传入模型 batch。

## 8. 正式 full release

正式产物根目录：

```text
/data/jiyuan.wang/EpisodeCatalogs/_20260820/
  egoscale_second1700h_batch01_full_v1/
```

关键绑定：

| 项目 | 值 |
|---|---|
| dataset | `egoscale_second1700h_batch01` |
| release | `20260820_full_v1` |
| Catalog ID | `egoscale_second1700h_batch01_full_20260820_full_v1` |
| Episodes | 100,448 |
| frames | 126,606,454 |
| Catalog SHA256 | `6f68e8fe6098d7692b365478f2a1ab73303199841cf05b05ee1663d429c614b6` |
| Catalog bytes | 1,114,655,769 |
| SQLite bytes | 16,945,152 |


构建全程没有中断，`resume_count=0`。正式完成后 partial/checkpoint 已自动删除。
断点能力通过独立故障注入测试验收：checkpoint 之后未提交的尾部会被截断，从最近
持久化 Episode 继续；输入清单变化会被拒绝。

静态验收结果：

- JSONL 行数、metadata Episodes、SQLite rows 均为 100,448；
- 实际 Catalog SHA、metadata SHA、SQLite binding SHA 完全一致；
- SQLite `PRAGMA quick_check=ok`；
- 全量统计与独立只读审计完全一致；
- 无 partial/checkpoint 残留。

真实流水线验收结果：

- 使用正式 Catalog/SQLite 完成 Episode 懒加载和 world-pose 真实读取；
- Catalog 绑定的小型 SamplePool 完成 grouped decode、Dataset、DataLoader 和
  ReadyBatchQueue 两批闭环；
- batch 为 `first_frame=[2,3,120,160]`、`future=[2,8,3,120,160]`、
  `state=[2,24]`、`action=[2,8,24]`，全部有限且 queue producer 无错误；
- 内置 release 名 `egoscale_second1700h_batch01` 可直接通过
  `open_training_data()` 使用；
- 与 `egoscale_first500h_shard00` 的 4-batch 混训 smoke 各消费 2 batch，证明
  新旧数据使用相同 Dataset/DataLoader/queue 和模型 batch 协议。

因此本批符合 EGOSCALE 增量接入约定：schema/语义不变时只新增不可变
Catalog/metadata/SQLite、release 注册和模型侧 SamplePool/source weight，不修改
训练主干。正式 production SamplePool 仍由模型训练端按其采样策略生成。

## 9. 训练端接入事项

本次更新不改变数据工厂的公开 API、batch 协议或 EGOSCALE reader 语义。训练端
合入本分支后，需要把 ThinkerWM 中 `libs/motus-data-factory` 指向合入后的数据工厂
提交，使下面的内置 release 注册可见：

```text
dataset: egoscale_second1700h_batch01
release: 20260820_full_v1
```

训练端仍负责以下训练策略和产物：

1. 基于本批 Catalog ID/SHA 生成互斥的 train/holdout `sampling_ranges` 或其他
   production SamplePool；不得复用 FIRST-500H 的 ranges/SamplePool。
2. 在训练 YAML 的 `data.sources` 中新增一个独立 source，填写上述 dataset、训练
   pool、可选 eval pool 和采样权重。
3. 按模型目标选择或覆盖相机、pose、文本和窗口采样配置；数据工厂默认继续使用
   `base_0`、`world_pose_with_validity`、`resize_mode=pad`。


YAML source 的最小结构与既有 indexed 数据源相同：

```yaml
- name: egoscale_second1700h_batch01
  dataset: egoscale_second1700h_batch01
  weight: <由训练策略确定>
  pool:
    type: sampling_ranges
    path: <训练端生成的 train ranges.sqlite>
  eval_pool:
    type: sampling_ranges
    path: <训练端生成的 holdout ranges.sqlite>
```


