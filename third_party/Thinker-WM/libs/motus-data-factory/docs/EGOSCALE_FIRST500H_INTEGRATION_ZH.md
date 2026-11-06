# EGOSCALE FIRST-500H 数据工厂接入

> 本文说明 `dev/20260816-datafactory-updates` 中
> EGOSCALE FIRST-500H 的数据语义、可配置读取模式、缺失值协议和分片发布方式。
> 数据工厂只负责完整、可复现地提供各模态，不决定模型使用哪些模态或如何计算
> loss。

## 1. 接入状态

本分支将 `/data/EGOSCALE-FIRST-500H` 接入统一的 Catalog + SamplePool
训练数据链路。

全量盘点共发现 48,691 个 LeRobot-v3 Episode，来自 29,314 个原始任务组。
为尽快提供训练数据，同时保持后续可增量扩容，先按原始任务组稳定哈希划分
10 个互不重叠的桶。首批 `shard00` 发布桶 0，包含：

- 4,848 个 Episode；
- 2,894 个原始任务组；
- 同一原始任务拆出的多个时间段始终落入同一分片；
- Catalog、metadata 和 SQLite index 分别独立发布，不覆盖后续分片。

剩余桶 1–9 合并为第二个不可变 Catalog，注册名称为
`egoscale_first500h_remaining90pct`，包含 43,843 个 Episode 和 26,420 个原始
任务组。两个 release 的交集为 0，并集严格等于本批全部 48,691 个 Episode。
release 版本均为 `20260816_shards_v1`。

## 2. 一个完整 Episode 的内容

每个顶层目录就是一个独立 Episode 包，主要文件如下：

```text
session_..._lerobot_v3/
├── source_manifest.json
├── calibration/
│   ├── undistorted_intrinsics.json
│   └── stereo_calibration.json
├── meta/
│   ├── info.json
│   ├── tasks.parquet
│   ├── episodes.parquet
│   ├── custom_annotation.json
│   └── custom_metadata.csv
├── data/chunk-000/episode_000000.parquet
└── videos/
    ├── base_0_camera/rgb/image/chunk-000/episode_000000.mp4
    └── base_1_camera/rgb/image/chunk-000/episode_000000.mp4
```

视频固定为 30 FPS、1920x1080。Parquet 一行对应一个视频帧，`frame_index`
连续，视频、head、左右手和 MANO 字段共享同一逐帧时间轴。

Catalog 保留 Episode 级任务名称、任务描述、任务标签、源视频帧范围，以及
`custom_annotation.json` 中每段细粒度动作描述和 `[start_frame,end_frame)`。
Catalog 是 split-neutral 的完整枚举，不在数据工厂内划分 train/test。

## 3. 标准四模态输出

### 3.1 Video

Reader 默认使用 `base_0`，支持：

- `base_0`：左目，默认；
- `base_1`：右目；
- `stereo_horizontal`：同一时刻的左目和右目按宽度水平拼接。

通过 release 的 `reader_options.egoscale.camera_mode` 修改。缩放默认保持宽高比并
补零，也可用 `resize_mode=interpolate`。

训练侧不需要重建 Catalog 或 SamplePool，只需在打开数据时覆盖 reader 配置：

```python
reader_options = {
    "egoscale": {
        "camera_mode": "base_0",  # base_0 | base_1 | stereo_horizontal
        "pose_profile": "camera_pose_with_validity",
        "resize_mode": "pad",
    }
}
```

三种模式使用完全相同的 `video_frame_indices`。双目模式不会改变时间轴，只改变
每帧图像的宽度；后续仍统一缩放到训练入口的 `video_size=(H,W)`。

### 3.2 State 和 Action

State 和 action 都是与视频逐帧对齐的人体运动观测。两者可由 SamplePool 提供
不同的 `state_indices` 和 `action_indices`，因此可以读取不同时刻；同一个 reader
实例下二者使用相同的坐标 profile。

标准输出固定为 24 维：

```text
[head_pose(7), head_valid(1),
 left_hand_eepose(7), left_hand_valid(1),
 right_hand_eepose(7), right_hand_valid(1)]
```

每个 pose 的 7 维保持原始数据定义，位置和旋转不会被数据工厂转换成增量控制量；
当前 reader 也不做归一化。

7 维布局为 `[pos xyz(3), quat xyzw(4)]`（pos 在前、quat 分量序 **xyzw**）——
20260817 经集群真实数据抽检确认（判别法：恒等式 T_cam = T_head⁻¹·T_world
在 xyzw 序下 4 个 episode 双手残差≈0、wxyz 下 7-29cm/66-128°；抽检脚本
`tmp/egoscale_spotcheck_20260817/check_quat_order.py`，主仓库侧）。原始
parquet 中无效帧的 pose 可为 NaN，reader 的零位姿补全会将其置零并置
validity=0（见 3.3 节）。

训练负责人可按任务选择以下 profile：

| `pose_profile` | 左右手字段 | 语义 | 维度 |
|---|---|---|---|
| `camera_pose_with_validity` | `left/right_hand/eepose` | 相对于头部/相机参考系的末端位姿，默认 | 24 |
| `world_pose_with_validity` | `left/right_hand/world_eepose` | 世界坐标系下的绝对末端位姿 | 24 |

例如，训练使用世界绝对末端位姿：

```python
pipeline = open_training_data(
    "egoscale_first500h_shard00",
    sample_pool="/path/to/egoscale_shard00.sample_pool.jsonl",
    batch_size=8,
    reader_options={
        "egoscale": {
            "camera_mode": "base_0",
            "pose_profile": "world_pose_with_validity",
            "resize_mode": "pad",
        }
    },
    # 其余 DataLoader、adapter 和 queue 参数照常传入
)
```

切换 profile 不需要重建 Catalog、SQLite 或 SamplePool。当前标准 batch 的
`pose_profile` 同时作用于 state 和 action；如果模型需要在一次 batch 中同时消费
两种坐标表示，可通过第 4 节的 source-native 接口读取两组原始字段，并在模型侧
自定义 adapter。

当前 state 和 action 均来自同一对齐的人体运动观测时间轴，默认不归一化。
这里的 action 是供模型端选择使用的视觉运动代理，
`is_control_command=False`，不是机器人真实控制命令。是否用于 loss、采用哪种
训练策略由模型端决定，数据工厂不区分训练 stage。

### 3.3 左右手缺失协议

一段 Episode 内可能出现以下正常情况：

- 某些帧只检测到左手或只检测到右手；
- 一只手在连续一段时间内离开画面；
- 整个 Episode 都没有检测到某一只手。

数据工厂不会删除这些 Episode，也不会用另一只手或前一帧伪造缺失位姿。默认
处理是：缺失手的 7 维 pose 置零，并把紧随其后的 validity 置为 `0`；有效手保持
真实 pose，validity 为 `1`。`head` 使用相同的 pose + validity 规则。由此，batch
shape 始终稳定为 `[N,24]`，同时保留逐时刻、逐手的缺失信息。

模型端必须保留并使用 validity：可以对无效手屏蔽 loss、作为显式条件输入，或在
生成 SamplePool 时过滤不符合训练策略的区间。不能仅凭 pose 为零判断缺失，也
不能在进入 loss 前丢弃 validity 后把补零值当成真实坐标。

### 3.4 Text

一个 Episode 可包含以下文本项：

- `task_name_en`；
- `task_description_en`；
- `task_detail_en`；
- `task_label`；
- 多个带视频范围的 `fine_action_en`。

模型侧 SamplePool 通过 `text_indices` 明确选择使用哪一项，不由 reader 猜测。

## 4. 访问完整 Episode 和原始字段

标准访问方式与已有数据集相同：

```python
from motus_data import open_dataset

repository = open_dataset("egoscale_first500h_shard00")
episode = repository.get_episode(
    "egoscale:session_..._lerobot_v3"
)

frames = episode.read_video([0, 15, 30])
state = episode.read_state([0, 15, 30])
action = episode.read_action([0, 15, 30])
texts = episode.read_text([0, 1])
```

除标准四模态外，EGOSCALE 的 tracks、world tracks、完整 MANO、原始时间戳等
字段可通过新增的向后兼容接口访问：

```python
print(episode.source_modality_schemas().keys())

raw = episode.read_source_modalities({
    "left_hand/tracks": [0, 15, 30],
    "left_hand/world_tracks": [0, 15, 30],
    "left_hand/eepose": [0, 15, 30],
    "left_hand/world_eepose": [0, 15, 30],
    "left_hand/mano/hand_pose": [0, 15, 30],
    "left_hand/mano/valid": [0, 15, 30],
    "head/timestamp": [0, 15, 30],
})
```

原始嵌套字段内部的 `None`/NaN 会补零；分析时必须同时读取对应的 validity
字段。该接口不改变标准 batch contract，已有数据源的调用方式不变。

## 5. 模型侧 SamplePool 和训练接入

模型侧仍负责给出 video/state/action/text 的全部显式索引。一条 selection 示例：

```json
{
  "episode_id": "egoscale:session_..._lerobot_v3",
  "video_frame_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8],
  "state_indices": [0],
  "action_indices": [0, 1, 2, 3, 4, 5, 6, 7],
  "text_indices": [1]
}
```

训练入口无需新增 source-specific 调用：

```python
from motus_data import open_training_data

pipeline = open_training_data(
    "egoscale_first500h_shard00",
    sample_pool="/path/to/egoscale_shard00.sample_pool.jsonl",
    batch_size=8,
    output_adapter="motus_native_multimodal_batch_v1",
    adapter_options={
        "num_video_frames": 8,
        "action_video_fps_ratio": 1,
    },
    num_workers=0,
    queue_enabled=True,
    queue_size=2,
)

batch, stats = pipeline.next()
```

模型端需要做的只有：

1. 针对该分片 Catalog 生成绑定同一 `catalog_id` 的 SamplePool；
2. 把所需的 `egoscale_first500h_shard00` 和/或
   `egoscale_first500h_remaining90pct` 加入混训 source 和权重配置；
3. 根据模型策略选择相机、文本和 state/action 索引。

现有 grouped sampler、分布式 rank/world_size、DataLoader、ReadyBatchQueue 和
batch adapter 接口均未改变。

## 6. 后续分片扩容

分片算法固定为：

```text
uint64_be(sha256(source_group_key)[:8]) % 10
```

当前冻结的 48,691 个 Episode 发布成两个训练 child：

| dataset | 哈希桶 | Episode | source group |
|---|---:|---:|---:|
| `egoscale_first500h_shard00` | 0 | 4,848 | 2,894 |
| `egoscale_first500h_remaining90pct` | 1–9 | 43,843 | 26,420 |

每个 child 拥有独立 Catalog、meta、SQLite、release 和 SamplePool。不要追加或
覆盖任何已发布文件，否则已有 SamplePool 的 `catalog_id`/SHA256 绑定会失效。

未来新增的数千小时、几十万 Episode 应作为新的不可变数据批次发布，而不是改写
本次 500H 的两个已发布分片：

1. 对新批次做结构和必需模态门禁，并冻结 Episode 清单；
2. 检查 `episode_id`/`source_group_key` 与既有 release 不重叠；
3. 以完整 `source_group_key` 为边界，约每 5,000 个 Episode 生成一个独立
   Catalog、meta 和 SQLite；
4. 在 `releases.yaml` 注册新的 child/release；
5. 模型端针对每个新 Catalog 生成匹配 `catalog_id` 的 SamplePool，并在混训配置中
   增加 source 和采样权重。

只要后续原始格式和模态语义不变，数据 reader、SamplePool 协议、DataLoader、
batch adapter 与 queue 都无需修改。这里的“增量构建”是**增加新的不可变
Catalog release**，不是向已发布 JSONL/SQLite 原地追加。若供应商改变 Parquet
schema、坐标语义、相机布局或标注格式，则必须先扩展 reader 并重新通过真实数据
smoke，不能只生成索引后直接训练。

首批单桶构建命令：

```bash
python tools/build_egoscale_release.py \
  --data-root /data/EGOSCALE-FIRST-500H \
  --storage-root /data/EGOSCALE-FIRST-500H \
  --output-root /data/jiyuan.wang/EpisodeCatalogs/_20260816/egoscale_first500h_shards_v1 \
  --data-version 20260816_shards_v1 \
  --shard-count 10 \
  --shard-index 0 \
  --workers 24
```

剩余九桶合并构建命令：

```bash
python tools/build_egoscale_release.py \
  --data-root /data/EGOSCALE-FIRST-500H \
  --storage-root /data/EGOSCALE-FIRST-500H \
  --output-root /data/jiyuan.wang/EpisodeCatalogs/_20260816/egoscale_first500h_shards_v1 \
  --data-version 20260816_shards_v1 \
  --shard-count 10 \
  --shard-indices 1-9 \
  --partition-name remaining90pct \
  --workers 24
```

不要对已进入训练的 release 使用 `--force`。

## 7. 第二批正式产物与验收

`egoscale_first500h_remaining90pct` 已完成正式发布：

```text
Episode                         43,843
source group                    26,420
frames                      53,375,880
Catalog bytes               474,328,334
SQLite bytes                  7,344,128
Catalog SHA256  2511eef2a6ad2698b32ab8fe409ecfae14cf97a804ee0386637978b98316b55a
Catalog build elapsed             914.7 s
```

正式门禁结果：

- 首片与剩余片 Episode ID 交集为 0，并集为 48,691；
- 两片 `source_group_key` 交集为 0，并集为 29,314；
- 两个 Catalog 均无重复 Episode ID；
- Catalog 实际 SHA256 与 metadata 一致；
- metadata Episode 数、JSONL 实际行数和 SQLite 记录数逐片一致；
- 剩余片实际哈希桶严格为 1–9；
- 世界绝对位姿 profile 的真实 Episode、SamplePool、DataLoader 和两批 queue smoke
  通过，state/action 均为有限 24D 数值；
- 首片和剩余片各消费两个 batch 的双 child 混训 queue smoke 通过；
- 全库单元回归 112 项通过。

验收报告保存在工作区 `outputs/egoscale_first500h_remaining90pct_release_20260816/`
中；该目录是运行产物，不进入 Git。

## 8. 已确认的数据边界

- 部分 Episode 整只左手或右手字段为 Arrow `null`；reader 已兼容。
- 部分帧的 tracks/pose 列表内部包含 `None`；reader 逐元素补零并保留有效位。
- 少量 Episode 没有细粒度动作文本覆盖，但仍有 Episode 级任务文本。
- 某些源任务被切成多个交付 Episode，细粒度 annotation 可能重复；Catalog 保留
  `source_group_key` 和源帧范围，模型侧可据此去重或筛选。
- `source_manifest.json` 的文件清单未列出 MP4，但真实双目 MP4 存在且抽检可解码。
- `info.json` 的部分逻辑 dtype 与实际 Parquet 物理类型不同；reader 以 Parquet
  为准并统一转换为稳定的 torch dtype。

这些边界均不会导致 silent skip；构建或读取遇到未知结构时直接报错。
