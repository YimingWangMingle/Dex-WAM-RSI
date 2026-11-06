# LIBERO 数据工厂接入与使用说明


## 1. 数据集梗概

数据位于：

```text
/data/chen.liao/Libero_parquet
```

它是 LeRobot v2 Parquet 格式的 LIBERO 数据，共四个 suite。数据工厂将其注册为
同一 family `libero` 下的四个独立 child：

| child / dataset | release | Episode | 时间步 | 英文任务 |
|---|---|---:|---:|---:|
| `libero_10` | `20260810_full_v1` | 379 | 101,469 | 10 |
| `libero_goal` | `20260810_full_v1` | 428 | 52,042 | 10 |
| `libero_object` | `20260810_full_v1` | 454 | 66,984 | 10 |
| `libero_spatial` | `20260810_full_v1` | 432 | 52,970 | 10 |
| 合计 | - | 1,693 | 273,465 | 40 |

数据频率为 10 Hz，约 7.60 小时。每个 Parquet 文件对应一个完整 Episode，每行对应
同一时间步，包含：

```text
image[t]         主相机 256x256 RGB PNG
wrist_image[t]   腕部相机 256x256 RGB PNG
state[t]         8D 绝对末端状态
actions[t]       7D 增量末端控制命令
text[0]          完整 Episode 的英文任务描述
```

不存在外置 MP4；图像以 PNG bytes 内嵌在 Parquet 中。


## 2. 模态语义和对齐

### 2.1 state 8D

```text
[eef_position_xyz,
 eef_axis_angle_xyz,
 gripper_left_qpos,
 gripper_right_qpos]
```

state 是当前 observation 的绝对末端状态。姿态是 axis-angle rotation vector，
不是普通 RPY Euler，也不是机器人关节角。

### 2.2 action 7D

```text
[delta_position_xyz,
 delta_axis_angle_xyz,
 gripper_command]
```

action 是在 `observation[t]` 之后执行的增量控制命令。最后一维为 `-1/+1` 夹爪
命令。这不是 `state[t+1] - state[t]` 的简单物理差值。

### 2.3 时间轴

四种模态使用同一个 Episode 局部 `[0,T)` 索引空间：

```text
len(image) = len(wrist_image) = len(state) = len(action) = T
timestamp[t] = t / 10
```

默认 48 个未来帧的典型 SamplePool 选择为：

```text
video_frame_indices = [t, t+1, ..., t+48]   # 49 帧
state_indices       = [t]                    # 1 个初始状态
action_indices      = [t, t+1, ..., t+47]   # 48 个控制命令
text_indices        = [0]                    # Episode 任务文本
```



## 3. 正式 Catalog 和 release

正式 Catalog 目录：

```text
/data/jiyuan.wang/EpisodeCatalogs/_20260810/libero_parquet_v1
```

每个 child 均发布：

```text
<child>.libero_parquet_v1.episode_catalog.jsonl
<child>.libero_parquet_v1.episode_catalog.meta.json
<child>.libero_parquet_v1.episode_catalog.index.sqlite
```

Catalog 是 split-neutral 的完整 Episode 枚举，不在数据工厂中固定 train/test。
训练、验证和展示分别使用哪些 Episode/selection，由模型侧提供不同 SamplePool 决定。

四个内置 release 已在 `motus_data/releases.yaml` 注册。一般调用时只写 dataset 名，
无需手工填写 Catalog、SQLite index 或原始数据路径。


## 4. 默认处理策略

不传覆盖参数时，四个 child 使用以下默认值：

| 项目 | 默认值 | 说明 |
|---|---|---|
| 相机 | `main` | 只读取主相机 |
| resize | `pad` | 保持宽高比缩放并居中补零 |
| 输出分辨率 | `480x640` | 参数顺序为 `(H,W)` |
| state/action 数值 | `zscore` | 使用官方全量统计量 |
| state 维度 | 8 | 保留原生维度 |
| action 维度 | 7 | 保留原生维度，不补零 |
| future video 帧数 | 48 | 加一帧条件帧，共读取 49 帧 |
| action/video 比例 | 1 | 默认 48 个未来帧对应 48 个 action |
| batch adapter | `motus_native_multimodal_batch_v1` | 支持 state/action 独立维度 |
| `num_workers` | 0 | 避免 multiprocessing shm 累计 |
| queue | 开启 | CPU ReadyBatchQueue |
| queue size | 2 | 每个 rank 的 ready batch 容量 |
| pin memory | `false` | 当前稳定默认值 |
| action 数值门禁 | 20 | 归一化后绝对值超过阈值会失败 |



## 5. 默认 z-score

默认 reader 对所有 8 个 state 维度和所有 7 个 action 维度执行：

```text
normalized = (raw - mean) / std
```

action 的 gripper 维也包含在 z-score 中。统计文件为：

```text
/data/chen.liao/Libero_parquet/meta/stats.json
```

四份 Catalog metadata 均绑定该文件的 SHA256。reader 在首次归一化时验证文件
指纹、维度、有限性和 `std > 0`；统计文件变化或路径指向不同内容时直接失败，不会
静默使用另一份统计量。

需要原始数值时，将 `normalization_mode` 改为 `raw`。raw 和 z-score 都返回
`float32`，索引、shape 和 batch 字段不变。


## 6. 默认模型 batch

LIBERO 推荐使用：

```text
motus_native_multimodal_batch_v1
```

这是 LIBERO 标准训练配方的默认 profile，但通用 API 为保持可扩展性，在不传
`output_adapter` 时仍返回原始 `IndexedEpisodeBatch`。模型 bridge 必须像本文示例
一样显式选择该 profile，不能假设 release 会自动替模型决定 batch 协议。

默认输出主要字段：

| 字段 | 默认 shape/type | 含义 |
|---|---|---|
| `first_frame` | `[B,3,480,640]` float | 条件帧，范围 `[0,1]` |
| `video_frames` | `[B,48,3,480,640]` float | 未来视频帧 |
| `initial_state` | `[B,8]` float | 显式选择的初始状态 |
| `action_sequence` | `[B,48,7]` float | 显式选择的 action |
| `t5_input_messages` | `list[str]` | 英文任务文本 |
| `vlm_input_messages` | `list[str]` | VLM prompt + 英文任务 |
| `first_frame_pils` | `list[PIL.Image]` | VLM 图像输入 |
| `robot_names` | `list[str]` | `libero-panda` |
| `source_datasets` | `list[str]` | `libero` |
| `source_children` | `list[str]` | 当前 suite child |
| `episode_ids` | `list[str]` | 完整 Episode 标识 |
| `frame_indices` | `list[list[int]]` | 显式 video indices |
| `state_dimension` | `int` | 固定为 8 |
| `action_dimension` | `int` | 固定为 7 |
| `normalization_modes` | `dict` | 默认 state/action 均为 `zscore` |
| `video_loss_weights` | `[B]` | 默认 1.0 |
| `action_loss_weights` | `[B]` | 默认 1.0 |

一个 batch 始终来自一个 child，batch 内维度和归一化模式保持一致。


## 7. 单 child 使用

### 7.1 查看完整 Episode

```python
from motus_data import open_dataset

repository = open_dataset("libero_10")
try:
    episode = repository.get_episode("libero:libero_10:000000")
    print(episode.info)
    print(episode.info.modality_schemas)
    print(episode.video.schema)
    print(episode.state.schema)
    print(episode.action.schema)

    frames = episode.read_video(
        [0, 10, 20],
        output_format="rgb_float32_chw",
        video_size=(480, 640),
    )
    state = episode.read_state([0, 10, 20])
    action = episode.read_action([0, 10, 20])
    text = episode.read_text([0])
finally:
    repository.close()
```

内置 release 默认返回 z-score state/action。只做原始数值分析时：

```python
repository = open_dataset(
    "libero_10",
    reader_options={
        "libero": {
            "camera_mode": "main",
            "resize_mode": "pad",
            "normalization_mode": "raw",
        }
    },
)
```

传入 `reader_options` 会覆盖该次调用的 release reader 配置，因此应把希望保留的
相机、resize 和归一化选项完整写出。

### 7.2 Dataset、DataLoader 和 queue

```python
from motus_data import open_training_data

pipeline = open_training_data(
    "libero_10",
    release="20260810_full_v1",
    sample_pool="/path/to/libero_10.train.sample_pool.jsonl",
    batch_size=2,
    output_adapter="motus_native_multimodal_batch_v1",
    adapter_options={
        "num_video_frames": 48,
        "action_video_fps_ratio": 1,
        "video_loss_weight": 1.0,
        "action_loss_weight": 1.0,
    },
    video_size=(480, 640),
    rank=RANK,
    world_size=WORLD_SIZE,
    num_workers=0,
    queue_enabled=True,
    queue_size=2,
    shuffle=True,
    drop_last=True,
)

try:
    pipeline.set_epoch(DATA_EPOCH)
    with pipeline:
        batch, queue_stats = pipeline.next()
        # 模型端从 batch 消费数据。
finally:
    pipeline.close()
```


## 8. 可调整开关

### 8.1 相机模式

`reader_options.libero.camera_mode`：

| 值 | 行为 |
|---|---|
| `main` | 只读取主相机，默认 |
| `wrist` | 只读取腕部相机 |
| `main_wrist_horizontal` | 同一时刻主相机在左、腕部相机在右，水平拼接 |

切换相机不需要重建 Catalog 或 SamplePool。

### 8.2 resize 模式

`reader_options.libero.resize_mode`：

| 值 | 行为 |
|---|---|
| `pad` | 保持比例缩放并居中补零，默认 |
| `interpolate` | 直接拉伸到目标 `(H,W)` |

目标分辨率由 `video_size=(H,W)` 调整。切换分辨率或 resize 不需要重建 Catalog 或
SamplePool。

### 8.3 归一化模式

`reader_options.libero.normalization_mode`：

| 值 | 行为 |
|---|---|
| `zscore` | 官方 mean/std，默认 |
| `raw` | 原始 float32 数值 |

切换归一化不需要重建 Catalog 或 SamplePool。若显式传入 `statistics_path`，文件
内容必须与 Catalog 绑定的 SHA256 一致，不能借此无审计替换统计量。

### 8.4 未来帧数

`adapter_options.num_video_frames` 默认 48。若改成 `N`：

```text
每条 selection 的 video index 数 = N + 1
batch.video_frames.shape[1]         = N
```

该开关必须与 SamplePool 同步。现有 49 帧 selection 不能直接交给配置为 16 帧的
adapter，adapter 会 fail closed，不会自动裁剪。

### 8.5 action/video 比例

`adapter_options.action_video_fps_ratio` 默认 1。若改成 `R`：

```text
每条 selection 的 action index 数 = N * R
batch.action_sequence.shape[1]      = N * R
```

该开关也必须与 SamplePool 同步，并由模型端保证所给 action indices 具有正确语义。
数据工厂不做重采样和时间轴推导。

### 8.6 loss 权重

adapter 支持：

```text
video_loss_weight
action_loss_weight
```

默认均为 1.0，合法范围 `[0,1]`。这些字段只是 batch 中提供给模型端的权重，不会
删除任何模态，也不会改变 reader 的读取行为。

### 8.7 自定义 adapter

模型端需要完全不同的 batch 协议时，可使用：

```python
output_adapter=None
```

此时 DataLoader 返回原始 `IndexedEpisodeBatch`，模型端可以提供自己的 callable 或
注册新的 adapter。应把模型协议变化集中在 adapter，不要修改 Catalog、reader 或
SamplePool 来适配单个模型。

### 8.8 DataLoader 和 queue

可调整：

```text
batch_size
num_workers
pin_memory
queue_enabled
queue_size
queue_timeout_seconds
queue_clone_tensors
shuffle
drop_last
seed
rank/world_size
```





