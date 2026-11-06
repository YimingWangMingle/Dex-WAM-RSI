# LIBERO Parquet 数据工厂接入

## 当前状态

本分支已完成 Libero 的数据访问和训练数据管线：

- 四个 split-neutral child Catalog builder；
- `LiberoParquetEpisodeReader`；
- Catalog SQLite offset index 发布工具；
- 四份全量 Catalog、metadata 和 SQLite offset index；
- 真实 Episode 四模态读取 smoke；
- 覆盖四十个 task、八十个 Episode 的 320 条 smoke SamplePool；
- 四 child 加权 mix、DataLoader 和 CPU ReadyBatchQueue；
- 原生 8D state / 7D action 的默认训练 adapter；
- 默认 z-score、可选 raw 的归一化实现；
- 微型 Parquet 单元测试和旧数据回归。

四个数据访问 release 已注册到 `releases.yaml`。数据工厂侧已经可以根据显式
SamplePool 完成切片读取、组 batch、DataLoader 和 queue；正式训练仍需模型侧提供
production SamplePool、采样权重及训练配置。

## 数据源协议

统一 source 为 `libero`，四个 child 独立提供 SamplePool 和采样权重：

```text
libero_10
libero_goal
libero_object
libero_spatial
```

每个 Parquet 文件是一个完整 Episode，每行是同一个 10 Hz 时间步：

```text
image[t] + wrist_image[t] + state[t]  -> observation[t]
actions[t]                            -> observation[t] 后执行的控制命令
text[0]                               -> 整个 Episode 的英文任务
```

Catalog 保留原生模态：

```text
video:  两路 256x256 RGB PNG
state:  float32 [T, 8]，绝对 EEF state
action: float32 [T, 7]，增量 EEF 控制命令 + 离散夹爪命令
text:   一个 task_en
```

数据工厂不从 video index 推导 state/action index。模型侧 SamplePool 必须显式
给出四组索引，reader 仅做边界校验和精确读取。

## 相机模式

reader 默认只返回主相机：

```python
from motus_data import LiberoParquetEpisodeReader

reader = LiberoParquetEpisodeReader(camera_mode="main")
```

可选模式：

```text
main                    主相机，默认
wrist                   腕部相机
main_wrist_horizontal   同时刻主相机和腕部相机水平拼接
```

相机模式属于 reader 配置，不写入 SamplePool，也不需要重建 Catalog。拼接模式
默认使用保持比例的 resize + 居中补零；需要复刻拉伸行为时可显式设置
`resize_mode="interpolate"`。

## 默认数值策略

reader 始终返回 `float32`，默认配置为：

```text
normalization_mode = zscore
state               [N, 8]
action              [N, 7]
公式                 (x - mean) / std
统计量               /data/chen.liao/Libero_parquet/meta/stats.json
```

state 的 8 维和 action 的 7 维分别使用官方全量 mean/std；action 的夹爪维也按同一
公式处理。Catalog metadata 记录 `stats.json` 的 SHA256，reader 在首次归一化前
核对该指纹，统计文件变化时直接失败，避免静默使用不同版本统计量。

模型侧需要原始值时可覆盖 release reader 配置：

```python
reader_options={
    "libero": {
        "camera_mode": "main",
        "resize_mode": "pad",
        "normalization_mode": "raw",
    }
}
```

不得截断 state、给 action 补零或强制二者维度相同。

## 默认 batch adapter

Libero 推荐使用注册 profile：

```text
motus_native_multimodal_batch_v1
```

默认 adapter 参数：

```text
num_video_frames=48
action_video_fps_ratio=1
video_loss_weight=1.0
action_loss_weight=1.0
```

对应每条 SamplePool selection 必须显式给出 49 个 video index、1 个 state index、
48 个 action index 和至少 1 个 text index。数据工厂不会从 video index 推导、修复
或替换其他模态索引。默认 batch 的主要字段为：

```text
first_frame       float [B, 3, H, W]
video_frames      float [B, 48, 3, H, W]
initial_state     float [B, 8]
action_sequence   float [B, 48, 7]
t5_input_messages list[str], len=B
vlm_input_messages list[str], len=B
source_datasets/source_children/episode_ids/index metadata
normalization_modes={"state": "zscore", "action": "zscore"}
```

训练端可通过 `adapter_options` 修改帧数、action/video 比例和 loss 权重；也可以令
`output_adapter=None` 获取原始 `IndexedEpisodeBatch`，再提供自己的 adapter。
旧 `motus_model_batch_v1` 保持不变，避免影响已有数据和训练接入。

## 构建四个 Catalog

集群原始数据和发布路径示例：

```bash
python -u tools/build_libero_release.py \
  --data-root /data/chen.liao/Libero_parquet \
  --storage-root /data/chen.liao/Libero_parquet \
  --output-root /data/jiyuan.wang/EpisodeCatalogs/_20260810/libero_parquet_v1 \
  --data-version libero_parquet_v1
```

`--data-root` 是构建时实际读取的位置；`--storage-root` 是写入 Catalog 的长期
运行时路径。正式产物二者均应为集群真实路径，不能写本地 SSHFS 挂载路径。

脚本逐 child 原子发布以下文件：

```text
<child>.libero_parquet_v1.episode_catalog.jsonl
<child>.libero_parquet_v1.episode_catalog.meta.json
<child>.libero_parquet_v1.episode_catalog.index.sqlite
```

已完整发布的 child 会校验 SHA256 后复用，因此进程在 child 边界中断后可安全
重跑。不要使用 `--force` 覆盖已经进入训练的正式 release。

## Episode 读取 smoke

```bash
python -u tools/smoke_libero_episode.py \
  --catalog /path/to/libero_10.libero_parquet_v1.episode_catalog.jsonl \
  --index /path/to/libero_10.libero_parquet_v1.episode_catalog.index.sqlite \
  --camera-mode main \
  --video-height 480 \
  --video-width 640
```

该命令从首、中、末三个时间步读取 video/state/action 和任务文本，输出 shape、
有限值检查与内容 SHA256。也应分别对 `wrist` 和
`main_wrist_horizontal` 运行一次。

## DataLoader 与 queue

训练端推荐调用：

```python
pipeline = open_training_data(
    "libero_10",
    sample_pool="/path/to/libero_10.sample_pool.jsonl",
    batch_size=2,
    output_adapter="motus_native_multimodal_batch_v1",
    adapter_options={
        "num_video_frames": 48,
        "action_video_fps_ratio": 1,
    },
    video_size=(480, 640),
    num_workers=0,
    queue_enabled=True,
    queue_size=2,
)
with pipeline:
    batch, stats = pipeline.next()
```

四 child 混训使用 `open_mixed_training_data()`，每个 child 独立提供 SamplePool 和
权重。默认 adapter 只定义数据输出，不决定模型如何使用各模态。

## 已完成验证

- 全量四 child：1,693 Episode、273,465 帧，Catalog/index 全部发布；
- 四个真实 suite 各构建一个 Episode Catalog 和 SQLite index：通过；
- 真实 `libero_10` Episode 的三种相机模式：通过；
- 真实输出 shape：video `[3,3,H,W]`、state `[3,8]`、action `[3,7]`；
- 数值有限性：通过；
- raw indexed mix：四 child 各消费两个 queue batch，通过；
- 默认 z-score 的 native adapter 单元闭环：state `[B,8]`、action `[B,T,7]`，通过；
- 真实 Parquet 默认链路：四 child 各消费 2 个 queue batch，state `[2,8]`、
  action `[2,48,7]`、未来视频 `[2,48,3,96,128]`，全部标记为 z-score，
  ReadyBatchQueue 正常关闭，通过；
- 480x640 长测：四 child 各消费 10 个 queue batch，共 40 batch；峰值 RSS 在
  第 3 个 batch 达到约 2248 MiB 后保持不变，`/dev/shm` 全程 0 MiB，queue
  正常关闭，通过；
- 真实比例 64-batch 配额：`14/16/17/17`，四 child 均非零；
- 统计文件 SHA256 与四份正式 Catalog metadata 完全一致；
- 全库回归测试通过。

Libero 数据工厂侧已通过验收。后续由模型端提供 production SamplePool 和 child
权重，并按本页默认输出协议完成短步训练握手。
