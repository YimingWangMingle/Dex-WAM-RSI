# Agibot Beta 数据工厂两层 API 使用指南

## 1. 文档定位

本文以第一个完整接入的 Agibot Beta 数据源为例，说明 Motus Data Factory
对外提供的两层能力：

1. **通用数据访问层**：任何用户都可以枚举 Episode、查看元数据和模态结构，
   并按自己给出的索引读取 video、state、action 和 text。
2. **训练数据准备层**：读取模型训练侧提供的 SamplePool，将其中每一行解释为
   一个样本，完成安全校验、同 Episode 联合解码、组 batch、DataLoader 和
   ReadyBatchQueue，供训练主线程消费。

数据工厂不替模型端决定采样策略。模型端选择哪些 Episode、哪些 video frame、
哪些 state/action timestep 和哪些 text，数据工厂负责精确读取、验证并保持原始
索引顺序。

模型端日常接入优先使用
[`PUBLIC_API_ZH.md`](PUBLIC_API_ZH.md) 中的 `open_dataset()` 和
`open_training_data()`。本文后续显式组装 Repository、sampler、DataLoader 和
queue 的代码主要用于理解内部结构、高级调试和自定义基础设施。

## 2. 当前完成情况

| 能力 | 当前状态 | 主要入口 |
|---|---|---|
| 完整 Episode 枚举 | 已完成 | `EpisodeRepository.iter_episode_ids()` |
| Episode 元数据和模态 schema | 已完成 | `get_episode()`、`EpisodeView.info` |
| 四模态独立索引读取 | 已完成 | `read_video/state/action/text()` |
| 长视频分块访问 | 已完成 | `iter_video_chunks()` |
| 有内存上限的完整 Episode 物化 | 已完成 | `materialize()`、`load_full_episode()` |
| 单个显式样本切片 | 已完成 | `EpisodeView.read_selection()` |
| SamplePool 文件绑定和全量索引校验 | 已完成 | `SamplePool` |
| Dataset `__getitem__` | 已完成 | `SamplePoolDataset[index]` |
| 同 Episode batch 联合读取 | 已完成 | `EpisodeGroupedBatchSampler` |
| 当前 Motus 模型 batch 转换 | 已完成 | `IndexedBatchCollator` |
| CPU DataLoader | 已完成 | PyTorch `DataLoader` |
| 有界后台 ready queue | 已完成 | `ReadyBatchQueue` |
| 多 rank 不重叠分片 | 已完成 | `create_batch_sampler()` |
| Catalog 低内存懒加载 | 已完成 | SQLite offset sidecar |
| 极简数据访问入口 | 已完成 | `open_dataset()` |
| 极简训练入口 | 已完成 | `open_training_data()` |

完整 Agibot Beta Catalog、索引 Repository、SamplePool 和真实 MP4/HDF5
grouped decode 已通过闭环验收。

2026-07-26 进一步完成了训练消费前的真实 queue 闭环：正式 Catalog/SQLite
sidecar 加载 8 条 SamplePool selection，形成并消费 2 个 batch。每批 4 条样本，
均精确输出 `49 video/1 state/48 action/1 text`，video union 为 97 帧，随后经
`motus_model_batch_v1`、CPU DataLoader 和 `ReadyBatchQueue` 由模拟训练主线程
成功取出。最终 marker 为 `REAL_AGIBOT_BETA_INDEXED_QUEUE_OK`。

当前需要明确的工程边界：

- 新的 Catalog/SamplePool 路径已经封装为独立的
  `open_training_data()`；旧 direct Dataset 使用的 YAML
  `build_data_pipeline(config)` 保持兼容，但不作为 indexed 路径入口。
- 当前 indexed 训练路径先完成了 Agibot Beta 单数据源。多个 Catalog/SamplePool
  的统一加权混训还需要在后续增加 indexed mix 层。

## 3. 两层数据流

### 3.1 第一层：数据访问

```text
Agibot Beta 原始 tar / MP4 / HDF5 / 标注
  -> 离线 Episode Catalog
  -> 可选 SQLite offset sidecar
  -> EpisodeRepository
  -> EpisodeView
  -> 按 video/state/action/text 各自索引空间访问
```

Catalog 只描述完整 Episode 的可访问事实，不包含 train/test、窗口、采样权重
或模型 token。

### 3.2 第二层：训练准备

```text
模型侧生成 SamplePool
  -> SamplePool 绑定 Catalog 并校验所有索引
  -> SamplePoolDataset
  -> EpisodeGroupedBatchSampler
  -> 同 Episode 多样本 union read
  -> IndexedEpisodeBatch
  -> 可替换模型 batch adapter
  -> CPU DataLoader
  -> rank-local ReadyBatchQueue
  -> 训练主线程 queue.next()
```

SamplePool 是模型侧拥有的 sample index 文件。数据工厂不会生成、推导、排序、
截断、padding 或修复其中的多模态索引。

## 4. Agibot Beta 正式数据产物

集群当前正式产物：

```text
/data/jiyuan.wang/EpisodeCatalogs/_20260723/agibot_beta/
  agibot_beta_full_20260723_concise_v2.episode_catalog.jsonl
  agibot_beta_full_20260723_concise_v2.episode_catalog.meta.json
  agibot_beta_full_20260723_concise_v2.episode_catalog.index.sqlite
```

三者的职责：

- `episode_catalog.jsonl`：不可变的完整 Episode 清单，是唯一语义真源。
- `episode_catalog.meta.json`：Catalog ID、Episode 数量和 SHA256。
- `episode_catalog.index.sqlite`：`episode_id -> JSONL byte offset` 的物理加速
  sidecar，不复制 Episode 语义，不改变 Catalog ID。

当前 Catalog：

```text
catalog_id = agibot_beta_full_20260723_concise_v2
episodes   = 167794
```

开发机挂载路径是：

```text
/home/ubt/pod-code/EpisodeCatalogs/_20260723/agibot_beta/
```

集群程序必须继续使用 `/data/jiyuan.wang/...` 路径。

## 5. 导入数据工厂

在集群环境中以标准库方式安装：

```bash
pip install /path/to/motus-data-factory
# 开发阶段在仓库根目录执行：pip install -e .
```

Python：

```python
from motus_data import EpisodeRepository
```

如果环境中安装过旧版本 `motus_data`，请先 `pip uninstall motus-data-factory`
再重新安装。

## 6. 第一层：通用 Episode 数据访问

### 6.1 打开 Repository

推荐对完整 Catalog 使用 SQLite 索引模式：

```python
from pathlib import Path

from motus_data import EpisodeRepository

root = Path(
    "/data/jiyuan.wang/EpisodeCatalogs/_20260723/agibot_beta"
)
catalog = root / (
    "agibot_beta_full_20260723_concise_v2."
    "episode_catalog.jsonl"
)
index = root / (
    "agibot_beta_full_20260723_concise_v2."
    "episode_catalog.index.sqlite"
)

repository = EpisodeRepository(
    catalog,
    index_path=index,
    verify_sha256=False,
    record_cache_size=256,
)

print(repository.catalog_id)
print(len(repository))
print(repository.indexed)
```

`verify_sha256=False` 只适用于已经完成离线全量审计并冻结发布的 Catalog 和
sidecar。初始化仍会严格检查 Catalog ID、协议、文件名、文件大小、行数和发布
SHA 的 sidecar 绑定，并在每次 cache miss 时校验读取到的 Episode 行。

需要每次启动都重新顺序扫描完整 Catalog SHA 时：

```python
repository = EpisodeRepository(
    catalog,
    index_path=index,
    verify_sha256=True,
)
```

不传 `index_path` 会进入兼容的全量内存模式。完整 Agibot Beta Catalog 在该
模式下每个进程约需要 3.3 GiB RSS，不适合作为多 rank 训练默认值。

使用结束后关闭文件和 SQLite 句柄：

```python
repository.close()
```

### 6.2 枚举 Episode

只枚举稳定 ID，不读取原始视频或 HDF5：

```python
for position, episode_id in enumerate(repository.iter_episode_ids()):
    print(position, episode_id)
    if position == 9:
        break
```

查看标准化后的 Episode 摘要：

```python
for position, info in enumerate(repository.iter_episode_infos()):
    print(
        info.episode_id,
        info.robot_name,
        info.video_count,
        info.state_count,
        info.action_count,
        info.text_count,
    )
    if position == 9:
        break
```

`iter_episode_infos()` 需要逐条解析 Catalog record。只需要 ID 时优先使用
`iter_episode_ids()`。

正式 concise-v2 Catalog 是 split-neutral 的，`info.split` 为 `None`。
train/test/demo 成员由不同 SamplePool 决定，不应对 v2 Catalog 调用
`iter_episode_infos(split="train")`。

### 6.3 获取一个完整逻辑 Episode

```python
episode = repository.get_episode("agibot_beta:327:648642")

print(episode.episode_id)
print(episode.info)
print(episode.metadata)

print(episode.video.count)
print(episode.video.schema)
print(episode.state.count)
print(episode.state.schema)
print(episode.action.count)
print(episode.action.schema)
print(episode.text.count)
print(episode.text.items)
```

`get_episode()` 只返回一个惰性的 `EpisodeView`，不会立刻解码完整视频。

`episode.info` 的关键字段：

```text
episode_id
catalog_id
source_dataset
source_child
robot_name
video_count/state_count/action_count/text_count
modality_schemas
metadata
```

每个 `ModalitySchema` 会说明：

```text
name
count
dtype
item_shape
index_semantics
metadata
```

### 6.4 分别按索引读取四种模态

四个索引空间相互独立，全部从 0 开始：

```python
frames = episode.read_video([10, 12, 15, 15])
states = episode.read_state([10, 12, 15])
actions = episode.read_action([11, 13, 16])
texts = episode.read_text([0, 2, 0])
```

数据工厂保证：

- 每个索引在自己的模态 count 内做 bounds check；
- 保留调用者给出的顺序；
- 保留重复索引；
- 不根据 video index 推导 state/action/text index；
- 不自动排序、补齐或修复索引。

默认视频输出：

```text
dtype  = torch.uint8
shape  = [N,H,W,3]
layout = RGB HWC
```

模型常用输出：

```python
frames = episode.read_video(
    [10, 12, 15],
    output_format="rgb_float32_chw",
    video_size=(480, 640),
)
```

对应：

```text
dtype  = torch.float32
shape  = [N,3,480,640]
range  = [0,1]
```

Agibot Beta 的 state/action 返回标准化 `float32 [N,26]`。其规范化方法与此前
通过 parity 的 direct training Dataset 一致。

文本返回 `tuple[TextItem, ...]`。每条 `TextItem` 包含：

```text
index
kind
text
frame_range
metadata
```

如果一段 Episode 的不同范围有不同文本描述，Catalog 会保留这些文本和其
`frame_range`。模型端仍需在 SamplePool 中明确提供最终使用的 `text_indices`。

### 6.5 按调用者给出的完整多模态索引读取一个切片

```python
selection = {
    "episode_id": episode.episode_id,
    "video_frame_indices": [100, 104, 108, 112],
    "state_indices": [100],
    "action_indices": [101, 102, 103],
    "text_indices": [0, 1],
}

sample = episode.read_selection(
    selection,
    video_output_format="rgb_float32_chw",
    video_size=(480, 640),
)
```

返回 `EpisodeSelectionData`：

```python
sample.info
sample.video_frame_indices
sample.state_indices
sample.action_indices
sample.text_indices
sample.video
sample.state
sample.action
sample.text
```

该接口适合查看、debug、模型侧选样工具验证和单样本推理。它与训练层使用同一
套索引检查和源数据 reader。

### 6.6 同一 Episode 一次读取多个切片

```python
samples = episode.read_selections(
    [selection_a, selection_b, selection_c],
    video_output_format="rgb_float32_chw",
    video_size=(480, 640),
)
```

`read_selections()` 会对同一 Episode 的 video/state/action 索引分别取 union，
只做一次必要的物理读取，再精确还原每个样本原来的顺序和重复项。

### 6.7 查看完整长视频

不要默认把完整 Episode 视频一次性放入内存。推荐分块：

```python
for chunk in episode.iter_video_chunks(
    chunk_size=64,
    output_format="rgb_uint8_hwc",
):
    print(chunk.indices, chunk.frames.shape)
    user_visual_processor(chunk.indices, chunk.frames)
```

只读取完整非视频模态：

```python
data = episode.materialize(
    modalities=("state", "action", "text"),
)
```

确实需要完整视频时必须显式设置上限：

```python
data = repository.load_full_episode(
    episode.episode_id,
    modalities=("video", "state", "action", "text"),
    max_video_bytes=6 * 1024**3,
    video_output_format="rgb_uint8_hwc",
)
```

估算大小超过上限时会在完整解码前抛出 `EpisodeMemoryLimitError`。

## 7. 第二层输入：模型侧 SamplePool

### 7.1 SamplePool 格式

第一行是 header：

```json
{"format":"motus_sample_pool_v2","pool_id":"agibot_beta_train_001","catalog_id":"agibot_beta_full_20260723_concise_v2","samples":1000000}
```

之后每行定义一个样本：

```json
{"episode_id":"agibot_beta:327:648642","video_frame_indices":[100,101,102,103],"state_indices":[100],"action_indices":[101,102,103],"text_indices":[0,1]}
```

四组索引全部由模型侧提供：

- `video_frame_indices`：至少一个；
- `state_indices`：可以为空；
- `action_indices`：可以为空；
- `text_indices`：可以为空。

数据工厂不会由一种模态的索引计算另一种模态的索引。

生产 SamplePool 的生成逻辑属于模型侧。数据工厂只要求：

1. `catalog_id` 与目标 Catalog 完全相同；
2. 每个 `episode_id` 存在；
3. 所有索引均在各自模态范围内；
4. 模型侧确认这些独立索引具有其需要的语义对齐关系。

训练、测试和展示使用不同 Pool：

```text
agibot_beta_train.sample_pool.jsonl
agibot_beta_test.sample_pool.jsonl
agibot_beta_demo.sample_pool.jsonl
```

Catalog 本身不划分 split。模型侧负责保证 train/test Pool 的 Episode 或 sample
不发生其不允许的重叠。

### 7.2 加载并验证 Pool

```python
from motus_data import SamplePool

pool = SamplePool(
    "/path/model_owned.sample_pool.jsonl",
    repository,
)

print(pool.pool_id)
print(len(pool))
print(pool.header)
print(pool.episode_groups)
```

构造 `SamplePool` 时会扫描 Pool 并完成以下检查，但不会解码 MP4/HDF5：

- header 协议合法；
- `catalog_id` 绑定正确；
- 每个 Episode 存在；
- 每行四类索引字段完整；
- 所有索引不越界；
- header 的 sample count 与实际行数一致。

失败会给出 SamplePool JSONL 行号，不会静默跳过或换样。

## 8. 第二层 Dataset：从 Pool 行得到训练样本

```python
from motus_data import SamplePoolDataset

dataset = SamplePoolDataset(
    repository,
    pool,
    video_output_format="rgb_float32_chw",
    video_size=(480, 640),
    action_magnitude_guard_threshold=100.0,
    action_magnitude_guard_sources=["agibot_beta"],
)
```

### 8.1 `__getitem__`

```python
sample = dataset[0]
selection = dataset.selection(0)
```

- `dataset.selection(0)` 返回 Pool 第 0 行索引定义的副本，不读取原始数据。
- `dataset[0]` 精确读取该行，返回 `EpisodeSelectionData`。

这里的整数 index 是 **SamplePool 行号**，不是 Episode 帧号，也不是 Catalog
Episode ordinal。

### 8.2 数值安全

在视频提取和解码前，Dataset 会：

- 对选中的 state/action 强制检查 finite；
- 默认对 Agibot Beta 使用绝对值阈值 100 的 magnitude guard。

异常样本直接失败，不会随机换一个样本。完整生产 Pool 在分布式训练前仍应做
一次只读 state/action 的离线数值审计；运行时 guard 是最后一道边界。

## 9. 同 Episode grouped DataLoader

### 9.1 为什么一个 batch 必须来自同一个 Episode

同一 Episode 中多个样本的窗口通常重叠。grouped batch 可以：

1. 合并多个样本的 video/state/action 索引；
2. HDF5 时间轴只加载一次；
3. tar 中 MP4 member 只提取一次；
4. 所需唯一视频帧只联合解码一次；
5. 再按每个 Pool selection 的原始索引精确重建样本。

该优化只减少物理 I/O，不改变样本内容。

### 9.2 构造 sampler

```python
batch_size = 4
rank = 0
world_size = 1
training_seed = 20260724

batch_sampler = dataset.create_batch_sampler(
    batch_size=batch_size,
    drop_last=True,
    shuffle=True,
    num_replicas=world_size,
    rank=rank,
    seed=training_seed,
)
```

约束：

- 一个 batch 只包含同一 Episode 的 Pool rows；
- complete batch 在 rank 间分片，不重叠；
- 所有 rank 获得相同 batch 数；
- `drop_last=True` 时，同一 Episode 不足一个 batch 的余数不会与其他 Episode
  混合；
- 不要再叠加 `DistributedSampler`。

生产 Pool 应保证每个需要参与训练的 Episode 至少有足够 selection 形成 batch。

### 9.3 获取模型无关的原始 indexed batch

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_sampler=batch_sampler,
    collate_fn=dataset.collate_fn,
    num_workers=0,
    pin_memory=False,
)

indexed_batch = next(iter(loader))
```

返回 `IndexedEpisodeBatch`：

```python
indexed_batch.info
indexed_batch.episode_id
indexed_batch.batch_size
indexed_batch.samples
```

这是数据工厂最稳定、模型无关的 batch 边界。每个 sample 仍携带四组原始索引
和解码结果。数据工厂不会在这一层强制 padding 或固定长度。

## 10. 转成当前 Motus 模型 batch

当前 Motus trainer 使用 `motus_model_batch_v1`：

```python
from motus_data import (
    IndexedBatchCollator,
    build_indexed_batch_adapter,
)

adapter = build_indexed_batch_adapter(
    "motus_model_batch_v1",
    num_video_frames=48,
    action_video_fps_ratio=1,
    video_loss_weight=1.0,
    action_loss_weight=0.0,
)

loader = DataLoader(
    dataset,
    batch_sampler=batch_sampler,
    collate_fn=IndexedBatchCollator(adapter),
    num_workers=0,
    pin_memory=False,
)

batch = next(iter(loader))
```

该 profile 要求每条 SamplePool selection 明确提供：

```text
49 video indices = 1 条件帧 + 48 future frames
1 state index
48 action indices（ratio=1）
至少 1 个 text index
```

不满足时 adapter 会硬失败，不会补齐或截断。

主要输出：

```text
first_frame          [B,3,H,W]
video_frames         [B,48,3,H,W]
initial_state        [B,26]
action_sequence      [B,48,26]
t5_input_messages    list[str]
vlm_input_messages   list[str]
first_frame_pils     list[PIL.Image]
robot_names
source_datasets
source_children
episode_ids
frame_indices
```

Stage1 video-only 可以把 `action_loss_weight=0.0`，但当前
`motus_model_batch_v1` 为兼容现有 trainer，仍要求 SamplePool 提供 state/action
并输出对应 tensor。真正不读取 state/action 的新模型 profile 应注册新的
adapter，而不是让 Dataset 猜测训练模式。

## 11. 接入 ReadyBatchQueue

```python
from motus_data.loading import ReadyBatchQueue

ready_queue = ReadyBatchQueue(
    loader,
    max_size=6,
    get_timeout_seconds=600.0,
    low_watermark=0,
    name=f"agibot_beta.train.rank{rank}",
    clone_tensors=False,
)

try:
    ready_queue.start()

    for step in range(max_steps):
        batch, queue_stats = ready_queue.next()

        # 模型侧职责：tokenizer/VLM/T5、搬卡、forward、loss、backward。
        batch["vlm_inputs"] = vlm_processor(
            batch["first_frame_pils"],
            batch["vlm_input_messages"],
        )
        batch["language_embedding"] = t5_encoder(
            batch["t5_input_messages"]
        )
        batch = move_batch_to_device(batch, device)
        loss = model_training_step(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        log_queue(queue_stats)
finally:
    ready_queue.close()
    repository.close()
```

ReadyBatchQueue：

- 每个 rank 独立创建；
- 后台线程持续消费 CPU DataLoader；
- 只保存已经 collate 完成的 batch；
- 有界，不会无限累积；
- DataLoader 耗尽后递增 data epoch，并对支持的 sampler 调用 `set_epoch()`；
- producer 异常会传递到训练主线程；
- `queue.next()` 返回 `(batch, queue_stats)`。

当前 Agibot Beta 验收和长期训练的保守配置是 `num_workers=0`，由 ready queue
线程隐藏 DataLoader 延迟。启用多 worker 前应单独验证内存、`/dev/shm`、
tar/HDF5 并发读取和每 worker 的 SamplePool/Repository 开销。

当 `num_workers>0` 且 queue 长时间保存共享内存 tensor 时，可评估
`clone_tensors=True`，代价是增加一次 CPU tensor copy。`num_workers=0` 时通常
没有解除 DataLoader shared-memory 生命周期的需要。

## 12. 分布式训练接入

每个训练 rank 都需要独立构造：

```text
EpisodeRepository
SamplePoolDataset
EpisodeGroupedBatchSampler
DataLoader
ReadyBatchQueue
```

示例：

```python
import torch.distributed as dist

rank = dist.get_rank()
world_size = dist.get_world_size()

batch_sampler = dataset.create_batch_sampler(
    batch_size=batch_size,
    drop_last=True,
    shuffle=True,
    num_replicas=world_size,
    rank=rank,
    seed=training_seed,
)
```

注意：

- 传全局 `rank/world_size`；
- 所有 rank 使用相同基础 seed，不要再手工 `seed + rank`；
- 不要额外增加 `DistributedSampler`；
- 每个 rank 使用自己的 `ReadyBatchQueue`；
- 完整 Catalog 应使用 SQLite sidecar，避免每 rank 反序列化 3.3 GiB Catalog。

不使用 queue 且模型侧显式管理 epoch 时：

```python
for epoch in range(num_epochs):
    batch_sampler.set_epoch(epoch)
    for batch in loader:
        train_step(batch)
```

## 13. 模型侧如何自定义输出

数据工厂稳定边界是：

```text
IndexedEpisodeBatch
```

`motus_model_batch_v1` 只是当前 Motus 的一个 adapter。下游模型可以注册自己的
profile：

```python
from motus_data import register_indexed_batch_adapter


class MyModelBatchAdapter:
    profile_name = "my_model_batch_v1"

    def adapt(self, batch):
        return {
            "episode_id": batch.episode_id,
            "frames": [sample.video for sample in batch.samples],
            "texts": [sample.text for sample in batch.samples],
        }


register_indexed_batch_adapter(
    "my_model_batch_v1",
    MyModelBatchAdapter,
)
```

然后：

```python
from motus_data import (
    IndexedBatchCollator,
    build_indexed_batch_adapter,
)

adapter = build_indexed_batch_adapter("my_model_batch_v1")

loader = DataLoader(
    dataset,
    batch_sampler=batch_sampler,
    collate_fn=IndexedBatchCollator(adapter),
    num_workers=0,
)
```

也可以直接使用 `IndexedEpisodeBatch`，在模型工程中编写自己的 collate/adapter。
需要保留同 Episode grouped sampler，否则会丢失 union decode 优化。

模型 adapter 可以决定：

- 条件帧和预测帧如何拆分；
- 是否使用 state/action；
- 是否 padding；
- 模型 tensor 的最终字段名和布局；
- tokenizer、VLM/T5 预处理；
- loss weight 和 branch 路由。

模型 adapter 不应：

- 修改或猜测 Pool 中的索引；
- 绕过 Dataset 的 bounds/finite/magnitude 校验；
- 把不同 Episode 强行塞入要求同 Episode 的 grouped read；
- 直接依赖 Agibot Beta tar/HDF5 的物理字段。

## 14. 用户可自定义的边界

### 14.1 只使用第一层

适用于：

- 数据浏览器；
- Episode 统计；
- 模型侧视觉分析；
- SamplePool 生成工具；
- 单样本 debug；
- 推理和展示。

只依赖：

```text
EpisodeRepository -> EpisodeView
```

### 14.2 使用 Dataset，但自定义 DataLoader

保留：

```text
EpisodeRepository
SamplePool
SamplePoolDataset
EpisodeGroupedBatchSampler
```

模型侧可以修改 DataLoader worker、pin memory 和外层 wrapper，但不能用普通
随机 sampler 替换 grouped batch sampler。

### 14.3 使用完整数据准备链路

保留：

```text
Repository -> Pool -> Dataset -> grouped sampler
-> model adapter -> DataLoader -> ReadyBatchQueue
```

训练端只消费 `queue.next()`。

## 15. 数据工厂与模型端职责

### 15.1 数据工厂负责

- 维护 Episode Catalog 和 source reader；
- 解析 Agibot Beta tar、MP4、HDF5 和文本；
- 提供统一 Episode 访问 API；
- 验证 Pool/Catalog 绑定和所有索引；
- 保持四类索引的顺序与重复；
- 标准化 Agibot Beta state/action；
- finite 和 magnitude 数据安全检查；
- 同 Episode union read；
- 分布式 grouped batch sampler；
- DataLoader 和 ReadyBatchQueue；
- 输出稳定的模型无关 `IndexedEpisodeBatch`。

### 15.2 模型侧负责

- 查看完整 Episode 后生成生产 SamplePool；
- 决定 train/test/demo Pool；
- 保证其提供的四类索引具有目标语义对齐；
- 决定每个样本长度、stride 和采样分布；
- 提供模型 batch adapter 或选择现有 profile；
- tokenizer、VLM、T5、搬卡、forward、loss、backward；
- optimizer、DeepSpeed、checkpoint 和训练调度。

## 16. 当前限制和下一步

当前 Agibot Beta 已经支持两层功能，但尚有以下后续工程项：

1. 增加多个 Catalog/SamplePool 的 indexed mix 层，再迁移到十源混训。
2. 在生产 SamplePool 上运行完整 state/action-only 离线数值审计。
3. 根据模型端最终 batch 协议新增或冻结正式 adapter。
4. 评估超大 SamplePool 的索引化/流式加载；当前 `SamplePool` 会在每个进程中
   扫描并保存全部 selection。
5. 按相同 Repository 协议继续接入 Agibot、RobotMind 和 RoboTwin。

旧的：

```python
build_dataset(config)
build_dataloader(config)
build_data_pipeline(config)
```

当前仍服务于历史 direct/YAML Dataset 路径。Catalog + SamplePool 的普通模型
接入应使用 `open_training_data()`；本文后面的显式组件构造只用于高级控制。

## 17. 最小完整示例

```python
from pathlib import Path

from torch.utils.data import DataLoader

from motus_data import (
    EpisodeRepository,
    IndexedBatchCollator,
    SamplePool,
    SamplePoolDataset,
    build_indexed_batch_adapter,
)
from motus_data.loading import ReadyBatchQueue


root = Path(
    "/data/jiyuan.wang/EpisodeCatalogs/_20260723/agibot_beta"
)
catalog = root / (
    "agibot_beta_full_20260723_concise_v2."
    "episode_catalog.jsonl"
)
index = root / (
    "agibot_beta_full_20260723_concise_v2."
    "episode_catalog.index.sqlite"
)
pool_path = Path("/path/model_owned.sample_pool.jsonl")

rank = 0
world_size = 1
batch_size = 4

repository = EpisodeRepository(
    catalog,
    index_path=index,
    verify_sha256=False,
)
pool = SamplePool(pool_path, repository)
dataset = SamplePoolDataset(
    repository,
    pool,
    video_output_format="rgb_float32_chw",
    video_size=(480, 640),
)
batch_sampler = dataset.create_batch_sampler(
    batch_size=batch_size,
    drop_last=True,
    shuffle=True,
    num_replicas=world_size,
    rank=rank,
    seed=20260724,
)
adapter = build_indexed_batch_adapter(
    "motus_model_batch_v1",
    num_video_frames=48,
    action_video_fps_ratio=1,
    action_loss_weight=0.0,
)
loader = DataLoader(
    dataset,
    batch_sampler=batch_sampler,
    collate_fn=IndexedBatchCollator(adapter),
    num_workers=0,
    pin_memory=False,
)
queue = ReadyBatchQueue(
    loader,
    max_size=6,
    get_timeout_seconds=600.0,
    name=f"agibot_beta.rank{rank}",
)

try:
    queue.start()
    for step in range(10):
        batch, stats = queue.next()
        print(
            step,
            batch["first_frame"].shape,
            batch["video_frames"].shape,
            batch["initial_state"].shape,
            batch["action_sequence"].shape,
            stats["ready_queue_size_after_get"],
        )
finally:
    queue.close()
    repository.close()
```

该示例完成：

```text
完整 Catalog 的低内存索引访问
-> 模型侧 Pool 校验
-> 同 Episode batch
-> union decode
-> Motus batch adapter
-> CPU DataLoader
-> ReadyBatchQueue
-> 模拟训练线程消费
```

## 18. 相关文档和代码入口

```text
Public API:
  motus_data/__init__.py

Episode access:
  motus_data/repository.py
  docs/EPISODE_REPOSITORY_V1.md

Catalog protocol:
  docs/EPISODE_CATALOG_SAMPLE_POOL_V2.md

SQLite sidecar:
  motus_data/catalog/sqlite_index.py
  docs/CATALOG_OFFSET_INDEX.md

SamplePool and grouped Dataset:
  motus_data/sample_pool.py
  docs/SAMPLE_POOL_DATASET_V1.md

Model batch adapters:
  motus_data/batch_adapters.py
  docs/INDEXED_BATCH_ADAPTERS.md

Ready queue:
  motus_data/loading/ready_queue.py

Current Motus batch contract:
  motus_data/contracts.py
  docs/BATCH_CONTRACT_V1.md
```
