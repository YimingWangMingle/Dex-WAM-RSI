# Motus Data Factory 功能与模型端接入说明

> 本文同时保留历史 direct/YAML Dataset 管线的接入说明。当前已经完成的
> Agibot Beta Catalog/Repository/SamplePool 两层 API 及其训练链路，请优先阅读
> [`AGIBOT_BETA_DATA_FACTORY_GUIDE_ZH.md`](AGIBOT_BETA_DATA_FACTORY_GUIDE_ZH.md)。
> 模型端的极简 indexed 入口请优先使用
> [`PUBLIC_API_ZH.md`](PUBLIC_API_ZH.md) 中的 `open_dataset()` 和
> `open_training_data()`。旧 `build_data_pipeline(config)` 继续服务 direct/YAML
> 路径。

## 1. 文档定位

本文面向下游模型训练工程，说明 Motus Data Factory：

- 负责什么，不负责什么；
- 向模型端输出什么；
- 模型端如何以标准方式接入；
- 模型端如何调整数据策略；
- 需要自定义 DataLoader、采样器或新数据源时应从哪里扩展；
- 当前实现状态以及十源完整迁移后的目标形态。

数据工厂与模型工程通过版本化 batch 契约解耦。当前契约名为：

```text
motus_model_batch_v1
```

模型端不应依赖某个数据源的目录结构、Manifest、HDF5 key、tar member、
视频解码器或 Dataset 类名。

## 2. 总体架构

```text
原始数据
  -> source-specific Dataset
  -> 运行时窗口采样与多模态对齐
  -> child-homogeneous weighted batch sampler
  -> CPU DataLoader + canonical collate
  -> rank-local bounded ReadyBatchQueue
  -> motus_model_batch_v1
  -> 模型端文本预处理、搬卡、forward、loss、backward
```

每个训练 rank 构建自己的 DataPipeline 和 ReadyBatchQueue。数据工厂在 CPU
侧准备好 batch，模型训练主线程调用 `pipeline.next()` 消费，不需要了解原始
数据来自 MP4、HDF5、tar 还是兼容 Manifest。

## 3. 职责边界

### 3.1 数据工厂负责

1. 扫描并解析各数据源的原始目录、标注和物理存储。
2. 解码视频帧、HDF5 JPEG、state 和 action。
3. 在一个 source Dataset 内动态选择训练窗口。
4. 使用同一时间基准精确对齐文本、视频、initial state 和 future action。
5. 做 episode 级 train/test 划分，避免同一 episode 的窗口跨 split 泄漏。
6. 执行数据源特有的数据安全规则、维度检查和坏样本重试。
7. 按 child 权重做多数据源采样，并保证每个 batch 只来自一个 child。
8. 完成 canonical collate，输出稳定的 CPU batch 字典。
9. 管理 DataLoader 和有界 ReadyBatchQueue。
10. 输出 queue、source、episode、frame index 等诊断信息。
11. 在 batch 离开数据边界前执行契约检查。

### 3.2 模型端负责

1. 提供训练 YAML 中的数据策略和训练策略。
2. T5/VLM tokenizer、processor、embedding 等模型相关预处理。
3. CPU 到 GPU 的设备搬运、dtype 和 autocast。
4. 根据 `robot_names` 选择本体 projector、padding 或模型分支。
5. Stage1/Stage2/Stage3 模式、branch 冻结和解冻。
6. video/action loss 的最终启用、组合和反向传播。
7. optimizer、gradient accumulation、DeepSpeed/Accelerate 和分布式通信。
8. checkpoint、resume/finetune、日志和评测。

数据工厂输出 `action_loss_weights` 和 `video_loss_weights`，但是否启用 action
branch、是否计算 action loss，最终仍由模型训练策略决定。

### 3.3 数据工厂明确不负责

- 不导入 Wan、DiT、VAE、T5、VLM、DeepSpeed 或 Accelerate。
- 不生成模型 token，不执行模型 forward/backward。
- 不决定哪些模型层冻结。
- 不将 batch 搬到 GPU。
- 不保存模型 checkpoint。
- 不让模型 YAML 任意改写物理 HDF5 key、字段顺序或真实本体维度。

## 4. 输出给模型端的数据

### 4.1 核心 tensor

| 字段 | 类型和形状 | 含义 |
|---|---|---|
| `first_frame` | float tensor `[B,C,H,W]` | 条件帧，范围 `[0,1]` |
| `video_frames` | float tensor `[B,T,C,H,W]` | 需要模型预测的未来帧，范围 `[0,1]` |
| `initial_state` | float tensor `[B,D]` | 条件帧时刻的机器人 state |
| `action_sequence` | float tensor `[B,T_action,D]` | 与未来时间段对齐的 action |
| `video_loss_weights` | float tensor `[B]` | 每个样本的视频 loss 权重 |
| `action_loss_weights` | float tensor `[B]` | 每个样本的 action loss 权重 |

当前 Motus Stage1 虽然不使用 action/state 进入模型分支，但现有 trainer 仍会
读取并搬运这些字段，因此 `motus_model_batch_v1` 仍要求它们存在。真正不需要
action/state 的新模型可以使用通用可选字段语义，但不能直接假设当前 Motus
trainer 也允许缺失。

### 4.2 文本和路由信息

| 字段 | 类型 | 模型端用途 |
|---|---|---|
| `t5_input_messages` | `list[str]`，长度 `B` | 交给模型侧 T5 tokenizer/encoder |
| `vlm_input_messages` | `list[str]`，长度 `B` | 交给模型侧 VLM processor |
| `first_frame_pils` | `list[PIL.Image]` | VLM 图像输入 |
| `robot_names` | `list[str]` | 本体 projector/branch 路由 |
| `source_datasets` | `list[str]` | 大数据源标识，例如 `robotmind` |
| `source_children` | `list[str]` | 子数据源/本体标识 |
| `state_action_policies` | `list[str]` | `real` 等 state/action 语义 |
| `vlm_inputs` | 初始为 `None` | 模型侧 VLM 预处理后的占位字段 |
| `language_embedding` | 初始为 `None` | 模型侧文本编码后的占位字段 |

`D` 可以随本体变化，例如 RobotMind Ark 为 14、TianYi 为 16、ArkMobile
为 20。数据工厂保证一个 batch 内只出现一个 child，因此 batch 内维度一致；
模型端根据 `robot_names` 做本体路由，不能假设所有 batch 都是 26 维。

### 4.3 诊断字段

当前还会输出：

- `episode_ids`
- `manifest_ids`：为兼容旧模型保留的历史字段名，direct Dataset 也会填稳定 ID
- `manifest_tasks`
- `frame_indices`
- `action_video_fps_ratios`
- `initial_state_valid`
- `action_sequence_valid`

这些字段适合日志、审计和 debug。模型主干不应把 `manifest_ids` 理解为训练
仍然依赖 Manifest。

## 5. 推荐接入方式

### 5.1 安装

把数据工厂作为标准 Python package 安装到模型环境：

```bash
pip install /path/to/motus-data-factory
# 开发阶段在仓库根目录执行：pip install -e .
```

安装后即可直接 `import motus_data`，无需设置 `PYTHONPATH`。

### 5.2 标准训练接入

```python
from motus_data import build_data_pipeline, validate_motus_model_batch_v1

pipeline = build_data_pipeline(
    config,
    split="train",
    rank=global_rank,
    world_size=world_size,
    seed=training_seed,
)

with pipeline:
    for step in range(max_steps):
        batch, queue_stats = pipeline.next()

        validate_motus_model_batch_v1(
            batch,
            expected_batch_size=config.training.batch_size,
            num_video_frames=config.common.num_video_frames,
        )

        batch["vlm_inputs"] = vlm_processor(
            batch["first_frame_pils"],
            batch["vlm_input_messages"],
        )
        batch["language_embedding"] = t5_encoder(batch["t5_input_messages"])

        batch = move_batch_to_device(batch, device)
        loss = model_training_step(batch)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
```

实际模型项目可以把文本预处理和搬卡封装到自己的 adapter 中。要求是 adapter
只消费契约字段，不导入 source-specific Dataset。

### 5.3 不需要 ReadyBatchQueue

```python
from motus_data import build_dataloader

dataloader = build_dataloader(
    config,
    split="train",
    rank=global_rank,
    world_size=world_size,
    seed=training_seed,
)

for batch in dataloader:
    loss = model_training_step(batch)
```

也可以保留 `build_data_pipeline`，通过 `queue_enabled=False` 关闭 queue。

## 6. 模型侧可以通过 YAML 调整什么

### 6.1 多源采样比例

每个 mix child 的 `weight` 是该 child 被选为一个 batch 数据源的相对权重：

```yaml
dataset:
  type: mix
  batches_per_epoch: 300000
  datasets:
    - type: agibot
      name: agibot_direct
      source_dataset: agibot
      weight: 4.0
      # source paths omitted

    - type: robotmind
      name: robotmind_tianyi
      source_dataset: robotmind
      embodiment: tianyi
      weight: 1.0
      # source paths omitted
```

权重会归一化。它控制 batch 来源概率，不表示在一个 batch 中按比例混合样本。
一个 batch 始终属于一个 child，这对不同本体维度和 grouped decode 都是硬约束。

### 6.2 通用输出策略

```yaml
common:
  video_height: 480
  video_width: 640
  num_video_frames: 48

training:
  batch_size: 12

system:
  num_workers: 0
  pin_memory: false
  persistent_workers: false
  prefetch_factor: null
  ready_batch_queue_enabled: true
  ready_batch_queue_size: 6
  ready_batch_queue_clone_tensors: true
  ready_batch_queue_timeout_seconds: 600
```

### 6.3 child 级策略

以下字段属于历史 direct Dataset 或兼容 Manifest adapter 的策略，例如：

- `split`
- `test_fraction`、`split_seed`
- `virtual_samples_per_episode`
- `max_decode_retries`
- `instruction_lang`
- `video_loss_weight`、`action_loss_weight`
- RobotMind 的 `video_frame_step` 和严格校验后的 `action_video_fps_ratio`
- Agibot Beta 的坏任务排除、action magnitude guard 和 decode backend

新的 Catalog/SamplePool indexed 路径不在 Catalog builder 中读取
`split/test_fraction/split_seed` 或任务 blacklist。Catalog 发布完整可访问
Episode；训练、测试、展示成员和窗口均由模型侧不同 SamplePool 决定。

模型端可以在启动前使用 OmegaConf 修改这些策略，不需要改数据工厂代码：

```python
from omegaconf import OmegaConf

config = OmegaConf.load(model_yaml)
config.training.batch_size = 8
config.system.ready_batch_queue_size = 3
config.dataset.datasets[0].weight = 2.0
```

### 6.4 不能由模型 YAML 任意修改的内容

- HDF5 物理 key。
- state/action 拼接顺序。
- 本体真实维度。
- Agibot annotation 记录的原生 action/video FPS ratio。
- tar member 的物理地址和 byte offset 语义。
- 保证模态对齐的数据安全约束。

这些属于数据事实。如果新数据版本改变了物理 schema，应在数据工厂增加明确的
新 schema/adapter 并做 parity，而不是在训练 YAML 中临时覆盖。

## 7. 模型侧需要自定义 DataLoader 时

优先通过 YAML 和 `build_dataloader` 调整。确实需要自定义 worker 初始化、
DataLoader wrapper 或外部 sampler 时，可以只使用数据工厂的 Dataset 和 collate。

```python
from torch.utils.data import DataLoader, DistributedSampler

from motus_data import build_dataset
from motus_data.collate import collate_samples

dataset = build_dataset(
    config,
    split="train",
    rank=global_rank,
    seed=training_seed,
)

collate_fn = getattr(dataset, "collate_fn", collate_samples)

if hasattr(dataset, "create_batch_sampler"):
    batch_sampler = dataset.create_batch_sampler(
        batch_size=config.training.batch_size,
        drop_last=True,
        shuffle=True,
        num_replicas=world_size,
        rank=global_rank,
        seed=training_seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=custom_num_workers,
        worker_init_fn=custom_worker_init_fn,
    )
else:
    distributed_sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=True,
            seed=training_seed,
            drop_last=True,
        )
        if world_size > 1
        else None
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        sampler=distributed_sampler,
        shuffle=distributed_sampler is None,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=custom_num_workers,
        worker_init_fn=custom_worker_init_fn,
    )
```

必须保留以下约束：

1. MixDataset 必须使用它的 `create_batch_sampler()`，否则可能破坏 child-homogeneous batch。
2. Agibot Beta 等 grouped/prebatched child 必须保留其 child batch sampler 和
   Dataset 的 `collate_fn`，否则会失去同 episode union decode。
3. 分布式训练必须传正确的 `rank/world_size`，不能让多个 rank 完全重复采样。
4. 每个 epoch 应调用 batch sampler、DistributedSampler 和 Dataset 中存在的
   `set_epoch(epoch)`。
5. 自定义 collate 的最终输出仍必须通过 `validate_motus_model_batch_v1()`。
6. 不能在 collate 中混合不同 child 的 state/action 维度。

如果只是想修改 queue 行为，不需要重写 DataLoader，直接修改 `system` 下的 queue
配置或调用 `build_data_pipeline(..., queue_enabled=False)`。

### 7.1 分布式 Dataset-only 模式

模型端自己构造 DataLoader 时，必须使用全局 `rank/world_size`：

```python
import torch.distributed as dist

rank = dist.get_rank()
world_size = dist.get_world_size()

dataset = build_dataset(
    config,
    split="train",
    rank=rank,
    seed=training_seed,
)
```

所有 rank 应传入同一个基础 `training_seed`。不要在模型端再写
`seed + rank`；Dataset 和专用 sampler 已经把 rank 纳入随机种子。

对于 MixDataset 或提供 `create_batch_sampler()` 的 grouped Dataset：

```python
batch_sampler = dataset.create_batch_sampler(
    batch_size=batch_size,
    drop_last=True,
    shuffle=True,
    num_replicas=world_size,
    rank=rank,
    seed=training_seed,
)

dataloader = DataLoader(
    dataset,
    batch_sampler=batch_sampler,
    collate_fn=getattr(dataset, "collate_fn", collate_samples),
    num_workers=num_workers,
)

for epoch in range(num_epochs):
    batch_sampler.set_epoch(epoch)
    for batch in dataloader:
        train_step(batch)
```

不能再在这个 DataLoader 外层添加 `DistributedSampler`，也不能同时传
`batch_size`、`shuffle`、`sampler` 或 `drop_last`。专用 batch sampler 已负责：

- rank/world-size 分配；
- child-homogeneous batch；
- Mix 权重；
- Agibot Beta 同 episode grouped decode；
- epoch 随机顺序，并把 epoch 传播给 child Dataset。

对于不提供 `create_batch_sampler()` 的单一普通 Dataset，使用 PyTorch
`DistributedSampler`：

```python
sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    seed=training_seed,
    drop_last=True,
)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
    shuffle=False,
    drop_last=True,
    collate_fn=collate_samples,
    num_workers=num_workers,
)

for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    for batch in dataloader:
        train_step(batch)
```

`sampler.set_epoch(epoch)` 改变跨 rank 的索引顺序；`dataset.set_epoch(epoch)`
改变同一个 episode/index 内动态窗口的随机种子。缺少前者会让每个 epoch 的
索引顺序重复，缺少后者会让动态窗口在索引重复时也重复。

当前 direct Dataset 的窗口 seed 由以下信息共同组成：

```text
global seed + epoch + rank + worker id + dataset name + logical sample index
```

以上 train/test 归属规则只适用于历史 direct Dataset。split-neutral Catalog
没有 train/test 字段；模型侧必须生成所需的 Pool，并在需要时审计 train/test
Pool 的 Episode/sample 交集。

`persistent_workers=True` 需要额外注意：worker 中持有独立 Dataset 副本，主
进程调用 `dataset.set_epoch()` 不会自动修改已经常驻的 worker 副本。当前稳定
生产配置使用 `num_workers=0`，不存在该问题。以后启用 persistent workers 时，
需要共享 epoch 状态、在 worker 中同步 epoch，或每个 epoch 重建 worker；不能
只在主进程修改一个普通 Python 整数。

如果训练按 `max_steps` 而不是显式 epoch 运行，应在 DataLoader iterator 耗尽并
重新创建时递增 `data_epoch`，然后在新 iterator 前调用相应的 `set_epoch()`。

## 8. 自定义新数据源

下游团队可以注册自己的 Dataset builder，而不修改核心 factory：

```python
from motus_data import register_dataset

@register_dataset("my_robot")
def build_my_robot(*, node, common, split, rank, seed, **kwargs):
    return MyRobotDataset(
        data_root=node["data_root"],
        split=split,
        video_size=(common["video_height"], common["video_width"]),
        num_video_frames=common["num_video_frames"],
        rank=rank,
        seed=seed,
    )
```

`MyRobotDataset.__getitem__()` 应返回 canonical sample 字段：

```python
{
    "first_frame": ...,             # [C,H,W]
    "video_frames": ...,            # [T,C,H,W]
    "initial_state": ...,           # [D]
    "action_sequence": ...,         # [T_action,D]
    "first_frame_pil": ...,
    "t5_input_message": ...,
    "vlm_input_message": ...,
    "robot_name": ...,
    "source_dataset": ...,
    "source_child": ...,
    "video_loss_weight": 1.0,
    "action_loss_weight": 1.0,
    "state_action_policy": "real",
    "episode_id": ...,
    "manifest_id": ...,
    "frame_indices": ...,
    "action_video_fps_ratio": ...,
}
```

推荐继承 `EpisodeDataset`，复用 rank/worker/epoch seed、虚拟样本和失败重试。
如果数据源能按 episode 共享解码，应像 Agibot Beta 一样实现自己的
`create_batch_sampler()` 和可选 `__getitems__()`，不要在通用 MixDataset 中
写数据源特例。

## 9. 自定义窗口采样策略

### 当前能力

- direct Agibot：在 annotation 的有效 frame 范围内随机选起点。
- direct Agibot Beta：在有效 action span 内随机选起点，并保留 grouped decode。
- direct RobotMind：随机起点，支持等间隔 `video_frame_step`。
- test/eval：确定性窗口。

### 完整方案中的扩展方式

后续将把窗口选择统一为显式 sampler protocol/registry。目标配置形式为：

```yaml
sampling:
  type: uniform_window
  random_start: true
  video_frame_step: 1
```

未来可以增加：

- contiguous/uniform stride；
- 多 stride 随机混合；
- 非均匀 future-dense；
- curriculum sampling；
- task/episode balanced sampling；
- 模型侧注册自定义 sampler。

无论使用哪种 sampler，它只能产生时间索引选择，视频/state/action 的实际读取
和对齐仍由 source Dataset 完成。模型端不应自行读取 MP4/HDF5 后再与数据工厂
输出拼接，否则会绕过对齐和安全检查。

当前如果需要尚未进入统一 registry 的复杂采样策略，应新增一个明确的 Dataset
或 sampler 插件并通过 source parity，不建议直接 monkey patch `__getitem__()`。

## 10. 当前实现状态

| 数据源 child | 当前数据工厂状态 | 是否完成集群真实验收 |
|---|---|---|
| Agibot | direct 原始 annotation + MP4/HDF5 | 是 |
| Agibot Beta | direct task_info + tar member physical catalog | 是 |
| RobotMind TianYi | direct HDF5 family adapter 已在本地实现 | 否，尚未同步集群 |
| RobotMind ArkMobile | schema 已实现 | 否 |
| RobotMind Ark | schema 已实现 | 否 |
| RobotMind TienKung | schema 已实现 | 否 |
| RoboTwin Aloha | 当前使用兼容 Manifest adapter | direct 待迁移 |
| RoboTwin UR5 | 当前使用兼容 Manifest adapter | direct 待迁移 |
| RoboTwin ARX | 当前使用兼容 Manifest adapter | direct 待迁移 |
| RoboTwin Piper | 当前使用兼容 Manifest adapter | direct 待迁移 |

兼容 Manifest adapter 仍可用于现有十源配置和迁移 parity，但最终目标不是继续
把所有数据源压成一个通用固定窗口 Manifest。

## 11. 十源完成后的目标形态

最终十个 child 为：

```text
Agibot
Agibot Beta
RoboTwin: Aloha / UR5 / ARX / Piper
RobotMind: TianYi / ArkMobile / Ark / TienKung
```

目标结构：

1. 每个 source family 负责自己的原始数据事实和运行时对齐。
2. 同一 family 的不同本体只提供小型 schema，不复制整套 Dataset。
3. 所有 child 输出同一个版本化 batch contract。
4. MixDataset 只做权重、child batch 选择和分布式采样，不理解源格式。
5. DataLoader 和 ReadyBatchQueue 对所有模型保持一致。
6. 模型端只通过 YAML 调整采样策略并消费 batch。
7. 兼容 Manifest loader 保留为回归/迁移工具，不作为 direct 数据源的必需链路。

## 12. 版本和兼容原则

- 在 `motus_model_batch_v1` 内可以增加不影响模型的诊断字段。
- 不能静默删除字段、改变 tensor 含义、维度顺序或取值范围。
- 如模型边界必须改变，应新增 `motus_model_batch_v2` 和明确 adapter。
- 每个新 source adapter 必须通过 fixture、真实 source audit、固定窗口 parity、
  DataLoader/queue/tiny optimizer gate。
- 只有 batch schema或模型解释变化时，才强制要求完整大模型 smoke；普通
  source adapter 迁移主要在数据边界完成验收。

## 13. 相关入口

```text
Public API:             motus_data/__init__.py
Dataset factory:        motus_data/factory.py
Batch contract:         motus_data/contracts.py
Canonical collate:      motus_data/collate.py
Weighted mixing:        motus_data/mixing/mix_dataset.py
DataLoader/pipeline:     motus_data/pipeline.py
ReadyBatchQueue:         motus_data/loading/ready_queue.py
Dataset plugin registry: motus_data/registry.py
Architecture:           docs/ARCHITECTURE.md
Batch contract detail:  docs/BATCH_CONTRACT_V1.md
```
