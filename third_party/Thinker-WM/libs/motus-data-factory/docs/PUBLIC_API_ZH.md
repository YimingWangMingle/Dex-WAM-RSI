# Motus Data Factory 极简公共 API

## 1. 面向使用者的两个入口

数据工厂的普通使用者只需要：

```python
from motus_data import open_dataset, open_training_data
```

- `open_dataset()`：查看和按索引读取完整 Episode。
- `open_training_data()`：根据模型侧 SamplePool 获得 DataLoader/queue batch。

Catalog、SQLite sidecar、source reader、数值安全规则、grouped sampler、
DataLoader、分布式 rank 和 ReadyBatchQueue 均由数据工厂内部管理。

## 2. 查看数据

```python
from motus_data import open_dataset

data = open_dataset("agibot_beta")
try:
    episode = data.get_episode("agibot_beta:327:648642")

    print(episode.info)
    print(episode.video.schema)
    print(episode.state.schema)
    print(episode.action.schema)
    print(episode.text.items)

    frames = episode.read_video([10, 20, 30])
    states = episode.read_state([10])
    actions = episode.read_action([11, 12, 13])
    texts = episode.read_text([0, 1])
finally:
    data.close()
```

`open_dataset("agibot_beta")` 自动使用当前冻结的默认 release：

```text
agibot_beta:20260723_concise_v2
```

其正式 Catalog、metadata 和 SQLite index 路径不需要由普通用户传入。

## 3. 训练取数据

当前 Motus 模型：

```python
from motus_data import (
    MotusModelBatchV1Adapter,
    open_training_data,
)

adapter = MotusModelBatchV1Adapter(
    num_video_frames=48,
    action_video_fps_ratio=1,
    action_loss_weight=0.0,
)

pipeline = open_training_data(
    "agibot_beta",
    sample_pool="/path/train.sample_pool.jsonl",
    batch_size=4,
    output_adapter=adapter,
    video_size=(480, 640),
)

with pipeline:
    for step in range(max_steps):
        batch, queue_stats = pipeline.next()

        batch["vlm_inputs"] = vlm_processor(
            batch["first_frame_pils"],
            batch["vlm_input_messages"],
        )
        batch["language_embedding"] = t5_encoder(
            batch["t5_input_messages"]
        )
        batch = move_batch_to_device(batch, device)
        loss = model_training_step(batch)
```

模型端必须提供的内容只有：

1. 数据集名称。
2. 模型侧生成的 SamplePool。
3. batch size。
4. 模型 batch adapter。
5. 模型需要的视频分辨率等少量输出策略。

## 4. 内部自动完成的工作

`open_training_data()` 内部完成：

```text
resolve frozen dataset release
-> open Catalog with SQLite lazy index
-> bind and validate SamplePool
-> build SamplePoolDataset
-> build same-Episode grouped batch sampler
-> detect torch.distributed rank/world_size
-> apply model batch adapter
-> build CPU DataLoader
-> start rank-local bounded ReadyBatchQueue
-> propagate producer errors
-> close queue and Repository
```

模型端不要再叠加 `DistributedSampler`，也不需要手工调用
`torch.distributed.get_rank()`。

## 5. 模型更新

数据工厂的稳定边界是 `IndexedEpisodeBatch`。模型输入格式变化时，只替换
adapter：

```python
pipeline = open_training_data(
    "agibot_beta",
    sample_pool=pool_path,
    batch_size=batch_size,
    output_adapter=new_model_adapter,
    video_size=(480, 640),
)
```

adapter 对象需要实现：

```python
class NewModelAdapter:
    profile_name = "new_model_batch_v1"

    def adapt(self, indexed_batch):
        return {
            "frames": ...,
            "texts": ...,
        }
```

也可以直接传入接收 `IndexedEpisodeBatch` 的 callable。

adapter 负责 CPU batch 组织，不应读取原始 tar/MP4/HDF5，也不应修改
SamplePool 索引。tokenizer、VLM/T5、GPU 搬运和模型 forward 仍在训练主线程。

## 6. 直接取得模型无关数据

不传 adapter：

```python
pipeline = open_training_data(
    "agibot_beta",
    sample_pool=pool_path,
    batch_size=4,
    output_adapter=None,
)

with pipeline:
    indexed_batch, stats = pipeline.next()
```

返回：

```text
IndexedEpisodeBatch
  info
  episode_id
  batch_size
  samples
```

每个 sample 保留原始 video/state/action/text 索引及对应解码数据。

## 7. 不使用 queue

调试时：

```python
pipeline = open_training_data(
    "agibot_beta",
    sample_pool=pool_path,
    batch_size=4,
    output_adapter=adapter,
    queue_enabled=False,
)
```

调用仍然是：

```python
batch, stats = pipeline.next()
```

## 8. 分布式行为

当 `torch.distributed` 已初始化时，数据工厂自动读取全局 rank/world size。
每个 rank 构造自己的 pipeline：

```python
pipeline = open_training_data(
    "agibot_beta",
    sample_pool=pool_path,
    batch_size=local_batch_size,
    output_adapter=adapter,
)
```

内部 grouped sampler 按完整 batch 分片，rank 间不重叠。所有 rank 应使用同一个
SamplePool 和基础 seed。

单元测试或特殊 launcher 可以显式覆盖：

```python
pipeline = open_training_data(
    "agibot_beta",
    sample_pool=pool_path,
    batch_size=4,
    output_adapter=adapter,
    rank=0,
    world_size=2,
)
```

## 9. Dataset release

普通调用只写：

```python
open_dataset("agibot_beta")
```

数据维护者负责在包内 `motus_data/releases.yaml` 中注册：

```text
dataset name
release version
Catalog relative path
SQLite index relative path
accepted video defaults
numeric safety policy
DataLoader/queue conservative defaults
```

默认 Catalog 根目录同样声明在 `releases.yaml` 的 `default_catalog_root`
字段（当前为 `/data/jiyuan.wang/EpisodeCatalogs`）。库不读取任何环境变量；
不同挂载环境应通过 API 的 `catalog_root=` 参数显式覆盖，或修改该 YAML。

发布新 Catalog 时在 `releases.yaml` 注册新 release，并在验收后切换默认
release。不要静默覆盖旧 Catalog 文件。

## 10. 自定义数据源插件

第三方无需修改库代码即可接入新数据集：实现 `EpisodeSourceReader`
子类，通过 `register_episode_source_reader()` 按 Catalog `source` 名
注册一次，之后 `open_dataset()` / `open_training_data()` /
`open_mixed_training_data()` 都会自动解析该 reader：

```python
from motus_data import (
    EpisodeDatasetRelease,
    open_dataset,
    register_episode_dataset_release,
    register_episode_source_reader,
)

register_episode_source_reader("my_source", MySourceReader)
register_episode_dataset_release(
    EpisodeDatasetRelease(dataset="my_dataset", release="v1", ...),
    default=True,
)
data = open_dataset("my_dataset")
```

reader 解析优先级：调用方 `readers=` 映射 > 全局注册表（含内置）。
完整插件契约、Catalog 构建要求与维护者收编清单见
`docs/ADDING_DATASETS_ZH.md`；可运行模板见
`examples/custom_episode_source.py`。

## 11. 自定义 SamplePool 注入点

`sample_pool` 参数除静态文件路径外，还接受用户自定义的 pool 对象或工厂，
用于运行时动态生成 selection 索引（索引选择策略归模型/训练代码所有；库只
负责统一格式、加载、打包、传输）：

```python
pipeline = open_training_data(
    "agibot_beta",
    sample_pool=my_pool,          # SupportsSamplePool 实例
    # sample_pool=my_factory,     # 或 callable: f(*, repository, seed) -> pool
    batch_size=4,
    output_adapter=adapter,
)
```

自定义 pool 必须实现 `motus_data.SupportsSamplePool` 协议：

```text
selections      按整数位置索引的 selection 序列（motus_sample_selection_v1 契约）
episode_groups  episode_id -> 该 episode 的 selection 位置元组（batch 内单 episode）
pool_id         诊断标识
__len__         selection 总数（缺省源权重）
set_epoch(epoch)  可选；实现后 sampler 链每个 epoch 翻转都会转发，
                  动态 pool 可据此刷新 selection
```

要求：同一位置在所有 rank/worker 上必须生成相同的 selection（用
`motus_data.sampling.compose_seed` 之类与 rank 无关的确定性种子）；越界与
格式校验仍在读取与 adapter 阶段强制执行。工厂形式在 Repository 打开之后
调用，可读取任意 Catalog 记录来构建内存索引结构，并与管线 seed 对齐。

当前数据工厂建议并由内置测试覆盖的动态 pool 使用固定采样拓扑：
`__len__`、每个 selection 位置所属的 `episode_id` 和 `episode_groups`
跨 epoch 保持不变，`set_epoch()` 只更新同一 Episode 内的
video/state/action/text 索引。若未来需要跨 epoch 更换 Episode 或重建分组，
应先升级 sampler/分布式协议，不能直接复用当前固定拓扑语义。

## 12. 高级覆盖

以下参数用于测试、迁移或性能实验，普通模型接入不需要：

```text
release
catalog_root
catalog_path/index_path
verify_sha256
record_cache_size
num_workers/pin_memory
queue_size/queue_timeout_seconds
rank/world_size
readers
```

更改这些参数不能绕过 Catalog/Pool 绑定、索引越界、finite 和 magnitude 安全
检查。

## 13. 当前边界

已经完成：

- 18 个内置 release 的完整 Episode 访问和 SQLite 懒加载；
- 静态 JSONL 与动态 `SupportsSamplePool` 两种 indexed SamplePool；
- grouped batch、分布式 sampler 和同 Episode 联合物理解码；
- 单源 `open_training_data()` 与多源 `open_mixed_training_data()`；
- child 权重、确定性 source quota/schedule 和 rank 一致分片；
- 可替换模型 adapter、CPU DataLoader 和 ReadyBatchQueue；
- RobotMind Agilex、OpenVid 和首批六个 EgoVerse child 的追加式 reader。

旧十源真实回归以及 18-child 真实 Catalog/SamplePool/reader/DataLoader/queue
门禁均已完成。18-child 门禁实际消费 72/72 batch，所有 child 均来自不同
Episode；当前合并态单元测试为 90/90。

尚未完成：

- 三个新增数据的模型侧 production train/val SamplePool 和最终 family 权重；
- OpenVid absent state/action 配置向 ThinkerWM bridge 的完整透传；
- `motus311` 下实际模型单卡、单节点八卡和三节点 IB/NCCL 训练门禁；
- 支持动态 pool 跨 epoch 改变 Episode 归属或 `episode_groups`。

这些扩展不会改变普通模型端的两个入口。
