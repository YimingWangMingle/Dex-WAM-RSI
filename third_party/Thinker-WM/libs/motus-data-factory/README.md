# Motus Data Factory

模型无关的多模态机器人数据加载、混合与 ready-batch queue 标准 Python 库。
详细针对数据工厂的两层能力进行说明：

1. **数据访问层**：统一查看完整 Episode，以及按显式索引读取 video、state、
   action、text。
2. **训练数据层**：执行模型侧提供的 SamplePool，完成同 Episode 联合解码、
   batch、分布式分片、DataLoader、CPU ReadyBatchQueue，并通过可替换 adapter
   输出模型需要的 batch。

> Agibot Beta 平行夹爪/灵巧手分型、原生 14D/24D state/action、独立归一化及
> 模型端迁移要求，参见
> [Agibot Beta 末端执行器分型更新说明](README_DEV_20260804_AGIBOT_BETA_EFFECTOR_PROFILES_ZH.md)。

作为标准库安装后即可在任意工程中 `import motus_data`，无需设置 PYTHONPATH
等环境变量。安装方式见第 1 节。

## 近期版本演进

当前发布主线关系：

```text
dev/20260728 基线 + dev/20260729 数据扩展
  -> release.20260807
  -> dev/20260804-agibot-beta-effector-profiles
  -> dev/20260809-datafactory-updates
  -> release.20260814
  -> dev/20260816-datafactory-updates
  -> dev/20260818-datafactory-updates
  -> main@6210dcf
  -> dev/20260820-egoscale-second-1700h-batch01
  -> dev/20260821-egoscale-second-1700h-batch02
```

| 版本 | 状态 | 更新概括 |
|---|---|---|
| `dev/20260728` | 已验证基线，后续汇入 `release.20260807` | 完成数据工厂标准 Python 库化和模型解耦，建立 YAML release 注册、Episode reader 插件、静态/动态 SamplePool、分布式 Dataset/DataLoader/ReadyBatchQueue 和可替换 batch adapter；旧 Episode-native 直读链路进入兼容退役阶段。 |
| `dev/20260729` | 已合入 `release.20260807` | 一次性完成 RobotMind Agilex、OpenVid 和 EgoVerse 六个 child 的 Catalog、reader、索引、质量门禁及混训接入，并构建 Python 3.11/Zarr v3 训练镜像和 18-child 回归链路。详见 [版本说明](README_DEV_20260729_UPDATE_ZH.md)。 |
| `dev/20260804-agibot-beta-effector-profiles` | 已发布独立分支 | 改进 Agibot Beta 夹爪/灵巧手混用导致的维度和归一化方式：全量审计 167,794 个 Episode、约 2.96 亿时间步，拆分为互斥的 14D/24D child，分别生成 Catalog、归一化统计、异常 Episode 门禁及双 child queue/model-batch 闭环，同时保留旧 26D release 兼容入口。详见 [版本说明](README_DEV_20260804_AGIBOT_BETA_EFFECTOR_PROFILES_ZH.md)。 |
| `dev/20260809-datafactory-updates` | 已汇入 `release.20260814` | 完成 LIBERO 四个 suite 的全链路接入：解析 1,693 个 Parquet Episode、273,465 个时间步和双相机图像，提供 raw/z-score、三种相机模式及原生 state 8D/action 7D adapter；Catalog、SamplePool、四 child 混训、96x128/480p queue 长测和 106 项全库回归全部通过。主体提交涉及约 18 个文件、4,000 行新增。详见 [版本说明](README_DEV_20260810_LIBERO_INTEGRATION_ZH.md)。 |
| `dev/20260816-datafactory-updates` | 已作为当前分支基线 | 新增 EGOSCALE FIRST-500H 分片式接入：全量盘点 48,691 个 LeRobot-v3 Episode，按原始任务组稳定哈希发布 4,848 Episode 首片和 43,843 Episode 剩余合并片；支持左目/右目/双目拼接、相对或世界绝对末端位姿、逐手缺失有效位、完整 Parquet 原生字段、任务/细粒度文本，以及 SamplePool、batch、DataLoader 和 ReadyBatchQueue 全链路。详见 [接入说明](docs/EGOSCALE_FIRST500H_INTEGRATION_ZH.md)。 |
| `dev/20260818-datafactory-updates` | 已汇入 `main@6210dcf` | 新增按训练权重在单个 data step 的全部 rank 间分配数据源的 `weighted_per_step` 策略，同时保留原 `synchronized` 兼容策略；实现确定性跨 rank 调度、跨 step 小数配额补偿和无 collective 本地重建，不改变 SamplePool、Catalog、DataLoader、ReadyBatchQueue 及 batch 协议。完成多 rank 分片互斥、权重精度、epoch 可复现性。详见 [训练接入说明](docs/CROSS_RANK_SOURCE_SAMPLING_ZH.md)。 |
| `dev/20260820-egoscale-second-1700h-batch01` | 已发布独立分支 | 基于训练负责人最新 `main@6210dcf` 接入 EGOSCALE SECOND-1700H batch_01：全量审计并发布 100,448 Episodes/58,756 source groups、1.266 亿帧的完整不可变 Catalog/SQLite，与 FIRST-500H 零 ID 重叠；正式 release、单源 queue 及新旧 EGOSCALE 混训闭环通过。详见 [接入与训练事项](docs/EGOSCALE_SECOND1700H_BATCH01_INTEGRATION_ZH.md)。 |
| `dev/20260821-egoscale-second-1700h-batch02` | 数据工厂验收完成 | 在 batch_01 最新版本上增量接入 EGOSCALE SECOND-1700H batch_02：全量审计并发布 48,241 Episodes/27,863 source groups、6,123 万帧的完整 Catalog/SQLite，所有必需文件和 31 字段通过，与已有 149,139 个 EGOSCALE Episodes 零重叠；内置 release、真实 queue、batch_01/batch_02 混训及 134 项全库回归通过，既有 reader、batch 和 queue 协议不变。详见 [接入与训练事项](docs/EGOSCALE_SECOND1700H_BATCH02_INTEGRATION_ZH.md)。 |

各版本均遵循增量兼容原则：新增数据通过 reader、release 和 adapter 注册接入，
不改变既有 Catalog/SamplePool、分布式分片、DataLoader 或 queue 的使用方法。

## 1. 安装与导入

数据工厂是标准 Python package，支持两种使用方式，任选其一。

### 1.1 方式一：直接拷贝源码（无需 pip 安装）

将本仓库的 `motus_data/` 目录整体复制到你的项目目录下，即可直接
`import motus_data`，不需要 pip 安装，也不需要设置 PYTHONPATH：

```text
your_project/
├── motus_data/       # 从本仓库整体拷贝
└── train.py
```

```python
from motus_data import open_dataset, open_training_data
```

此方式下需手动安装 Python 依赖（见 1.3）。

### 1.2 方式二：pip 安装为标准库

在目标环境中安装：

```bash
# 直接使用
pip install .

# 开发模式（改动代码即时生效）
pip install -e .
```

必选依赖（PyTorch、OmegaConf、NumPy、Pillow）自动安装。机器人数据源相关
功能需要可选依赖：

```bash
pip install -e ".[robot]"   # h5py、scipy、opencv-python、decord
pip install -e ".[egoverse]"  # Python >=3.11、zarr 3.1.5
pip install -e ".[libero]"  # pyarrow，读取 LIBERO LeRobot v2 Parquet
pip install -e ".[egoscale]"  # pyarrow，读取 EGOSCALE LeRobot v3 Parquet
```

安装后无需设置 PYTHONPATH 或任何环境变量，在任意目录即可导入。

### 1.3 依赖说明与导入验证

两种方式所需的 Python 依赖相同：

- 必选：PyTorch、OmegaConf、NumPy、Pillow
- 机器人数据源相关功能：h5py、scipy、opencv-python、decord
- EgoVerse Zarr v3：Python >=3.11、zarr 3.1.5
- LIBERO LeRobot v2 Parquet：pyarrow >=14
- EGOSCALE LeRobot v3 Parquet：pyarrow >=14；视频解码需要系统 `ffmpeg`

直接拷贝方式下需手动安装：

```bash
pip install torch omegaconf numpy pillow
pip install h5py scipy opencv-python decord
# 仅在 Python >=3.11 环境安装 EgoVerse reader 依赖
pip install zarr==3.1.5
pip install "pyarrow>=14"
```

验证导入（pip 安装方式可在任意目录运行；直接拷贝方式请在 `motus_data/`
所在的项目目录中运行）：

```bash
python - <<'PY'
from motus_data import episode_dataset_releases

print("registered releases:")
for release_id in episode_dataset_releases():
    print(" ", release_id)
PY
```

运行单元测试（需在本仓库根目录）：

```bash
python -m pytest
```

所有运行期 API（`open_dataset`、`open_training_data` 等）都接受显式的
`catalog_root=` 参数。不显式传入时使用内置默认 Catalog 根目录；默认值和
已注册的 release 列表都声明在包内的 `motus_data/releases.yaml` 中（随包
分发，pip 安装和直接拷贝目录都会带上），需要修改默认值时直接编辑该文件，
不依赖任何环境变量。

当前默认 Catalog 根目录为：

```text
/data/jiyuan.wang/EpisodeCatalogs
```

代码发布到其他环境后，原始数据、Catalog 和 SamplePool 路径必须按实际挂载修改。

## 2. 三大核心功能

Motus Data Factory 提供三个独立且可组合的功能模块。

### 为什么这样设计

三个功能对应机器人数据从「物理存储」到「模型 batch」的三个阶段，每一层解决
一类问题，层与层之间用显式契约（Catalog、SamplePool、batch contract）连接。

**第一层把物理格式抽象成统一 Episode 视图。** 机器人数据源的物理形态高度
异构：文件 MP4、tar 内 MP4/HDF5、HDF5 内 JPEG，且各源的字段顺序、维度和
时间轴比例都不同。如果每个模型、每次数据分析都直接面对物理格式，同一套解析
逻辑会在各处重复实现并悄悄分叉。因此第一层把物理细节固化在 source reader
中，对外只暴露「完整 Episode + 显式索引读取」的统一逻辑视图——训练、审计、
调试、可视化看到的都是同一个事实来源。

**第二层把「取哪些帧」从数据管线中剥离为独立的 SamplePool 声明文件。**
采样策略本质上是训练策略：窗口怎么取、video/state/action/text 怎么对齐，只有
模型侧能决定。但它也不应该硬编码进某个模型的 Dataset 类——那样既无法审计、
无法复现，也无法跨模型复用。SamplePool 用纯文本 JSONL 把每条样本的四组显式
索引固化下来：可以独立保存、版本管理和审计；数据工厂只校验和执行，索引越界
直接报错而不是静默修复，策略的正确性始终由模型侧显式负责。

**第三层把「怎么高效取出来」收敛为通用数据流。** 确定取什么之后，同 Episode
多 selection 联合解码（同一帧只解码一次）、组 batch、按 rank/world_size 分片、
后台预取队列，是所有模型都需要的通用工程能力，但每个训练项目各自实现都容易
踩坑（最典型的就是再叠加 DistributedSampler 或交给 Accelerate 造成二次分片）。
这层在库中一次做对，模型侧只通过一个可替换的 adapter 声明自己的 batch 协议。

三层可以独立使用（只查数据、只生成 SamplePool、直接接训练），也可以串成完整
链路：**Catalog 是事实，SamplePool 是策略，DataLoader 是执行**。这正是第 4 节
责任边界的来源——数据工厂负责事实与执行，模型侧负责策略。

### 功能一：灵活查看 Episode

无需了解底层 tar/MP4/HDF5 格式，直接按 `episode_id` 查看任何数据集的完整内容。

```python
from motus_data import open_dataset, episode_dataset_releases

# 列出所有已注册的数据集
for release_id in episode_dataset_releases():
    print(release_id)
# agibot:20260727_full_v1
# agibot_beta:20260723_concise_v2
# robotmind_ark:20260727_full_v1
# robotmind_tianyi:20260727_full_v1
# ... 共 11 个数据集

# 遍历某个数据集的每个 episode
data = open_dataset("agibot_beta")
try:
    print(f"episodes: {len(data)}")

    for episode_id in data.iter_episode_ids():
        ep = data.get_episode(episode_id)
        info = ep.info
        print(f"  {episode_id}: "
              f"video={info.video_count} "
              f"state={info.state_count} "
              f"action={info.action_count} "
              f"robot={info.robot_name}")

        # 查看文本
        for item in ep.text.items:
            print(f"    text[{item.index}] {item.kind}: {item.text[:80]}...")

        # 按帧索引直接解码
        frames = ep.read_video([0, 10, 20])
        states = ep.read_state([0])
        actions = ep.read_action([1, 2, 3])
        print(f"    frames={frames.shape}, state={states.shape}")
finally:
    data.close()
```

支持的方法一览：

| 方法 | 说明 |
|---|---|
| `iter_episode_ids()` | 迭代所有 episode_id |
| `get_episode(id)` | 按 episode_id 获取完整 EpisodeView |
| `ep.video.read(indices)` | 按帧索引解码视频 |
| `ep.state.read(indices)` | 按行索引读取 state |
| `ep.action.read(indices)` | 按行索引读取 action |
| `ep.text.read(indices)` | 按索引读取文本条目 |
| `ep.read_selection(dict)` | 一次按四组索引读取全模态数据 |

### 功能二：可自定义的预处理（SamplePool JSONL）

SamplePool 是你控制「从每个 episode 取哪些帧」的声明文件。它是一个纯文本 JSONL，不绑定任何模型架构。

**文件结构：**

第一行 header，后续每行一条 selection：

```jsonl
{"format":"motus_sample_pool_v2","pool_id":"my_pool","catalog_id":"agibot_beta_20260723_concise_v2","samples":3}
{"episode_id":"agibot_beta:327:648642","video_frame_indices":[0,10,20,30,40],"state_indices":[0],"action_indices":[1,2,3,4,5],"text_indices":[0,1]}
{"episode_id":"agibot_beta:327:648642","video_frame_indices":[50,60,70,80,90],"state_indices":[50],"action_indices":[51,52,53,54,55],"text_indices":[0,2]}
{"episode_id":"agibot_beta:328:123456","video_frame_indices":[0,16,32,48,64],"state_indices":[0],"action_indices":[1,2,3,4,5],"text_indices":[0]}
```

每条 selection 指定四组显式索引：

| 字段 | 含义 |
|---|---|
| `episode_id` | 目标 episode |
| `video_frame_indices` | 要解码的视频帧编号 |
| `state_indices` | 要读取的 state 行号 |
| `action_indices` | 要读取的 action 行号 |
| `text_indices` | 要读取的文本条目编号 |

**用脚本自动生成：**

```python
import json
from motus_data import open_dataset
from motus_data.index_contracts import MOTUS_SAMPLE_POOL_HEADER_V2

data = open_dataset("agibot_beta")
selections = []

for episode_id in data.iter_episode_ids():
    ep = data.get_episode(episode_id)
    # 你的预处理策略：例如每 16 帧取一个 49 帧窗口
    for start in range(0, ep.info.video_count - 48, 16):
        selections.append({
            "episode_id": episode_id,
            "video_frame_indices": list(range(start, start + 49)),
            "state_indices": [start],
            "action_indices": list(range(start + 1, start + 49)),
            "text_indices": [0, 1],
        })

header = {
    "format": MOTUS_SAMPLE_POOL_HEADER_V2,
    "pool_id": "my_custom_pool",
    "catalog_id": data.catalog_id,
    "samples": len(selections),
}

with open("my_pool.sample_pool.jsonl", "w") as f:
    f.write(json.dumps(header, ensure_ascii=False) + "\n")
    for sel in selections:
        f.write(json.dumps(sel, ensure_ascii=False) + "\n")

data.close()
print(f"done: {len(selections)} selections")
```

你的 SamplePool 可以独立保存、版本管理和复用，与模型代码解耦。

### 功能三：便捷获取数据流（DataLoader）

把 SamplePool 变成可训练的数据流，内部自动完成索引校验、同 episode 联合解码、分组 batch、分布式分片和后台预取。

**自定义下游处理：** 返回原始 `IndexedEpisodeBatch`，每条 sample 保留完整解码数据，你可以完全自主决定后续处理逻辑。

```python
from motus_data import open_training_data

pipeline = open_training_data(
    "agibot_beta",
    sample_pool="my_pool.sample_pool.jsonl",
    batch_size=4,
    output_adapter=None,        # 不传 adapter，拿到原始数据
    video_size=(480, 640),
)

with pipeline:
    indexed_batch, stats = pipeline.next()
    for sample in indexed_batch.samples:
        print(sample.info.episode_id, sample.video.shape, sample.state.shape)
        # 完全自主处理：拼接、归一化、tokenize 等
```

**可选：使用内置 adapter 直接输出模型 batch：** 传入 `MotusModelBatchV1Adapter`（或其他 adapter callable），pipeline 自动将原始数据转换为模型直接消费的 dict。

`MotusModelBatchV1Adapter` 约定每条 selection 的 video 为 `num_video_frames+1` 个连续帧（第 0 帧作条件帧，后 `num_video_frames` 帧作预测目标），action 为 `num_video_frames × action_video_fps_ratio` 个连续动作，state 恰好 1 个。你需要确保 SamplePool 中的索引满足这些长度要求。

```python
from motus_data import open_training_data, MotusModelBatchV1Adapter

adapter = MotusModelBatchV1Adapter(
    num_video_frames=48,
    action_video_fps_ratio=1,
)

pipeline = open_training_data(
    "agibot_beta",
    sample_pool="my_pool.sample_pool.jsonl",
    batch_size=4,
    output_adapter=adapter,
    video_size=(480, 640),
)

with pipeline:
    for step in range(max_steps):
        batch, queue_stats = pipeline.next()
        # batch 已是模型可直接消费的 dict：
        #   first_frame, video_frames, initial_state, action_sequence,
        #   t5_input_messages, vlm_input_messages, robot_names, ...
        print(f"step {step}: "
              f"first_frame={batch['first_frame'].shape}, "
              f"video_frames={batch['video_frames'].shape}")
```

**分布式支持：** pipeline 自动获取 rank 和 world_size，不需要手动传入或配置 `DistributedSampler`。

- 当 `torchrun` / `torch.distributed` 已初始化时，自动读取 `dist.get_rank()` 和 `dist.get_world_size()`
- 单机单卡或未初始化时，默认 `rank=0, world_size=1`
- 也可以显式覆盖：

```python
pipeline = open_training_data(
    "agibot_beta",
    sample_pool="my_pool.sample_pool.jsonl",
    batch_size=4,
    output_adapter=None,
    rank=2,          # 显式指定 rank
    world_size=8,    # 显式指定 world_size
)
```

**多节点行为：** `dist.get_rank()` 返回的是全局 rank（跨所有节点的全局编号，例如 4 节点 × 8 GPU = world_size 32，rank 范围 0–31）。`EpisodeGroupedBatchSampler` 按全局 rank 对完整 batch 做分片，无论进程在哪个节点上，每个 rank 拿到互不重叠的 selection。

- 不需要区分节点内 rank（local rank）和全局 rank（global rank），数据工厂只关心全局 rank
- 跨节点自动生效：每个节点上的进程独立调用 `open_training_data()`，使用同一个 SamplePool 和相同的 `seed` 即可
- **不要在模型侧再叠加 `DistributedSampler` 或交给 Accelerate 做分片**，否则会造成样本遗漏或 rank 间步数不一致

**多数据集混合训练：** 用 `open_mixed_training_data()` 按权重组合多个 SamplePool，每个 source 可以来自不同数据集、使用不同的预处理策略：

```python
from motus_data import (
    open_mixed_training_data,
    IndexedTrainingSource,
)

pipeline = open_mixed_training_data(
    sources=[
        IndexedTrainingSource(
            name="agibot_beta",
            dataset="agibot_beta",
            sample_pool="pools/agibot_beta_train.sample_pool.jsonl",
            weight=0.6,                         # 占 60% 的 batch
        ),
        IndexedTrainingSource(
            name="tianyi",
            dataset="robotmind_tianyi",
            sample_pool="pools/tianyi_train.sample_pool.jsonl",
            weight=0.3,                         # 占 30% 的 batch
        ),
        IndexedTrainingSource(
            name="ark",
            dataset="robotmind_ark",
            sample_pool="pools/ark_train.sample_pool.jsonl",
            weight=0.1,                         # 占 10% 的 batch
        ),
    ],
    batch_size=4,
    batches_per_epoch=1000,                     # 每 epoch 总共 1000 个 batch
    output_adapter=None,                        # 同样支持不传或传入 adapter
    video_size=(480, 640),
)

with pipeline:
    for step in range(max_steps):
        indexed_batch, stats = pipeline.next()
        # stats["mix_source_name"]   → 当前 batch 来自哪个 source
        # stats["mix_source_family"] → 当前 batch 的 family
        print(f"step {step}: source={stats['mix_source_name']}")
```

也可以不写 `IndexedTrainingSource`，直接传 dict：

```python
pipeline = open_mixed_training_data(
    sources=[
        {
            "name": "agibot_beta",
            "dataset": "agibot_beta",
            "sample_pool": "pools/agibot_beta_train.sample_pool.jsonl",
            "weight": 0.6,
        },
        {
            "name": "tianyi",
            "dataset": "robotmind_tianyi",
            "sample_pool": "pools/tianyi_train.sample_pool.jsonl",
            "weight": 0.4,
        },
    ],
    batch_size=4,
    batches_per_epoch=1000,
)
```

每个 epoch 内部按权重分配各 source 的 batch 数量。同一个 rank 的 batch 始终
来自单一 source，避免 batch 内跨数据集维度冲突。跨 rank 的 source 调度提供两种
策略：

- `synchronized`：默认兼容模式；所有 rank 在同一个 data step 使用相同 source。
- `weighted_per_step`：把一个 data step 的全部 rank 视为全局 rank-slot，按 source
  权重分配；每个 source 在单步获得 `floor(weight * world_size)` 或
  `ceil(...)` 个 rank，小数余量跨 step 累积补偿。

```python
pipeline = open_mixed_training_data(
    sources,
    batch_size=4,
    batches_per_epoch=1000,
    seed=20260818,
    rank=global_rank,
    world_size=world_size,
    cross_rank_source_policy="weighted_per_step",
)
```

`weighted_per_step` 不需要 distributed collective：每个 rank 使用相同的
`seed/epoch/weights/world_size` 确定性重建全局当步分配，再只读取自己的 source。
小权重 source 不保证每一步出现，但累计 `rank x data_step` 占比与配置权重的误差
被限制在整数 rank-slot 精度内。SamplePool、Catalog、DataLoader、ready queue 和
batch 协议均不改变。详细接入与验证说明见
[`docs/CROSS_RANK_SOURCE_SAMPLING_ZH.md`](docs/CROSS_RANK_SOURCE_SAMPLING_ZH.md)。

## 3. 仓库内容

本仓库即 `motus-data-factory` 标准 Python 库（import 名 `motus_data`）。
它是模型无关的：不导入 WAN、T5、VLM、DeepSpeed、Accelerate 或任何特定训练模型。

主要目录：

```text
motus_data/
  catalog/                  # Episode Catalog 构建和 SQLite offset index
  datasets/                 # 数据源解析与历史兼容 Dataset
  loading/ready_queue.py    # CPU ReadyBatchQueue
  repository.py             # 完整 Episode 统一访问
  source_readers.py         # Agibot/RoboTwin/RobotMind 物理读取
  sample_pool.py            # SamplePool 校验、Dataset、grouped sampler
  batch_adapters.py         # 模型 batch 协议适配
  indexed_pipeline.py       # 单源 indexed 训练流水线
  indexed_mix.py            # 多源 weighted indexed 训练流水线
  releases.py               # 已冻结 Catalog release（默认值见 releases.yaml）
  releases.yaml             # 内置默认 Catalog 根目录和 release 注册表
  contracts.py              # batch contract 校验

tools/                      # Catalog、真实解码和十源审计工具
docs/                       # 协议和专项文档
tests/                      # 单元与回归测试
examples/                   # 审计、对比与 fixture 生成示例
```

## 4. 责任边界

### 4.1 数据工厂负责

- 解析原始 MP4、tar、HDF5、PT 和 JPEG 数据。
- 固化每个数据源真实字段、顺序、维度和视觉预处理。
- 发布完整 Episode Catalog 和低内存 SQLite offset index。
- 提供完整 Episode 的懒加载访问。
- 校验并执行 SamplePool 中四组显式索引。
- 做数值范围、越界、非有限值和 source schema 检查。
- 对同一 Episode 的多个 selection 做 union read，复用物理解码。
- 组 batch、按 rank/world size 分片、管理 epoch。
- 提供可选的 CPU ReadyBatchQueue。
- 通过 adapter 转换成模型需要的 batch 协议。

### 4.2 模型训练端负责

- 决定训练、验证和展示分别使用哪些 selection。
- 生成 production SamplePool。
- 保证 SamplePool 中 video/state/action/text 索引具有训练策略所需的语义对齐。
- 决定 child 采样权重、batch size、分辨率、训练步数和 queue 配置。
- 决定训练使用哪些模态、冻结层、loss 权重、resume/finetune。
- 负责 H2D、forward、loss、backward、optimizer、scheduler 和 checkpoint。

数据工厂不会修改模型侧给出的selection。索引越界或格式错误会直接失败，不会静默修复。

## 5. 当前支持的数据源

Catalog 是 split-neutral 的完整 Episode 枚举。训练集和验证集成员由不同
SamplePool 决定。

| child | family | release | Episode 数 | state/action 维度 | 当前时间轴比例 | 接入状态 |
|---|---|---:|---:|---:|---:|---|
| `agibot` | `agibot` | `20260727_full_v1` | 18,081 | 26 | 6 | `dev/20260728` 已支持 |
| `agibot_beta` | `agibot_beta` | `20260723_concise_v2` | 167,794 | 26 | 1 | `dev/20260728` 已支持 |
| `robotwin_aloha` | `robotwin` | `20260727_full_v1` | 27,244 | 14 | 1 | `dev/20260728` 已支持 |
| `robotwin_ur5` | `robotwin` | `20260727_full_v1` | 24,824 | 14 | 1 | `dev/20260728` 已支持 |
| `robotwin_arx` | `robotwin` | `20260727_full_v1` | 26,909 | 14 | 1 | `dev/20260728` 已支持 |
| `robotwin_piper` | `robotwin` | `20260727_full_v1` | 20,350 | 14 | 1 | `dev/20260728` 已支持 |
| `robotmind_tianyi` | `robotmind` | `20260727_full_v1` | 7,145 | 16 | 1 | `dev/20260728` 已支持 |
| `robotmind_arkmobile` | `robotmind` | `20260727_full_v1` | 4,295 | 20 | 1 | `dev/20260728` 已支持 |
| `robotmind_ark` | `robotmind` | `20260727_full_v1` | 14,857 | 14 | 1 | `dev/20260728` 已支持 |
| `robotmind_tienkung` | `robotmind` | `20260727_full_v1` | 10,945 | 16 | 1 | `dev/20260728` 已支持 |
| `robotmind_agilex` | `robotmind` | `20260728_full_v1` | 22,206 | 14 | 1 | `dev/20260729` 新增 |
| `openvid` | `openvid` | `20260730_full_v1` | 1,019,957 | absent | 1 | `dev/20260729` 新增 |
| `libero_10` | `libero` | `20260810_full_v1` | 379 | state 8 / action 7 | 1 | 当前分支新增 |
| `libero_goal` | `libero` | `20260810_full_v1` | 428 | state 8 / action 7 | 1 | 当前分支新增 |
| `libero_object` | `libero` | `20260810_full_v1` | 454 | state 8 / action 7 | 1 | 当前分支新增 |
| `libero_spatial` | `libero` | `20260810_full_v1` | 432 | state 8 / action 7 | 1 | 当前分支新增 |
| `egoverse_aria_bimanual` | `egoverse` | `20260730_first6_v1` | 1,845 | 14 | 1 | `dev/20260729` 新增 |
| `egoverse_aria_left_arm` | `egoverse` | `20260730_first6_v1` | 333 | 7 | 1 | `dev/20260729` 新增 |
| `egoverse_aria_right_arm` | `egoverse` | `20260730_first6_v1` | 315 | 7 | 1 | `dev/20260729` 新增 |
| `egoverse_eva_bimanual` | `egoverse` | `20260730_first6_v1` | 2,876 | 16 | 1 | `dev/20260729` 新增 |
| `egoverse_scale_bimanual` | `egoverse` | `20260730_first6_v1` | 17,037 | 14 | 1 | `dev/20260729` 新增 |
| `egoverse_mecka_freeform_bimanual` | `egoverse` | `20260730_first6_v1` | 50,266 | 14 | 1 | `dev/20260729` 新增 |
| `egoscale_first500h_shard00` | `egoscale` | `20260816_shards_v1` | 4,848 | 24 | 1 | 当前分支首批分片 |
| `egoscale_first500h_remaining90pct` | `egoscale` | `20260816_shards_v1` | 43,843 | 24 | 1 | 当前分支剩余合并分片 |
| `egoscale_second1700h_batch01` | `egoscale` | `20260820_full_v1` | 100,448 | 24 | 1 | 单一完整 Catalog 已验收 |
| `egoscale_second1700h_batch02` | `egoscale` | `20260821_full_v1` | 48,241 | 24 | 1 | 单一完整 Catalog 已验收 |

当前表格列出的 26 个非重叠基础 child 总计 1,636,352 个 Episode。另有
Agibot Beta 平行夹爪/灵巧手两个替代 release，因与旧 Agibot Beta Catalog 重叠，
不重复计入总数。既有 18-child 门禁与 LIBERO 四-child 独立门禁均已通过；这些
只验收数据链路，不代表模型侧 production SamplePool 或训练采样权重已经确定。

补充说明：

- Agibot 使用文件 MP4 和 HDF5，视频与 proprio/action 是独立索引空间。
- Agibot Beta 使用 tar 内 MP4/HDF5，Catalog 记录完整 Episode，不记录训练窗口。
- RoboTwin 原始 MP4 每帧已经是三个 `320x120` 视角纵向组成的 `320x360`
  复合图。Reader 把它作为一个视觉输入，并按旧训练逻辑等比例缩放、居中补零。
- RobotMind 使用 HDF5 内 JPEG 和 schema 固化的数值字段；TianYi、ArkMobile、
  Ark、TienKung、Agilex 共用 reader，但字段顺序、主视角和维度分别冻结。
- Agilex 使用 `camera_front` 作为主视角，state/action 均按
  `left_arm,left_gripper,right_arm,right_gripper` 顺序读取 14 维 puppet
  时间轴；其 Catalog 直接从完整原始 HDF5 Episode 构建，不依赖旧 Manifest。
- RobotMind Catalog 同时保留中英文文本，当前训练 SamplePool 选择英文项。
- OpenVid 只有视频和英文 caption；Motus adapter 在兼容层提供 26 维全零
  state/action 占位，显式标记 `state_action_policy=absent`，action loss 必须为 0。
- LIBERO 使用每 Episode 一个 LeRobot v2 Parquet，默认读取主相机；reader 可切换
  腕部相机或双视角水平拼接。state/action 保留原生 8D/7D，reader 默认使用
  Catalog 绑定的官方统计量做 z-score，也支持显式切换为 raw。推荐训练 adapter
  `motus_native_multimodal_batch_v1` 保留独立的 state/action 维度，不截断、不补零。
- EgoVerse 首批六个 Zarr child 已注册 production release；其 profile、原始
  state/action 语义、quarantine 质量边界和构建门禁见
  `docs/EGOVERSE_ZARR_INTEGRATION_ZH.md`。
- EGOSCALE FIRST-500H 每个包是一个完整 LeRobot-v3 Episode，包含两路 1080p
  MP4、逐帧 head/双手 pose、tracks、world pose、MANO 与英文分层文本。默认
  左目和 24D 世界绝对 pose+validity state/action，也可无重建切换到右目、双目
  拼接或相对末端位姿。缺手 pose 补零且显式保留 validity；action 是人体运动观测
  代理，不是机器人控制命令。本批以 4,848 Episode 首片和 43,843 Episode
  剩余合并片发布；后续每个独立到货批次默认发布一个完整不可变 Catalog release，
  不覆盖任何已发布产物。SECOND-1700H batch_01 和 batch_02 已按该规则分别发布。
- 面向已接入 `dev/20260728` 的协作者，本次新增内容、模型接入限制、
  `motus311` 镜像差异和后续门禁汇总在
  `README_DEV_20260729_UPDATE_ZH.md`。

## 6. 第一层：访问和分析完整 Episode

普通数据使用者优先使用：

```python
from motus_data import open_dataset
```

`open_dataset()` 返回 `EpisodeRepository`。它通过 SQLite sidecar 按
`episode_id` 懒加载 Catalog 行，不会在每个训练 rank 中把超大 Catalog 全量展开
成 Python 对象。

### 6.1 打开数据源

```python
from motus_data import open_dataset

repository = open_dataset(
    "agibot_beta",
    release="20260723_concise_v2",
    catalog_root="/data/jiyuan.wang/EpisodeCatalogs",
    use_index=True,
)

try:
    print("catalog_id:", repository.catalog_id)
    print("episode_count:", len(repository))
    print("lazy_indexed:", repository.indexed)
finally:
    repository.close()
```

省略 `release` 时使用该 child 注册的默认 release：

```python
repository = open_dataset("robotmind_tianyi")
```

### 6.2 枚举和过滤 Episode 元信息

只枚举 ID，不解码视频：

```python
from itertools import islice
from motus_data import open_dataset

repository = open_dataset("robotwin_aloha")
try:
    episode_ids = list(islice(repository.iter_episode_ids(), 10))
    print(episode_ids)
finally:
    repository.close()
```

查看统一 Episode 信息：

```python
from itertools import islice
from motus_data import open_dataset

repository = open_dataset("robotmind_arkmobile")
try:
    for info in islice(repository.iter_episode_infos(), 5):
        print(
            info.episode_id,
            info.source_dataset,
            info.source_child,
            info.robot_name,
            info.video_count,
            info.state_count,
            info.action_count,
            info.text_count,
        )
        for name, schema in info.modality_schemas.items():
            print(
                " ",
                name,
                "count=", schema.count,
                "dtype=", schema.dtype,
                "item_shape=", schema.item_shape,
                "index_semantics=", schema.index_semantics,
            )
finally:
    repository.close()
```

当前 v2 Catalog 是 split-neutral 的，所以不能通过 Catalog 的 `split=train/test`
筛选。训练/验证成员应从对应 SamplePool 获得。

### 6.3 获取一个完整 Episode 的逻辑视图

```python
from motus_data import open_dataset

repository = open_dataset("agibot_beta")
try:
    episode_id = next(repository.iter_episode_ids())
    episode = repository.get_episode(episode_id)

    print(episode.info)
    print("metadata:", episode.metadata)
    print("video frames:", episode.video.count)
    print("state rows:", episode.state.count)
    print("action rows:", episode.action.count)
    print("text items:", episode.text.count)
finally:
    repository.close()
```

`get_episode()` 不会立即解码完整视频。返回的 `EpisodeView` 是统一逻辑视图，
各模态只在调用 `read()` 时读取。

### 6.4 查看文本及其范围信息

```python
from motus_data import open_dataset

repository = open_dataset("agibot")
try:
    episode = repository.get_episode(next(repository.iter_episode_ids()))
    for item in episode.text.items:
        print(
            "index=", item.index,
            "kind=", item.kind,
            "frame_range=", item.frame_range,
            "text=", item.text,
            "metadata=", item.metadata,
        )
finally:
    repository.close()
```

若一个 MP4 的不同时间段对应不同文本，Catalog 会保存多个 `TextItem` 及其
`frame_range`/metadata。数据工厂只公开这些事实；模型侧最终在 SamplePool 中明确
指定 `text_indices`。

### 6.5 按帧索引读取视频

读取原始 RGB uint8 HWC：

```python
repository = open_dataset("robotwin_aloha")
try:
    episode = repository.get_episode(next(repository.iter_episode_ids()))
    frames = episode.video.read(
        [0, 10, 20],
        output_format="rgb_uint8_hwc",
        video_size=(480, 640),
    )
    print(frames.shape, frames.dtype)  # [3, 480, 640, 3], torch.uint8
finally:
    repository.close()
```

读取训练适配常用的 float CHW：

```python
from motus_data import open_dataset

repository = open_dataset("robotwin_aloha")
try:
    episode = repository.get_episode(next(repository.iter_episode_ids()))
    frames = episode.video.read(
        [0, 1, 2],
        output_format="rgb_float32_chw",
        video_size=(480, 640),
    )
    print(frames.shape)  # [3, 3, 480, 640]
finally:
    repository.close()
```

不要为了浏览长 Episode 一次性解码所有帧。推荐分块：

```python
from motus_data import open_dataset

repository = open_dataset("agibot_beta")
try:
    episode = repository.get_episode(next(repository.iter_episode_ids()))
    for chunk in episode.video.iter_chunks(
        chunk_size=32,
        start=0,
        stop=min(episode.video.count, 320),
        output_format="rgb_uint8_hwc",
        video_size=(240, 320),
    ):
        print(chunk.indices[0], chunk.indices[-1], chunk.frames.shape)
finally:
    repository.close()
```

### 6.6 读取 state 和 action

```python
repository = open_dataset("robotmind_tianyi")
try:
    episode = repository.get_episode(next(repository.iter_episode_ids()))

    state = episode.state.read([0, 10, 20])
    action = episode.action.read(range(1, 49))

    print("state:", state.shape, state.dtype)
    print("action:", action.shape, action.dtype)
    print("state finite:", bool(state.isfinite().all()))
    print("action finite:", bool(action.isfinite().all()))
finally:
    repository.close()
```

字段拼接顺序和物理维度由 source reader 固定，调用方不能通过 YAML 重新定义。

### 6.7 一次读取一条显式多模态 selection

```python
repository = open_dataset("robotmind_tianyi")
try:
    episode_id = next(repository.iter_episode_ids())
    episode = repository.get_episode(episode_id)

    selection = {
        "episode_id": episode_id,
        "video_frame_indices": list(range(0, 49)),
        "state_indices": [0],
        "action_indices": list(range(1, 49)),
        "text_indices": [0],
    }

    sample = episode.read_selection(
        selection,
        video_output_format="rgb_float32_chw",
        video_size=(480, 640),
    )

    print(sample.video.shape)   # [49, 3, 480, 640]
    print(sample.state.shape)   # [1, D]
    print(sample.action.shape)  # [48, D]
    print([item.text for item in sample.text])
finally:
    repository.close()
```

### 6.8 同 Episode 多 selection 联合读取

```python
from motus_data import open_dataset

repository = open_dataset("robotmind_tianyi")
try:
    episode = repository.get_episode(next(repository.iter_episode_ids()))
    selections = [
        {
            "episode_id": episode.episode_id,
            "video_frame_indices": list(range(0, 49)),
            "state_indices": [0],
            "action_indices": list(range(1, 49)),
            "text_indices": [0],
        },
        {
            "episode_id": episode.episode_id,
            "video_frame_indices": list(range(8, 57)),
            "state_indices": [8],
            "action_indices": list(range(9, 57)),
            "text_indices": [0],
        },
    ]
    samples = episode.read_selections(
        selections,
        video_output_format="rgb_float32_chw",
        video_size=(480, 640),
    )
    print([sample.video.shape for sample in samples])
finally:
    repository.close()
```

`read_selections()` 会先校验全部索引和数值，再对各模态取 union。同一视频帧只解码
一次，之后按每条 selection 的原始顺序重建样本。重复索引和给定顺序会保留。

### 6.9 完整 materialize 的内存保护

state/action/text 可以直接 materialize：

```python
from motus_data import open_dataset

repository = open_dataset("robotmind_tianyi")
try:
    episode = repository.get_episode(next(repository.iter_episode_ids()))
    data = episode.materialize(modalities=("state", "action", "text"))
    print(data.state.shape, data.action.shape, len(data.text))
finally:
    repository.close()
```

完整视频必须由调用者提供最大内存预算：

```python
from motus_data import open_dataset

repository = open_dataset("robotmind_tianyi")
try:
    episode = repository.get_episode(next(repository.iter_episode_ids()))
    data = episode.materialize(
        modalities=("video", "state", "action", "text"),
        max_video_bytes=4 * 1024**3,
        video_output_format="rgb_uint8_hwc",
    )
finally:
    repository.close()
```

预计占用超过预算时会抛出 `EpisodeMemoryLimitError`，不会无界占用内存。

## 7. SamplePool 协议

SamplePool 是模型侧给数据工厂的样本选择文件。一个 child 通常对应一个 train
SamplePool；需要验证时再提供独立 val SamplePool。

### 7.1 文件结构

JSONL 第一行是 header：

```json
{
  "format": "motus_sample_pool_v2",
  "pool_id": "robotmind_tianyi_train_v1",
  "catalog_id": "robotmind_tianyi_20260727_full_v1",
  "samples": 1000000,
  "meta": {
    "purpose": "production_train",
    "production_sampling_policy": true
  }
}
```

后续每行是一条 selection：

```json
{
  "episode_id": "robotmind:robotmind-tianyi:close_drawer:episode_0001",
  "video_frame_indices": [0, 1, 2, 3, 4],
  "state_indices": [0],
  "action_indices": [1, 2, 3, 4],
  "text_indices": [0]
}
```

四组索引都由模型侧显式提供：

- `video_frame_indices`：需要解码的帧编号。
- `state_indices`：需要读取的 state 行。
- `action_indices`：需要读取的 action 行。
- `text_indices`：Catalog 中的文本项编号。

### 7.2 当前 ThinkerWM batch profile 的长度要求

`motus_model_batch_v1` 在 `num_video_frames=48` 时要求：

- video：49 个显式索引，第 1 个作为条件帧，后 48 个作为预测目标。
- state：恰好 1 个显式索引。
- action：`48 * action_video_fps_ratio` 个显式索引。
- text：至少 1 个显式索引。

因此 Agibot 当前每条 selection 需要 288 个 action 索引；当前其余 child 需要
48 个。这个要求来自当前 ThinkerWM adapter，不是 Catalog 的固有限制。未来模型
协议变化时可以更换 adapter。

### 7.3 production SamplePool 必须满足

- `catalog_id` 与目标 Catalog 精确一致。
- `episode_id` 存在。
- 所有索引非负且不越界。
- train 和 validation selection 集合不重叠。
- 每个 Episode 有足够 selection 支持目标 batch size 和 `drop_last`。
- 文件规模足以覆盖目标 world size 和训练计划。
- index 语义由模型侧确认，不能依赖数据工厂自动推导。
- header 的 `samples` 与实际 selection 行数一致。

当前工程示例的 SamplePool 位于：

```text
/data/jiyuan.wang/SamplePools/_20260727/ten_source_smoke_v1
```

每个 child 只有 8 条，且 header 明确：

```json
"production_sampling_policy": false
```


完整协议：

- `docs/EPISODE_CATALOG_SAMPLE_POOL_V2.md`
- `docs/SAMPLE_POOL_DATASET_V1.md`

## 8. 第二层：单源训练数据流水线

### 8.1 直接获得完整 DataLoader + queue

下例从 `torchrun` 环境读取分布式参数；普通单进程运行时自动使用
`rank=0/world_size=1`。把 `sample_pool_path` 替换成与目标 Catalog 绑定的真实
SamplePool：

```python
import os

from motus_data import (
    open_training_data,
    validate_motus_model_batch_v1,
)

global_rank = int(os.environ.get("RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
max_steps = 10
sample_pool_path = "/path/to/robotmind_tianyi.train.sample_pool.jsonl"

pipeline = open_training_data(
    "robotmind_tianyi",
    release="20260727_full_v1",
    catalog_root="/data/jiyuan.wang/EpisodeCatalogs",
    sample_pool=sample_pool_path,
    batch_size=4,
    output_adapter="motus_model_batch_v1",
    adapter_options={
        "num_video_frames": 48,
        "action_video_fps_ratio": 1,
        "video_loss_weight": 1.0,
        "action_loss_weight": 0.0,
        "state_action_policy": "real",
    },
    video_size=(480, 640),
    rank=global_rank,
    world_size=world_size,
    seed=20260727,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    pin_memory=False,
    queue_enabled=True,
    queue_size=2,
    queue_timeout_seconds=600,
    queue_clone_tensors=True,
)

try:
    pipeline.start()
    for step in range(max_steps):
        batch, queue_stats = pipeline.next()

        validate_motus_model_batch_v1(
            batch,
            expected_batch_size=4,
            num_video_frames=48,
        )

        print(batch["first_frame"].shape)
        print(batch["video_frames"].shape)
        print(batch["initial_state"].shape)
        print(batch["action_sequence"].shape)
        print(batch["t5_input_messages"])
        print(queue_stats)
finally:
    pipeline.close()
```

`pipeline.close()` 必须执行。推荐使用 `try/finally` 或上下文管理器，避免后台
producer 线程残留。

### 8.2 不使用 queue

```python
import os
from motus_data import open_training_data

rank = int(os.environ.get("RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
pool_path = "/path/to/robotmind_tianyi.train.sample_pool.jsonl"

pipeline = open_training_data(
    "robotmind_tianyi",
    sample_pool=pool_path,
    batch_size=4,
    output_adapter="motus_model_batch_v1",
    adapter_options={
        "num_video_frames": 48,
        "action_video_fps_ratio": 1,
    },
    rank=rank,
    world_size=world_size,
    queue_enabled=False,
)

try:
    batch, stats = pipeline.next()
finally:
    pipeline.close()
```

也可以访问 `pipeline.dataloader`，但仍应让 pipeline 管理 Repository 和 sampler
生命周期。

### 8.3 自定义模型 batch adapter

数据工厂 canonical grouped batch 是 `IndexedEpisodeBatch`。模型可以提供 callable：

```python
from motus_data import open_training_data

def my_model_adapter(indexed_batch):
    samples = indexed_batch.samples
    return {
        "episode_ids": [sample.info.episode_id for sample in samples],
        "video": [sample.video for sample in samples],
        "state": [sample.state for sample in samples],
        "action": [sample.action for sample in samples],
        "text": [
            [item.text for item in sample.text]
            for sample in samples
        ],
    }

pool_path = "/path/to/robotmind_tianyi.train.sample_pool.jsonl"
pipeline = open_training_data(
    "robotmind_tianyi",
    sample_pool=pool_path,
    batch_size=2,
    output_adapter=my_model_adapter,
    queue_enabled=False,
)
```

模型协议变化应集中在 adapter，不应修改 source reader、Catalog 或 SamplePool。

## 9. 第二层：十源 weighted mix

```python
import os

from motus_data import IndexedTrainingSource, open_mixed_training_data

global_rank = int(os.environ.get("RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
max_steps = 10

sources = [
    IndexedTrainingSource(
        name="agibot",
        dataset="agibot",
        family="agibot",
        release="20260727_full_v1",
        catalog_root="/data/jiyuan.wang/EpisodeCatalogs",
        sample_pool="/path/to/agibot.train.sample_pool.jsonl",
        weight=18081,
        adapter_options={"action_video_fps_ratio": 6},
    ),
    IndexedTrainingSource(
        name="agibot_beta",
        dataset="agibot_beta",
        family="agibot_beta",
        release="20260723_concise_v2",
        catalog_root="/data/jiyuan.wang/EpisodeCatalogs",
        sample_pool="/path/to/agibot_beta.train.sample_pool.jsonl",
        weight=167794,
        adapter_options={"action_video_fps_ratio": 1},
    ),
    IndexedTrainingSource(
        name="robotmind_tianyi",
        dataset="robotmind_tianyi",
        family="robotmind",
        release="20260727_full_v1",
        catalog_root="/data/jiyuan.wang/EpisodeCatalogs",
        sample_pool="/path/to/robotmind_tianyi.train.sample_pool.jsonl",
        weight=7145,
        adapter_options={"action_video_fps_ratio": 1},
    ),
]

pipeline = open_mixed_training_data(
    sources,
    batch_size=4,
    batches_per_epoch=1000,
    output_adapter="motus_model_batch_v1",
    adapter_options={
        "num_video_frames": 48,
        "video_loss_weight": 1.0,
        "action_loss_weight": 0.0,
        "state_action_policy": "real",
    },
    video_size=(480, 640),
    seed=20260727,
    rank=global_rank,
    world_size=world_size,
    queue_enabled=True,
    queue_size=2,
    queue_timeout_seconds=600,
    queue_clone_tensors=True,
    num_workers=0,
    pin_memory=False,
    shuffle=True,
    drop_last=True,
)

try:
    for step in range(max_steps):
        batch, stats = pipeline.next()
        print(
            stats["mix_source_name"],
            stats["mix_source_family"],
            batch["source_children"],
        )
finally:
    pipeline.close()
```

混训约束：

- 一个 batch 始终只属于一个 child，避免本体维度冲突。
- 所有 rank 使用相同 child source schedule。
- 各 rank 在该 child 的 SamplePool 内消费互不重叠 selection。
- `family` 用于大数据源统计，`name` 用于 child 级诊断。
- 显式 `weight` 控制 child batch 比例。
- 省略 `weight` 时按该 child SamplePool selection 数量加权。
- `batches_per_epoch` 通过 largest-remainder 生成接近精确比例的整数配额。

不要在模型端再次添加 `DistributedSampler`，也不要把数据工厂 DataLoader 再交给
Accelerate 做分片。这一点很重要！！！之前尝试多次卡bug在这里

## 10. 当前 ThinkerWM batch 输出

`motus_model_batch_v1` 的主要字段：

| 字段 | 形状/类型 | 含义 |
|---|---|---|
| `first_frame` | `[B,C,H,W]` float | 条件帧，范围 `[0,1]` |
| `video_frames` | `[B,T,C,H,W]` float | 未来视频帧 |
| `initial_state` | `[B,D]` float | 显式 state selection |
| `action_sequence` | `[B,T_action,D]` float | 显式 action selection |
| `first_frame_pils` | `list[PIL.Image]` | VLM 图像输入 |
| `t5_input_messages` | `list[str]` | T5 文本 |
| `vlm_input_messages` | `list[str]` | VLM prompt |
| `robot_names` | `list[str]` | 本体路由 |
| `source_datasets` | `list[str]` | family，例如 `robotmind` |
| `source_children` | `list[str]` | child，例如 `robotmind_ark` |
| `episode_ids` | `list[str]` | Episode 标识 |
| `manifest_ids` | `list[str]` | 兼容旧 trainer 的 selection 稳定 ID，但其实没啥用，这里仅仅只是方便回退版本 |
| `frame_indices` | `list[list[int]]` | 显式视频索引 |
| `action_video_fps_ratios` | `list[int]` | 当前 batch 时间轴比例 |
| `video_loss_weights` | `[B]` | 视频 loss 权重 |
| `action_loss_weights` | `[B]` | action loss 权重 |
| `vlm_inputs` | 初始为 `None` | 模型侧预处理占位 |
| `language_embedding` | 初始为 `None` | 模型侧编码占位 |

不同 child 的 `D` 可以不同，但一个 batch 内保证一致。数据工厂不根据训练阶段删除
模态；模型端决定实际消费哪些字段以及对应的 loss。

Libero 等 state/action 原生维度不同的数据，使用新增且向后兼容的
`motus_native_multimodal_batch_v1`。其公共字段与上表一致，但额外显式输出：

| 字段 | 形状/类型 | 含义 |
|---|---|---|
| `state_dimension` | `int` | 当前 batch 原生 state 维度，Libero 为 8 |
| `action_dimension` | `int` | 当前 batch 原生 action 维度，Libero 为 7 |
| `normalization_modes` | `dict` | state/action 当前为 `raw` 或 `zscore` |
| `batch_adapter_profile` | `str` | 固定为 native profile 名称 |

该 profile 默认需要 49 个显式 video index，并输出 1 个条件帧、48 个未来帧、
1 个 initial state 和 48 个 action。帧数与 action/video 比例可通过
`adapter_options` 修改。归一化由各 source reader 配置，Libero 默认 z-score、
支持 raw；adapter 不再次修改数值。

## 11. 模型接入要点

模型训练端接入数据工厂时，只需要遵守以下通用约定：

- **BatchSource 生命周期**：训练开始时调用 `pipeline.start()`；每一步用
  `batch, queue_stats = pipeline.next()` 取数；退出（包括异常退出）时必须
  `pipeline.close()`，推荐使用 `with pipeline:` 或 `try/finally`。
  ReadyBatchQueue 有后台 producer 线程，Repository 持有 Catalog 文件描述符和
  SQLite 连接，不关闭会造成线程和文件句柄残留。
- **不要二次分片**：数据工厂已经按 `rank/world_size` 完成分片。不要在模型端
  再叠加 `DistributedSampler`，也不要把数据工厂的 DataLoader 交给 Accelerate
  的 `accelerator.prepare()` 做分片，否则会造成样本遗漏、rank 步数不一致或
  分布式 hang。
- **协议变化集中在 adapter**：模型 batch 协议变化时应新增或替换
  output adapter，不要修改 source reader、Catalog 或 SamplePool，也不要让
  Dataset 针对特定训练 stage 特判。
- **validation**：建议只在 rank0 创建，使用 `rank=0/world_size=1`、no queue、
  no shuffle、`drop_last=False`，并为每个 child 提供独立 val SamplePool。

## 12. 深入文档

版本更新和模型接入：

- `README_DEV_20260729_UPDATE_ZH.md`
- `README_DEV_20260804_AGIBOT_BETA_EFFECTOR_PROFILES_ZH.md`
- `README_DEV_20260810_LIBERO_INTEGRATION_ZH.md`
- `docs/EGOSCALE_FIRST500H_INTEGRATION_ZH.md`
- `docs/CROSS_RANK_SOURCE_SAMPLING_ZH.md`
- `docs/EGOSCALE_SECOND1700H_BATCH01_INTEGRATION_ZH.md`
- `docs/EGOSCALE_SECOND1700H_BATCH02_INTEGRATION_ZH.md`

数据访问和协议：

- `docs/PUBLIC_API_ZH.md`
- `docs/EPISODE_REPOSITORY_V1.md`
- `docs/EPISODE_CATALOG_SAMPLE_POOL_V2.md`
- `docs/SAMPLE_POOL_DATASET_V1.md`
- `docs/CATALOG_OFFSET_INDEX.md`
- `docs/INDEXED_BATCH_ADAPTERS.md`
- `docs/BATCH_CONTRACT_V1.md`

应先阅读本 README 的第 6-10 节理解数据，再按第 11 节的要点完成训练接入。
