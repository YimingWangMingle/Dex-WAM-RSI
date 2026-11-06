# dev/20260729 数据工厂更新说明


## 1. 版本边界

- 基于已验证的数据工厂基线：`dev/20260728`。
- 本次继续迭代的分支：`dev/20260729`。
- 当前新增三个接入目标：RobotMind Agilex、EgoVerse、OpenVid。
- Catalog 默认根目录仍为 `/data/jiyuan.wang/EpisodeCatalogs`。
- 原始数据在集群上的目录：
  - RobotMind Agilex：`/data/RoboMIND/Agilex`
  - EgoVerse：`/data/EgoVerse`
  - OpenVid：`/data/OpenVid`


## 2. 本次新增数据

| 接入目标 | 训练 child | Episode 数 | 主要模态 | release |
|---|---|---:|---|---|
| RobotMind Agilex | `robotmind_agilex` | 22,206 | HDF5 内 JPEG、英文/中文文本、14D state/action | `20260728_full_v1` |
| OpenVid | `openvid` | 1,019,957 | ZIP 内视频、英文 caption；无真实 state/action | `20260730_full_v1` |
| EgoVerse | 6 个 child | 72,672 | Zarr v3 内 JPEG、文本、7D/14D/16D state/action | `20260730_first6_v1` |

EgoVerse 六个 child：

| child | Episode 数 | state/action 维度 | 数据语义 |
|---|---:|---:|---|
| `egoverse_aria_bimanual` | 1,845 | 14 | 双臂观测位姿 |
| `egoverse_aria_left_arm` | 333 | 7 | 左臂观测位姿 |
| `egoverse_aria_right_arm` | 315 | 7 | 右臂观测位姿 |
| `egoverse_eva_bimanual` | 2,876 | 16 | observed state / commanded action |
| `egoverse_scale_bimanual` | 17,037 | 14 | 双臂观测位姿 |
| `egoverse_mecka_freeform_bimanual` | 50,266 | 14 | 双臂观测位姿 |

语义注意事项：

- Agilex 主视角为 `camera_front`。state/action 均按
  `left_arm,left_gripper,right_arm,right_gripper` 顺序读取 14 维 puppet
  时间轴。
- OpenVid 原始数据只有视频和英文 caption。数据工厂可以为旧 ThinkerWM batch
  协议生成 26 维全零 state/action 兼容占位，但这些占位不是真实机器人信号。
- EgoVerse 中只有 EVA bimanual 的 action 是 commanded pose/gripper。Aria、
  Scale、Mecka 当前暴露的 action 是未来观测到的末端位姿，不应在未评估语义前
  当作真实控制指令使用。


## 3. 对 dev/20260728 的兼容性

本次三个新数据接入没有改动态 SamplePool 的核心语义。以下文件采用
`dev/20260728@26702a0` 的最新实现：

- `motus_data/indexed_pipeline.py`
- `motus_data/indexed_mix.py`
- `motus_data/sample_pool.py`

它们新增 `SupportsSamplePool` 对象/工厂入口和 `set_epoch()` 转发。静态 JSONL
路径仍解析为原 `SamplePool(path, repository)`，现有 Catalog 绑定、selection
校验、grouped sampler、DataLoader 和 queue 行为不变。以下公共主链仍未被本次
三数据接入修改：

- `motus_data/pipeline.py`
- `motus_data/loading/`

`open_dataset()` 和 `open_mixed_training_data()` 的调用方式不变；
`open_training_data()` 的 `sample_pool=` 只做了向后兼容的类型扩展，可继续传
原文件路径，也可传 `SupportsSamplePool` 实例或 factory。

原有十个 release、默认 Catalog 根目录、YAML 的数据工厂配置层级、queue 生命周期、
rank/world-size 分片方式均保持不变。共享模块中的变化均为带旧默认值的兼容扩展：

- RobotMind reader 新增 Agilex schema，不改变已有四个 RobotMind child。
- Catalog 仅在 `storage.type=none` 时允许 state/action 数量为 0。
- batch adapter 的缺失模态兼容模式默认关闭，旧数据仍按原规则校验。
- collate 仅对使用缺失模态占位的样本增加标记，旧数据 batch key 不变。
- 新增 Zarr、普通 ZIP 和分片 ZIP reader，不替换旧 reader。

因此，Agilex 和 EgoVerse 的加入不要求模型端修改既有数据工厂调用接口。新数据
只通过 release、SamplePool、YAML source 和机器人维度配置进入训练。OpenVid
还存在第 5 节说明的模型 bridge 门禁。

## 4. 模型端如何加入新数据

模型端继续使用原有单份训练 YAML，不增加第二份启动 YAML。对每个新 child：

1. 基于对应 immutable Catalog 提供 production train/val SamplePool。它可以是
   静态 JSONL，也可以是满足 `SupportsSamplePool` 的动态对象或 factory。
2. 保证 train/val selection 不重叠，并满足目标 batch size、rank 数和
   `drop_last` 的 grouped batch 数量要求。
3. 在 `dataset.data_factory.sources` 中追加 child、release、SamplePool 和权重。
4. 在 `dataset.robot_names` 中追加本体名和 state/action 维度。
5. 保持原有 `build_training_data_source()`、`pipeline.start()/next()/close()`
   和 queue 消费方式不变。

当前动态 SamplePool 可以原样用于 Agilex 和 EgoVerse。OpenVid 也使用同一
SamplePool 接口，但仍必须满足第 5 节的 absent state/action 配置。

建议使用的 robot name：

| child | robot name | 维度 |
|---|---|---:|
| `robotmind_agilex` | `robotmind-agilex` | 14 |
| `egoverse_aria_bimanual` | `egoverse-aria-bimanual` | 14 |
| `egoverse_aria_left_arm` | `egoverse-aria-left-arm` | 7 |
| `egoverse_aria_right_arm` | `egoverse-aria-right-arm` | 7 |
| `egoverse_eva_bimanual` | `egoverse-eva-bimanual` | 16 |
| `egoverse_scale_bimanual` | `egoverse-scale-bimanual` | 14 |
| `egoverse_mecka_freeform_bimanual` | `egoverse-mecka-freeform-bimanual` | 14 |
| `openvid` | `openvid-null` | 26（兼容占位） |


## 5. OpenVid 额外门禁

现有 ThinkerWM bridge 只向数据工厂传递 `action_video_fps_ratio` 和
`state_action_policy`，尚未传递 OpenVid 所需的缺失模态选项。正式加入 OpenVid
前，应在 bridge 中集中新增以下可选 per-source 字段，并保持旧默认值不变：

- `allow_missing_state_action`
- `placeholder_state_action_dim`
- `video_loss_weight`
- `action_loss_weight`

Stage1 video-only 中，OpenVid 至少需要：

```yaml
state_action_policy: absent
allow_missing_state_action: true
placeholder_state_action_dim: 26
action_loss_weight: 0.0
```

这只能保证 batch 协议和 loss 权重正确。Stage2 若混入 OpenVid，模型端还必须
根据 absent policy 移除其 action token/conditioning；只把 action loss 设为 0
不足以阻止占位 action 进入模型。


## 6. motus311 镜像差异

候选镜像全名：

```text
ubhub.ubtrobot.com/ogi/chen-liao-world-model:v0.0.4-py311-egoverse-test
```

它基于上一版
`ubhub.ubtrobot.com/ogi/chen-liao-world-model:v0.0.3` 构建，不替换原镜像内
已有环境。

| 项目 | v0.0.3 / legacy `motus` | 新镜像的 `motus311` |
|---|---|---|
| OS / CUDA / system NCCL | Ubuntu 22.04.5 / CUDA 12.8.1 / NCCL 2.25.1 | 继承基础镜像，不变 |
| Python | 3.10.19 | 3.11.14 |
| Torch / torchvision | 2.7.1+cu128 / 0.22.1+cu128 | 版本保持一致 |
| FlashAttention | 2.8.3 | 2.8.3，针对 CPython 3.11 和 H200 sm90 重编译 |
| EgoVerse Zarr | 不支持所需 Zarr v3 运行时 | `zarr==3.1.5` |
| Zarr 核心依赖 | 不作为 EgoVerse 固定栈提供 | `numcodecs==0.16.5`、`donfig==0.8.1.post1`、`google-crc32c==1.8.0` |
| OpenCV | 原环境保留 | 重新确保 headless wheel 最后安装，避免 Pod 缺少 `libGL.so.1` |

镜像内共有三个 Conda 环境：

- `base`：Python 3.13，非训练环境。
- `motus`：上一版 Python 3.10 训练环境，完整保留，作为旧流程回退点。
- `motus311`：本次新增的 Python 3.11 训练候选环境。

镜像 PATH 默认指向 `motus311`，但训练脚本若显式执行
`conda activate motus`，会覆盖默认 PATH。包含 EgoVerse 的训练必须明确激活：

```bash
source /home/fengodchen/miniconda3/etc/profile.d/conda.sh
conda activate motus311
python --version
python -c "import torch, zarr; print(torch.__version__, zarr.__version__)"
```

镜像没有安装 EgoVerse 官方仿真、Ray、MuJoCo 或 Aria SDK；当前数据工厂只读取
训练所需的 Zarr payload，不依赖这些组件。

## 7. motus311 训练风险与验收门禁

已经通过：

- 镜像构建期固定版本和 import 验证。
- Torch CUDA 12.8、NCCL backend availability、FlashAttention、DeepSpeed、
  PyAV、Decord、OpenCV、HDF5、Zarr 导入。
- 集群 `motus311` 环境中的真实 EgoVerse payload、batch 和 queue smoke。
- 当前数据工厂 84/84 测试及 18-child 数据链路门禁。
- `pip check` 相对 v0.0.3 未新增依赖冲突。

仍需模型训练端验收：

1. 使用实际 ThinkerWM 模型完成单 GPU forward/backward/optimizer smoke。
2. 完成单节点 8 GPU 分布式 smoke。
3. 完成三节点 IB/NCCL smoke，再开始长训。
4. 验证实际使用的 VAE、VLM、T5、FlashAttention、DeepSpeed 配置在
   Python 3.11 下无 ABI 或运行期回归。
5. 验证退出、异常和 dataloader producer 关闭流程，不残留进程或文件句柄。

主要风险：

- Python 从 3.10 升至 3.11，`motus311` 是按 v0.0.3 包版本重建的环境，不是
  legacy `motus` 的字节级复制。少用到的 Python/CUDA 扩展仍可能有 ABI 差异。
- FlashAttention wheel 针对 H200 compute capability 9.0 构建；换用其他 GPU
  架构前必须重新验证或重建。
- 当前镜像标签仍带 `test`，在完成模型单卡、单机多卡和三节点门禁前不应作为
  无条件生产基线。
- 旧启动脚本若继续激活 `motus`，普通旧数据可能仍能运行，但 EgoVerse 会因
  Python/Zarr 版本不满足而失败。
- 新环境保留旧环境功能的依据是关键包版本、导入和数据工厂回归；这不等价于
  所有模型训练路径已经逐一运行。旧 `motus` 环境和 v0.0.3 镜像仍应保留为回退点。



