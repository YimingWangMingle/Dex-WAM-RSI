# Agibot Beta 末端执行器分型更新说明

本文面向模型训练端负责人，说明
`dev/20260804-agibot-beta-effector-profiles` 分支对 Agibot Beta 数据读取、
Catalog、归一化和训练接入协议所做的更新。

本次改动只修复数据工厂中的 Agibot Beta state/action 语义和数值问题，
不改变 SamplePool、DataLoader、混训、ReadyBatchQueue 或模型 batch 的公共调用方式。
数据工厂仍统一提供完整多模态数据，不区分任何训练阶段；模型端自行决定使用哪些模态、
训练哪些分支以及计算哪些 loss。


## 1. 更新摘要

旧版 Agibot Beta 实际混有两种不同末端执行器：

- 平行夹爪：原始 `state/effector/position` 为 `(T, 2)`。
- 灵巧手：原始 `state/effector/position` 为 `(T, 12)`，表示左右手各 6 个关节角。

旧 reader 无条件执行 `reshape(T, 2, -1)`，再写入统一补零的 26 维向量。
这会把灵巧手的 12 个关节误解释成左右臂各 6 个“附加维度”，并继续使用按夹爪数据
统计的旧 z-score 常数。夹爪补零维的 std 约为 `1.171e-4`，灵巧手关节值除以该
错误 std 后会被放大到数千，已观察到极值约 `5960`。

本次修复后的原则是：

1. 依据每个 Episode 的真实 HDF5 schema 分型，不按任务名猜测。
2. 将 Agibot Beta 暴露为两个独立 child。
3. 两个 child 均输出原生 state/action 维度，不做结构性补零。
4. 两个 child 分别基于各自完整 Catalog 统计 mean/std，并独立归一化。
5. 两个 child 使用各自的 Catalog、SamplePool、采样权重和模型端维度映射。
6. 保留旧 release 和旧 Catalog 不变，避免影响正在运行的旧训练任务。


## 2. 新增的两个训练 child

| child / dataset | `robot_name` | Episode 数 | 任务族数 | 原始 effector | state/action 维度 |
|---|---|---:|---:|---|---:|
| `agibot_beta_parallel_gripper` | `agibot-beta-parallel-gripper` | 160,336 | 198 | `(T, 2)` | 14 |
| `agibot_beta_dexterous_hand` | `agibot-beta-dexterous-hand` | 7,458 | 19 | `(T, 12)` | 24 |

两个 child 的 Episode 集合互斥，合计 167,794 个 Episode，与旧完整 Catalog
数量一致。完整扫描发现灵巧手涉及 19 个任务族，不应只依赖最初人工发现的
`595/608/620/660/730` 五个任务 ID。

### 2.1 原生特征顺序

平行夹爪为 14 维：

```text
left.position.xyz
left.rotation.rotvec_xyz
left.gripper.position
right.position.xyz
right.rotation.rotvec_xyz
right.gripper.position
```

灵巧手为 24 维：

```text
left.position.xyz
left.rotation.rotvec_xyz
left.hand.joint_0 ... left.hand.joint_5
right.position.xyz
right.rotation.rotvec_xyz
right.hand.joint_0 ... right.hand.joint_5
```

state 和 action 使用相同的原生维度与特征顺序，沿各自 SamplePool 提供的显式
索引读取。数据工厂不把 14 维补到 24/26 维，也不把 24 维截断为夹爪表示。


## 3. Catalog 与 release

旧 Catalog 保持只读且不变：

```text
dataset: agibot_beta
release: 20260723_concise_v2
catalog_id: agibot_beta_full_20260723_concise_v2
episodes: 167794
sha256: b37786f8cbb9af1e60a30f90db3112967a04fdb4c060396d3e3dedbfbc998d96
```

旧 release 继续保留 legacy 26D 行为，仅用于兼容已有训练。新任务不要继续使用
旧 release 训练 action 分支。

新 Catalog 位于：

```text
/data/jiyuan.wang/EpisodeCatalogs/_20260805/agibot_beta_effector_profiles_v1/
```

| dataset | release | Catalog ID | SHA256 |
|---|---|---|---|
| `agibot_beta_parallel_gripper` | `20260805_full_v1` | `agibot_beta_parallel_gripper_20260805_full_v1` | `bcbde021ad1ceda4a9a162422cf4c70c649abcdcdfc1db417e7c9a21364dd84c` |
| `agibot_beta_dexterous_hand` | `20260805_full_v1` | `agibot_beta_dexterous_hand_20260805_full_v1` | `f722c77db0746178f41019976b80ed572f981b0bda39443ce6f2f91658422911` |

每份 Catalog 均包含对应的 `.meta.json` 和 `.index.sqlite`。已验证：

- SQLite `quick_check` 通过。
- JSONL、metadata 和 SQLite Episode 数一致。
- 两个新 Catalog 无交集。
- 两者并集与旧完整 Catalog 一致，没有缺失或新增 Episode ID。


## 4. 独立归一化

两个 child 分别在完整 Catalog 上统计 population mean/std。state 和 action
来自同一 proprio 时间轴，使用同一份统计量：

```text
normalized = (raw - mean) / applied_scale
```

为防止近常量维度再次产生极端放大，若经验 std 小于 `1e-3`，该维
`applied_scale` 固定为 `1.0`，即只中心化、不除以极小 std。本次两份全量统计中
没有触发该规则的维度，因此 `applied_scale == empirical_std`。

归一化产物目录：

```text
/data/jiyuan.wang/NormalizationArtifacts/_20260805/agibot_beta_effector_profiles_v1/
```

| child | 时间步数 | 维度 | artifact SHA256 |
|---|---:|---:|---|
| 平行夹爪 | 286,302,843 | 14 | `af50693b75162d46fcb0f5f9258657057d754916054a20126209ee2fb6f32a95` |
| 灵巧手 | 9,294,459 | 24 | `aa65a6004ef2b25f4b571170fbf8c176b1547b5f3d251c9aef1ad1e2b6f51f26` |

两次统计均未发现 non-finite 值或全局近常量维度。

### 4.1 平行夹爪 14D 统计量

顺序与第 2.1 节一致：

```text
mean = [
   0.659385,  0.290213,  0.680633,
  -0.149765,  1.621520, -1.145700,
  65.903300,
   0.667516, -0.308570,  0.681128,
  -1.674830,  0.514346, -0.275348,
  68.835600
]

std = [
  0.077175, 0.097405, 0.184519,
  0.832193, 1.382950, 1.152700,
  38.280800,
  0.081267, 0.099392, 0.181821,
  0.772167, 0.682519, 0.446378,
  38.230100
]
```

按全局原始端点估算，归一化后范围约为 `[-8.95, 6.91]`，不再出现旧实现中
数千量级的放大。

### 4.2 灵巧手 24D 统计量

```text
mean = [
   0.670577,  0.264678,  0.673792,
   0.234589,  0.575705, -0.357179,
   0.177614, -0.004279,  0.092586,
   0.098512,  0.133998, -0.959958,
   0.671702, -0.255609,  0.684275,
  -1.135740,  0.910348, -0.299940,
   0.114614,  0.127992,  0.232880,
   0.235958,  0.279941, -0.959718
]

std = [
  0.132901, 0.086516, 0.158560,
  1.477570, 1.379890, 1.656310,
  0.167298, 0.275221, 0.450623,
  0.458771, 0.500429, 0.240222,
  0.102125, 0.102629, 0.158283,
  0.523065, 0.747014, 0.566288,
  0.193116, 0.354618, 0.476968,
  0.478267, 0.524423, 0.240694
]
```

release 会自动绑定与 Catalog ID、分型、特征 schema 和 SHA256 匹配的归一化
产物。模型端通过内置 release 使用时不需要手工传入 mean/std 路径；任何不匹配
都会 fail closed，而不是静默读取错误统计量。


## 5. 已知异常与质量门禁

全量归一化审计只发现一个严重异常 Episode：

```text
episode_id: agibot_beta:554:913085
feature: right.hand.joint_5
feature_index: 23
timesteps: 361-366
raw_value: -11.0955085754
normalized_value: about -42.11067
```

这是唯一一个出现 `abs(z) > 20` 的 Episode，共 6 个异常值。为保持 Catalog 是
完整数据枚举，新 Catalog 仍保留该 Episode；但内置灵巧手 release 已在
`reader_options.excluded_episode_ids` 中排除它，读取数值模态时会在视频解码前
明确报错。同时两个新 release 的 action magnitude guard 阈值均为 `20`。

归一化统计仍覆盖完整 Catalog，包括这 6 个值。相对于 9,294,459 个灵巧手时间步，
其对全局统计影响可忽略；运行期会整段拒绝该 Episode。


## 6. 对现有数据工厂接口的影响

以下公共接口和协议均保持不变：

- `open_dataset()`
- `open_training_data()`
- `open_mixed_training_data()`
- 静态或动态 SamplePool 协议
- grouped sampler 与同 Episode 联合视频解码
- rank/world-size 分片和 `set_epoch()`
- DataLoader、ReadyBatchQueue 生命周期和消费方式
- `motus_model_batch_v1` 的字段名与总体结构

本次对共享模块增加的是带旧默认值的 release 级 `reader_options` 透传，用于绑定
归一化 artifact 和质量排除列表。没有配置该字段的旧数据集保持原行为。

旧 `agibot_beta@20260723_concise_v2` 仍输出 legacy 26D；两个新 release 分别输出
原生 14D/24D。新旧路径不会自动混用。


## 7. 模型端接入步骤

### 7.1 准备两个独立 SamplePool

模型端需要为两个 child 分别生成 SamplePool，并严格绑定对应 Catalog ID：

```text
agibot_beta_parallel_gripper_20260805_full_v1
agibot_beta_dexterous_hand_20260805_full_v1
```

旧 `agibot_beta_full_20260723_concise_v2` 的 SamplePool 不能直接用于新 Catalog。
两个 child 与 RobotMind/RobotWin 的多本体模式一致：各自一个 SamplePool、各自一个
source 和采样权重，在 family 统计层可统一归入 `agibot_beta`。

### 7.2 在训练配置中注册两个 source

以下为语义示例。外层 YAML 层级沿用模型项目当前 bridge 的单训练配置，不要求增加
第二份启动 YAML：

```yaml
data_factory:
  sources:
    - name: agibot_beta_parallel_gripper
      dataset: agibot_beta_parallel_gripper
      release: 20260805_full_v1
      sample_pool: /path/to/agibot_beta_parallel_gripper.sample_pool.jsonl
      family: agibot_beta
      weight: 1.0

    - name: agibot_beta_dexterous_hand
      dataset: agibot_beta_dexterous_hand
      release: 20260805_full_v1
      sample_pool: /path/to/agibot_beta_dexterous_hand.sample_pool.jsonl
      family: agibot_beta
      weight: 1.0
```

采样权重由模型端按训练策略设置，不要求两个 child 等权，也不由数据工厂按样本量
强制决定。

等价的 Python source 定义如下，后续 pipeline 调用方式不变：

```python
from motus_data import IndexedTrainingSource, open_mixed_training_data

sources = [
    IndexedTrainingSource(
        name="agibot_beta_parallel_gripper",
        dataset="agibot_beta_parallel_gripper",
        release="20260805_full_v1",
        sample_pool="/path/to/agibot_beta_parallel_gripper.sample_pool.jsonl",
        family="agibot_beta",
        weight=1.0,
    ),
    IndexedTrainingSource(
        name="agibot_beta_dexterous_hand",
        dataset="agibot_beta_dexterous_hand",
        release="20260805_full_v1",
        sample_pool="/path/to/agibot_beta_dexterous_hand.sample_pool.jsonl",
        family="agibot_beta",
        weight=1.0,
    ),
]

pipeline = open_mixed_training_data(
    sources=sources,
    batch_size=TRAIN_BATCH_SIZE,
    batches_per_epoch=STEPS_PER_EPOCH,
    output_adapter="motus_model_batch_v1",
    num_workers=0,
    queue_enabled=True,
    queue_size=READY_BATCH_QUEUE_SIZE,
    # 其余分布式、视频尺寸和 adapter 参数沿用现有配置。
)
```

### 7.3 注册模型侧原生维度

模型端 action/state encoder、decoder 或 projector 需要识别新的 `robot_name`：

```yaml
# 键名按模型项目现有 ActionBranch 配置调整，映射语义如下。
robot_state_action_dims:
  agibot-beta-parallel-gripper: 14
  agibot-beta-dexterous-hand: 24
```

每个 batch 保持 child 同质，不会在同一 batch 内混入 14D 和 24D。模型端不要再
把两个 child 强制 pad 到旧 26D，也不要保留只接受 `agibot-a2-d` 或固定 26D 的
硬编码检查。

### 7.4 checkpoint 处理

旧 checkpoint 不包含以下新名字对应的独立参数：

```text
agibot-beta-parallel-gripper
agibot-beta-dexterous-hand
```

因此模型端需要：

1. 为 14D 和 24D 分别创建 action/state encoder、decoder 或 projector。
2. 对新增参数执行明确初始化并参与训练，或实现经过验证的旧权重迁移。
3. 加载旧 checkpoint 时允许新增参数缺失，随后核对 missing/unexpected keys；不能
   静默认为新 action 分支已经获得有效预训练权重。

这属于模型结构与 checkpoint 策略变化。数据工厂只保证返回正确的原生维度和
`robot_name`，不会替模型端补齐或迁移分支权重。


## 8. 模型 batch 预期

继续使用现有 `motus_model_batch_v1` adapter 时，关键张量为：

| child | `initial_state` | `action_sequence` |
|---|---|---|
| 平行夹爪 | `[B, 14]` | `[B, T, 14]` |
| 灵巧手 | `[B, 24]` | `[B, T, 24]` |

`robot_names` 分别为 `agibot-beta-parallel-gripper` 和
`agibot-beta-dexterous-hand`。视频、首帧、文本及其他 batch 字段的协议不变。

数据工厂始终读取和提供完整可用模态。模型端是否把 state/action 送入网络、是否冻结
对应分支、是否计算 action loss，均由模型训练配置决定。


## 9. 验证结果

本次实现已完成以下验证：

- 全部单元测试 `100/100` 通过。
- 两份 Catalog 的数量、哈希、SQLite、互斥性和并集完整性通过。
- 两份归一化 artifact 的 Catalog ID、profile、schema、维度和 SHA256 绑定通过。
- 真实 reader 分别抽检 3 个平行夹爪和 3 个灵巧手 Episode。
- 抽检归一化最大绝对值约为 `2.23` 和 `2.51`。
- 已知异常 Episode 在视频解码前被拒绝。
- 两个真实 SamplePool 完成切片、同 Episode grouped decode、MP4 解码、collate、
  DataLoader、加权混训和 ReadyBatchQueue 闭环。
- queue 共生产并消费 4 个 batch，两个 child 各 2 个，无 producer exception。
- `motus_model_batch_v1` 最终形状分别验证为 `[2,14]/[2,2,14]` 和
  `[2,24]/[2,2,24]`。


## 10. 模型端迁移检查表

- [ ] 使用包含本次更新的 `motus_data/` 版本。
- [ ] 分别生成并验证两个新 Catalog ID 对应的 SamplePool。
- [ ] 训练 YAML 中增加两个独立 source，并设置各自采样权重。
- [ ] 注册 14D 和 24D 两个新 `robot_name`。
- [ ] 检查并移除模型侧固定 26D 或旧 robot name 的硬编码。
- [ ] 明确初始化或迁移两个新 action/state 分支参数。
- [ ] 加载旧 checkpoint 后人工核对 missing/unexpected keys。
- [ ] 启动长训前分别做单 child batch smoke 和双 child 混训 queue smoke。
- [ ] 在日志中确认实际采到两个 child，张量维度和 `robot_names` 正确。
- [ ] 不再使用旧 Agibot Beta SamplePool 训练新的 action/state 分支。


## 11. 主要实现文件

```text
motus_data/agibot_beta_profiles.py
motus_data/agibot_beta_normalization.py
motus_data/catalog/agibot_beta_profiles.py
motus_data/repository.py
motus_data/reader_registry.py
motus_data/releases.py
motus_data/releases.yaml
motus_data/indexed_pipeline.py
motus_data/indexed_mix.py
```

审计、构建和闭环 smoke 脚本位于：

```text
examples/audit_agibot_beta_effector_profiles.py
examples/build_agibot_beta_profile_catalogs.py
examples/build_agibot_beta_normalization.py
examples/audit_agibot_beta_normalized_outliers.py
examples/smoke_agibot_beta_profile_releases.py
examples/smoke_agibot_beta_profile_mixed_queue.py
examples/smoke_agibot_beta_profile_model_batch.py
```

旧 `docs/AGIBOT_BETA_DATA_FACTORY_GUIDE_ZH.md` 描述的是 legacy 单 Catalog/26D
路径。涉及新的 action/state 训练时，以本文和两个 `20260805_full_v1` release
为准。
