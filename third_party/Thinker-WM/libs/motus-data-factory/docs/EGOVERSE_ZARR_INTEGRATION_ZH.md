# EgoVerse Zarr 接入说明

## 状态与边界

当前代码已实现 EgoVerse `processed_v3` Zarr 数据族的原生 Episode reader、
支持 checkpoint/resume/quarantine 的 Catalog builder，并已注册首批
production release。第一批包含以下六个 child：

1. `egoverse_aria_bimanual`
2. `egoverse_aria_left_arm`
3. `egoverse_aria_right_arm`
4. `egoverse_eva_bimanual`
5. `egoverse_scale_bimanual`
6. `egoverse_mecka_freeform_bimanual`

以上 profile 已完成全量 Catalog/SQLite、随机真实 payload、逐 child queue、
18-child 混训兼容和默认 release API 门禁。六源共发布 72,672 个 Episode，
另有 806 个原始 Episode 因标注范围、NaN/Inf、Zarr checksum 或字段/长度问题
被显式写入 quarantine sidecar，不会进入训练 Catalog。

六个默认 release 版本均为 `20260730_first6_v1`。训练端可以直接调用
`open_dataset("egoverse_<profile>")` 或
`open_training_data("egoverse_<profile>", ...)`，无需自行传 Catalog/index。

原始数据目录严格只读。所有 Catalog、index、日志和临时文件必须写到
`/data/jiyuan.wang` 下的项目或数据资产目录，不能写到 `/data/EgoVerse`。
production Catalog 中必须记录集群路径 `/data/EgoVerse/...`，不能记录本地挂载
路径 `/home/ubt/egoverse/...`。

## 数据契约

一个 `.zarr` store 对应一个完整 Episode。有效索引范围统一使用 root attribute
`total_frames`：

```text
video:  [0, total_frames)
state:  [0, total_frames)
action: [0, total_frames)
```

Zarr 数组的物理第一维可能因 shard padding 大于 `total_frames`。Catalog builder
会验证所有必需数组不短于有效长度，reader 永远不会把 padding 暴露给用户。

第一版 reader 保留原始数值语义：

- 人类采集源 Aria、Scale、Mecka：state/action 都按模型分别给出的索引读取
  observed end-effector pose 原始行。
- EVA：state 读取 observed pose/gripper；action 读取 commanded pose/gripper。
- 不在数据工厂中隐式构造 future action horizon，也不根据 video index 推导
  state/action index。
- head-relative、插值、future horizon 等转换只有在模型负责人确认显式 profile
  后才能增加，不能静默改变当前 raw contract。

数据工厂不会决定模型是否使用这些字段。Catalog 和 reader 会显式提供：

- Aria/Scale/Mecka：
  `action.source_signal=observed_ee_pose_timeline`，
  `action.is_control_command=false`；
- EVA：
  `action.source_signal=commanded_ee_pose_gripper_timeline`，
  `action.is_control_command=true`；
- 所有 profile：`action.indices=model_provided`。

即使 action 来源是未来观测轨迹，它仍按模型端给定的 `action_indices` 正常读取、
组 batch 和进入 queue；是否参与 loss 由模型训练端决定。

文本包含：

- Episode 级英文 `task_description`。
- 英文 annotation span，范围为 `[start_idx, end_idx)`。
- 没有任何文本的 Episode 仍保留在 split-neutral Catalog 中，并以
  `missing_instruction` 标记；文本训练的 SamplePool 负责过滤。

## Profile

不同输出维度必须使用不同 child/profile：

| profile | state | action | 维度 |
|---|---|---|---:|
| `aria_bimanual` | left/right observed pose | left/right observed pose | 14 |
| `aria_left_arm` | left observed pose | left observed pose | 7 |
| `aria_right_arm` | right observed pose | right observed pose | 7 |
| `eva_bimanual` | left/right observed pose+gripper | left/right commanded pose+gripper | 16 |
| `eva_left_arm` | left observed pose+gripper | left commanded pose+gripper | 8 |
| `eva_right_arm` | right observed pose+gripper | right commanded pose+gripper | 8 |
| `scale_bimanual` | left/right observed pose | left/right observed pose | 14 |
| `mecka_freeform_bimanual` | left/right observed pose | left/right observed pose | 14 |
| `mecka_flagship_bimanual` | left/right observed pose | left/right observed pose | 14 |

`images.front_1` 是当前统一主视角。默认 resize 模式为等比例缩放后居中补零；
这会保持 Mecka `360x640` 的画面比例。模型端只有在明确需要拉伸时才通过
`EgoVerseEpisodeReader(resize_mode="interpolate")` 覆盖。Lightwheel 使用
MP4/JSON/MCAP，不属于这个 Zarr reader；`processed_v3/scale_old` 也不应接入。

## 构建小 Catalog

安装可选依赖需要 Python 3.11：

```bash
pip install -e ".[egoverse]"
```

以 Aria 双臂为例：

```bash
PYTHONPATH=. python tools/build_episode_catalog.py \
  --family egoverse \
  --data-root /data/EgoVerse/processed_v3/aria \
  --profile aria_bimanual \
  --output /data/jiyuan.wang/EpisodeCatalogs/_smoke/egoverse_aria_bimanual.jsonl \
  --catalog-id egoverse_aria_bimanual_smoke_v1 \
  --data-version smoke_v1 \
  --source-child egoverse_aria_bimanual \
  --max-episodes 2 \
  --resume \
  --checkpoint-interval 100
```

CLI 默认执行 numeric finite、首尾 JPEG、长度和协议校验，并生成 SQLite offset
index。取消某项检查必须显式使用 `--no-validate-numeric-finite` 或
`--no-validate-jpeg`，production 构建不应关闭。

`--resume` 会保留隐藏的 partial Catalog 和原子 checkpoint。checkpoint 记录
构建参数指纹、源 Episode inventory 指纹、已处理位置、Catalog 安全字节偏移、
错误计数和质量统计。重启同一命令时先截断到最近一次已 fsync 的安全点再继续；
参数或源 inventory 变化会拒绝恢复。构建成功后 partial/state 自动删除，只发布
完整 Catalog 和 metadata。

## 首批六源正式构建

首批 release 固定包含：

- `aria_bimanual`
- `aria_left_arm`
- `aria_right_arm`
- `eva_bimanual`
- `scale_bimanual`
- `mecka_freeform_bimanual`

使用统一入口串行构建六份 Catalog、metadata 和 SQLite index：

```bash
PYTHONPATH=. python -u tools/build_egoverse_first6_release.py \
  --egoverse-root /data/EgoVerse \
  --output-root \
    /data/jiyuan.wang/EpisodeCatalogs/_20260730/egoverse_first6_v1 \
  --data-version 20260730_first6_v1 \
  --checkpoint-interval 100 \
  --quarantine-episode-errors
```

该命令可原样重跑。未完成 child 从最近一次 fsync checkpoint 继续；已完成
child 会先核验 Catalog SHA256、metadata 和 SQLite 绑定后跳过；若 Catalog 和
metadata 已发布但只缺 SQLite index，则只补索引。每完成一个 child 都会原子
更新 release report。正式 release 注册必须等六个 child 全部构建并完成真实
随机访问、batch 和 queue 验收后进行。

`--quarantine-episode-errors` 不会关闭校验。每个 Episode 仍完整执行有限值、
必需字段、长度、首尾 JPEG 和文本范围校验；任一必需模态失败时，整段 Episode
不会进入 Catalog，而是逐条写入与 Catalog 绑定的 quarantine JSONL。metadata
记录该 sidecar 的行数、字节数和 SHA256。构建完成后必须审核各 child 的隔离
数量、比例和错误类别；store inventory 错误仍保持 fail-closed。

## 访问与 SamplePool

Catalog 注册 release 后，访问方式与其他数据源一致：

```python
from motus_data import open_dataset

repository = open_dataset("egoverse_aria_bimanual")
episode = repository.get_episode("egoverse:aria_bimanual:<episode-name>")

print(episode.info)
frames = episode.video.read([0, 3, 9])
state = episode.state.read([0, 2])
action = episode.action.read([4, 7])
texts = episode.text.items
repository.close()
```

模型端 SamplePool 仍提供四组独立索引：

```json
{
  "episode_id": "egoverse:aria_bimanual:<episode-name>",
  "video_frame_indices": [0, 2, 4, 6, 8],
  "state_indices": [0],
  "action_indices": [2, 4, 6, 8],
  "text_indices": [0]
}
```

数据工厂只校验并执行这些索引。后续 grouped DataLoader、batch adapter 和
ReadyBatchQueue 接口不因新增 EgoVerse 而改变。
