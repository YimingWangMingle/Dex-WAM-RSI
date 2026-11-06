# EGOSCALE SECOND-1700H batch_02 接入说明

## 1. 版本与数据范围

本次开发分支：

```text
dev/20260821-egoscale-second-1700h-batch02
```

基线为已交付的 batch_01 分支最新提交：

```text
dev/20260820-egoscale-second-1700h-batch01@70225fa
```

batch_02 是与 batch_01 同一批次的后续传输数据，继续使用同一 EGOSCALE 数据协议。
本批发布为一个完整不可变 Catalog/SQLite，没有分桶，也没有拆成多个训练 child。

全量只读审计结果：

| 项目 | 数量 |
|---|---:|
| Episode | 48,241 |
| source group | 27,863 |
| frames | 61,230,397 |
| texts | 1,117,297 |
| fine actions | 973,277 |

48,241 个 Episode 全部通过正式 builder 解析，缺失必需文件、空文件、缺列、增列和
解析错误均为 0。与 FIRST-500H 和 SECOND-1700H batch_01 已发布数据合计
149,139 个 Episode、88,070 个 source group 的交集均为 0。

## 2. 数据协议

本批与 batch_01 保持相同的 Episode、模态和时间轴协议：

- 每个 Episode 使用一份 LeRobot-v3 Parquet 时间轴；
- 视频支持左目 `base_0`、右目 `base_1` 和双目水平拼接；
- state/action 支持 camera-relative 与 world-absolute pose profile；
- 默认使用 `base_0`、`world_pose_with_validity`、`resize_mode=pad`；
- 模型输出继续使用 head、左手、右手 pose 及逐项 validity 组成的 24D 协议；
- Episode 级任务文本和带帧范围的细粒度动作文本均完整保留；
- 手部缺失或非有限 pose 继续通过既有 validity/finite 门禁处理。

全部 Episode 均包含同一组 31 个 Parquet 字段。其中 46,815 条保持 canonical 物理
顺序，1,426 条仅物理列顺序不同；reader 按字段名访问，因此不影响运行时读取。
本批不需要新增 reader、batch adapter 或模型输入协议。

## 3. 数据工厂 Release

内置 release 注册：

```text
dataset: egoscale_second1700h_batch02
release: 20260821_full_v1
```

正式产物根目录：

```text
/data/jiyuan.wang/EpisodeCatalogs/_20260821/
  egoscale_second1700h_batch02_full_v1/
```

本 release 只新增 Catalog/metadata/SQLite 和注册项，不修改 batch_01、FIRST-500H
或其他数据集的冻结产物。正式绑定如下：

| 项目 | 值 |
|---|---|
| Catalog ID | `egoscale_second1700h_batch02_full_20260821_full_v1` |
| Catalog SHA256 | `f7a580753279b6c0a8763e2c8ed20745e791a429dcda86de8e5cde6d04eb597b` |
| Catalog bytes | 538,173,052 |
| SQLite bytes | 8,138,752 |
| Catalog 构建耗时 | 2,391.8 秒（约 39.9 分钟） |

构建未发生中断，`resume_count=0`。完成后 partial/checkpoint 已自动删除。独立静态
校验确认 JSONL、metadata、report 和 SQLite 均为 48,241 Episode；实际 SHA、
metadata SHA、report SHA 和 SQLite binding SHA 完全一致，SQLite
`PRAGMA quick_check=ok`。

真实流水线验收已通过：

- 正式 Catalog/SQLite 可直接访问 Episode，并完成真实 MP4 与 Parquet 读取；
- world-pose SamplePool、Dataset、DataLoader 和 ReadyBatchQueue 连续两批通过；
- batch 为 `first_frame=[2,3,120,160]`、`future=[2,8,3,120,160]`、
  `state=[2,24]`、`action=[2,8,24]`，全部有限且 queue producer 无错误；
- 内置 release 名可直接打开；
- batch_01/batch_02 四批混训各消费两批，输出协议完全一致；
- 新增注册后全库回归 `134/134` 通过。

## 4. 训练端接入事项

训练端更新到包含本 release 注册的数据工厂提交后，按既有 indexed 数据源方式接入：

1. 基于 batch_02 的 Catalog ID/SHA 生成独立且互斥的 train/holdout
   `sampling_ranges` 或其他 production SamplePool；不得复用 batch_01 的 pool。
2. 在训练 YAML 的 `data.sources` 中新增 `egoscale_second1700h_batch02`，配置训练
   pool、可选 eval pool 和采样权重。
3. 根据训练目标覆盖相机、pose、文本和窗口策略；未覆盖时使用上述数据工厂默认值。
4. 先完成 CPU 多源采样和短步 GPU 握手，再加入正式长训。

YAML source 的最小结构：

```yaml
- name: egoscale_second1700h_batch02
  dataset: egoscale_second1700h_batch02
  weight: <由训练策略确定>
  pool:
    type: sampling_ranges
    path: <训练端生成的 train ranges.sqlite>
  eval_pool:
    type: sampling_ranges
    path: <训练端生成的 holdout ranges.sqlite>
```

数据工厂负责 Catalog/SQLite、Episode 访问、显式索引执行、batch、DataLoader 和
ReadyBatchQueue。训练端不需要修改训练主干；SamplePool、train/holdout 划分、
source weight 和训练 YAML 仍由训练端生成和维护。
