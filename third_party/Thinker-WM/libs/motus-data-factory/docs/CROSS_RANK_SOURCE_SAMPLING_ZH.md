# 跨 Rank 按权重分配数据源

## 1. 目标与边界

`open_mixed_training_data()` 支持两种跨 rank source 调度：

| 策略 | 同一个 data step 的跨 rank 行为 | 兼容性 |
|---|---|---|
| `synchronized` | 所有 rank 使用同一个 source | 默认，保持历史行为 |
| `weighted_per_step` | 按 source 权重分配全部 rank 槽位 | 显式开启 |

两种策略都保持以下不变量：

- 每个 rank 内的一个 batch 只来自一个 source；
- 每个 source 继续使用自己的 Catalog、SamplePool、reader 和 adapter；
- `EpisodeGroupedBatchSampler` 继续按全局 rank 分片样本；
- DataLoader、ready queue 和 batch 输出协议不变；
- source schedule 在 `seed + epoch` 下确定且可复现。

该策略只定义**取数据时**的 source 分配。复用时，同一批
rank/source 分配会随该 batch 一起保持不变。

## 2. `weighted_per_step` 语义

假设 `world_size=24`，权重为：

```text
source_a = 0.628653
source_b = 0.1800862
...
```

每个 data step 有 24 个 rank-slot。`source_a` 的理论槽位数为
`0.628653 * 24 = 15.087672`，因此单步分到 15 或 16 个 rank；不足一个 rank 的
小数信用会累积到后续 step。小权重 source 可能若干 step 才获得一个 rank，但在
一段训练窗口内的累计比例会逼近配置权重。

每一步先确定各 source 的整数槽位数，再用确定性 shuffle 把 source 分配给 rank。
因此不会把某个 source 永久绑定到固定 GPU。所有 rank 独立执行同一算法，不增加
NCCL/Gloo 通信。

## 3. 数据工厂 API

```python
from motus_data import IndexedTrainingSource, open_mixed_training_data

pipeline = open_mixed_training_data(
    sources=[
        IndexedTrainingSource(
            name="source_a",
            dataset="source_a",
            sample_pool=source_a_pool,
            weight=0.7,
        ),
        IndexedTrainingSource(
            name="source_b",
            dataset="source_b",
            sample_pool=source_b_pool,
            weight=0.3,
        ),
    ],
    batch_size=8,
    batches_per_epoch=10000,
    seed=20260818,
    rank=global_rank,
    world_size=world_size,
    cross_rank_source_policy="weighted_per_step",
    queue_enabled=True,
    queue_size=2,
)
```

新增统计字段：

```text
mix_cross_rank_source_policy
mix_rank
mix_world_size
```

原有 `mix_source_name`、`mix_source_weight`、`mix_consumed_by_source` 等字段保持
不变。

## 4. 训练包装层接入

模型训练项目若通过自己的 YAML/factory 包装数据工厂，只需把配置值原样传入：

```yaml
dataset:
  cross_rank_source_policy: weighted_per_step
  sources:
    - {name: source_a, dataset: source_a, weight: 0.7, pool: ...}
    - {name: source_b, dataset: source_b, weight: 0.3, pool: ...}
```

```python
pipeline = open_mixed_training_data(
    sources,
    # 其余既有参数保持不变
    cross_rank_source_policy=str(
        config.dataset.get("cross_rank_source_policy", "synchronized")
    ),
)
```

不配置该字段时仍使用 `synchronized`，已有训练 YAML 行为不变。

## 5. 已完成的 CPU 验证

单元测试覆盖：

- 默认 `synchronized` 行为不变；
- 当前 14 source 权重在 24 rank、1000 data-step 下的模拟；
- 每个 source 累计槽位与理论权重误差不超过 1 个 rank-slot；
- 相同 seed/epoch 可复现，不同 epoch 重新调度；
- 同一 source 的跨 rank SamplePool batch 仍不重叠；
- ready queue 开启时策略和统计字段正确透传；
- 未知策略 fail-fast。

## 6. 多机多卡验收项

CPU 验证不能替代模型分布式反向测试。模型侧合入后应检查：

1. ZeRO/DDP `find_unused_parameters=false` 下无未使用参数或 collective 超时；
2. 模态 dropout 继续使用跨 rank 一致的决定，不能随 source 独立；
3. 不同 action 序列长度的 source 同步反向正常；
4. 同一 global step 的 `mix_source_name` 分布符合配置权重；
5. 同 source rank 的 `episode_id/frame_indices` 不重复；
6. 异构 source 解码耗时是否形成明显 straggler。
