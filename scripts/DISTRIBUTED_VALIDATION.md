# RoboDojo 分布式 holdout

`configs/rsi_robodojo_v1.yaml` 的 `system.val_distributed: true` 将同一份
15 条 holdout × 8 个窗口槽位分配到全部 rank。16 卡时前 8 卡各 8 个，后
8 卡各 7 个；不使用训练采样器的补齐或自动跨 epoch 行为。

每卡逐窗口推理，microbatch=1。每次仍生成完整 3 块（12 帧、24 个动作），
视频 CFG=5；训练 batch size 和每 1000 步验证/保存的周期不变。
`val_num_batches: 15` 与训练 batch size=8 共同校验全局 120 个槽位，
不再表示每卡要跑 15 批。此路径目前只支持单源完整 holdout、复制模型权重
（如当前 ZeRO-1），显式拒绝 ZeRO-3。

每轮所有 rank 最多推理一个窗口，再共同交换标量和错误。没有剩余窗口的
rank 也参加通信。错误会终止本轮并由训练器记录失败；验证前已保存 checkpoint。
`system.distributed_timeout_seconds: 3600` 通过 Accelerate 的
`InitProcessGroupKwargs` 显式设置实际进程组超时。逐轮同步不等于重置已挂起
NCCL 操作的计时器；单次推理或存储卡死仍可能超时。

指标按全部窗口汇总，而非各 rank 均值的简单平均；`*_std` 表示窗口指标的
总体标准差。完整三块的动作/视频误差用于 checkpoint 比较，同时保留
`action_first_block_*`，观察部署实际执行的前 8 个动作。闭环成功率和真实
设备延迟必须使用部署的一块执行协议单独测量。

推理噪声固定为 `val_seed + 全局窗口编号`，与 rank 数量无关，评测后恢复
训练 RNG。相较旧的 batch=8 顺序取噪声，随机样本发生改变，因此旧、新协议
下的单次指标不是逐位等价；之后跨 checkpoint 使用此固定协议比较。
选中的窗口保持原样：当前 120 个槽位中有 118 个不同的时间窗口，两个重复
来自原有采样结果，不是分布式补齐产生的。此次没有改选窗口。

全部窗口出图（20261023 起）：`val_a2v` 与 `val_depth` 同时开启时，每个窗口
推理并算完原精度指标后，由负责该窗口的 rank 立即把 RGB / 深度 / A2V 对比视频
和 meta 写到 `checkpoint_dir/validation_samples_tmp/holdout-*/window_NNNN/`（共享
存储；每 rank 同时最多编码一个窗口），只把轻量条目交回 rank 0；编码失败与推理
失败一样在本轮共享。rank 0 的 `publish_validation_artifacts` 校验 120 个窗口全覆盖
且身份一致后，才把目录移入 monitor 的 `eval/step_*/revision_*/<task>/window_NNNN/`
并最后写完成记录；无论发布成败，暂存目录都会删除。旧的“每任务 1 个代表窗口”已移除。
`configs/robodojo_joint24_v1.yaml`（单未来块）每窗口只推理 1 块 24 步，
`action_first_block_*` 即完整 24 步；部署只执行前 16 步。

CPU 回归：

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=.:libs/motus-data-factory:libs/motus-monitor \
  python scripts/test_distributed_validation.py
CUDA_VISIBLE_DEVICES= PYTHONPATH=.:libs/motus-data-factory:libs/motus-monitor \
  python scripts/test_training_stop_and_timing.py
```

前者验证真实评价函数在不均匀/空分片、非主 rank 失败时的行为，以及单进程
与多进程指标一致、训练 RNG 恢复。后者还覆盖真实训练循环的分布式验证后
保存停止及验证失败保留 checkpoint。CPU 测试不证明 H200 的峰值显存或耗时；
正式启动前还需在空闲 GPU 上运行一次完整模型的验证冒烟。
