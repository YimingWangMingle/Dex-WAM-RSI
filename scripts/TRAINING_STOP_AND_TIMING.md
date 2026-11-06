# 保存后停止与训练步计时

适用于包含 `utils/training_control.py` 接入的新启动训练进程。修改磁盘代码不会改变已经运行的 Python 进程。

## 请求保存后停止

在仓库根目录执行，`--run-dir` 指向**本次运行的 checkpoint 目录**，即其中包含 `step_timing/` 和 `checkpoint_step_*` 的目录：

```bash
python scripts/request_training_stop.py --run-dir /absolute/path/to/checkpoints/config_name/run_name
```

命令写入该目录的 `.stop_requested` 后立即返回；重复执行不会覆盖已有请求。rank 0 在每步末检查请求并把决定广播给全部 rank。所有 rank 完成本步（含正在进行的验证/保存），共同保存模型、优化器、调度器与随机数状态，再退出。恰好命中定期保存或最后一步时，同一个 checkpoint 只保存一次。

确认 `stop_completed.json` 的新时间戳与 step，以及训练进程已退出。回执的 `status: checkpoint_saved` 表示 checkpoint 已完整落盘，资源清理可能仍在进行；不能把旧回执或发出请求本身当作本次停止完成。成功后删除请求文件，保留回执。保存失败则保留请求，且不产生本次成功回执；已有回执不会被当作当前保存成功。

请求只针对这个运行目录，不需要找 PID。它不会中断正在执行的 GPU 操作，也不能挽救已经崩溃或卡死的进程。不要用终止 torchrun 的信号替代此命令：launcher 可能直接结束 workers，无法保证集体保存完成。初始化完成前脚本会拒绝没有 `step_timing/` 的目录。

## 日志计时

每个正常日志周期、最后一步及停止步，在验证、保存和 GC 完成后输出：

```text
Total: ...s, Data: ...s, Prepare: ...s, Update: ...s, Validation: ...s, Checkpoint: ...s, PeriodicEval: ...s, Other: ...s
```

- `Total`：从本步开始至步末汇总点的墙钟耗时，包括下列各阶段；汇总日志自身及随后监控输出的写入时间不在内。
- `Data`：取 batch 的等待与整理，不含 `Prepare`，避免重复相加。
- `Prepare`：搬运输入、VAE/历史输入等训练准备。复用同一批 latent 的后续更新步为 0。
- `Update`：前向、反向、参数更新和指标同步，并非仅 `optimizer.step()`。
- `Validation`：holdout 验证及其同步等待。
- `Checkpoint`：本步完整 checkpoint 保存及其同步等待；最后一步的保存也计入。
- `PeriodicEval`：可选周期评测；`Other`：GC、停止请求同步及其他未分段开销。

这些是各 rank 的墙钟时间，控制台显示 rank 0；不是逐个 CUDA kernel 强制同步计时。

TensorBoard/WandB 增加 `timing/{total,data,prepare,update,validation,checkpoint,periodic_eval,other}_seconds`。`train/step_time`（WandB 的 `step_time`）从旧的参数更新阶段耗时改为完整步耗时；旧曲线比较请改看 `timing/update_seconds`。JSONL 保留原 `step_time_seconds`（更新阶段）和 `loop_total_seconds`，并增加 `wall_*_seconds` 分项供监控使用。验证失败时也写出该步截至失败的耗时和 checkpoint 开销。

CPU 双进程回归：

```bash
CUDA_VISIBLE_DEVICES= python scripts/test_training_stop_and_timing.py
```

测试执行实际 trainer 的 `train`/`save_checkpoint` 方法，以微型 CPU 模型替代大模型计算，覆盖普通完成、停止步、定期保存重合、验证中请求、保存中请求、保存失败、checkpoint 重载和计时。
