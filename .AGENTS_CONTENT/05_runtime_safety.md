# 运行时安全与诊断

## 环境变量

- `MOTUS_NONFINITE_FAIL_FAST=1` —— 跨 rank 检查损失/梯度/参数，写诊断快速失败。全量参数/梯度扫描（4 处逐 tensor isfinite + 跨 rank allreduce）按 `diagnostics.nonfinite_scan_interval` 降频（默认 1；生产 50），与本开关绑定——置 0 关闭扫描；每步 loss 标量检查不受影响。
- `MOTUS_TRAIN_OUTLIER_LOSS_ZERO_THRESHOLD`（默认 50，<=0 关闭）—— 训练侧离群样本 loss 置零（20260820 起语义改为**统一 delta-EEF 空间**，不再检查 raw |action|/|state| 绝对值）：分两阶段——`prepare_training_context` 前 `_sanitize_nonfinite_batch` 对 raw actions/state 做逐样本非有限防御扫描 + nan_to_num（摘除维 NaN 传播不到统一张量、后置检查兜不住，必须前置）；context 后 `_zero_outlier_sample_loss_weights` 按统一 action 张量 [B,T,U]（pos=delta 物理量、rot6/夹爪∈[-1,1]，合法值远低于 1，50 即数据损坏级离群）样本级 abs max 检测并置零其 action/video loss 权重。计数入 `outlier_loss_zeroed_samples`；离群 episode id 跨 rank 汇总入 step_record `outlier_episodes` 并写 monitor 事件。配套：关掉库内两个 guard，避免坏样本 raise 杀死 ready-queue producer 拖挂训练：`MOTUS_ACTION_MAGNITUDE_GUARD_THRESHOLD=0` + `MOTUS_ACTION_FINITE_GUARD_ENABLED=0`。
- `MOTUS_SKIP_BACKWARD`、`MOTUS_SKIP_OPTIMIZER_WHEN_LR_ZERO` —— 调试开关（通常不设）。
- `MOTUS_FFMPEG_BIN` —— ffmpeg 路径（缺省集群内置静态 ffmpeg；本机无 ffmpeg 时 `data_prepare` 回退 imageio_ffmpeg 自带二进制）。

## 离群保护与诊断

动作/状态离群保护由 motus_data 按 release 执行（`releases.yaml` 的 `action_magnitude_guard_threshold/sources`）；`MOTUS_ACTION_MAGNITUDE_GUARD_*` 与 `dataset_timing/` 已随旧 `data/` 包移除。

诊断输出写本次运行的 checkpoint 目录：`step_timing/`、`action_diagnostics/`、`startup_timing.json`。关键日志字段：`Time`、`QueueWait`、`QueueSize`、`QueueEmpty`、`Source`、`Seen`（step_record 另含 `mix_source_name` 等 mix_* 字段）。
