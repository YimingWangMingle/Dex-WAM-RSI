#!/usr/bin/env bash
# RoboDojo Joint24 v1 训练入口（configs/robodojo_joint24_v1.yaml），每个节点各执行一次。
# 16 卡 = 2 节点 × 8 卡；梯度走 Pod IP + IB，不走 SSH 隧道端口。
#
#   RUN_NAME=RoboDojo_joint24_v1_20261023_16gpu NODE_RANK=0 MASTER_ADDR=<rank0 Pod IP> \
#     bash scripts/launch_robodojo_joint24_v1.sh
#
# 首次启动：按 YAML finetune.checkpoint_path 从 rsi_robodojo_v1@10000 只加载权重。
# 本运行中断后续训：同一 RUN_NAME 加 RESUME_CHECKPOINT=<本运行的 checkpoint_step_N>，
# 恢复模型/优化器/调度器/RNG/数据位置；只接受本配置自己的输出目录，绝不续旧 AR 运行。
# save/val 间隔与步数只由 YAML 决定（2000/2000/20000），拒绝环境变量覆盖。
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT/third_party/Thinker-WM/configs/robodojo_joint24_v1.yaml"
ENV_BIN=/home/fengodchen/miniconda3/envs/motus/bin
OUT_BASE=/data/xiongwei.zhao/thinkerwam-depth/outputs
CKPT_BASE=/data/xiongwei.zhao/thinkerwam-depth/checkpoints/robodojo_joint24_v1/robodojo_joint24_v1

: "${RUN_NAME:?set RUN_NAME identically on every node}"
: "${NODE_RANK:?set NODE_RANK (0..NNODES-1)}"
: "${MASTER_ADDR:?set MASTER_ADDR to the rank-0 Pod IP}"
NNODES=${NNODES:-2}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
RDZV_PORT=${RDZV_PORT:-29624}

for name in SAVE_INTERVAL VAL_INTERVAL MAX_STEPS LOG_INTERVAL FINETUNE_CHECKPOINT CHECKPOINT_DIR; do
  if [[ -n "${!name:-}" ]]; then
    echo "ERROR: $name override is not allowed; edit $CONFIG instead" >&2
    exit 1
  fi
done
if [[ "${MOTUS_FP8_LINEAR:-0}" != "0" ]]; then
  echo "ERROR: this recipe keeps the BF16 backbone; unset MOTUS_FP8_LINEAR" >&2
  exit 1
fi
if [[ $((NNODES * NPROC_PER_NODE)) -ne 16 ]]; then
  echo "WARNING: world size $((NNODES * NPROC_PER_NODE)) differs from the 16-GPU target recipe" >&2
fi

if [[ ! "$RUN_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
  echo "ERROR: RUN_NAME must be a simple directory name" >&2
  exit 1
fi
if [[ ! "$NODE_RANK" =~ ^[0-9]+$ ]] || (( NODE_RANK >= NNODES )); then
  echo "ERROR: NODE_RANK must be in 0..NNODES-1" >&2
  exit 1
fi
OUT_DIR="$OUT_BASE/$RUN_NAME"
mkdir -p "$OUT_DIR"
# Keep this fd open across exec: a live node cannot be launched twice.
exec 9>"$OUT_DIR/.launch_node${NODE_RANK}.lock"
flock -n 9 || { echo "ERROR: this run/node is already active" >&2; exit 1; }
GUARD_ARGS=()
RESUME_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  GUARD_ARGS=(--resume "$RESUME_CHECKPOINT")
  RESUME_ARGS=(--resume_checkpoint "$RESUME_CHECKPOINT")
fi
"$ENV_BIN/python" -B "$ROOT/third_party/Thinker-WM/utils/robodojo_launch_guard.py" \
  --config "$CONFIG" --output "$OUT_DIR" --checkpoints "$CKPT_BASE/$RUN_NAME" \
  --node-rank "$NODE_RANK" "${GUARD_ARGS[@]}"
LOG="$OUT_DIR/train_node${NODE_RANK}.log"

export MOTUS_TEMPORAL_ATTN=flex
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$ROOT:$ROOT/third_party/Thinker-WM/libs/motus-data-factory:$ROOT/third_party/Thinker-WM/libs/motus-monitor"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
# SSH sessions do not inherit the Job's NCCL env; without NVLS=0 NCCL init fails with 'invalid argument'.
export NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 NCCL_IB_TIMEOUT=22 NCCL_IB_RETRY_CNT=7
export NCCL_IB_GID_INDEX=3 NCCL_IB_ROCE_VERSION_NUM=2 NCCL_IB_CUDA_SUPPORT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$ROOT"
echo "run=$RUN_NAME node=$NODE_RANK/$NNODES master=$MASTER_ADDR:$RDZV_PORT config=$CONFIG resume=${RESUME_CHECKPOINT:-none} log=$LOG"
exec "$ENV_BIN/torchrun" \
  --nnodes="$NNODES" --nproc_per_node="$NPROC_PER_NODE" --node_rank="$NODE_RANK" \
  --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$RDZV_PORT" --rdzv_id="$RUN_NAME" \
  --max_restarts=0 \
  --module thinker_wm.train.train --config "$CONFIG" --deepspeed "$ROOT/third_party/Thinker-WM/configs/zero1.json" \
  --run_name "$RUN_NAME" --report_to tensorboard "${RESUME_ARGS[@]}" \
  >>"$LOG" 2>&1
