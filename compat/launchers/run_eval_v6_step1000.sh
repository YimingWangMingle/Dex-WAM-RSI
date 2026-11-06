#!/usr/bin/env bash
# V6 step 1000 family-holdout 18, same 8-GPU split as V5 baseline.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${PY:-/home/fengodchen/miniconda3/envs/motus/bin/python}"
CKPT="${CKPT:-/data/xiongwei.zhao/thinkerwam-depth/checkpoints/depth_ablation_treatment_v6/depth_ablation_treatment_v6/depth_ablation_treatment_v6/checkpoint_step_1000}"
RUNDIR="${RUNDIR:-/data/xiongwei.zhao/thinkerwam-depth/motus_monitor_data/runs/depth_ablation_v6_step1000_family_holdout}"
LOG_ROOT="${LOG_ROOT:-/data/xiongwei.zhao/thinkerwam-depth/outputs/eval_family_holdout18/logs/v6_step1000}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$LOG_ROOT" "$RUNDIR"

echo "ckpt=$CKPT"
echo "rundir=$RUNDIR"
pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  log="$LOG_ROOT/rank${gpu}.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$REPO/scripts/eval_family_holdout_rank.py" \
    --config "$REPO/configs/depth_ablation_treatment_v6.yaml" \
    --run-dir "$RUNDIR" \
    --rank "$gpu" \
    --world-size 8 \
    --device cuda:0 \
    --checkpoints "$CKPT" \
    --steps 1000 \
    >"$log" 2>&1 &
  pids+=("$!")
  echo "launched rank $gpu pid ${pids[-1]} log=$log"
done

rc=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "rank pid $pid failed" >&2
    rc=1
  fi
done
if [[ $rc -ne 0 ]]; then
  echo "eval had rank failures" >&2
  exit "$rc"
fi

"$PY" -u "$REPO/scripts/eval_family_holdout_rank.py" \
  --gather --run-dir "$RUNDIR" --step 1000 --world-size 8 \
  | tee "$LOG_ROOT/step1000_round.json"
echo "V6_STEP1000_DONE"
