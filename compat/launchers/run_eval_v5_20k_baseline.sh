#!/usr/bin/env bash
# V5 step 20000 family-holdout 18, with motion/background AbsRel.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${PY:-/home/fengodchen/miniconda3/envs/motus/bin/python}"
CKPT="${CKPT:-/data/xiongwei.zhao/thinkerwam-depth/checkpoints/depth_ablation_treatment_v5/depth_ablation_treatment_v5/depth_ablation_treatment_v5/depth_ablation_treatment_v5/checkpoint_step_20000}"
RUNDIR="${RUNDIR:-/data/xiongwei.zhao/thinkerwam-depth/motus_monitor_data/runs/depth_ablation_v5_step20k_motion_baseline}"
LOG_ROOT="${LOG_ROOT:-/data/xiongwei.zhao/thinkerwam-depth/outputs/eval_family_holdout18/logs/v5_20k_baseline}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$LOG_ROOT" "$RUNDIR"

echo "ckpt=$CKPT"
echo "rundir=$RUNDIR"
pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  log="$LOG_ROOT/rank${gpu}.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$REPO/scripts/eval_family_holdout_rank.py" \
    --config "$REPO/configs/depth_ablation_treatment_v5.yaml" \
    --run-dir "$RUNDIR" \
    --rank "$gpu" \
    --world-size 8 \
    --device cuda:0 \
    --checkpoints "$CKPT" \
    --steps 20000 \
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
  --gather --run-dir "$RUNDIR" --step 20000 --world-size 8 \
  | tee "$LOG_ROOT/step20000_round.json"
echo "BASELINE_DONE"
