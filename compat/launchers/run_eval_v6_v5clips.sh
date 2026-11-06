#!/usr/bin/env bash
# Official V6 offline eval on the frozen V5 family-holdout 18 clips.
# depth_temporal_smooth_alpha follows the yaml (currently 0).
#
#   STEP=2000 bash scripts/run_eval_v6_v5clips.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${PY:-/home/fengodchen/miniconda3/envs/motus/bin/python}"
STEP="${STEP:?set STEP (e.g. STEP=2000)}"
CKPT_ROOT="${CKPT_ROOT:-/data/xiongwei.zhao/thinkerwam-depth/checkpoints/depth_ablation_treatment_v6/depth_ablation_treatment_v6/depth_ablation_treatment_v6}"
CKPT="${CKPT:-$CKPT_ROOT/checkpoint_step_${STEP}}"
RUNDIR="${RUNDIR:-/data/xiongwei.zhao/thinkerwam-depth/motus_monitor_data/runs/depth_ablation_v6_step${STEP}_v5clips}"
LOG_ROOT="${LOG_ROOT:-/data/xiongwei.zhao/thinkerwam-depth/outputs/eval_family_holdout18/logs/v6_step${STEP}_v5clips}"
CLIPS="${CLIPS:-$REPO/scripts/family_holdout_18_v5.json}"
CONFIG="${CONFIG:-$REPO/configs/depth_ablation_treatment_v6.yaml}"
# Extra eval_family_holdout_rank.py flags, e.g. EXTRA_EVAL_ARGS=--no-depth-read-by-action
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$LOG_ROOT" "$RUNDIR"

echo "step=$STEP"
echo "ckpt=$CKPT"
echo "rundir=$RUNDIR"
echo "clips=$CLIPS"
echo "extra=${EXTRA_EVAL_ARGS:-none}"
echo "config=$CONFIG (depth EMA from yaml)"
pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  log="$LOG_ROOT/rank${gpu}.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$REPO/scripts/eval_family_holdout_rank.py" \
    --config "$CONFIG" \
    --run-dir "$RUNDIR" \
    --rank "$gpu" \
    --world-size 8 \
    --device cuda:0 \
    --checkpoints "$CKPT" \
    --steps "$STEP" \
    --clips "$CLIPS" \
    $EXTRA_EVAL_ARGS \
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
  --gather --run-dir "$RUNDIR" --step "$STEP" --world-size 8 \
  | tee "$LOG_ROOT/step${STEP}_round.json"
echo "V6_STEP${STEP}_V5CLIPS_DONE"
