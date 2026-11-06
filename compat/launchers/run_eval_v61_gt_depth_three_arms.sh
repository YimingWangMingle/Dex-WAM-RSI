#!/usr/bin/env bash
# Three-arm action-from-depth eval on V6.1@3000, V5 family-holdout 18.
# Does not touch the live V7.1 training job. Needs 8 free GPUs.
#
#   A noderead     --no-depth-read-by-action
#   B gen-depth    --depth-read-by-action
#   C gt-depth     --depth-read-by-action --teacher-force-depth
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${PY:-/home/fengodchen/miniconda3/envs/motus/bin/python}"
STEP="${STEP:-3000}"
CKPT="${CKPT:-/data/xiongwei.zhao/thinkerwam-depth/checkpoints/depth_ablation_treatment_v6_1/depth_ablation_treatment_v6_1/depth_ablation_treatment_v6_1/checkpoint_step_3000}"
CLIPS="${CLIPS:-$REPO/scripts/family_holdout_18_v5.json}"
CONFIG="${CONFIG:-$REPO/configs/depth_ablation_treatment_v6_1.yaml}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

run_arm() {
  local name="$1"
  shift
  local rundir="/data/xiongwei.zhao/thinkerwam-depth/motus_monitor_data/runs/depth_ablation_v6_1_step${STEP}_v5clips_${name}"
  local log_root="/data/xiongwei.zhao/thinkerwam-depth/outputs/eval_family_holdout18/logs/v6_1_step${STEP}_${name}"
  mkdir -p "$log_root" "$rundir"
  echo "==== arm $name extra=$* ===="
  echo "rundir=$rundir"
  EXTRA_EVAL_ARGS="$*" \
    STEP="$STEP" CKPT="$CKPT" RUNDIR="$rundir" LOG_ROOT="$log_root" \
    CLIPS="$CLIPS" CONFIG="$CONFIG" \
    bash "$REPO/scripts/run_eval_v6_v5clips.sh"
}

run_arm noderead --no-depth-read-by-action
run_arm gendepth --depth-read-by-action
run_arm gtdepth --depth-read-by-action --teacher-force-depth
echo "THREE_ARMS_DONE"
