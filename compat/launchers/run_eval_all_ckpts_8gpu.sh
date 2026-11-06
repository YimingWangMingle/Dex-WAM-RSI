#!/usr/bin/env bash
# 8-GPU family-holdout eval over V1-V5 checkpoints.
#   bash scripts/run_eval_all_ckpts_8gpu.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${PY:-/home/fengodchen/miniconda3/envs/motus/bin/python}"
CKPT_ROOT="${CKPT_ROOT:-/data/xiongwei.zhao/thinkerwam-depth/checkpoints}"
MONITOR="${MONITOR:-/data/xiongwei.zhao/thinkerwam-depth/motus_monitor_data/runs}"
LOG_ROOT="${LOG_ROOT:-/data/xiongwei.zhao/thinkerwam-depth/outputs/eval_family_holdout18/logs}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$LOG_ROOT"

V5_CKPT="$CKPT_ROOT/depth_ablation_treatment_v5/depth_ablation_treatment_v5/depth_ablation_treatment_v5/depth_ablation_treatment_v5"

run_version() {
  local ver="$1" cfg="$2" rundir="$3"
  shift 3
  local steps=() ckpts=()
  while [[ $# -gt 0 ]]; do
    steps+=("$1")
    ckpts+=("$2")
    shift 2
  done
  echo "=== $ver steps=${steps[*]} ==="
  local pids=()
  local gpu
  for gpu in 0 1 2 3 4 5 6 7; do
    local log="$LOG_ROOT/${ver}_rank${gpu}.log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$REPO/scripts/eval_family_holdout_rank.py" \
      --config "$cfg" \
      --run-dir "$rundir" \
      --rank "$gpu" \
      --world-size 8 \
      --device cuda:0 \
      --checkpoints "${ckpts[@]}" \
      --steps "${steps[@]}" \
      >"$log" 2>&1 &
    pids+=("$!")
    echo "  launched rank $gpu pid ${pids[-1]} log=$log"
  done
  local rc=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      echo "  rank pid $pid failed" >&2
      rc=1
    fi
  done
  if [[ $rc -ne 0 ]]; then
    echo "version $ver had rank failures" >&2
    return "$rc"
  fi
  local step
  for step in "${steps[@]}"; do
    "$PY" -u "$REPO/scripts/eval_family_holdout_rank.py" \
      --gather --run-dir "$rundir" --step "$step" --world-size 8 \
      | tee "$LOG_ROOT/${ver}_step${step}_round.json"
  done
}

# Occupiers yield on foreign mem, but they still burn SM until the model loads.
pkill -f 'scripts/train_fourGPU.py' || true
sleep 2

run_version v1 \
  "$REPO/configs/depth_ablation_treatment_v4.yaml" \
  "$MONITOR/depth_ablation_treatment_v1__20260824_164626" \
  2000 "$CKPT_ROOT/depth_ablation_treatment_v1/depth_ablation_treatment_v1/checkpoint_step_2000"

run_version v2 \
  "$REPO/configs/depth_ablation_treatment_v4.yaml" \
  "$MONITOR/depth_ablation_treatment_v2__20260825_164637" \
  1000 "$CKPT_ROOT/depth_ablation_treatment_v2/depth_ablation_treatment_v2/checkpoint_step_1000" \
  2000 "$CKPT_ROOT/depth_ablation_treatment_v2/depth_ablation_treatment_v2/checkpoint_step_2000"

run_version v3 \
  "$REPO/configs/depth_ablation_treatment_v4.yaml" \
  "$MONITOR/depth_ablation_treatment_v3__20260825_224801" \
  1000 "$CKPT_ROOT/depth_ablation_treatment_v3/depth_ablation_treatment_v3/checkpoint_step_1000" \
  2000 "$CKPT_ROOT/depth_ablation_treatment_v3/depth_ablation_treatment_v3/checkpoint_step_2000" \
  3000 "$CKPT_ROOT/depth_ablation_treatment_v3/depth_ablation_treatment_v3/checkpoint_step_3000"

run_version v4 \
  "$REPO/configs/depth_ablation_treatment_v4.yaml" \
  "$MONITOR/depth_ablation_treatment_v4__20260826_121927" \
  1000 "$CKPT_ROOT/depth_ablation_treatment_v4/depth_ablation_treatment_v4/checkpoint_step_1000" \
  2000 "$CKPT_ROOT/depth_ablation_treatment_v4/depth_ablation_treatment_v4/checkpoint_step_2000" \
  3000 "$CKPT_ROOT/depth_ablation_treatment_v4/depth_ablation_treatment_v4/checkpoint_step_3000"

run_version v5 \
  "$REPO/configs/depth_ablation_treatment_v5.yaml" \
  "$MONITOR/depth_ablation_treatment_v5__20260827_160011" \
  1000 "$V5_CKPT/checkpoint_step_1000" \
  2000 "$V5_CKPT/checkpoint_step_2000" \
  3000 "$V5_CKPT/checkpoint_step_3000"

echo "ALL_DONE"
