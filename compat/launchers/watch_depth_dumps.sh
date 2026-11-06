#!/usr/bin/env bash
# Restart dead dump workers without stealing the other job's GPUs.
# AgiBot: shards 0-3 on GPUs 0-3 only.
# EgoScale: shards 0-3 on GPUs 4-7.
set -u
ROOT="/code1/zxw_data/pretrain/ThinkerWAM-Depth"
PY="/home/fengodchen/miniconda3/envs/motus/bin/python"
AGI_OUT="/data/xiongwei.zhao/agibot_da3_depth/v1"
EGO_OUT="/data/xiongwei.zhao/egoscale_ffs_depth/v1"
LOG="${AGI_OUT}/watch.log"
mkdir -p "$AGI_OUT" "$EGO_OUT" "${AGI_OUT}/logs"
ts() { date -Iseconds; }
alive() { kill -0 "$1" 2>/dev/null; }

restart_agi_shard() {
  local i="$1"
  CUDA_VISIBLE_DEVICES="$i" nohup "$PY" -u "$ROOT/scripts/dump_agibot_da3_depth.py" --shard "${i}/8" \
    >> "${AGI_OUT}/logs/shard${i}.log" 2>&1 &
  local pid=$!
  echo "$pid" >> "${AGI_OUT}/run.pids"
  echo "$(ts) RESTART agibot shard ${i}/8 gpu=${i} pid=${pid}" | tee -a "$LOG"
}

restart_ego_shard() {
  local i="$1"
  local gpu=$((4 + i))
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" -u "$ROOT/scripts/dump_egoscale_ffs_depth.py" --shard "${i}/4" \
    >> "${EGO_OUT}/shard_${i}.log" 2>&1 &
  local pid=$!
  echo "$pid" >> "${EGO_OUT}/run.pids"
  echo "$(ts) RESTART egoscale shard ${i}/4 gpu=${gpu} pid=${pid}" | tee -a "$LOG"
}

check_pool() {
  local name="$1" out="$2"
  shift 2
  local expected=("$@")
  local pids=()
  if [[ -f "${out}/run.pids" ]]; then
    mapfile -t pids < "${out}/run.pids"
  fi
  local live=0
  for p in "${pids[@]}"; do
    [[ -z "$p" ]] && continue
    if alive "$p"; then live=$((live + 1)); fi
  done
  echo "$(ts) CHECK ${name} live=${live} expected=${#expected[@]}" >> "$LOG"
  if [[ "$name" == agibot ]]; then
    for i in "${expected[@]}"; do
      if ! pgrep -f "dump_agibot_da3_depth.py --shard ${i}/8" >/dev/null; then
        restart_agi_shard "$i"
      fi
    done
  else
    for i in "${expected[@]}"; do
      if ! pgrep -f "dump_egoscale_ffs_depth.py --shard ${i}/4" >/dev/null; then
        restart_ego_shard "$i"
      fi
    done
  fi
}

if [[ -f "${AGI_OUT}/STOP" || -f "${EGO_OUT}/STOP" ]]; then
  echo "$(ts) STOP file present, not restarting" | tee -a "$LOG"
  exit 0
fi

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
check_pool agibot "$AGI_OUT" 0 1 2 3
check_pool egoscale "$EGO_OUT" 0 1 2 3
