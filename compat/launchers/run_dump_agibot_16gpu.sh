#!/bin/bash
# 2 pod × 8 GPU AgiBot DA3 dump. No NCCL. Same jobs.json on shared /data.
#
# Pod-0 (this host, hostname * -0 *):
#   bash scripts/run_dump_agibot_16gpu.sh 0
# Pod-1:
#   bash scripts/run_dump_agibot_16gpu.sh 1
set -euo pipefail
NODE="${1:?usage: $0 <0|1>}"
if [[ "$NODE" != "0" && "$NODE" != "1" ]]; then
  echo "NODE must be 0 or 1" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/home/fengodchen/miniconda3/envs/motus/bin/python}"
OUT="${OUT:-/data/xiongwei.zhao/agibot_da3_depth/v1}"
LOGDIR="${LOGDIR:-$OUT/logs_round2}"
mkdir -p "$LOGDIR"

echo "node=$NODE host=$(hostname) py=$PY out=$OUT"
for local in 0 1 2 3 4 5 6 7; do
  shard=$((NODE * 8 + local))
  echo "launch shard ${shard}/16 on cuda:${local}"
  CUDA_VISIBLE_DEVICES=$local "$PY" scripts/dump_agibot_da3_depth.py \
    --shard "${shard}/16" \
    >"$LOGDIR/shard${shard}.log" 2>&1 &
  echo $! >"$LOGDIR/shard${shard}.pid"
done
echo "started 8 workers. logs: $LOGDIR/shard*.log"
echo "tail -f $LOGDIR/shard$((NODE * 8)).log"
wait
echo "node=$NODE all 8 shards finished"
