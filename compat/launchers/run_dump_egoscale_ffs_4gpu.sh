#!/usr/bin/env bash
# Launch 4 EgoScale FFS dump workers. Do not steal DA3 GPUs unless those
# workers are stopped first. Default cards: 4 5 6 7.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/fengodchen/miniconda3/envs/motus/bin/python}"
OUT="${OUT:-/data/xiongwei.zhao/egoscale_ffs_depth/v1}"
GPUS="${GPUS:-4,5,6,7}"
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
IFS=',' read -r -a cards <<< "$GPUS"
n="${#cards[@]}"
: > "$OUT/run.pids"
i=0
for gpu in "${cards[@]}"; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$ROOT/scripts/dump_egoscale_ffs_depth.py" --shard "$i/$n" \
    >> "$OUT/shard_${i}.log" 2>&1 &
  echo $! >> "$OUT/run.pids"
  echo "started shard $i/$n on gpu $gpu pid $!"
  i=$((i + 1))
done
echo "logs $OUT/shard_*.log"
