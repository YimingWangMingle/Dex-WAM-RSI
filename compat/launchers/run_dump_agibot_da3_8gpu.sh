#!/usr/bin/env bash
set -euo pipefail
ROOT="/code1/zxw_data/pretrain/ThinkerWAM-Depth"
PY="/home/fengodchen/miniconda3/envs/motus/bin/python"
OUT="/data/xiongwei.zhao/agibot_da3_depth/v1"
LOG="${OUT}/logs"
mkdir -p "${LOG}"
cd "${ROOT}"
if [[ ! -f "${OUT}/jobs.json" ]]; then
  "${PY}" -u scripts/dump_agibot_da3_depth.py --make-jobs
fi
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
: > "${OUT}/run.pids"
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="${i}" nohup "${PY}" -u scripts/dump_agibot_da3_depth.py --shard "${i}/8" \
    >> "${LOG}/shard${i}.log" 2>&1 &
  echo $! >> "${OUT}/run.pids"
  echo "started shard ${i} pid $!"
done
echo "pids: $(tr '\n' ' ' < "${OUT}/run.pids")"
echo "status: ${PY} ${ROOT}/scripts/dump_agibot_da3_depth.py --status"
