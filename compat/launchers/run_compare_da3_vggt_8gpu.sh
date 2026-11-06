#!/usr/bin/env bash
set -euo pipefail
ROOT="/code1/zxw_data/pretrain/ThinkerWAM-Depth"
PY="/home/fengodchen/miniconda3/envs/motus/bin/python"
OUT="/data/xiongwei.zhao/agibot_da3_depth/da3_vs_vggt"
LOG="${OUT}/logs"
mkdir -p "${LOG}"
cd "${ROOT}"
"${PY}" -u scripts/compare_da3_vggt.py --make-jobs --n-jobs 8
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="${i}" "${PY}" -u scripts/compare_da3_vggt.py --shard "${i}/8" \
    > "${LOG}/shard${i}.log" 2>&1 &
  echo "started shard ${i} pid $!"
done
wait
"${PY}" -u scripts/compare_da3_vggt.py --reduce | tee "${OUT}/summary.txt"
echo DONE
