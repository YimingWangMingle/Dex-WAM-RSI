#!/usr/bin/env bash
# 8-GPU RynnWorld-4D Stage-3 inference on the encoded family-holdout 18 clips.
set -euo pipefail

PY="/home/fengodchen/miniconda3/envs/motus/bin/python"
RW="/code1/zxw_data/pretrain/world-motion-action-model/code/RynnWorld-4D-main"
OUT="/code1/zxw_data/pretrain/ThinkerWAM-Depth/tmp/rynnworld4d_holdout18"
JSON="${OUT}/holdout18.json"
CKPT="/code1/zxw_data/checkpoint/RynnWorld-4D"
MODEL="/code1/zxw_data/checkpoint/Wan-2.2"
LOG="${OUT}/logs"
INFER_OUT="${OUT}/infer"

mkdir -p "${LOG}" "${INFER_OUT}"

SFT="${CKPT}/pytorch_model/mp_rank_00_model_states.pt"
HASH="$("${PY}" - <<PY
import hashlib
print(hashlib.md5("${SFT}".encode()).hexdigest())
PY
)"
LOCAL="/tmp/sft_${HASH}.pt"
if [[ ! -f "${LOCAL}" ]]; then
  echo "pre-copy SFT -> ${LOCAL}"
  cp -f "${SFT}" "${LOCAL}"
fi

cd "${RW}"
# 18 clips, round-robin across 8 GPUs
for gpu in $(seq 0 7); do
  idxs=()
  for i in $(seq 0 17); do
    if (( i % 8 == gpu )); then
      idxs+=("${i}")
    fi
  done
  joined=$(IFS=,; echo "${idxs[*]}")
  echo "gpu=${gpu} samples=${joined}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u inference-sft.py \
    --model_path "${MODEL}" \
    --checkpoint_path "${CKPT}" \
    --json_path "${JSON}" \
    --output_dir "${INFER_OUT}" \
    --per_dataset_samples 0 \
    --max_samples 40 \
    --fusion_mode joint \
    --share_ffn False \
    --joint_start_layer 0 \
    --joint_end_layer 30 \
    --joint_every_n_layers 3 \
    --joint_frame_wise True \
    --joint_use_rope True \
    --joint_unidirectional True \
    --zero_fusion False \
    --use_ema True \
    --num_inference_steps 50 \
    --guidance_scale 1.0 \
    --keep_on_gpu True \
    --sample_indices "${joined}" \
    > "${LOG}/gpu${gpu}.log" 2>&1 &
done

wait
echo "all inference ranks finished"
