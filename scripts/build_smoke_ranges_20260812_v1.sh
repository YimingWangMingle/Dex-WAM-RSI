#!/usr/bin/env bash
# H200 上为 stage2_20260812_v1 的 13 个新源构建冒烟抽样 ranges（--max-episodes 200、
# 无窗口过滤、split all）。生产全量 ranges 构建命令见 configs/stage2_20260812_v1.yaml
# 头注 5（train/holdout 各一份 + 窗口过滤参数）。
# 用法（H200，motus311 环境）：
#   bash scripts/build_smoke_ranges_20260812_v1.sh
set -euo pipefail

REPO=/code1/llm_team/chen.liao/ThinkerWM/thinkerwm-pretrain
CAT=/data/jiyuan.wang/EpisodeCatalogs
OUT=/data/chen.liao/thinkerwm-pretrain/training_dataset_config/sampling_ranges/stage2_20260812_v1_smoke
PY=/home/fengodchen/miniconda3/envs/motus311/bin/python
mkdir -p "$OUT"
cd "$REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# Agibot-Beta effector profile×2（dataset 传家族名 agibot_beta：文本走
# episode_context+action_segment 特例）
for n in parallel_gripper dexterous_hand; do
  "$PY" data_prepare/build_sampling_ranges.py \
    --catalog "$CAT/_20260805/agibot_beta_effector_profiles_v1/agibot_beta_${n}.20260805_full_v1.episode_catalog.jsonl" \
    --dataset agibot_beta --num-video-frames 48 --max-episodes 200 \
    --output "$OUT/agibot_beta_${n}.ranges.sqlite" --force
done

# EgoVerse×6
for n in aria_bimanual aria_left_arm aria_right_arm eva_bimanual scale_bimanual mecka_freeform_bimanual; do
  "$PY" data_prepare/build_sampling_ranges.py \
    --catalog "$CAT/_20260730/egoverse_first6_v1/egoverse_${n}.20260730_first6_v1.episode_catalog.jsonl" \
    --dataset egoverse --num-video-frames 48 --max-episodes 200 \
    --output "$OUT/egoverse_${n}.ranges.sqlite" --force
done

# Libero×4
for n in libero_10 libero_goal libero_object libero_spatial; do
  "$PY" data_prepare/build_sampling_ranges.py \
    --catalog "$CAT/_20260810/libero_parquet_v1/${n}.libero_parquet_v1.episode_catalog.jsonl" \
    --dataset libero --num-video-frames 48 --max-episodes 200 \
    --output "$OUT/${n}.ranges.sqlite" --force
done

# OpenVid（纯视频源：state/action=0，自动识别 absent）
"$PY" data_prepare/build_sampling_ranges.py \
  --catalog "$CAT/_20260730/openvid/openvid_1m_full_20260730.episode_catalog.jsonl" \
  --dataset openvid --num-video-frames 48 --max-episodes 200 \
  --output "$OUT/openvid.ranges.sqlite" --force

echo "SMOKE_RANGES_BUILD_OK -> $OUT"
ls "$OUT"
