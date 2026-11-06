#!/usr/bin/env bash
# H200 上为 stage2_20260812_v1 的 13 个新源构建生产全量 ranges（train/holdout 各
# 一份，holdout 1%）。robotwin×4 复用 stage2_20260807 ranges，不在本脚本内。
# 参数依据 configs/stage2_20260812_v1.yaml 头注 5；冒烟抽样版见
# scripts/build_smoke_ranges_20260812_v1.sh。
# 用法（H200，motus311 环境；全量构建耗时较长，建议 nohup/后台）：
#   bash scripts/build_full_ranges_20260812_v1.sh
# 已存在的输出默认跳过（断点续建）；FORCE=1 强制重建：
#   FORCE=1 bash scripts/build_full_ranges_20260812_v1.sh
set -euo pipefail

REPO=/code1/llm_team/chen.liao/ThinkerWM/thinkerwm-pretrain
CAT=/data/jiyuan.wang/EpisodeCatalogs
OUT=/data/chen.liao/thinkerwm-pretrain/training_dataset_config/sampling_ranges/stage2_20260812_v1
PY=/home/fengodchen/miniconda3/envs/motus311/bin/python
mkdir -p "$OUT"
cd "$REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

build_one() {
  # $1 = 输出路径；其余参数原样传给 build_sampling_ranges.py
  local out="$1"; shift
  if [[ -f "$out" && "${FORCE:-0}" != "1" ]]; then
    echo "SKIP_EXISTS $out"
    return 0
  fi
  "$PY" data_prepare/build_sampling_ranges.py "$@" --output "$out" --force
}

# Agibot-Beta effector profile×2：沿用 0807 窗口过滤（静止 eps 0.02/K=12），
# 离群阈值对齐新 profile release guard 20；--workers 16 加速大 catalog 扫描
for n in parallel_gripper dexterous_hand; do
  for split in train holdout; do
    build_one "$OUT/agibot_beta_${n}.${split}.ranges.sqlite" \
      --catalog "$CAT/_20260805/agibot_beta_effector_profiles_v1/agibot_beta_${n}.20260805_full_v1.episode_catalog.jsonl" \
      --dataset agibot_beta --num-video-frames 48 \
      --action-guard-threshold 20 \
      --window-static-eps 0.02 --window-max-static-frames 12 --window-outlier-threshold 20 \
      --workers 16 \
      --split "$split" --holdout-fraction 0.01
  done
done

# EgoVerse×6：raw 物理量纲未标定，首轮不加窗口过滤
for n in aria_bimanual aria_left_arm aria_right_arm eva_bimanual scale_bimanual mecka_freeform_bimanual; do
  for split in train holdout; do
    build_one "$OUT/egoverse_${n}.${split}.ranges.sqlite" \
      --catalog "$CAT/_20260730/egoverse_first6_v1/egoverse_${n}.20260730_first6_v1.episode_catalog.jsonl" \
      --dataset egoverse --num-video-frames 48 \
      --split "$split" --holdout-fraction 0.01
  done
done

# Libero×4：z 空间窗口过滤同 agibot（guard 20 对齐 release）
for n in libero_10 libero_goal libero_object libero_spatial; do
  for split in train holdout; do
    build_one "$OUT/${n}.${split}.ranges.sqlite" \
      --catalog "$CAT/_20260810/libero_parquet_v1/${n}.libero_parquet_v1.episode_catalog.jsonl" \
      --dataset libero --num-video-frames 48 \
      --action-guard-threshold 20 \
      --window-static-eps 0.02 --window-max-static-frames 12 --window-outlier-threshold 20 \
      --workers 16 \
      --split "$split" --holdout-fraction 0.01
  done
done

# OpenVid（纯视频源：不支持窗口过滤；空 caption episode 构建期剔除）
for split in train holdout; do
  build_one "$OUT/openvid.${split}.ranges.sqlite" \
    --catalog "$CAT/_20260730/openvid/openvid_1m_full_20260730.episode_catalog.jsonl" \
    --dataset openvid --num-video-frames 48 \
    --split "$split" --holdout-fraction 0.01
done

echo "FULL_RANGES_BUILD_OK -> $OUT"
ls "$OUT"
