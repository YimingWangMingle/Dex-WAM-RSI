#!/usr/bin/env bash
# H200 上为 20260819 文本变更（configs/stage2_20260818_v1.yaml 头注 4）重建
# 四族 segmented text ranges：agibot_beta×2 + egoverse×6 + egoscale×2 共 10 源
# × train/holdout（holdout 1%，split seed 与旧构建一致用缺省值）。
# 新 payload kind "segmented"：episode 级 context + 窗口区间内全部重叠片段；
# 起点域 = 片段覆盖并集（不被任何片段覆盖的起点剔除；零片段但有 context 的
# episode——egoverse aria×3/eva 仅 task——回退 fixed 不裁剪，20260819 拍板）。
# 按用户 20260819 拍板：产物落 tmp 目录（非生产 training_dataset_config），
# catalog 只读引用 /data/jiyuan.wang/EpisodeCatalogs 既有 frozen release。
# 注意：必须从含 segmented 分支的新代码运行（tmp checkout，
# /code1/llm_team/chen.liao/ThinkerWM/tmp/thinkerwm-pretrain-segtext-20260819，
# 可用 REPO=... 覆盖）；libero/openvid 不在本脚本（沿用旧目录）。
# 用法（H200，motus311 环境；egoscale remaining90pct catalog 474MB/43843
# episode，全量构建耗时较长，建议 nohup/后台）：
#   bash scripts/build_segmented_ranges_20260818_v1.sh
# 已存在的输出默认跳过（断点续建）；FORCE=1 强制重建：
#   FORCE=1 bash scripts/build_segmented_ranges_20260818_v1.sh
set -euo pipefail

REPO=${REPO:-/code1/llm_team/chen.liao/ThinkerWM/tmp/thinkerwm-pretrain-segtext-20260819}
CAT=/data/jiyuan.wang/EpisodeCatalogs
OUT=/data/chen.liao/thinkerwm-pretrain/tmp/sampling_ranges/segmented_20260818_v1
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

# Agibot-Beta effector profile×2（窗口过滤同 0812_v2：静止 eps 0.02/K=12，
# guard/离群阈值对齐 profile release guard 20；catalog 20260805_full_v1）
for n in parallel_gripper dexterous_hand; do
  for split in train holdout; do
    build_one "$OUT/agibot_beta_${n}.${split}.ranges.sqlite" \
      --catalog "$CAT/_20260805/agibot_beta_effector_profiles_v1/agibot_beta_${n}.20260805_full_v1.episode_catalog.jsonl" \
      --dataset agibot_beta --num-video-frames 48 \
      --video-stride --action-fps-ratio 6 --max-stride 10 \
      --action-guard-threshold 20 \
      --window-static-eps 0.02 --window-max-static-frames 12 --window-outlier-threshold 20 \
      --workers 16 \
      --split "$split" --holdout-fraction 0.01
  done
done

# EgoVerse×6（raw 物理量纲未标定，不加窗口过滤；catalog 20260730_first6_v1）
for n in aria_bimanual aria_left_arm aria_right_arm eva_bimanual scale_bimanual mecka_freeform_bimanual; do
  for split in train holdout; do
    build_one "$OUT/egoverse_${n}.${split}.ranges.sqlite" \
      --catalog "$CAT/_20260730/egoverse_first6_v1/egoverse_${n}.20260730_first6_v1.episode_catalog.jsonl" \
      --dataset egoverse --num-video-frames 48 \
      --video-stride --action-fps-ratio 6 --max-stride 10 \
      --split "$split" --holdout-fraction 0.01
  done
done

# EGOSCALE×2（raw 未标定人体观测，不加窗口过滤；catalog 20260816_shards_v1）
for n in shard00_of10 remaining90pct; do
  for split in train holdout; do
    build_one "$OUT/egoscale_first500h_${n/_of10/}.${split}.ranges.sqlite" \
      --catalog "$CAT/_20260816/egoscale_first500h_shards_v1/egoscale_first500h.${n}.20260816_shards_v1.episode_catalog.jsonl" \
      --dataset egoscale --num-video-frames 48 \
      --video-stride --action-fps-ratio 6 --max-stride 10 \
      --workers 16 \
      --split "$split" --holdout-fraction 0.01
  done
done

echo "SEGMENTED_RANGES_BUILD_OK -> $OUT"
ls "$OUT"
