#!/usr/bin/env bash
# H200 上为 stage2_20260812_v2（变 FPS 训练轮，20260813 起并入 v1 五族 17 源
# 混合）构建生产全量 ranges：17 源 × train/holdout（holdout 1%），catalog 与
# split seed 同 v1/0807。真实源统一 --video-stride --action-fps-ratio 6
# --max-stride 10（1:6 video_stride 布局 + 时长倍率 k∈[1,10]，max_stride 列
# 建库预计算；纯视频源 openvid 自 20260813 起支持 video_stride，且不受
# ratio 6 烘焙约束、用户拍板 ratio 1——49 帧窗口不再大面积退化）。
# 窗口过滤参数同 v1：agibot_beta 两 profile 与 libero 用 guard 20 + 静止
# eps 0.02/K=12 + 离群 20；egoverse（raw 未标定）与 openvid（纯视频）不过滤。
# 注意：robotwin×4 的 ranges 已按同参数先行构建（断点续建自动跳过）；
# libero 短 episode 在 289 帧窗口下多数只有 s=0 单起点，yaml 侧用
# windows_per_episode: 16 补齐采样位（用户 20260813 拍板）。
# 用法（H200，motus311 环境；全量构建耗时较长，建议 nohup/后台）：
#   bash scripts/build_full_ranges_20260812_v2.sh
# 已存在的输出默认跳过（断点续建）；FORCE=1 强制重建：
#   FORCE=1 bash scripts/build_full_ranges_20260812_v2.sh
set -euo pipefail

REPO=/code1/llm_team/chen.liao/ThinkerWM/thinkerwm-pretrain
CAT=/data/jiyuan.wang/EpisodeCatalogs
OUT=/data/chen.liao/thinkerwm-pretrain/training_dataset_config/sampling_ranges/stage2_20260812_v2
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

# Agibot-Beta effector profile×2（20260805 拆分 release；窗口过滤同 v1：
# 静止 eps 0.02/K=12，guard/离群阈值对齐新 profile release guard 20）
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

# Libero×4（z 空间窗口过滤同 agibot，guard 20 对齐 release；catalog
# 20260810 libero_parquet_v1）
for n in libero_10 libero_goal libero_object libero_spatial; do
  for split in train holdout; do
    build_one "$OUT/${n}.${split}.ranges.sqlite" \
      --catalog "$CAT/_20260810/libero_parquet_v1/${n}.libero_parquet_v1.episode_catalog.jsonl" \
      --dataset libero --num-video-frames 48 \
      --video-stride --action-fps-ratio 6 --max-stride 10 \
      --action-guard-threshold 20 \
      --window-static-eps 0.02 --window-max-static-frames 12 --window-outlier-threshold 20 \
      --workers 16 \
      --split "$split" --holdout-fraction 0.01
  done
done

# OpenVid（纯视频源：不支持窗口过滤；空 caption episode 构建期剔除；
# 20260813 起支持 video_stride/max_stride 变倍率抽帧。不吃 action 分支、
# 不受 ratio 6 烘焙约束，用户 20260813 拍板 ratio 1：49 帧窗口，p50 166
# 帧不再退化，有效 FPS = 原生/k ∈ [3,30]）
for split in train holdout; do
  build_one "$OUT/openvid.${split}.ranges.sqlite" \
    --catalog "$CAT/_20260730/openvid/openvid_1m_full_20260730.episode_catalog.jsonl" \
    --dataset openvid --num-video-frames 48 \
    --video-stride --action-fps-ratio 1 --max-stride 10 \
    --workers 16 \
    --split "$split" --holdout-fraction 0.01
done

# robotwin×4（catalog 20260727 ten_source_full_v1；静止 eps 0.02/K=12 +
# 离群 50 窗口过滤同 0807）
for n in robotwin_aloha robotwin_ur5 robotwin_arx robotwin_piper; do
  for split in train holdout; do
    build_one "$OUT/${n}.${split}.ranges.sqlite" \
      --catalog "$CAT/_20260727/ten_source_full_v1/${n}.20260727_full_v1.episode_catalog.jsonl" \
      --dataset "$n" --num-video-frames 48 \
      --video-stride --action-fps-ratio 6 --max-stride 10 \
      --window-static-eps 0.02 --window-max-static-frames 12 --window-outlier-threshold 50 \
      --workers 16 \
      --split "$split" --holdout-fraction 0.01
  done
done

echo "FULL_RANGES_BUILD_OK -> $OUT"
ls "$OUT"
