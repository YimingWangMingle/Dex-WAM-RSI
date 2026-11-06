#!/usr/bin/env bash
# 为 stage2_20260815_v1（固定 5fps 轮，k=1 旧路径）重建 libero×4 的 5fps ranges：
#   1) 0815_v1 固定 5fps 机制下，30fps 源经 --action-fps-ratio 6 精确得 5fps、
#      ranges 沿用 stage2_20260812_v2 目录无需重建；libero 原生 10Hz，改 5fps
#      需 --action-fps-ratio 2（10/2=5），故单独重建到新目录
#      stage2_20260815_v1（0812_v2 目录里的 libero ranges 是 ratio 6 变 FPS 布局，
#      不能混用）。
#   2) 布局/过滤参数与 0812_v2 libero 完全一致：--video-stride（49 帧跨 2*48=96
#      原生步、动作 96 步）+ --max-stride 10（max_stride 列与现有 ranges 结构一致；
#      5fps 轮 yaml 不写 pool.stride_max，该列无害残留，03_config.md 已注明）+ 窗口
#      过滤（guard 20 对齐 release、静止 eps 0.02/K=12、离群 20）。
#   3) libero 短 episode（p50 259 帧）在 97 帧窗口下多数只有 s=0 单起点；本脚本
#      只建 ranges，采样位补齐（windows_per_episode: 16）属 yaml 侧配置，加回训练
#      时由 stage2_20260815_v1.yaml 的 source 条目携带（当前 yaml 尚未加回）。
# 用法（H200，motus311 环境；全量构建耗时较长，建议 nohup/后台）：
#   bash scripts/build_libero_ranges_20260815_v1.sh
# 已存在的输出默认跳过（断点续建）；FORCE=1 强制重建：
#   FORCE=1 bash scripts/build_libero_ranges_20260815_v1.sh
set -euo pipefail

REPO=/code1/llm_team/chen.liao/ThinkerWM/thinkerwm-pretrain
CAT=/data/jiyuan.wang/EpisodeCatalogs
OUT=/data/chen.liao/thinkerwm-pretrain/training_dataset_config/sampling_ranges/stage2_20260815_v1
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

# Libero×4（z 空间窗口过滤同 agibot，guard 20 对齐 release；catalog
# 20260810 libero_parquet_v1）。ratio 2 = 原生 10Hz → 5fps。
for n in libero_10 libero_goal libero_object libero_spatial; do
  for split in train holdout; do
    build_one "$OUT/${n}.${split}.ranges.sqlite" \
      --catalog "$CAT/_20260810/libero_parquet_v1/${n}.libero_parquet_v1.episode_catalog.jsonl" \
      --dataset libero --num-video-frames 48 \
      --video-stride --action-fps-ratio 2 --max-stride 10 \
      --action-guard-threshold 20 \
      --window-static-eps 0.02 --window-max-static-frames 12 --window-outlier-threshold 20 \
      --workers 16 \
      --split "$split" --holdout-fraction 0.01
  done
done

echo "LIBERO_RANGES_BUILD_OK -> $OUT"
ls "$OUT"
