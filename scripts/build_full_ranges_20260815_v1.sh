#!/usr/bin/env bash
# H200 上为 stage2_20260815_v1 新增的 egoscale×2 源构建生产全量 ranges
# （train/holdout 各一份，holdout 1%）。旧 8 源沿用 stage2_20260812_v2 目录
# 无需重建，不在本脚本内。参数对齐 0812_v2 的 egoverse 待遇：--video-stride
# --action-fps-ratio 6 --max-stride 10（1:6 video_stride 布局；本配置不写
# stride_max → k=1，max_stride 列为无害残留、保留与 0812_v2 同格式）；
# raw 人体观测数据未标定，不加窗口过滤（同 egoverse）。
# catalog：EGOSCALE FIRST-500H release 20260816_shards_v1（datafactory 分支
# dev/20260816-datafactory-updates 接入，见 configs/stage2_20260815_v1.yaml
# 头注 6 与库内 docs/EGOSCALE_FIRST500H_INTEGRATION_ZH.md）。
# 用法（H200，motus311 环境；remaining90pct catalog 474MB/43843 episode，
# 全量构建耗时较长，建议 nohup/后台）：
#   bash scripts/build_full_ranges_20260815_v1.sh
# 已存在的输出默认跳过（断点续建）；FORCE=1 强制重建：
#   FORCE=1 bash scripts/build_full_ranges_20260815_v1.sh
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

echo "FULL_RANGES_BUILD_OK -> $OUT"
ls "$OUT"
