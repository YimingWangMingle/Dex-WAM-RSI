#!/usr/bin/env bash
# H200 上为 stage2_20260818_v1 加回的 openvid 源构建生产全量 ranges
# （train/holdout 各一份，holdout 1%）。其余 14 源 ranges 沿用
# stage2_20260812_v2 / stage2_20260815_v1 目录不变，不在本脚本内。
# 与 0812_v2 时代 openvid ranges 的差异（固定 5fps 轮语义，见
# configs/stage2_20260818_v1.yaml 头注）：
#   1) --action-fps-ratio 6（放弃 0812_v2 的 ratio 1 变 FPS 拍板）：30fps
#      原生经 ratio 6 精确得 5fps，与全配置其他源一致；
#   2) --min-episode-frames 289（= 48*6+1，新建库期最小时长过滤）：有效
#      长度不足一个完整无填充窗口的 episode 整条剔除（约存活 16.1%、
#      ~16.5 万条；旧行为是保留 s=0 单起点 + 末帧填充退化窗口）；
#   3) --max-stride 10 仅为与 egoverse/egoscale ranges 同格式的无害残留
#      （本配置不写 stride_max → k=1，max_stride 列保留）。
# 纯视频源：不支持窗口过滤；空 caption episode 构建期剔除（既有行为）。
# 用法（H200，motus311 环境；catalog ~102 万条，全量构建耗时较长，建议
# nohup/后台）：
#   bash scripts/build_openvid_ranges_20260818_v1.sh
# 已存在的输出默认跳过（断点续建）；FORCE=1 强制重建：
#   FORCE=1 bash scripts/build_openvid_ranges_20260818_v1.sh
set -euo pipefail

REPO=/code1/llm_team/chen.liao/ThinkerWM/thinkerwm-pretrain
CAT=/data/jiyuan.wang/EpisodeCatalogs
OUT=/data/chen.liao/thinkerwm-pretrain/training_dataset_config/sampling_ranges/stage2_20260818_v1
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

# OpenVid（纯视频源：不支持窗口过滤；空 caption episode 构建期剔除；
# ratio 6 + min 289 帧过滤，见脚本头注）
for split in train holdout; do
  build_one "$OUT/openvid.${split}.ranges.sqlite" \
    --catalog "$CAT/_20260730/openvid/openvid_1m_full_20260730.episode_catalog.jsonl" \
    --dataset openvid --num-video-frames 48 \
    --video-stride --action-fps-ratio 6 --max-stride 10 \
    --min-episode-frames 289 \
    --workers 16 \
    --split "$split" --holdout-fraction 0.01
done

echo "FULL_RANGES_BUILD_OK -> $OUT"
ls "$OUT"
