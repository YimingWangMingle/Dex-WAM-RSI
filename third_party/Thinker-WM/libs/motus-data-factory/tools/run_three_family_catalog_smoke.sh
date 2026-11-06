#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${MOTUS_CATALOG_SMOKE_ROOT:-/data/jiyuan.wang/EpisodeCatalogs/_20260727/ten_source_smoke}"
MANIFEST_ROOT="/data/jiyuan.wang/Manifests/_0713/holdout_1pct_seed20260713/manifests"

mkdir -p "${OUT_ROOT}"

python "${ROOT}/tools/build_episode_catalog.py" \
  --family agibot \
  --output "${OUT_ROOT}/agibot.smoke1.episode_catalog.jsonl" \
  --catalog-id agibot_20260727_smoke1 \
  --data-version 20260727_smoke1 \
  --source-child agibot \
  --annotation-folders /code1/llm_team/chen.liao/Motus/datasets/agibot/new_task_info \
  --data-root-folders /code1/llm_team/chen.liao/Motus/datasets/agibot \
  --dataset-names agibot-world \
  --max-episodes 1 \
  --force

python "${ROOT}/tools/build_episode_catalog.py" \
  --family robotwin \
  --output "${OUT_ROOT}/robotwin_aloha.smoke1.episode_catalog.jsonl" \
  --catalog-id robotwin_aloha_20260727_smoke1 \
  --data-version 20260727_smoke1 \
  --source-child robotwin_aloha \
  --manifest "${MANIFEST_ROOT}/robotwin_aloha_manifest.holdout1pct_seed20260713.motus_manifest.jsonl" \
  --max-episodes 1 \
  --force

python "${ROOT}/tools/build_episode_catalog.py" \
  --family robotmind \
  --output "${OUT_ROOT}/robotmind_tianyi.smoke1.episode_catalog.jsonl" \
  --catalog-id robotmind_tianyi_20260727_smoke1 \
  --data-version 20260727_smoke1 \
  --source-child robotmind_tianyi \
  --manifest "${MANIFEST_ROOT}/robotmind_tianyi_manifest.holdout1pct_seed20260713.motus_manifest.jsonl" \
  --max-episodes 1 \
  --force

python "${ROOT}/tools/audit_three_family_catalog_smoke.py" \
  --root "${OUT_ROOT}"

echo "THREE_FAMILY_CATALOG_SMOKE_OK output=${OUT_ROOT}"
