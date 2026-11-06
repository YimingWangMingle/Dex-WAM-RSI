#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${MOTUS_CATALOG_SMOKE_ROOT:-/data/jiyuan.wang/EpisodeCatalogs/_20260727/ten_source_smoke}"
MANIFEST_ROOT="/data/jiyuan.wang/Manifests/_0713/holdout_1pct_seed20260713/manifests"

mkdir -p "${OUT_ROOT}"

build_child() {
  local family="$1"
  local child="$2"
  local manifest="$3"
  python "${ROOT}/tools/build_episode_catalog.py" \
    --family "${family}" \
    --output "${OUT_ROOT}/${child}.smoke1.episode_catalog.jsonl" \
    --catalog-id "${child}_20260727_smoke1" \
    --data-version 20260727_smoke1 \
    --source-child "${child}" \
    --manifest "${manifest}" \
    --max-episodes 1 \
    --force
}

build_child robotwin robotwin_ur5 \
  "${MANIFEST_ROOT}/robotwin_ur5_manifest.holdout1pct_seed20260713.motus_manifest.jsonl"
build_child robotwin robotwin_arx \
  "${MANIFEST_ROOT}/robotwin_arx_manifest.holdout1pct_seed20260713.motus_manifest.jsonl"
build_child robotwin robotwin_piper \
  "${MANIFEST_ROOT}/robotwin_piper_manifest.holdout1pct_seed20260713.motus_manifest.jsonl"
build_child robotmind robotmind_arkmobile \
  "${MANIFEST_ROOT}/arkmobile_full_manifest_with_test.jsonl"
build_child robotmind robotmind_ark \
  "${MANIFEST_ROOT}/robotmind_ark_manifest.jsonl"
build_child robotmind robotmind_tienkung \
  "${MANIFEST_ROOT}/robotmind_tienkung_manifest.jsonl"

python "${ROOT}/tools/audit_episode_catalogs.py" \
  --expected-count 6 \
  --catalog "${OUT_ROOT}/robotwin_ur5.smoke1.episode_catalog.jsonl" \
  --catalog "${OUT_ROOT}/robotwin_arx.smoke1.episode_catalog.jsonl" \
  --catalog "${OUT_ROOT}/robotwin_piper.smoke1.episode_catalog.jsonl" \
  --catalog "${OUT_ROOT}/robotmind_arkmobile.smoke1.episode_catalog.jsonl" \
  --catalog "${OUT_ROOT}/robotmind_ark.smoke1.episode_catalog.jsonl" \
  --catalog "${OUT_ROOT}/robotmind_tienkung.smoke1.episode_catalog.jsonl"

echo "REMAINING_SIX_CATALOG_SMOKE_OK output=${OUT_ROOT}"
