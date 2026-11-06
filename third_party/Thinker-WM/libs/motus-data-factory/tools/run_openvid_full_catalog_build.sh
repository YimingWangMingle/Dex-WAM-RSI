#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${OPENVID_DATA_ROOT:-/data/OpenVid}"
CSV_PATH="${OPENVID_CSV_PATH:-${DATA_ROOT}/data/train/OpenVid-1M.csv}"
OUTPUT_ROOT="${OPENVID_CATALOG_ROOT:-/data/jiyuan.wang/EpisodeCatalogs/_20260730/openvid}"
CATALOG="${OUTPUT_ROOT}/openvid_1m_full_20260730.episode_catalog.jsonl"
METADATA="${OUTPUT_ROOT}/openvid_1m_full_20260730.episode_catalog.meta.json"
INDEX="${OUTPUT_ROOT}/openvid_1m_full_20260730.episode_catalog.index.sqlite"

mkdir -p "${OUTPUT_ROOT}"
exec 9>"${OUTPUT_ROOT}/.openvid_full_catalog_build.lock"
if ! flock -n 9; then
  echo "another OpenVid Catalog build is already running" >&2
  exit 1
fi

test -d "${DATA_ROOT}"
test -f "${CSV_PATH}"

if { test -f "${CATALOG}" && ! test -f "${METADATA}"; } \
  || { ! test -f "${CATALOG}" && test -f "${METADATA}"; }; then
  echo "inconsistent final Catalog state: ${CATALOG} ${METADATA}" >&2
  exit 1
fi

if ! test -f "${CATALOG}"; then
  env PYTHONPATH="${PROJECT_ROOT}" "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/tools/build_episode_catalog.py" \
    --family openvid \
    --data-root "${DATA_ROOT}" \
    --csv-path "${CSV_PATH}" \
    --output "${CATALOG}" \
    --catalog-id openvid_1m_full_20260730 \
    --data-version OpenVid-1M_20260730 \
    --source-child openvid \
    --resume \
    --checkpoint-interval 10000 \
    --no-build-index
else
  echo "Catalog already complete: ${CATALOG}"
fi

if ! test -f "${INDEX}"; then
  env PYTHONPATH="${PROJECT_ROOT}" "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/tools/build_catalog_index.py" \
    --catalog "${CATALOG}"
else
  echo "Catalog index already exists: ${INDEX}"
fi

echo "OPENVID_RELEASE_READY catalog=${CATALOG} metadata=${METADATA} index=${INDEX}"
