#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${MOTUS_FULL_CATALOG_ROOT:-/data/jiyuan.wang/EpisodeCatalogs/_20260727/ten_source_full_v1}"
LOG_ROOT="${MOTUS_FULL_CATALOG_LOG_ROOT:-${ROOT}/outputs/ten_source_catalog_full_20260727}"
MANIFEST_ROOT="/data/jiyuan.wang/Manifests/_0713/holdout_1pct_seed20260713/manifests"
DATA_VERSION="20260727_full_v1"
TARGET="${1:-all}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

catalog_path() {
  printf '%s/%s.%s.episode_catalog.jsonl' "${OUT_ROOT}" "$1" "${DATA_VERSION}"
}

prepare_child() {
  local child="$1"
  local output
  output="$(catalog_path "${child}")"
  local base="${output%.jsonl}"
  local metadata="${base}.meta.json"
  local index="${base}.index.sqlite"
  if [[ -f "${output}" && -f "${metadata}" && -f "${index}" ]]; then
    echo "CATALOG_CHILD_SKIP child=${child} reason=complete output=${output}"
    return 1
  fi
  if [[ -f "${output}" && -f "${metadata}" && ! -e "${index}" ]]; then
    echo "CATALOG_CHILD_INDEX_ONLY child=${child} output=${output}"
    python "${ROOT}/tools/build_catalog_index.py" --catalog "${output}"
    echo "CATALOG_CHILD_DONE child=${child} output=${output}"
    return 1
  fi
  if [[ -e "${output}" || -e "${metadata}" || -e "${index}" ]]; then
    echo "CATALOG_CHILD_INVALID_PARTIAL child=${child} output=${output}" >&2
    echo "Expected no artifacts, catalog+metadata, or all three artifacts." >&2
    exit 1
  fi
  return 0
}

build_agibot() {
  local child="agibot"
  local output
  output="$(catalog_path "${child}")"
  prepare_child "${child}" || return 0
  echo "CATALOG_CHILD_START child=${child} output=${output}"
  python "${ROOT}/tools/build_episode_catalog.py" \
    --family agibot \
    --output "${output}" \
    --catalog-id "agibot_${DATA_VERSION}" \
    --data-version "${DATA_VERSION}" \
    --source-child "${child}" \
    --annotation-folders /code1/llm_team/chen.liao/Motus/datasets/agibot/new_task_info \
    --data-root-folders /code1/llm_team/chen.liao/Motus/datasets/agibot \
    --dataset-names agibot-world
  echo "CATALOG_CHILD_DONE child=${child} output=${output}"
}

build_manifest_child() {
  local family="$1"
  local child="$2"
  local manifest="$3"
  local output
  output="$(catalog_path "${child}")"
  prepare_child "${child}" || return 0
  echo "CATALOG_CHILD_START child=${child} output=${output}"
  python "${ROOT}/tools/build_episode_catalog.py" \
    --family "${family}" \
    --output "${output}" \
    --catalog-id "${child}_${DATA_VERSION}" \
    --data-version "${DATA_VERSION}" \
    --source-child "${child}" \
    --manifest "${manifest}"
  echo "CATALOG_CHILD_DONE child=${child} output=${output}"
}

run_child() {
  local child="$1"
  case "${child}" in
    agibot)
      build_agibot
      ;;
    robotwin_aloha|robotwin_ur5|robotwin_arx|robotwin_piper)
      build_manifest_child robotwin "${child}" \
        "${MANIFEST_ROOT}/${child}_manifest.holdout1pct_seed20260713.motus_manifest.jsonl"
      ;;
    robotmind_tianyi)
      build_manifest_child robotmind "${child}" \
        "${MANIFEST_ROOT}/robotmind_tianyi_manifest.holdout1pct_seed20260713.motus_manifest.jsonl"
      ;;
    robotmind_arkmobile)
      build_manifest_child robotmind "${child}" \
        "${MANIFEST_ROOT}/arkmobile_full_manifest_with_test.jsonl"
      ;;
    robotmind_ark)
      build_manifest_child robotmind "${child}" \
        "${MANIFEST_ROOT}/robotmind_ark_manifest.jsonl"
      ;;
    robotmind_tienkung)
      build_manifest_child robotmind "${child}" \
        "${MANIFEST_ROOT}/robotmind_tienkung_manifest.jsonl"
      ;;
    *)
      echo "Unknown child: ${child}" >&2
      exit 2
      ;;
  esac
}

CHILDREN=(
  agibot
  robotwin_aloha
  robotwin_ur5
  robotwin_arx
  robotwin_piper
  robotmind_tianyi
  robotmind_arkmobile
  robotmind_ark
  robotmind_tienkung
)

if [[ "${TARGET}" == "all" ]]; then
  for child in "${CHILDREN[@]}"; do
    child_log="${LOG_ROOT}/${child}.${DATA_VERSION}.out"
    echo "CATALOG_CHILD_LOG child=${child} log=${child_log}"
    run_child "${child}" 2>&1 | tee "${child_log}"
  done
  catalog_args=()
  for child in "${CHILDREN[@]}"; do
    catalog_args+=(--catalog "$(catalog_path "${child}")")
  done
  python "${ROOT}/tools/audit_episode_catalogs.py" \
    --expected-count 9 \
    "${catalog_args[@]}"
  echo "FULL_NINE_CATALOG_BUILD_OK output=${OUT_ROOT}"
else
  run_child "${TARGET}"
  python "${ROOT}/tools/audit_episode_catalogs.py" \
    --expected-count 1 \
    --catalog "$(catalog_path "${TARGET}")"
  echo "FULL_CHILD_CATALOG_BUILD_OK child=${TARGET} output=${OUT_ROOT}"
fi
