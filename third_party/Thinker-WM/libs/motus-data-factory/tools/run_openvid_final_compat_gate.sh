#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "OPENVID_FINAL_GATE_STAGE=existing_ten_source"
"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/audit_ten_source_indexed_mix.py" \
  --batch-size 2 \
  --height 64 \
  --width 64 \
  --queue-size 2 \
  --weighted-batches 4 \
  --quota-batches 1000 \
  --seed 20260730

echo "OPENVID_FINAL_GATE_STAGE=eleven_source"
"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/audit_eleven_source_indexed_mix.py" \
  --batch-size 2 \
  --height 64 \
  --width 64 \
  --queue-size 2 \
  --quota-batches 1000 \
  --seed 20260730

echo "OPENVID_FINAL_COMPAT_GATE_OK"
