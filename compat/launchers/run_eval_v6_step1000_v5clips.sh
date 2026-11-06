#!/usr/bin/env bash
# Back-compat wrapper. Prefer: STEP=1000 bash scripts/run_eval_v6_v5clips.sh
set -euo pipefail
export STEP="${STEP:-1000}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_eval_v6_v5clips.sh"
