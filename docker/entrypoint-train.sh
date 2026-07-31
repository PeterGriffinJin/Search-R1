#!/usr/bin/env bash
set -euo pipefail

cd "${SEARCH_R1_REPO:-/workspace/Search-R1}"
source scripts/local_4xa800/env.sh
mkdir -p "$SEARCH_R1_RUNTIME_ROOT" "$SEARCH_R1_CHECKPOINTS" "$RAY_TMPDIR"

exec "$@"
