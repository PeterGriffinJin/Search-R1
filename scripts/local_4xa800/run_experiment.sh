#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

PRESET="${EXPERIMENT_PRESET:-${1:-}}"
if [[ -z "$PRESET" ]]; then
  echo "Set EXPERIMENT_PRESET or pass a preset name." >&2
  exit 1
fi

PRESET_FILE="$SEARCH_R1_REPO/configs/experiments/${PRESET%.env}.env"
if [[ ! -f "$PRESET_FILE" ]]; then
  echo "Preset not found: $PRESET_FILE" >&2
  exit 1
fi

set -a
source "$PRESET_FILE"
set +a

exec "$SCRIPT_DIR/train_grpo.sh"
