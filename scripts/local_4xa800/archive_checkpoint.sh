#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

EXPERIMENT_NAME="${1:?usage: archive_checkpoint.sh EXPERIMENT_NAME [STEP]}"
STEP="${2:-latest}"
ACTOR_ROOT="$SEARCH_R1_CHECKPOINTS/$EXPERIMENT_NAME/actor"

if [[ "$STEP" == "latest" ]]; then
  STEP_NUMBER=-1
  for checkpoint in "$ACTOR_ROOT"/global_step_*; do
    [[ -d "$checkpoint" ]] || continue
    candidate="${checkpoint##*_}"
    if (( candidate > STEP_NUMBER )); then
      STEP_NUMBER="$candidate"
    fi
  done
  if (( STEP_NUMBER < 0 )); then
    echo "No checkpoint found for $EXPERIMENT_NAME." >&2
    exit 1
  fi
else
  STEP_NUMBER="$STEP"
fi

SOURCE="$ACTOR_ROOT/global_step_$STEP_NUMBER"
DEST_ROOT="$SEARCH_R1_CHECKPOINT_ARCHIVE/$EXPERIMENT_NAME/actor"
DEST="$DEST_ROOT/global_step_$STEP_NUMBER"

test -d "$SOURCE"
mkdir -p "$DEST_ROOT"
if [[ ! -f "$DEST/model.safetensors.index.json" ]]; then
  rm -rf "$DEST.tmp"
  cp -r "$SOURCE" "$DEST.tmp"
  mv "$DEST.tmp" "$DEST"
fi

echo "$DEST"
