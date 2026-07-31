#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$CONDA_BIN" || ! -x "$CONDA_BIN" ]]; then
  echo "Conda is required. Install Miniconda and retry." >&2
  exit 1
fi

export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$SEARCH_R1_RUNTIME_ROOT/conda-pkgs}"
mkdir -p "$SEARCH_R1_RUNTIME_ROOT/envs" "$CONDA_PKGS_DIRS" "$TMPDIR"

if [[ ! -x "$SEARCH_R1_ENV/bin/python" ]]; then
  "$CONDA_BIN" create -y -p "$SEARCH_R1_ENV" python=3.9 pip
fi

if [[ ! -x "$RETRIEVER_ENV/bin/python" ]]; then
  "$CONDA_BIN" create -y -p "$RETRIEVER_ENV" python=3.10 pip
fi

echo "Training environment: $SEARCH_R1_ENV"
echo "Retriever environment: $RETRIEVER_ENV"
