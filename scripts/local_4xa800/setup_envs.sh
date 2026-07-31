#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
TRAIN_CONSTRAINTS="$SEARCH_R1_REPO/constraints/train.txt"
RETRIEVER_CONSTRAINTS="$SEARCH_R1_REPO/constraints/retriever.txt"

if [[ ! -x "$SEARCH_R1_ENV/bin/python" || ! -x "$RETRIEVER_ENV/bin/python" ]]; then
  echo "Conda environments are missing. Create them before running this script." >&2
  exit 1
fi

"$SEARCH_R1_ENV/bin/python" -m pip install \
  -c "$TRAIN_CONSTRAINTS" torch==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu121
"$SEARCH_R1_ENV/bin/python" -m pip install \
  -c "$TRAIN_CONSTRAINTS" numpy opencv-python-headless
"$SEARCH_R1_ENV/bin/python" -m pip install \
  -c "$TRAIN_CONSTRAINTS" vllm
"$SEARCH_R1_ENV/bin/python" -m pip install \
  -c "$TRAIN_CONSTRAINTS" flash-attn --no-build-isolation
"$SEARCH_R1_ENV/bin/python" -m pip install \
  -c "$TRAIN_CONSTRAINTS" -e "$SEARCH_R1_REPO" wandb swanlab

"$RETRIEVER_ENV/bin/python" -m pip install \
  -c "$RETRIEVER_CONSTRAINTS" \
  Cython numpy pandas pyjnius scikit-learn scipy tqdm pyyaml requests \
  fastapi uvicorn
"$RETRIEVER_ENV/bin/python" -m pip install \
  -c "$RETRIEVER_CONSTRAINTS" torch==2.4.0 \
  --index-url https://download.pytorch.org/whl/cpu
"$RETRIEVER_ENV/bin/python" -m pip install \
  -c "$RETRIEVER_CONSTRAINTS" \
  transformers sentencepiece pillow onnxruntime openai tiktoken flask
"$RETRIEVER_ENV/bin/python" -m pip install \
  -c "$RETRIEVER_CONSTRAINTS" pyserini --no-deps
