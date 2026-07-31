#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
N_GPUS="${#GPU_IDS[@]}"

cd "$SEARCH_R1_REPO"
"$SEARCH_R1_ENV/bin/python" -c \
  "import torch, ray, transformers, vllm, flash_attn, verl; \
assert torch.cuda.device_count() == $N_GPUS; \
print('training env:', torch.__version__, ray.__version__, transformers.__version__, vllm.__version__)"

"$SEARCH_R1_ENV/bin/torchrun" --standalone --nproc-per-node="$N_GPUS" \
  "$SCRIPT_DIR/verify_nccl.py"

"$RETRIEVER_ENV/bin/python" -c \
  "from pyserini.search.lucene import LuceneSearcher; \
import fastapi, uvicorn; print('retriever env: OK')"

java -version
