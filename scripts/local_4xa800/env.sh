#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SEARCH_R1_REPO="${SEARCH_R1_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export SEARCH_R1_ROOT="${SEARCH_R1_ROOT:-/mnt/private/nijianqiao/search-r1-run}"
export SEARCH_R1_RUNTIME_ROOT="${SEARCH_R1_RUNTIME_ROOT:-/dev/shm/njq-searchr1}"
export SEARCH_R1_ENV="${SEARCH_R1_ENV:-$SEARCH_R1_RUNTIME_ROOT/envs/searchr1}"
export RETRIEVER_ENV="${RETRIEVER_ENV:-$SEARCH_R1_RUNTIME_ROOT/envs/retriever}"

export HF_HOME="${HF_HOME:-$SEARCH_R1_ROOT/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SEARCH_R1_RUNTIME_ROOT/pip-cache}"
export TMPDIR="${TMPDIR:-$SEARCH_R1_RUNTIME_ROOT/tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-$SEARCH_R1_RUNTIME_ROOT/ray}"

export SEARCH_R1_DATA="${SEARCH_R1_DATA:-$SEARCH_R1_ROOT/data/nq_hotpotqa_train}"
export SEARCH_R1_MODEL="${SEARCH_R1_MODEL:-$SEARCH_R1_ROOT/models/Qwen2.5-3B}"
export SEARCH_R1_BM25_INDEX="${SEARCH_R1_BM25_INDEX:-$SEARCH_R1_ROOT/indexes/wiki-18-bm25/bm25}"
export SEARCH_R1_CORPUS_ARCHIVE="${SEARCH_R1_CORPUS_ARCHIVE:-$SEARCH_R1_ROOT/corpus/wiki-18.jsonl.gz}"
export SEARCH_R1_CORPUS="${SEARCH_R1_CORPUS:-$SEARCH_R1_RUNTIME_ROOT/wiki-18.jsonl}"
export SEARCH_R1_CHECKPOINTS="${SEARCH_R1_CHECKPOINTS:-$SEARCH_R1_RUNTIME_ROOT/checkpoints}"
export SEARCH_R1_CHECKPOINT_ARCHIVE="${SEARCH_R1_CHECKPOINT_ARCHIVE:-$SEARCH_R1_ROOT/checkpoints}"
export SEARCH_R1_LOGS="${SEARCH_R1_LOGS:-$SEARCH_R1_ROOT/logs}"

export TOKENIZERS_PARALLELISM=true
export WANDB_MODE="${WANDB_MODE:-offline}"
export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-21-openjdk-amd64}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-unused-for-local-bm25}"
export SWANLAB_WORKSPACE="${SWANLAB_WORKSPACE:-nileonx}"
export SWANLAB_MODE="${SWANLAB_MODE:-online}"
export SWANLAB_LOGDIR="${SWANLAB_LOGDIR:-$SEARCH_R1_ROOT/swanlab}"
export SWANLAB_ROOT="${SWANLAB_ROOT:-$SEARCH_R1_ROOT/.swanlab}"
export SWANLAB_API_KEY_FILE="${SWANLAB_API_KEY_FILE:-$SEARCH_R1_ROOT/.secrets/swanlab_api_key}"
if [[ -z "${SWANLAB_API_KEY:-}" && -r "$SWANLAB_API_KEY_FILE" ]]; then
  IFS= read -r SWANLAB_API_KEY < "$SWANLAB_API_KEY_FILE"
  export SWANLAB_API_KEY
fi

if [[ "${USE_CLASH_PROXY:-0}" == "1" ]]; then
  export HTTP_PROXY="${CLASH_HTTP_PROXY:-http://127.0.0.1:7890}"
  export HTTPS_PROXY="${CLASH_HTTP_PROXY:-http://127.0.0.1:7890}"
  unset ALL_PROXY
  export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
fi
