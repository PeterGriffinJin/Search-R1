#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

HF_CLI="$SEARCH_R1_ENV/bin/huggingface-cli"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
if [[ "${DISABLE_DOWNLOAD_PROXY:-0}" == "1" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi

if [[ ! -x "$HF_CLI" ]]; then
  echo "huggingface-cli is unavailable; run setup_envs.sh first." >&2
  exit 1
fi

mkdir -p \
  "$SEARCH_R1_ROOT/indexes/wiki-18-bm25" \
  "$(dirname "$SEARCH_R1_CORPUS_ARCHIVE")" \
  "$(dirname "$SEARCH_R1_CORPUS")" \
  "$SEARCH_R1_DATA" \
  "$SEARCH_R1_MODEL"

"$HF_CLI" download PeterJinGo/wiki-18-bm25-index \
  --repo-type dataset \
  --local-dir "$SEARCH_R1_ROOT/indexes/wiki-18-bm25" \
  --max-workers 4 &
INDEX_PID=$!

"$HF_CLI" download PeterJinGo/nq_hotpotqa_train \
  --repo-type dataset \
  --local-dir "$SEARCH_R1_DATA" \
  --max-workers 2 &
DATA_PID=$!

"$HF_CLI" download Qwen/Qwen2.5-3B \
  --local-dir "$SEARCH_R1_MODEL" \
  --max-workers 4 &
MODEL_PID=$!

(
  "$HF_CLI" download PeterJinGo/wiki-18-corpus \
    --repo-type dataset \
    --local-dir "$(dirname "$SEARCH_R1_CORPUS_ARCHIVE")" \
    --max-workers 2
  if [[ ! -s "$SEARCH_R1_CORPUS" ]]; then
    tar -xzOf "$SEARCH_R1_CORPUS_ARCHIVE" > "$SEARCH_R1_CORPUS.tmp"
    mv "$SEARCH_R1_CORPUS.tmp" "$SEARCH_R1_CORPUS"
  fi
) &
CORPUS_PID=$!

STATUS=0
wait "$INDEX_PID" || STATUS=1
wait "$DATA_PID" || STATUS=1
wait "$MODEL_PID" || STATUS=1
wait "$CORPUS_PID" || STATUS=1

test -f "$SEARCH_R1_BM25_INDEX/segments_1"
test -f "$SEARCH_R1_DATA/train.parquet"
test -f "$SEARCH_R1_DATA/test.parquet"
test -f "$SEARCH_R1_MODEL/config.json"
test -s "$SEARCH_R1_CORPUS"
exit "$STATUS"
