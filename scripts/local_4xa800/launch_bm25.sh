#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

if [[ ! -f "$SEARCH_R1_BM25_INDEX/segments_1" ]]; then
  echo "BM25 index is missing; run download_assets.sh first." >&2
  exit 1
fi
if [[ ! -s "$SEARCH_R1_CORPUS" ]]; then
  if [[ ! -s "$SEARCH_R1_CORPUS_ARCHIVE" ]]; then
    echo "Wiki corpus archive is missing; run download_assets.sh first." >&2
    exit 1
  fi
  mkdir -p "$(dirname "$SEARCH_R1_CORPUS")"
  tar -xzOf "$SEARCH_R1_CORPUS_ARCHIVE" > "$SEARCH_R1_CORPUS.tmp"
  mv "$SEARCH_R1_CORPUS.tmp" "$SEARCH_R1_CORPUS"
fi

INDEX_PATH="$SEARCH_R1_BM25_INDEX"
if [[ "${STAGE_BM25_TO_SHM:-0}" == "1" ]]; then
  STAGED_INDEX="$SEARCH_R1_RUNTIME_ROOT/wiki-18-bm25"
  if [[ ! -f "$STAGED_INDEX/segments_1" ]]; then
    mkdir -p "$STAGED_INDEX"
    cp -r "$SEARCH_R1_BM25_INDEX/." "$STAGED_INDEX/"
  fi
  INDEX_PATH="$STAGED_INDEX"
fi

exec "$RETRIEVER_ENV/bin/python" \
  "$SEARCH_R1_REPO/search_r1/search/bm25_server.py" \
  --index_path "$INDEX_PATH" \
  --corpus_path "$SEARCH_R1_CORPUS" \
  --topk "${RETRIEVER_TOPK:-3}" \
  --host "${BM25_HOST:-127.0.0.1}" \
  --port "${BM25_PORT:-8000}"
