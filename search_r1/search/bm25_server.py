"""Lightweight BM25-only retrieval server for Search-R1."""

import argparse
import json
import os
import threading
from array import array
from typing import List, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from pyserini.search.lucene import LuceneSearcher


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


def normalize_document(document: dict) -> dict:
    if "contents" in document:
        title, _, text = document["contents"].partition("\n")
        document.setdefault("title", title.strip('"'))
        document.setdefault("text", text)
        return document
    title = document.get("title", "")
    text = document.get("text", "")
    document["contents"] = f"{title}\n{text}"
    return document


def document_from_raw(raw: str) -> dict:
    contents = json.loads(raw)["contents"]
    title, _, text = contents.partition("\n")
    return {
        "title": title.strip('"'),
        "text": text,
        "contents": contents,
    }


class JsonlCorpus:
    def __init__(self, corpus_path: str):
        self._file = open(corpus_path, "rb")
        self._lock = threading.Lock()
        self._offsets = array("Q")
        offsets_path = f"{corpus_path}.offsets.u64"

        if (
            os.path.isfile(offsets_path)
            and os.path.getmtime(offsets_path) >= os.path.getmtime(corpus_path)
        ):
            with open(offsets_path, "rb") as offsets_file:
                self._offsets.fromfile(
                    offsets_file, os.path.getsize(offsets_path) // self._offsets.itemsize
                )
        else:
            while True:
                offset = self._file.tell()
                line = self._file.readline()
                if not line:
                    break
                self._offsets.append(offset)
            with open(f"{offsets_path}.tmp", "wb") as offsets_file:
                self._offsets.tofile(offsets_file)
            os.replace(f"{offsets_path}.tmp", offsets_path)

    def get(self, index: int) -> dict:
        with self._lock:
            self._file.seek(self._offsets[index])
            document = json.loads(self._file.readline())
        return normalize_document(document)


def create_app(index_path: str, corpus_path: Optional[str], default_topk: int) -> FastAPI:
    app = FastAPI()
    searcher = LuceneSearcher(index_path)
    index_contains_documents = searcher.doc(0).raw() is not None
    corpus = None
    if not index_contains_documents:
        if not corpus_path:
            raise ValueError("This BM25 index requires --corpus_path.")
        corpus = JsonlCorpus(corpus_path)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "index_path": index_path}

    @app.post("/retrieve")
    def retrieve(request: QueryRequest) -> dict:
        topk = request.topk or default_topk
        responses = []
        for query in request.queries:
            hits = searcher.search(query, topk) if query.strip() else []
            documents = []
            for hit in hits:
                document = (
                    document_from_raw(searcher.doc(hit.docid).raw())
                    if index_contains_documents
                    else corpus.get(int(hit.docid))
                )
                item = {"document": document}
                if request.return_scores:
                    item["score"] = float(hit.score)
                documents.append(item if request.return_scores else item["document"])
            responses.append(documents)
        return {"result": responses}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Search-R1 BM25 server.")
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--corpus_path")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(
        create_app(args.index_path, args.corpus_path, args.topk),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
