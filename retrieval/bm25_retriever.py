from __future__ import annotations

from typing import Sequence

from rank_bm25 import BM25Okapi

from utils.common import tokenize
from utils.schemas import DocumentChunk


class BM25Retriever:
    def __init__(self) -> None:
        self.chunks: list[DocumentChunk] = []
        self.index: BM25Okapi | None = None

    def build(self, chunks: Sequence[DocumentChunk]) -> None:
        self.chunks = list(chunks)
        self.index = BM25Okapi([tokenize(c.text) for c in self.chunks])

    def search(self, query: str, top_k: int = 8, excluded_ids: set[str] | None = None) -> list[tuple[DocumentChunk, float]]:
        if self.index is None:
            raise RuntimeError("BM25Retriever.build() must be called before search().")
        scores = self.index.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        excluded_ids = excluded_ids or set()
        for idx in order:
            c = self.chunks[idx]
            if c.chunk_id in excluded_ids:
                continue
            out.append((c, float(scores[idx])))
            if len(out) >= top_k:
                break
        return out
