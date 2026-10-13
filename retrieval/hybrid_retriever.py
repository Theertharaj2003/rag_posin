from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from utils.common import normalize
from utils.schemas import DocumentChunk, RetrievalResult


class HybridRetriever:
    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, dense_weight: float = 0.6, bm25_weight: float = 0.4) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.chunks: list[DocumentChunk] = []

    def build(self, chunks: Sequence[DocumentChunk]) -> None:
        self.chunks = list(chunks)
        self.dense.build(chunks)
        self.bm25.build(chunks)

    def search(self, query: str, top_k: int = 8, excluded_ids: set[str] | None = None) -> list[RetrievalResult]:
        d = self.dense.search(query, top_k=max(top_k * 2, 10), excluded_ids=excluded_ids)
        b = self.bm25.search(query, top_k=max(top_k * 2, 10), excluded_ids=excluded_ids)
        d_scores = normalize([s for _, s in d])
        b_scores = normalize([s for _, s in b])
        merged: dict[str, RetrievalResult] = {}
        for (c, _), score in zip(d, d_scores):
            merged[c.chunk_id] = RetrievalResult(chunk=c, dense_score=score)
        for (c, _), score in zip(b, b_scores):
            if c.chunk_id not in merged:
                merged[c.chunk_id] = RetrievalResult(chunk=c)
            merged[c.chunk_id].bm25_score = score
        for r in merged.values():
            r.hybrid_score = self.dense_weight * r.dense_score + self.bm25_weight * r.bm25_score
        return sorted(merged.values(), key=lambda x: x.hybrid_score, reverse=True)[:top_k]


class Reranker:
    def __init__(self, model_name: str, enabled: bool = True) -> None:
        self.enabled = enabled
        self.model = None
        if enabled:
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(model_name)
            except Exception as exc:
                self.enabled = False
                self.load_error = str(exc)

    def rerank(self, query: str, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        if not results or not self.enabled or self.model is None:
            return results[:top_k]
        pairs = [(query, r.chunk.text) for r in results]
        scores = self.model.predict(pairs, show_progress_bar=False)
        for r, s in zip(results, scores):
            r.rerank_score = float(s)
        return sorted(results, key=lambda x: x.rerank_score, reverse=True)[:top_k]
