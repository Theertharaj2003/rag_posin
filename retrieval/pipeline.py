from __future__ import annotations

from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever, Reranker
from utils.schemas import DocumentChunk, RetrievalResult


class RetrievalPipeline:
    def __init__(self, settings) -> None:
        self.dense = DenseRetriever(settings.dense_model, settings.device)
        self.bm25 = BM25Retriever()
        self.hybrid = HybridRetriever(self.dense, self.bm25, settings.dense_weight, settings.bm25_weight)
        self.reranker = Reranker(settings.reranker_model, settings.rerank_enabled)
        self.settings = settings

    def build(self, chunks: list[DocumentChunk]) -> None:
        self.hybrid.build(chunks)

    def retrieve(self, query: str, excluded_ids: set[str] | None = None, top_k: int | None = None) -> list[RetrievalResult]:
        top_k = top_k or self.settings.top_k
        candidates = self.hybrid.search(query, top_k=max(top_k, self.settings.rerank_top_k), excluded_ids=excluded_ids)
        return self.reranker.rerank(query, candidates, top_k=min(top_k, self.settings.final_k if top_k > self.settings.final_k else top_k))
