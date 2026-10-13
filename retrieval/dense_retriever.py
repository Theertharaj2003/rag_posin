from __future__ import annotations

from typing import Sequence

import numpy as np

from utils.schemas import DocumentChunk


class DenseRetriever:
    def __init__(self, model_name: str, device: str = "auto") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Run: pip install -r requirements.txt"
            ) from exc
        kwargs = {}
        if device != "auto":
            kwargs["device"] = device
        self.model = SentenceTransformer(model_name, **kwargs)
        self.chunks: list[DocumentChunk] = []
        self.embeddings: np.ndarray | None = None

    def build(self, chunks: Sequence[DocumentChunk]) -> None:
        self.chunks = list(chunks)
        self.embeddings = np.asarray(
            self.model.encode(
                [c.text for c in self.chunks],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )

    def search(self, query: str, top_k: int = 8, excluded_ids: set[str] | None = None) -> list[tuple[DocumentChunk, float]]:
        if self.embeddings is None:
            raise RuntimeError("DenseRetriever.build() must be called before search().")
        q = np.asarray(self.model.encode([query], normalize_embeddings=True, show_progress_bar=False))[0]
        scores = self.embeddings @ q
        order = np.argsort(-scores)
        out = []
        excluded_ids = excluded_ids or set()
        for idx in order:
            c = self.chunks[int(idx)]
            if c.chunk_id in excluded_ids:
                continue
            out.append((c, float(scores[int(idx)])))
            if len(out) >= top_k:
                break
        return out
