from __future__ import annotations

from utils.common import tokenize
from utils.schemas import RetrievalResult


def lexical_consistency(target: RetrievalResult, peers: list[RetrievalResult]) -> float:
    if not peers:
        return 0.5
    a = set(tokenize(target.chunk.text))
    scores = []
    for p in peers:
        if p.chunk.chunk_id == target.chunk.chunk_id:
            continue
        b = set(tokenize(p.chunk.text))
        scores.append(len(a & b) / max(1, len(a | b)))
    return sum(scores) / len(scores) if scores else 0.5
