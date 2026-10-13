from __future__ import annotations

from utils.schemas import RetrievalResult


def quarantine(results: list[RetrievalResult], threshold: float) -> set[str]:
    ids = set()
    for r in results:
        if r.poison_risk >= threshold or r.verification_label == "contradiction":
            r.quarantined = True
            ids.add(r.chunk.chunk_id)
    return ids
