from __future__ import annotations

from verification.evidence_verifier import EvidenceVerifier
from utils.common import clamp, tokenize
from utils.schemas import RetrievalResult


def claim_attribution(claim: str, results: list[RetrievalResult], verifier: EvidenceVerifier) -> list[dict]:
    out = []
    q = set(tokenize(claim))
    for r in results:
        e = set(tokenize(r.chunk.text))
        lexical = len(q & e) / max(1, len(q))
        v = verifier.verify(claim, r.chunk.text)
        score = clamp(0.45 * lexical + 0.35 * v.entailment + 0.20 * r.trust_score)
        out.append({
            "claim": claim,
            "chunk_id": r.chunk.chunk_id,
            "source": r.chunk.source,
            "score": score,
            "verification": v.label,
        })
    return sorted(out, key=lambda x: x["score"], reverse=True)
