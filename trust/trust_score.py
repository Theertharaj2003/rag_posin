from __future__ import annotations

from dataclasses import dataclass

from utils.common import clamp
from utils.schemas import RetrievalResult


@dataclass
class TrustComponents:
    relevance: float
    source_reliability: float
    consistency: float
    evidence_support: float
    poison_risk: float
    trust: float


class TrustEngine:
    def __init__(self, weights: dict[str, float]) -> None:
        self.weights = weights

    def score(self, result: RetrievalResult, consistency: float, evidence_support: float) -> TrustComponents:
        w = self.weights
        trust = (
            w["relevance_weight"] * result.hybrid_score
            + w["source_reliability_weight"] * result.chunk.source_reliability
            + w["consistency_weight"] * consistency
            + w["evidence_support_weight"] * evidence_support
            - w["poison_risk_weight"] * result.poison_risk
        )
        return TrustComponents(
            relevance=result.hybrid_score,
            source_reliability=result.chunk.source_reliability,
            consistency=consistency,
            evidence_support=evidence_support,
            poison_risk=result.poison_risk,
            trust=clamp(trust),
        )
