from __future__ import annotations

from dataclasses import dataclass, field

from generation.llm import GroundedGenerator
from poisoning.detector import PoisonDetector
from trust.consistency import lexical_consistency
from trust.trust_score import TrustEngine
from verification.attribution import claim_attribution
from verification.claim_extractor import extract_claims
from verification.evidence_verifier import EvidenceVerifier
from healing.quarantine import quarantine
from utils.schemas import RetrievalResult


@dataclass
class HealingTrace:
    rounds: list[dict] = field(default_factory=list)
    quarantined_ids: list[str] = field(default_factory=list)
    recovered: bool = False


class SelfHealingRAG:
    def __init__(self, retriever, detector: PoisonDetector, trust_engine: TrustEngine, verifier: EvidenceVerifier, generator: GroundedGenerator, settings) -> None:
        self.retriever = retriever
        self.detector = detector
        self.trust_engine = trust_engine
        self.verifier = verifier
        self.generator = generator
        self.settings = settings

    def _score_and_verify(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        for r in results:
            assessment = self.detector.assess(query, r.chunk)
            r.poison_risk = assessment.risk
        for r in results:
            peers = [x for x in results if x.chunk.chunk_id != r.chunk.chunk_id]
            consistency = lexical_consistency(r, peers)
            vr = self.verifier.verify(query, r.chunk.text)
            r.verification_label = vr.label
            r.verification_score = vr.score
            r.verified = vr.label == "entailment"
            components = self.trust_engine.score(r, consistency, vr.entailment)
            r.trust_score = components.trust
        return sorted(results, key=lambda x: x.trust_score, reverse=True)

    def answer(
        self,
        query: str,
        initial_excluded: set[str] | None = None,
        use_local_model: bool = False,
    ) -> tuple[object, HealingTrace, list[RetrievalResult]]:
        excluded = set(initial_excluded or set())
        trace = HealingTrace()
        final_results: list[RetrievalResult] = []

        for round_id in range(1, self.settings.max_healing_rounds + 2):
            results = self.retriever.retrieve(query, excluded_ids=excluded, top_k=self.settings.top_k)
            results = self._score_and_verify(query, results)
            q_ids = quarantine(results, self.settings.quarantine_threshold)
            excluded |= q_ids
            trusted = [r for r in results if r.verified and not r.quarantined and r.trust_score >= self.settings.poison_threshold]
            trace.rounds.append({
                "round": round_id,
                "retrieved": [r.to_dict() for r in results],
                "trusted_count": len(trusted),
                "quarantined": sorted(q_ids),
            })
            final_results = results
            if len(trusted) >= self.settings.min_trusted_evidence or round_id > self.settings.max_healing_rounds:
                break
            if not q_ids:
                # No obvious evidence was quarantined, so another retrieval round
                # would be unlikely to change the candidate set.
                break

        trusted = [r for r in final_results if r.verified and not r.quarantined]
        generation = self.generator.generate(query, trusted, use_local_model=use_local_model)
        claims = extract_claims(generation.answer)
        attributions = [claim_attribution(c, trusted, self.verifier) for c in claims]
        generation.attributions = attributions
        trace.quarantined_ids = sorted(excluded)
        trace.recovered = len(trusted) >= self.settings.min_trusted_evidence
        return generation, trace, final_results
