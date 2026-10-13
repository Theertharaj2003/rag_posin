from __future__ import annotations

from pathlib import Path

from data.demo_dataset import QUESTIONS, build_chunks, write_demo_data
from evaluation.metrics import attack_success_rate, attribution_precision, binary_metrics
from evaluation.reporting import save_run
from generation.llm import GroundedGenerator
from healing.healing_controller import SelfHealingRAG
from poisoning.attack_types import AttackType
from poisoning.detector import PoisonDetector
from poisoning.poison_generator import generate_poison
from retrieval.pipeline import RetrievalPipeline
from trust.trust_score import TrustEngine
from verification.evidence_verifier import EvidenceVerifier


def make_pipeline(settings, chunks):
    r = RetrievalPipeline(settings)
    r.build(chunks)
    return r


def naive_answer(query: str, results):
    selected = results[:2]
    text = " ".join(r.chunk.text for r in selected)
    return text, any(r.chunk.is_poisoned for r in selected)


def score_trust_only(query, results, settings, detector, verifier, trust_engine):
    for r in results:
        r.poison_risk = detector.assess(query, r.chunk).risk
        vr = verifier.verify(query, r.chunk.text)
        r.verification_label = vr.label
        r.verification_score = vr.score
        r.verified = vr.label == "entailment"
        components = trust_engine.score(r, 0.5, vr.entailment)
        r.trust_score = components.trust
    return sorted(results, key=lambda x: x.trust_score, reverse=True)


def run_experiment(settings, root: Path) -> dict:
    write_demo_data(root)
    clean = build_chunks()
    detector = PoisonDetector()
    verifier = EvidenceVerifier(settings.nli_model, settings.entailment_threshold, settings.nli_contradiction_threshold)
    trust_engine = TrustEngine(settings.trust_weights)
    generator = GroundedGenerator(settings.generator_model, settings.device)
    records = []

    for attack in AttackType:
        corpus = generate_poison(clean, attack, ratio=0.40, seed=settings.seed)
        retriever = make_pipeline(settings, corpus)
        healer = SelfHealingRAG(retriever, detector, trust_engine, verifier, generator, settings)
        for q in QUESTIONS:
            query = q["question"]
            baseline_results = retriever.retrieve(query, top_k=settings.top_k)
            _, baseline_poison = naive_answer(query, baseline_results)

            trust_results = score_trust_only(query, retriever.retrieve(query, top_k=settings.top_k), settings, detector, verifier, trust_engine)
            trust_selected = trust_results[:2]
            trust_poison = any(r.chunk.is_poisoned for r in trust_selected)

            generation, trace, healed_results = healer.answer(query)
            healed_poison = any(r.chunk.is_poisoned for r in healed_results[:2] if not r.quarantined)
            attrs = [a for group in getattr(generation, "attributions", []) for a in group[:1]]

            records.append({
                "question_id": q["id"],
                "attack_type": attack.value,
                "baseline_poisoned_top2": int(baseline_poison),
                "trust_poisoned_top2": int(trust_poison),
                "healed_poisoned_top2": int(healed_poison),
                "recovered": int(trace.recovered),
                "quarantined_count": len(trace.quarantined_ids),
                "trusted_count": sum(1 for r in healed_results if r.verified and not r.quarantined),
                "attribution_precision": attribution_precision(attrs),
                "answer": generation.answer,
            })

    summary = {
        "baseline_attack_success_rate": attack_success_rate([bool(r["baseline_poisoned_top2"]) for r in records]),
        "trust_attack_success_rate": attack_success_rate([bool(r["trust_poisoned_top2"]) for r in records]),
        "healed_attack_success_rate": attack_success_rate([bool(r["healed_poisoned_top2"]) for r in records]),
        "recovery_rate": sum(r["recovered"] for r in records) / max(1, len(records)),
        "mean_attribution_precision": sum(r["attribution_precision"] for r in records) / max(1, len(records)),
        "notes": "These are local experimental measurements on the included controlled demo corpus, not claims about external benchmarks.",
    }
    save_run(records, summary, root / "results", "rag_experiment")
    return summary
