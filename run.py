from __future__ import annotations

import argparse
import json
from pathlib import Path

from config.settings import get_settings
from data.demo_dataset import QUESTIONS, build_chunks, write_demo_data
from evaluation.reporting import save_run
from utils.common import set_seed


def build_system(settings, chunks):
    from retrieval.pipeline import RetrievalPipeline
    from generation.llm import GroundedGenerator
    from healing.healing_controller import SelfHealingRAG
    from poisoning.detector import PoisonDetector
    from trust.trust_score import TrustEngine
    from verification.evidence_verifier import EvidenceVerifier
    retriever = RetrievalPipeline(settings)
    retriever.build(chunks)
    detector = PoisonDetector()
    trust = TrustEngine(settings.trust_weights)
    verifier = EvidenceVerifier(settings.nli_model, settings.entailment_threshold, settings.nli_contradiction_threshold)
    generator = GroundedGenerator(settings.generator_model, settings.device)
    return SelfHealingRAG(retriever, detector, trust, verifier, generator, settings), retriever


def demo(root: Path) -> None:
    write_demo_data(root)
    print("Demo data prepared.")
    print("Next: python run.py --mode retrieve --query \"What is retrieval augmented generation?\"")


def retrieve_mode(settings, root: Path, query: str) -> None:
    system, retriever = build_system(settings, build_chunks())
    results = retriever.retrieve(query, top_k=settings.top_k)
    print(f"\nQUERY: {query}\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r.chunk.chunk_id} | hybrid={r.hybrid_score:.4f} | source={r.chunk.source}")
        print(r.chunk.text)
        print()


def pipeline_mode(settings, root: Path, query: str) -> None:
    system, _ = build_system(settings, build_chunks())
    generation, trace, results = system.answer(query)
    print(f"\nQUERY: {query}\n")
    print("ANSWER:", generation.answer)
    print("MODEL:", generation.model)
    print("RECOVERED:", trace.recovered)
    print("QUARANTINED:", trace.quarantined_ids)
    print("ATTRIBUTIONS:")
    print(json.dumps(generation.attributions, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Trust-aware self-healing RAG research framework")
    parser.add_argument("--mode", choices=["demo", "retrieve", "pipeline", "experiment"], default="demo")
    parser.add_argument("--query", default="What is retrieval augmented generation?")
    args = parser.parse_args()
    settings = get_settings()
    set_seed(settings.seed)
    root = Path(__file__).resolve().parent

    if args.mode == "demo":
        demo(root)
    elif args.mode == "retrieve":
        retrieve_mode(settings, root, args.query)
    elif args.mode == "pipeline":
        pipeline_mode(settings, root, args.query)
    else:
        from experiments.runner import run_experiment
        summary = run_experiment(settings, root)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
