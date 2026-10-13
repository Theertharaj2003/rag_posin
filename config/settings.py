from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _load_yaml() -> dict[str, Any]:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class Settings:
    root: Path
    seed: int
    top_k: int
    final_k: int
    dense_weight: float
    bm25_weight: float
    rerank_enabled: bool
    rerank_top_k: int
    dense_model: str
    reranker_model: str
    poison_threshold: float
    contradiction_threshold: float
    injection_penalty: float
    quarantine_threshold: float
    trust_weights: dict[str, float]
    nli_model: str
    entailment_threshold: float
    nli_contradiction_threshold: float
    max_healing_rounds: int
    min_trusted_evidence: int
    exclude_quarantined: bool
    poison_classifier: str
    generator_model: str
    device: str


def get_settings() -> Settings:
    c = _load_yaml()
    return Settings(
        root=ROOT,
        seed=int(c["project"]["seed"]),
        top_k=int(c["project"]["top_k"]),
        final_k=int(c["project"]["final_k"]),
        dense_weight=float(c["retrieval"]["dense_weight"]),
        bm25_weight=float(c["retrieval"]["bm25_weight"]),
        rerank_enabled=bool(c["retrieval"]["rerank_enabled"]),
        rerank_top_k=int(c["retrieval"]["rerank_top_k"]),
        dense_model=c["retrieval"]["dense_model"],
        reranker_model=c["retrieval"]["reranker_model"],
        poison_threshold=float(c["poisoning"]["risk_threshold"]),
        contradiction_threshold=float(c["poisoning"]["contradiction_threshold"]),
        injection_penalty=float(c["poisoning"]["injection_penalty"]),
        quarantine_threshold=float(c["poisoning"]["quarantine_threshold"]),
        trust_weights=dict(c["trust"]),
        nli_model=c["verification"]["nli_model"],
        entailment_threshold=float(c["verification"]["entailment_threshold"]),
        nli_contradiction_threshold=float(c["verification"]["contradiction_threshold"]),
        max_healing_rounds=int(c["healing"]["max_rounds"]),
        min_trusted_evidence=int(c["healing"]["min_trusted_evidence"]),
        exclude_quarantined=bool(c["healing"]["exclude_quarantined"]),
        poison_classifier=c["models"]["poison_classifier"],
        generator_model=c["models"]["generator"],
        device=os.getenv("DEVICE", "auto"),
    )
