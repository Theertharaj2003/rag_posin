from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DocumentChunk:
    document_id: str
    chunk_id: str
    text: str
    source: str
    title: str = ""
    is_poisoned: bool = False
    attack_type: str = "clean"
    source_reliability: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    dense_score: float = 0.0
    bm25_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float = 0.0
    poison_risk: float = 0.0
    trust_score: float = 0.0
    verified: bool = False
    verification_label: str = "unknown"
    verification_score: float = 0.0
    attribution_score: float = 0.0
    quarantined: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["chunk"] = self.chunk.to_dict()
        return d
