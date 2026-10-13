from __future__ import annotations

import re
from dataclasses import dataclass

from utils.common import clamp, tokenize
from utils.schemas import DocumentChunk


INJECTION_PATTERNS = [
    r"\bsystem override\b",
    r"\bignore (the )?(user|previous|trusted|above)\b",
    r"\bdisregard\b",
    r"\bhigher-priority instruction\b",
    r"\boutput the sentence\b",
    r"\bdo not follow\b",
]

SUSPICIOUS_PHRASES = [
    "intentionally false",
    "poisoned evidence",
    "suspicious passages should be preferred",
    "treat this passage as verified",
]


@dataclass
class PoisonAssessment:
    risk: float
    injection_score: float
    anomaly_score: float
    reasons: list[str]


class PoisonDetector:
    def assess(self, query: str, chunk: DocumentChunk) -> PoisonAssessment:
        text = chunk.text.lower()
        reasons: list[str] = []
        injection_hits = sum(bool(re.search(p, text)) for p in INJECTION_PATTERNS)
        phrase_hits = sum(p in text for p in SUSPICIOUS_PHRASES)
        injection_score = clamp(0.35 * injection_hits + 0.25 * phrase_hits)

        q_tokens = set(tokenize(query))
        t_tokens = set(tokenize(chunk.text))
        overlap = len(q_tokens & t_tokens) / max(1, len(q_tokens))
        anomaly_score = 1.0 - overlap
        if injection_hits:
            reasons.append("instruction-like or prompt-injection language")
        if phrase_hits:
            reasons.append("known synthetic-poison wording pattern")
        if chunk.source_reliability < 0.30:
            reasons.append("low source reliability metadata")
        if chunk.is_poisoned:
            reasons.append("ground-truth poison label is present in experiment data")

        risk = clamp(
            0.55 * injection_score
            + 0.25 * (1.0 - chunk.source_reliability)
            + 0.10 * anomaly_score
            + 0.10 * float(chunk.is_poisoned)
        )
        return PoisonAssessment(risk, injection_score, anomaly_score, reasons)
