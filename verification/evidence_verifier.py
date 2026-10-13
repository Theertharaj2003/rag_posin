from __future__ import annotations

from dataclasses import dataclass

from utils.common import clamp, tokenize


@dataclass
class VerificationResult:
    label: str
    score: float
    entailment: float
    contradiction: float


class EvidenceVerifier:
    def __init__(self, model_name: str, entailment_threshold: float = 0.55, contradiction_threshold: float = 0.55) -> None:
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold
        self.pipeline = None
        self.load_error = None
        try:
            from transformers import pipeline
            self.pipeline = pipeline("text-classification", model=model_name)
        except Exception as exc:
            self.load_error = str(exc)

    def verify(self, claim: str, evidence: str) -> VerificationResult:
        if self.pipeline is not None:
            try:
                result = self.pipeline({"text": claim, "text_pair": evidence}, truncation=True)[0]
                label = result["label"].lower()
                score = float(result["score"])
                if "entail" in label:
                    return VerificationResult("entailment", score, score, 0.0)
                if "contrad" in label:
                    return VerificationResult("contradiction", score, 0.0, score)
                return VerificationResult("neutral", score, 0.0, 0.0)
            except Exception:
                pass

        # Deterministic fallback for smoke tests/offline development.
        q = set(tokenize(claim))
        e = set(tokenize(evidence))
        overlap = len(q & e) / max(1, len(q))
        contradiction_terms = {"false", "not", "never", "incorrect", "no", "doesn't", "doesnt"}
        contradiction = 0.0
        if contradiction_terms & e and overlap > 0.25:
            contradiction = 0.65
        entailment = clamp(overlap)
        if contradiction >= self.contradiction_threshold:
            return VerificationResult("contradiction", contradiction, entailment, contradiction)
        if entailment >= self.entailment_threshold:
            return VerificationResult("entailment", entailment, entailment, contradiction)
        return VerificationResult("neutral", max(entailment, contradiction), entailment, contradiction)
