from __future__ import annotations

from dataclasses import dataclass, field

from generation.prompt_templates import SYSTEM_PROMPT
from utils.common import sentence_split
from utils.schemas import RetrievalResult


@dataclass
class GenerationResult:
    answer: str
    model: str
    attributions: list = field(default_factory=list)


class GroundedGenerator:
    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.model_name = model_name
        self.pipe = None
        self.load_error = None
        self.device = device

    def _load_pipeline(self) -> None:
        if self.pipe is not None:
            return
        try:
            from transformers import pipeline
            kwargs = {"model": self.model_name}
            if self.device != "auto":
                kwargs["device"] = 0 if self.device.startswith("cuda") else -1
            self.pipe = pipeline("text-generation", **kwargs)
        except Exception as exc:
            self.load_error = str(exc)
            raise RuntimeError(
                f"Could not load the configured generator '{self.model_name}': {self.load_error}"
            ) from exc

    def generate(self, query: str, evidence: list[RetrievalResult], use_local_model: bool = False) -> GenerationResult:
        verified = [r for r in evidence if r.verified and not r.quarantined]
        if not verified:
            return GenerationResult("Insufficient verified evidence to answer this question safely.", "abstention")

        if use_local_model:
            self._load_pipeline()
            context = "\n".join(f"[{r.chunk.chunk_id}] {r.chunk.text}" for r in verified)
            prompt = f"{SYSTEM_PROMPT}\nQuestion: {query}\nEvidence:\n{context}\nAnswer:"
            result = self.pipe(prompt, max_new_tokens=180, do_sample=False)[0]["generated_text"]
            answer = result.split("Answer:", 1)[-1].strip()
            return GenerationResult(answer, self.model_name)

        # Extractive fallback. This makes the complete control flow executable
        # before downloading a large generator and provides a transparent baseline.
        sentences = []
        q_tokens = set(query.lower().split())
        for r in verified:
            for s in sentence_split(r.chunk.text):
                score = len(q_tokens & set(s.lower().split()))
                sentences.append((score, s, r.chunk.chunk_id))
        sentences.sort(reverse=True)
        selected = []
        seen = set()
        for _, s, cid in sentences:
            if s not in seen:
                selected.append(f"{s} [{cid}]")
                seen.add(s)
            if len(selected) >= 2:
                break
        return GenerationResult(" ".join(selected), "extractive_fallback")
