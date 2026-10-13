from __future__ import annotations

from utils.common import sentence_split


def extract_claims(answer: str) -> list[str]:
    return sentence_split(answer)
