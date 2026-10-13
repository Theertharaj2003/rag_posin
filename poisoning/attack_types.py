from enum import Enum


class AttackType(str, Enum):
    FACTUAL = "factual"
    CONTRADICTORY = "contradictory"
    INDIRECT_INJECTION = "indirect_injection"
