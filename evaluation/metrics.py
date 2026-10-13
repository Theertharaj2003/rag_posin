from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def binary_metrics(y_true: Iterable[int], y_pred: Iterable[int], y_score: Iterable[float] | None = None) -> dict[str, float]:
    y_true = list(y_true)
    y_pred = list(y_pred)
    out = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if y_score is not None and len(set(y_true)) > 1:
        out["auroc"] = float(roc_auc_score(y_true, list(y_score)))
    return out


def retrieval_recall(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(retrieved_ids) & relevant_ids) / len(relevant_ids)


def mean_reciprocal_rank(rank_lists: list[list[str]], relevant: list[set[str]]) -> float:
    rr = []
    for ranked, rel in zip(rank_lists, relevant):
        value = 0.0
        for i, item in enumerate(ranked, 1):
            if item in rel:
                value = 1.0 / i
                break
        rr.append(value)
    return float(np.mean(rr)) if rr else 0.0


def attribution_precision(attributions: list[dict], threshold: float = 0.55) -> float:
    if not attributions:
        return 0.0
    return sum(a["score"] >= threshold for a in attributions) / len(attributions)


def attack_success_rate(answers_contaminated: list[bool]) -> float:
    return sum(answers_contaminated) / len(answers_contaminated) if answers_contaminated else 0.0


def recovery_rate(before_failed: list[bool], after_success: list[bool]) -> float:
    eligible = [a and b for a, b in zip(before_failed, after_success)]
    return sum(eligible) / max(1, sum(before_failed))
