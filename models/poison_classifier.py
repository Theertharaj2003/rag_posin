from __future__ import annotations

from pathlib import Path


class LearnedPoisonClassifier:
    """Optional fine-tuned sequence classifier.

    The base DeBERTa checkpoint is not a poison classifier by itself. Train a
    checkpoint with models/train_poison_detector.py, then point model_path to it.
    """

    def __init__(self, model_path: str | Path):
        self.model_path = str(model_path)
        self.pipeline = None
        self.load_error = None
        try:
            from transformers import pipeline
            self.pipeline = pipeline("text-classification", model=self.model_path)
        except Exception as exc:
            self.load_error = str(exc)

    def predict_risk(self, text: str) -> float:
        if self.pipeline is None:
            raise RuntimeError(f"Learned poison model is unavailable: {self.load_error}")
        result = self.pipeline(text, truncation=True)[0]
        label = str(result["label"]).lower()
        score = float(result["score"])
        if label in {"1", "poison", "poisoned", "label_1"}:
            return score
        return 1.0 - score
