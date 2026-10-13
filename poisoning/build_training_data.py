from __future__ import annotations

import csv
from pathlib import Path

from data.demo_dataset import build_chunks
from poisoning.attack_types import AttackType
from poisoning.poison_generator import generate_poison


def build_training_file(root: Path, output: Path, repeats: int = 20) -> None:
    clean = build_chunks()
    rows = []
    for c in clean:
        for _ in range(repeats):
            rows.append({"text": c.text, "label": 0})
    for attack in AttackType:
        poisoned = generate_poison(clean, attack, ratio=1.0, seed=42)
        for c in poisoned:
            for _ in range(repeats):
                rows.append({"text": c.text, "label": 1})
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    out = root / "data" / "processed" / "poison_training.csv"
    build_training_file(root, out)
    print(out)
