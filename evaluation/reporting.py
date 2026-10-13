from __future__ import annotations

import csv
from pathlib import Path

from utils.common import json_dump


def save_run(records: list[dict], summary: dict, root: Path, stem: str = "experiment") -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / f"{stem}.json"
    csv_path = root / f"{stem}.csv"
    json_dump({"summary": summary, "records": records}, json_path)
    if records:
        fields = sorted({k for r in records for k in r.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    return json_path, csv_path
