from __future__ import annotations

import json
from pathlib import Path

from utils.schemas import DocumentChunk


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary > start + max_chars // 2:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return json.dumps(data, ensure_ascii=False)
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def ingest_directory(input_dir: Path, output_json: Path, source_reliability: float = 0.75) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md", ".json", ".pdf"}:
            continue
        text = read_file(path)
        for i, part in enumerate(chunk_text(text)):
            chunks.append(DocumentChunk(
                document_id=path.stem,
                chunk_id=f"{path.stem}_c{i+1}",
                text=part,
                source=str(path),
                title=path.stem,
                source_reliability=source_reliability,
            ))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps([c.to_dict() for c in chunks], indent=2, ensure_ascii=False), encoding="utf-8")
    return chunks
