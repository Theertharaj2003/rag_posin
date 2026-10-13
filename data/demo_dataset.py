from __future__ import annotations

from pathlib import Path

from utils.schemas import DocumentChunk
from utils.common import json_dump


DEMO_DOCS = [
    {
        "document_id": "doc_rag_001",
        "title": "Retrieval-Augmented Generation Overview",
        "source": "research_notes",
        "source_reliability": 0.90,
        "text": (
            "Retrieval-Augmented Generation combines a retrieval component with a language model. "
            "The retriever selects external evidence relevant to a user query, and the generator uses that evidence to formulate an answer. "
            "The retrieval corpus can be a collection of documents, web pages, or enterprise records."
        ),
    },
    {
        "document_id": "doc_rag_002",
        "title": "RAG Retrieval and Generation",
        "source": "research_notes",
        "source_reliability": 0.90,
        "text": (
            "A typical RAG pipeline first converts documents into chunks and embeddings, stores them in a searchable index, "
            "retrieves candidate passages for a query, and supplies selected passages to a generator. "
            "Hybrid retrieval can combine lexical matching such as BM25 with semantic vector retrieval."
        ),
    },
    {
        "document_id": "doc_trust_001",
        "title": "Evidence Trust in RAG",
        "source": "research_notes",
        "source_reliability": 0.88,
        "text": (
            "Trust-aware RAG does not treat every retrieved passage as equally reliable. "
            "A trust score can combine query relevance, source reliability, agreement with other evidence, support for the claim, "
            "and estimated poisoning risk. Low-trust evidence can be down-ranked or quarantined."
        ),
    },
    {
        "document_id": "doc_security_001",
        "title": "Knowledge Poisoning",
        "source": "security_notes",
        "source_reliability": 0.92,
        "text": (
            "Knowledge poisoning occurs when malicious or misleading content is inserted into a retrieval corpus so that a RAG system "
            "retrieves the manipulated evidence. A successful defense should detect suspicious evidence and avoid grounding an answer in it."
        ),
    },
    {
        "document_id": "doc_healing_001",
        "title": "Self-Healing Retrieval",
        "source": "research_notes",
        "source_reliability": 0.86,
        "text": (
            "A self-healing RAG workflow can quarantine suspicious passages, re-retrieve alternatives, verify the replacement evidence, "
            "and regenerate the answer. The system should expose uncertainty or abstain when adequate verified evidence is unavailable."
        ),
    },
]

QUESTIONS = [
    {
        "id": "q1",
        "question": "What is retrieval augmented generation?",
        "answer": "RAG combines retrieval of external evidence with language-model generation.",
    },
    {
        "id": "q2",
        "question": "What is knowledge poisoning in RAG?",
        "answer": "Knowledge poisoning inserts malicious or misleading content into the retrieval corpus to manipulate generated answers.",
    },
    {
        "id": "q3",
        "question": "How can a self-healing RAG system recover from suspicious evidence?",
        "answer": "It can quarantine suspicious evidence, retrieve alternatives, verify them, and regenerate the answer.",
    },
    {
        "id": "q4",
        "question": "What signals can be used in a trust score?",
        "answer": "Relevance, source reliability, cross-source consistency, evidence support, and poisoning risk can contribute to trust.",
    },
]


def build_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            document_id=d["document_id"],
            chunk_id=f"{d['document_id']}_c1",
            text=d["text"],
            source=d["source"],
            title=d["title"],
            source_reliability=d["source_reliability"],
        )
        for d in DEMO_DOCS
    ]


def write_demo_data(root: Path) -> None:
    clean = [c.to_dict() for c in build_chunks()]
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    json_dump(clean, processed / "clean_chunks.json")
    json_dump(QUESTIONS, processed / "evaluation_questions.json")
