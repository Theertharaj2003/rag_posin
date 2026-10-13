from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings
from data.demo_dataset import build_chunks
from data.ingest import chunk_text
from experiments.runner import make_pipeline
from generation.llm import GroundedGenerator
from healing.healing_controller import SelfHealingRAG
from poisoning.detector import PoisonDetector
from trust.trust_score import TrustEngine
from utils.schemas import DocumentChunk
from verification.evidence_verifier import EvidenceVerifier


def _extract_pdf_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    uploaded_file.seek(0)
    reader = PdfReader(uploaded_file)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n\n".join(pages).strip()


def _build_chunks_from_documents(documents: list[tuple[str, str]]) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for document_index, (document_name, text) in enumerate(documents, start=1):
        document_id = f"doc{document_index}_{Path(document_name).stem}"
        for chunk_index, part in enumerate(chunk_text(text, max_chars=1200, overlap=150), start=1):
            chunks.append(
                DocumentChunk(
                    document_id=document_id,
                    chunk_id=f"{document_id}_c{chunk_index}",
                    text=part,
                    source=document_name,
                    title=document_name,
                    source_reliability=0.75,
                )
            )
    return chunks


@st.cache_resource(show_spinner="Loading retrieval and verification models...")
def build_demo_app():
    settings = get_settings()
    retriever = make_pipeline(settings, build_chunks())
    detector = PoisonDetector()
    trust = TrustEngine(settings.trust_weights)
    verifier = EvidenceVerifier(settings.nli_model, settings.entailment_threshold, settings.nli_contradiction_threshold)
    generator = GroundedGenerator(settings.generator_model, settings.device)
    return SelfHealingRAG(retriever, detector, trust, verifier, generator, settings), settings


def build_document_app(documents: list[tuple[str, str]]):
    settings = get_settings()
    chunks = _build_chunks_from_documents(documents)
    if not chunks:
        raise ValueError("No searchable text was extracted from the uploaded PDFs.")
    retriever = make_pipeline(settings, chunks)
    detector = PoisonDetector()
    trust = TrustEngine(settings.trust_weights)
    verifier = EvidenceVerifier(settings.nli_model, settings.entailment_threshold, settings.nli_contradiction_threshold)
    generator = GroundedGenerator(settings.generator_model, settings.device)
    return SelfHealingRAG(retriever, detector, trust, verifier, generator, settings), chunks


def summarize_results(results):
    if not results:
        return {
            "avg_hybrid": 0.0,
            "avg_poison": 0.0,
            "avg_trust": 0.0,
            "verified_count": 0,
            "quarantined_count": 0,
        }
    return {
        "avg_hybrid": round(sum(r.hybrid_score for r in results) / len(results), 4),
        "avg_poison": round(sum(r.poison_risk for r in results) / len(results), 4),
        "avg_trust": round(sum(r.trust_score for r in results) / len(results), 4),
        "verified_count": sum(1 for r in results if r.verified),
        "quarantined_count": sum(1 for r in results if r.quarantined),
    }


def describe_models(settings, system, generation) -> list[dict[str, str]]:
    retriever = system.retriever
    reranker = retriever.reranker
    verifier = system.verifier
    generator = system.generator
    reranker_active = bool(reranker.enabled and reranker.model is not None)
    verifier_active = verifier.pipeline is not None
    generated_with_configured_model = generation.model == settings.generator_model
    reranker_error = getattr(reranker, "load_error", "")
    verifier_error = verifier.load_error or ""
    generator_error = generator.load_error or ""

    return [
        {
            "Component": "Dense retrieval / embeddings",
            "Model or method": settings.dense_model,
            "Status": "Loaded and used",
            "Details": "Encodes document chunks and the question; dense similarity contributes to the hybrid retrieval score.",
        },
        {
            "Component": "Lexical retrieval",
            "Model or method": "BM25 (rank-bm25)",
            "Status": "Used",
            "Details": "Ranks chunks by term overlap; this is a statistical retrieval method, not a neural model.",
        },
        {
            "Component": "Hybrid retrieval",
            "Model or method": f"Dense {settings.dense_weight:.0%} + BM25 {settings.bm25_weight:.0%}",
            "Status": "Used",
            "Details": "Combines normalized dense and BM25 scores before optional reranking.",
        },
        {
            "Component": "Cross-encoder reranker",
            "Model or method": settings.reranker_model,
            "Status": "Loaded and used" if reranker_active else "Not active",
            "Details": (
                f"Reranks retrieved chunks; configured={settings.rerank_enabled}."
                if reranker_active
                else (
                    f"Hybrid retrieval order is used without cross-encoder reranking. Load error: {reranker_error}"
                    if reranker_error
                    else "Hybrid retrieval order is used without cross-encoder reranking."
                )
            ),
        },
        {
            "Component": "Evidence verification (NLI)",
            "Model or method": settings.nli_model,
            "Status": "Initialized" if verifier_active else "Lexical fallback active",
            "Details": (
                f"Checks whether evidence entails the question/claim. Entailment threshold: {settings.entailment_threshold:.2f}; "
                f"contradiction threshold: {settings.nli_contradiction_threshold:.2f}. "
                "The implementation falls back to token-overlap heuristics if NLI inference fails."
                if verifier_active
                else (
                    "The configured NLI model did not initialize; verification uses deterministic token-overlap heuristics. "
                    f"Load error: {verifier_error}"
                    if verifier_error
                    else "The configured NLI model did not initialize; verification uses deterministic token-overlap heuristics."
                )
            ),
        },
        {
            "Component": "Poison detection",
            "Model or method": "Built-in rule-based detector",
            "Status": "Used",
            "Details": (
                f"Checks instruction-like phrases, source reliability, query overlap, and experiment poison labels. "
                f"The configured classifier ({settings.poison_classifier}) is not currently called by this detector."
            ),
        },
        {
            "Component": "Answer generation",
            "Model or method": generation.model,
            "Status": (
                "Configured model used"
                if generated_with_configured_model
                else "Extractive fallback used"
                if generation.model == "extractive_fallback"
                else "Abstained"
            ),
            "Details": (
                f"Configured generator: {settings.generator_model}. "
                + (
                    "The answer was produced by the configured text-generation model."
                    if generated_with_configured_model
                    else (
                        "Extractive sentence selection was used. The configured generator loads only when explicitly requested; "
                        "it was not loaded for this run."
                        + (f" Previous load error: {generator_error}" if generator_error else "")
                    )
                    if generation.model == "extractive_fallback"
                    else "No verified evidence was available, so the system abstained."
                )
            ),
        },
        {
            "Component": "Execution device",
            "Model or method": settings.device,
            "Status": "Configured",
            "Details": "DEVICE environment setting passed to model components; 'auto' lets the libraries select a device.",
        },
    ]


def describe_configured_models(settings) -> list[dict[str, str]]:
    return [
        {
            "Pipeline stage": "Semantic retrieval",
            "Configured model or method": settings.dense_model,
            "Purpose and configuration": "Embeds document chunks and queries for semantic similarity.",
        },
        {
            "Pipeline stage": "Lexical retrieval",
            "Configured model or method": "BM25 (rank-bm25)",
            "Purpose and configuration": "Finds term-matching passages; blended with dense retrieval "
            f"at {settings.bm25_weight:.0%} (dense weight: {settings.dense_weight:.0%}).",
        },
        {
            "Pipeline stage": "Reranking",
            "Configured model or method": settings.reranker_model,
            "Purpose and configuration": (
                f"Cross-encoder reranks candidates; enabled={settings.rerank_enabled}, "
                f"rerank candidate count={settings.rerank_top_k}."
            ),
        },
        {
            "Pipeline stage": "Evidence verification",
            "Configured model or method": settings.nli_model,
            "Purpose and configuration": (
                "Natural-language inference estimates entailment and contradiction; "
                f"thresholds={settings.entailment_threshold:.2f}/{settings.nli_contradiction_threshold:.2f}. "
                "A lexical heuristic fallback is used if model loading/inference fails."
            ),
        },
        {
            "Pipeline stage": "Poison detection",
            "Configured model or method": "Rule-based detector",
            "Purpose and configuration": (
                f"Scores injection phrases, source reliability, query overlap, and experiment labels. "
                f"The configured classifier {settings.poison_classifier} is not currently connected."
            ),
        },
        {
            "Pipeline stage": "Answer generation",
            "Configured model or method": settings.generator_model,
            "Purpose and configuration": (
                "Configured local text-generation model. It is loaded only when explicitly requested; "
                "the current default answer path uses extractive sentence selection."
            ),
        },
        {
            "Pipeline stage": "Self-healing",
            "Configured model or method": "Trust scoring + quarantine + re-retrieval",
            "Purpose and configuration": (
                f"Up to {settings.max_healing_rounds} healing rounds; requires "
                f"{settings.min_trusted_evidence} trusted evidence chunks; "
                f"quarantines at risk threshold {settings.quarantine_threshold:.2f}."
            ),
        },
        {
            "Pipeline stage": "Execution device",
            "Configured model or method": settings.device,
            "Purpose and configuration": "Device selection passed to compatible model components; 'auto' delegates selection to the libraries.",
        },
    ]


st.set_page_config(page_title="Trust-Aware Self-Healing RAG", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1440px; padding-top: 2.2rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(39, 68, 104, .12), rgba(72, 126, 154, .08));
        border: 1px solid rgba(120, 150, 180, .22);
        padding: 1rem 1.1rem;
        border-radius: 14px;
    }
    [data-testid="stMetricLabel"] p {font-size: .88rem;}
    [data-testid="stMetricValue"] {font-size: 1.55rem;}
    div.stButton > button[kind="primary"] {min-height: 2.9rem; font-weight: 650;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Trust-Aware and Self-Healing RAG")
st.caption("Ask questions over your documents and inspect the evidence and safety signals behind each answer.")

left, input_column, right = st.columns([1, 2, 1])
with input_column:
    with st.container(border=True):
        st.subheader("Upload and ask")
        uploaded_files = st.file_uploader(
            "Upload one or more PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )
        query = st.text_input("Your question", "What is knowledge poisoning in RAG?")
        use_local_generator = st.checkbox(
            f"Use configured generator ({get_settings().generator_model})",
            value=False,
            help="Loads the local generation model on the first run. This may require a large download and substantial memory.",
        )
        run_button = st.button("Run analysis", type="primary", width="stretch")
        st.caption("Without a PDF, the built-in demo corpus will be used. Retrieval and verification models may download on first run.")

st.subheader("Configured models and pipeline")
with st.container(border=True):
    configured_model_rows = describe_configured_models(get_settings())
    for row in configured_model_rows:
        model_column, detail_column = st.columns([1, 2], vertical_alignment="center")
        model_column.markdown(f"**{row['Pipeline stage']}**")
        model_column.code(row["Configured model or method"], language=None)
        detail_column.write(row["Purpose and configuration"])
    st.caption(
        "Configured values are listed here. After analysis, open the Models used tab to see which components loaded and ran, including fallbacks."
    )

documents: list[tuple[str, str]] = []
input_signature_parts = [query.strip(), f"use_local_generator={use_local_generator}"]
if uploaded_files:
    try:
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.getvalue()
            input_signature_parts.append(
                f"{uploaded_file.name}:{hashlib.sha256(file_bytes).hexdigest()}"
            )
            uploaded_file.seek(0)
            text = _extract_pdf_text(uploaded_file)
            if text:
                documents.append((uploaded_file.name, text))
            else:
                st.warning(f"No text could be extracted from {uploaded_file.name}. Scanned PDFs may need OCR.")
        if documents:
            st.caption(
                f"Loaded {len(documents)} PDF(s) · "
                f"{sum(len(text) for _, text in documents):,} extracted characters"
            )
    except Exception as exc:
        st.error(f"Could not read an uploaded PDF: {exc}")
        st.stop()
input_signature = hashlib.sha256(
    "\n".join(input_signature_parts).encode("utf-8")
).hexdigest()

if run_button:
    st.session_state.pop("analysis_report", None)
    try:
        if uploaded_files:
            if not documents:
                st.error("No searchable text was extracted. Upload a text-based PDF and try again.")
                st.stop()
            with st.spinner("Processing PDFs and running the trust-aware RAG pipeline..."):
                system, chunks = build_document_app(documents)
                generation, trace, results = system.answer(
                    query,
                    use_local_model=use_local_generator,
                )
            settings = get_settings()
            doc_names = [name for name, _ in documents]
        else:
            with st.spinner("Running the demo corpus..."):
                system, settings = build_demo_app()
                generation, trace, results = system.answer(
                    query,
                    use_local_model=use_local_generator,
                )
            doc_names = ["demo_dataset"]

        summary = summarize_results(results)
        model_details = describe_models(settings, system, generation)
        evidence_rows = [
            {
                "chunk_id": r.chunk.chunk_id,
                "source": r.chunk.source,
                "hybrid_score": round(r.hybrid_score, 4),
                "poison_risk": round(r.poison_risk, 4),
                "trust_score": round(r.trust_score, 4),
                "verification_label": r.verification_label,
                "verified": r.verified,
                "quarantined": r.quarantined,
                "text": r.chunk.text,
            }
            for r in results
        ]
        st.session_state["analysis_report"] = {
            "documents": doc_names,
            "query": query,
            "use_local_generator": use_local_generator,
            "answer": generation.answer,
            "model": generation.model,
            "summary": summary,
            "recovered": trace.recovered,
            "quarantined_ids": trace.quarantined_ids,
            "rounds": trace.rounds,
            "attributions": generation.attributions,
            "models": model_details,
            "evidence": evidence_rows,
            "input_signature": input_signature,
        }
    except Exception as exc:
        st.error(str(exc))
        st.info("The first run may download model weights. If the app still fails, check the Python environment and the requirements installation.")

report = st.session_state.get("analysis_report")
if report:
    st.markdown("---")
    if report.get("input_signature") != input_signature:
        st.warning("The PDFs or question changed since this report was created. Run analysis again to refresh these results.")
    st.subheader("Grounded answer")
    st.write(report["answer"])
    st.caption(f"Generator: {report['model']} · Sources: {', '.join(report['documents'])}")

    summary = report["summary"]
    evidence_rows = report["evidence"]
    evidence_count = len(evidence_rows)
    verified_rate = summary["verified_count"] / evidence_count if evidence_count else 0.0
    quarantined_rate = summary["quarantined_count"] / evidence_count if evidence_count else 0.0
    attribution_scores = [
        max(
            item["score"]
            for item in group
            if isinstance(item, dict) and isinstance(item.get("score"), (int, float))
        )
        for group in report["attributions"]
        if isinstance(group, list)
        and any(
            isinstance(item, dict) and isinstance(item.get("score"), (int, float))
            for item in group
        )
    ]
    average_attribution = (
        sum(attribution_scores) / len(attribution_scores) if attribution_scores else 0.0
    )
    metric_columns = st.columns(5)
    metric_columns[0].metric("Recovery", "Recovered" if report["recovered"] else "Not recovered")
    metric_columns[1].metric("Verified evidence", f"{summary['verified_count']}/{evidence_count}")
    metric_columns[2].metric("Average trust", f"{summary['avg_trust']:.3f}")
    metric_columns[3].metric("Average poison risk", f"{summary['avg_poison']:.3f}")
    metric_columns[4].metric("Quarantined", summary["quarantined_count"])

    evidence_df = pd.DataFrame(evidence_rows)
    st.subheader("Analysis metrics")
    st.caption("Measured for this question and its retrieved evidence; these are not benchmark classifier metrics.")
    metrics_df = pd.DataFrame(
        [
            {"Metric": "Retrieved chunks", "Value": evidence_count, "How to read it": "Evidence candidates returned for this question."},
            {"Metric": "Average hybrid retrieval score", "Value": f"{summary['avg_hybrid']:.3f}", "How to read it": "Combined dense and lexical retrieval score; not a probability."},
            {"Metric": "Average trust score", "Value": f"{summary['avg_trust']:.3f}", "How to read it": "Mean trust score assigned to retrieved evidence."},
            {"Metric": "Average estimated poison risk", "Value": f"{summary['avg_poison']:.3f}", "How to read it": "Mean heuristic/model risk score; not ground truth."},
            {"Metric": "Verified evidence rate", "Value": f"{verified_rate:.1%}", "How to read it": f"{summary['verified_count']} of {evidence_count} retrieved chunks."},
            {"Metric": "Quarantine rate", "Value": f"{quarantined_rate:.1%}", "How to read it": f"{summary['quarantined_count']} of {evidence_count} retrieved chunks."},
            {"Metric": "Recovery", "Value": "Yes" if report["recovered"] else "No", "How to read it": "Whether trusted evidence met the configured recovery requirement."},
            {"Metric": "Attributed claims", "Value": len(report["attributions"]), "How to read it": "Answer claims for which attribution details were produced."},
            {"Metric": "Mean attribution score", "Value": f"{average_attribution:.3f}" if attribution_scores else "N/A", "How to read it": "Mean claim-to-evidence score; N/A when no claims were attributed."},
            {"Metric": "Precision / recall / F1 / accuracy", "Value": "N/A", "How to read it": "Requires labeled ground-truth examples; not inferable from a single PDF question."},
        ]
    )
    st.dataframe(metrics_df, width="stretch", hide_index=True)

    tabs = st.tabs(["Overview", "Evidence", "Models used", "Attribution and trace", "Export"])
    with tabs[0]:
        overview_columns = st.columns(3)
        overview_columns[0].metric("PDFs analyzed", len(report["documents"]) if report["documents"] != ["demo_dataset"] else "Demo")
        overview_columns[1].metric("Chunks retrieved", len(evidence_rows))
        overview_columns[2].metric("Sources represented", evidence_df["source"].nunique() if not evidence_df.empty else 0)
        st.subheader("Retrieved evidence scores")
        if not evidence_df.empty:
            score_chart = evidence_df.set_index("chunk_id")[
                ["hybrid_score", "trust_score", "poison_risk"]
            ]
            st.bar_chart(score_chart, width="stretch")
            verification_counts = evidence_df["verification_label"].value_counts()
            st.subheader("Verification outcomes")
            st.bar_chart(verification_counts, width="stretch")
        st.info(
            "These are query-level retrieval and safety signals, not benchmark accuracy measurements. "
            "Poison risk is an estimate; verify important answers against the source documents."
        )
    with tabs[1]:
        st.subheader("Evidence table")
        st.dataframe(
            evidence_df.drop(columns=["text"], errors="ignore"),
            width="stretch",
            hide_index=True,
        )
        st.subheader("Evidence snippets")
        for row in evidence_rows:
            with st.expander(
                f"{row['chunk_id']} · trust {row['trust_score']:.3f} · risk {row['poison_risk']:.3f}"
            ):
                st.caption(f"Source: {row['source']} · Verification: {row['verification_label']}")
                st.write(row["text"])
    with tabs[2]:
        st.subheader("Models and methods used")
        st.caption("This reflects the active pipeline and the backend used for this specific answer, not only config.yaml.")
        st.dataframe(
            pd.DataFrame(report["models"]),
            width="stretch",
            hide_index=True,
        )
        with st.expander("Model and method notes"):
            st.markdown(
                """
                - **Dense embeddings** represent each document chunk and question as vectors for semantic search.
                - **BM25** provides lexical matching and is combined with dense search in the hybrid retriever.
                - **Cross-encoder reranking** scores each query/chunk pair when the reranker successfully loads.
                - **NLI verification** estimates entailment and contradiction between a question/claim and evidence. If it cannot load, the app uses its lexical heuristic fallback.
                - **Poison detection** currently uses built-in rules; the configured DeBERTa classifier is not wired into this detector.
                - **Answer generation** uses extractive sentence selection by default; enable the generator option in the input form to explicitly load and use the configured Qwen model.
                """
            )
    with tabs[3]:
        st.subheader("Claim attribution")
        st.json(report["attributions"])
        st.subheader("Self-healing trace")
        st.json({
            "query": report["query"],
            "recovered": report["recovered"],
            "quarantined_ids": report["quarantined_ids"],
            "rounds": report["rounds"],
        })
    with tabs[4]:
        st.subheader("Download results")
        csv_evidence_df = evidence_df.drop(columns=["text"], errors="ignore")
        export_data = {
            "documents": report["documents"],
            "query": report["query"],
            "use_local_generator": report["use_local_generator"],
            "answer": report["answer"],
            "generator": report["model"],
            "metrics": report["summary"],
            "recovered": report["recovered"],
            "quarantined_ids": report["quarantined_ids"],
            "attributions": report["attributions"],
            "models": report["models"],
            "evidence": report["evidence"],
            "rounds": report["rounds"],
        }
        download_columns = st.columns(2)
        download_columns[0].download_button(
            "Download evidence CSV",
            data=csv_evidence_df.to_csv(index=False).encode("utf-8"),
            file_name="rag_evidence.csv",
            mime="text/csv",
            width="stretch",
        )
        download_columns[1].download_button(
            "Download full report JSON",
            data=json.dumps(export_data, ensure_ascii=False, indent=2),
            file_name="rag_report.json",
            mime="application/json",
            width="stretch",
        )
else:
    st.info("Upload one or more PDFs, enter a question, and click 'Run analysis' to view results.")
