# Trust-Aware and Self-Healing RAG

Research implementation for:

**Trust-Aware and Self-Healing Retrieval-Augmented Generation Framework for Knowledge Poisoning Detection, Source Attribution, and Secure Evidence Grounding**

## What this implementation contains

1. Document ingestion and chunking
2. Dense retrieval with BGE-style Sentence Transformers
3. BM25 lexical retrieval
4. Hybrid retrieval
5. Cross-encoder reranking
6. Controlled knowledge-poisoning generation (factual, contradictory, indirect-instruction)
7. Rule-based poisoning risk detection
8. Optional DeBERTa classifier hook for learned poisoning detection
9. Dynamic evidence trust scoring
10. NLI-based evidence verification
11. Claim-level source attribution
12. Self-healing retrieval controller with quarantine and re-retrieval
13. Baseline vs defended experiment runners
14. Retrieval/security/faithfulness/attribution/recovery metrics
15. CSV + JSON experiment logging
16. Streamlit demonstration UI
17. Pytest unit tests and a lightweight smoke-test mode

## Research caution

This repository is an experimental research framework. It does not claim that the proposed method is optimal or that poisoning/hallucination can be eliminated. Thresholds and weights are configurable and must be tuned only on training/development data, then frozen for test evaluation.

## Recommended research models

- Embeddings: `BAAI/bge-base-en-v1.5`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Poison classifier: `microsoft/deberta-v3-base` (fine-tune on labeled poison data before using as a learned detector)
- NLI verifier: `facebook/bart-large-mnli`
- Optional local generator: `Qwen/Qwen2.5-7B-Instruct` or another compatible instruct model

The code does **not** download a 7B generator during the first setup. Retrieval, trust, verification, and healing can be tested first.

## Quick start on Windows PowerShell

```powershell
cd trust_aware_rag
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python run.py --mode demo
```

If PowerShell blocks activation, use:

```powershell
.\.venv\Scripts\python.exe run.py --mode demo
```

## First real-model run

After dependencies are installed and model downloads are permitted:

```powershell
python run.py --mode retrieve --query "What is retrieval augmented generation?"
```

Then run the complete defense pipeline:

```powershell
python run.py --mode pipeline --query "What is retrieval augmented generation?"
```

Run the controlled experiment:

```powershell
python run.py --mode experiment
```

Launch the UI:

```powershell
streamlit run app/main.py
```

The Streamlit configuration disables the source file watcher because imported NLP libraries can trigger noisy watcher errors in some Windows environments.

The Streamlit UI supports one or more PDF uploads and shows the grounded answer, a visible analysis-metrics table, per-query retrieved-evidence scores, verification outcomes, recovery status, attribution details, the self-healing trace, and a detailed Models used tab. A pre-analysis overview lists configured models and methods; after analysis, the Models used tab distinguishes configured models from components actually used (for example, the default extractive answer path and rule-based poison detector). Extractive answers are the default. Enable **Use configured generator** to load and use the configured local generator; this may download large model weights and requires substantial memory. Results can be downloaded as an evidence CSV or a full JSON report. Precision, recall, F1, and accuracy are shown as unavailable in this single-query view because they require labeled ground-truth examples. Scanned PDFs without an embedded text layer require OCR before upload; the displayed scores describe retrieved evidence and are not ground-truth benchmark scores.

## Project structure

```text
trust_aware_rag/
├── app/
├── config/
├── data/
│   ├── raw/
│   ├── clean/
│   ├── poisoned/
│   └── processed/
├── models/
├── retrieval/
├── poisoning/
├── trust/
├── verification/
├── healing/
├── generation/
├── evaluation/
├── experiments/
├── tests/
├── results/
├── logs/
├── requirements.txt
├── .env.example
├── config.yaml
└── run.py
```

## Research experiment design

The experiment runner creates four conceptual conditions:

- **clean_rag** — normal RAG over clean evidence
- **poisoned_rag** — normal RAG over a corpus containing controlled poison
- **trust_rag** — poisoning-aware retrieval/trust filtering without recovery
- **self_healing_rag** — poisoning-aware trust + verification + re-retrieval + regeneration

Metrics are computed from observed runs; no performance numbers are hard-coded.

## Recommended dissertation ablations

A. Dense retrieval only
B. BM25 only
C. Hybrid retrieval
D. Hybrid + reranker
E. Hybrid + trust scoring
F. Hybrid + trust + verification
G. Full self-healing framework

For each setting vary attack type and poisoning ratio. Keep the test queries fixed across methods.
