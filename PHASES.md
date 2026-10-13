# Implementation phases

## Phase 1 — Environment
- Virtual environment
- Dependencies
- Configuration
- Seed control
- Tests

## Phase 2 — Data ingestion
- PDF/TXT/MD/JSON reader
- Chunking with overlap
- Metadata preservation

## Phase 3 — Dense retrieval
- Sentence Transformer embeddings
- Normalized cosine similarity

## Phase 4 — Hybrid retrieval
- BM25 lexical retrieval
- Dense + BM25 weighted fusion
- Cross-encoder reranking

## Phase 5 — Poisoning simulation
- Factual poison
- Contradictory poison
- Indirect prompt-injection poison
- Ground-truth poison labels

## Phase 6 — Poison detection
- Injection indicators
- Synthetic-poison indicators
- Source reliability
- Retrieval/query anomaly
- Optional learned DeBERTa classifier training pipeline

## Phase 7 — Trust engine
`Trust = w1*Relevance + w2*SourceReliability + w3*Consistency + w4*EvidenceSupport - w5*PoisonRisk`

Weights are configuration variables and must be tuned on development data only.

## Phase 8 — Evidence verification
- NLI entailment/contradiction/neutral labels
- Offline lexical fallback for development

## Phase 9 — Source attribution
- Claim extraction
- Claim/evidence overlap
- NLI support
- Trust-weighted attribution score

## Phase 10 — Self-healing
- Quarantine suspicious evidence
- Exclude quarantined chunks
- Re-retrieve alternatives
- Re-score and verify
- Regenerate
- Expose uncertainty/abstain if evidence remains insufficient

## Phase 11 — Generation
- Safe system prompt
- Optional local instruct model
- Extractive fallback during development

## Phase 12 — Evaluation
- Retrieval: Recall@K, MRR
- Poisoning: precision, recall, F1, AUROC when labels/scores exist
- Security: attack success rate
- Generation: faithfulness/answer correctness should be evaluated against a manually validated test set
- Attribution: attribution precision/evidence coverage
- Recovery: self-healing recovery rate

## Phase 13 — UI and reproducibility
- Streamlit demo
- CSV/JSON experiment logging
- Unit tests
- Configuration-driven experiments
