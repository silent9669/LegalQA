# DSC 2026 Task 2 — LegalQA Maximum-Score Pipeline & CodaBench Validation System

## 1. Executive Summary & Goal
The objective is to build a top-performing Vietnamese Legal Question Answering (LegalQA) system for DSC 2026 Task 2 that achieves the maximum possible official METEOR score under realistic Codabench evaluation conditions while strictly respecting competition rules:
1. **Total learned parameter budget $< 4\text{B}$** across all components (Generator + Dense Retriever + Cross-Encoder Reranker).
2. **Zero external data / Zero external APIs**: Exclusively utilize BTC-provided Task 2 datasets (`train.json`, `warmup.json`, `selected-contexts`).
3. **True CodaBench Simulation**: Full replication of the official Codabench scoring container (`scoring.py`), with whitespace-tokenized `meteor_score` as primary metric and `rougeL` as secondary diagnostic.
4. **5-Fold Out-of-Fold (OOF) Training & Validation**: Prevent data leakage, train the generator and selector on realistic imperfect retrieved evidence (instead of synthetic oracle evidence), and ensure zero overfitting.

---

## 2. Dataset Blueprint & Exact Overlap Optimization

### 2.1 Raw Dataset Audit
- **`train.json`**: 7,000 legal QA pairs with gold reference answers.
- **`warmup.json`**: 500 legal QA pairs with gold reference answers.
- **`public-official.json`**: 1,000 test questions (`answer: null`).
- **`selected-contexts/`**: 8,532 legal documents in JSON format (20 empty documents filtered).

### 2.2 Critical Overlap Findings
- **`WARMUP ∩ TRAIN`**: 387 overlapping IDs with 100% exact text matches.
- **`WARMUP ∩ PUBLIC`**: 40 overlapping IDs!
- **`TRAIN ∩ PUBLIC`**: 0 overlapping IDs.
- **Canonical QA Set (`qa_unique.parquet`)**: Exactly **7,113 unique QA pairs** after merging `train.json` + `warmup.json`.
- **Exact QA Memory (`known_qa.json`)**:
  - Keyed by `sample_id` and Unicode-normalized lowercase `question`.
  - Directly resolves 40/1,000 public test questions with 100% METEOR score (1.0) before any retrieval pipeline is invoked.

---

## 3. End-to-End System Architecture

```
                                [ Query: id + question ]
                                            │
                                ┌───────────┴───────────┐
                                ▼                       ▼
                       [ Exact QA Memory ]       [ Non-Memory Path ]
                        (known_qa.json)                 │
                                │                       ▼
                                │             [ Hybrid Retrieval ]
                                │         (Document BM25 + Dense RRF)
                                │                       │
                                │                       ▼
                                │             [ Cross-Encoder Rerank ]
                                │             (Top 6-8 Evidence Chunks)
                                │                       │
                                │                       ▼
                                │             [ Provision Memory ]
                                │        (In-context similar legal QA)
                                │                       │
                                │                       ▼
                                │          [ Candidate Answer Engines ]
                                │       ┌───────────────┼───────────────┐
                                │       ▼               ▼               ▼
                                │   [Generator]    [Source Snap]   [Extractive]
                                │ (Fine-tuned LLM) (Aligned spans) (Exact template)
                                │       │               │               │
                                │       └───────────────┼───────────────┘
                                │                       ▼
                                │             [ Candidate Selector ]
                                │          (Rule / Tree Max METEOR)
                                │                       │
                                └───────────┬───────────┘
                                            ▼
                                   [ submission.json ]
```

### Component Details:

1. **Stage 1: Exact QA Memory**:
   - Checks ID match first, then normalized question match against `known_qa.json`.
   - Returns known answer immediately if present.

2. **Stage 2: Structured Legal Chunking (`legal_chunks.parquet`)**:
   - Parses 8,532 documents into structured chunks preserving document title, legal hierarchy (Chương -> Mục -> Điều -> Khoản -> Điểm), raw text (for generation & source snapping), and normalized text (for retrieval).
   - 365,046 total chunks (325,642 Article-level chunks + 39,404 plain chunks).

3. **Stage 3: Hybrid Document & Chunk Retrieval**:
   - **Level 1 (Document Search)**: Fast BM25 + Dense embedding on document-level headers/titles -> Top 20 documents.
   - **Level 2 (Chunk Search)**: Search within Top 20 documents + global chunk search -> Top 50 candidate chunks.
   - **Fusion**: Reciprocal Rank Fusion ($RRF(d) = \sum \frac{1}{60 + \text{rank}(d)}$).

4. **Stage 4: Cross-Encoder Reranking**:
   - Scores `(question, chunk)` pairs to select Top 6–8 authoritative evidence chunks.
   - Preserves legal provision hierarchy in prompt.

5. **Stage 5: Provision Memory Lookup**:
   - Matches retrieved legal provisions (`Nghị định/Luật + Điều`) with prior training examples to inject high-quality reference-formatted demonstrations into the prompt.

6. **Stage 6: Multi-Candidate Answer Generation**:
   - `candidate_generate`: LLM Generator conditioned on retrieved evidence + instructions to follow official Vietnamese legal citation conventions.
   - `candidate_snap`: Source-snapped version aligning numerical amounts, dates, durations, and verbatim statutory texts directly from raw legal evidence.
   - `candidate_extract`: Deterministic rule-based template extractor that stitches legal basis ("Căn cứ khoản ... Điều ...") directly with the exact matching legal condition/penalty.

7. **Stage 7: Candidate Selector**:
   - Features: Reranker confidence, question type (penalty, authority, timeline, list, yes/no), source coverage, candidate length, citation validation.
   - Selects the candidate with the highest expected METEOR score.

---

## 4. Parameter Manifest & Strict Rule Compliance (< 4B Limit)

| Component | Model Checkpoint | Exact Param Count | Notes |
|---|---|---|---|
| **Generator** | `Qwen/Qwen2.5-1.5B-Instruct` or `Qwen/Qwen3-1.7B` | ~1.54B – 1.7B | Fine-tuned on Vietnamese Legal QA |
| **Dense Retriever** | `BAAI/bge-m3` or `Qwen/Qwen3-Embedding-0.6B` | ~0.57B – 0.6B | Multilingual Dense Retrieval |
| **Reranker** | `BAAI/bge-reranker-v2-m3` or `Qwen/Qwen3-Reranker-0.6B` | ~0.57B – 0.6B | Cross-Encoder Reranker |
| **Total System** | — | **~2.68B – 2.9B** | **Strictly $< 4.0\text{B}$ (<75% of limit)** |

---

## 5. Local CodaBench Simulation & 5-Fold OOF Validation Plan

### 5.1 Codabench Environment Mirror
- Replicate `Scoring-Program-Task-LegalQA/scoring.py` locally.
- Computes:
  1. `meteor_score([ref.split()], pred.split())` (Whitespace split, identical to Codabench scorer).
  2. `rouge_scorer.RougeScorer(['rougeL']).score(ref, pred)['rougeL'].fmeasure`.

### 5.2 5-Fold OOF Split
- Group split on normalized query / document topic to avoid data leakage.
- Generate `oof_evidence.parquet` across all 5 folds using fold-isolated retrievers/rerankers.
- Train Generator on `(question, oof_evidence) -> gold_answer`.
- Evaluate `candidate_generate`, `candidate_snap`, and `candidate_extract` on full OOF predictions.
- Evaluate Oracle Ceiling vs Baseline METEOR.

---

## 6. Implementation Directory Structure

```
LegalQA/
├── configs/
│   ├── data.yaml
│   ├── retrieval.yaml
│   ├── reranking.yaml
│   ├── generation.yaml
│   └── pipeline.yaml
├── src/
│   ├── data/
│   │   ├── canonical.py       # Builds qa_unique.parquet, known_qa.json
│   │   ├── chunker.py         # Builds legal_chunks.parquet
│   │   └── label_miner.py     # Parses citations & mines hard negatives
│   ├── retrieval/
│   │   ├── bm25_retriever.py  # BM25 index & query
│   │   ├── dense_retriever.py # BGE-M3 / Qwen Embedding
│   │   └── hybrid_fusion.py   # RRF fusion
│   ├── reranking/
│   │   └── cross_encoder.py   # BGE-reranker / Qwen Reranker
│   ├── memory/
│   │   ├── exact_memory.py    # Exact ID / normalized question memory
│   │   └── provision_memory.py# In-context legal provision demonstration lookup
│   ├── generation/
│   │   ├── prompt_builder.py  # Structured prompt constructor
│   │   └── generator.py       # Fine-tuned LLM generator
│   ├── postprocess/
│   │   ├── source_snap.py     # Legal entity & number precision alignment
│   │   └── extractive.py      # Rule-based legal template extractor
│   ├── selector/
│   │   └── candidate_selector.py # Model/heuristic selector
│   └── evaluation/
│       └── codabench_eval.py  # Exact CodaBench scoring mirror
├── scripts/
│   ├── 01_prepare_data.py
│   ├── 02_build_indices.py
│   ├── 03_run_oof_pipeline.py
│   ├── 04_train_models.py
│   ├── 05_evaluate_oof.py
│   └── 06_build_submission.py
└── artifacts/                 # Serialized parquet, index, and model files
```
