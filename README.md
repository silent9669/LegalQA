# DSC 2026 Task 2 — LegalQA Maximum-Score Pipeline

This repository implements the end-to-end Legal Question Answering system for DSC 2026 Task 2, designed to maximize the official **METEOR** score under strict competition constraints ($< 4\text{B}$ total learned parameter budget, zero external data/APIs, fully offline self-contained inference).

---

## 1. Component Implementation Status

To ensure engineering rigor and clear team visibility, every pipeline component is classified under a strict 5-tier status taxonomy:

| Component | Status | Details |
|---|---|---|
| **Exact QA Memory** | **Implemented** | Exact ID lookup + normalized-question lookup across 7,113 canonical QA pairs. |
| **BM25 Inverted Index Retriever** | **Implemented** | Lexical retrieval over 365,046 canonical chunks (18 columns, 425.5 MB). |
| **Lexical Overlap Reranker** | **Implemented** | Zero-parameter Jaccard/overlap token reranking (`SimpleLexicalReranker`). |
| **Article Stitcher** | **Implemented** | Multi-part article context expansion and sibling chunk assembly. |
| **Extractive Answer Generator** | **Implemented** | Deterministic extractive candidate generation from retrieved provisions. |
| **Source Snapping** | **Implemented** | Align penalties, dates, citations, and durations to source context. |
| **Rule-based Candidate Selector** | **Implemented** | Penalty keyword and extractive confidence heuristic selection. |
| **Exact CodaBench Scorer** | **Implemented** | Whitespace-tokenized NLTK METEOR and unstemmed ROUGE-L matching official platform. |
| **Artifact Governance & Audit** | **Implemented** | Manifest generation (`artifacts.json`, `models.json`) and SHA-256 integrity verification. |
| **Submission v1 (BM25 Baseline)** | **Measured** | Official CodaBench score: `METEOR = 0.2952`, `ROUGE-L = 0.3957`. |
| **Submission v2 (Article Stitcher)** | **Measured** | Official CodaBench score: `METEOR = 0.3781`, `ROUGE-L = 0.3718` (Top 26 Leaderboard). |
| **Sampled OOF Validation (Legacy)** | **Measured (Legacy Signal)** | `0.3542 ± 0.0223` on random 100-sample splits; superseded by full benchmark validation plan. |
| **Hybrid RRF Fusion** | **Configured but unused** | `src/retrieval/hybrid_fusion.py` implemented; inactive in current default prediction flow. |
| **Neural Configurations** | **Configured but unused** | `configs/models.yaml`, `retrieval.yaml`, `reranking.yaml`, `generation.yaml` defined. |
| **Dense Retriever (`Qwen3-Embedding-0.6B`)** | **Planned** | 595.8M parameters; planned for post-validation neural bake-off. |
| **Neural Cross-Encoder (`Qwen3-Reranker-0.6B`)** | **Planned** | 595.8M parameters; planned for post-validation neural bake-off. |
| **Grounded Generator (`Qwen3-1.7B`)** | **Planned** | 2,031.7M conservative parameters; planned for post-validation neural bake-off. |
| **Provision Demonstration Memory** | **Planned** | In-context demonstration lookup from grounded training provisions. |
| **Learned Candidate Selector** | **Planned** | Feature-based classifier to select optimal candidate answer per query type. |
| **Benchmark-Equivalent Full OOF Validation** | **Planned** | 7,113-row question-blocked and document-held-out 5-fold cross-validation. |
| **Pretrained Checkpoint Rule Legality** | **Unverified** | Awaiting formal clarification from organizers on pretrained checkpoint usage. |

---

## 2. Active Prediction Pipeline Flow

The active runtime prediction pipeline (`src/pipeline.py`) operates fully offline with deterministic zero-parameter components:

$$\text{Query} \xrightarrow{\text{Step 1}} \text{Exact Memory} \xrightarrow{\text{Miss}} \text{BM25 Retrieval} \xrightarrow{\text{Step 2}} \text{Lexical Rerank} \xrightarrow{\text{Step 3}} \text{Article Stitcher} \xrightarrow{\text{Step 4}} \text{Extractive + Snap} \xrightarrow{\text{Step 5}} \text{Rule Selector} \rightarrow \text{Prediction}$$

1. **Exact QA Memory Lookup**: Checks `sample_id` and normalized question against 7,113 canonical training pairs ($100\%$ precision on overlap).
2. **Lexical BM25 Retrieval**: Retrieves top-25 candidate chunks from `artifacts/chunks/legal_chunks.parquet` (365,046 legal clauses/points).
3. **Lexical Overlap Reranker**: Re-scores candidates using token intersection, Jaccard similarity, and citation alignment to top-8 evidence chunks.
4. **Article Stitcher**: Expands the top ranked chunk by finding all sibling parts of the same article (`doc_id`, `dieu`) and concatenating them in numerical order.
5. **Multi-Candidate Synthesis & Selection**: Generates extractive, snapped, and memory candidates; applies penalty-keyword rules to select the final response.

---

## 3. Project Structure

```text
LegalQA/
├── README.md                          # Pipeline documentation & status taxonomy
├── requirements.txt                   # Production dependencies
├── validation.py                      # Validation CLI entrypoint
│
├── configs/                           # Modular YAML configurations
│   ├── pipeline.yaml                  # Main pipeline orchestration settings
│   ├── models.yaml                    # Model parameter manifest (JSON-compatible YAML)
│   ├── retrieval.yaml                 # BM25 & dense retrieval parameters
│   ├── reranking.yaml                 # Cross-encoder reranker parameters
│   └── generation.yaml                # Generator & LoRA fine-tuning parameters
│
├── artifacts/                         # Authoritative data hierarchy
│   ├── raw/                           # Official Task 2 inputs (train, warmup, public, contexts)
│   ├── chunks/                        # legal_chunks.parquet (365,046 rows, 425.5 MB), chunks_output.jsonl
│   ├── data/                          # qa_unique.parquet (7,113 rows), known_qa.json
│   ├── labels/                        # retrieval_labels.parquet (7,113 rows)
│   ├── manifests/                     # artifacts.json (8,544 files), models.json (< 4B budget)
│   ├── submissions/                   # submission.json, submission.json.zip
│   └── archive/                       # Quarantined legacy artifacts (git-ignored)
│
├── src/                               # Modular source implementation
│   ├── data/                          # canonical.py, chunker.py, label_miner.py, artifact_manifest.py
│   ├── evaluation/                    # codabench_eval.py (Exact CodaBench METEOR & ROUGE-L)
│   ├── memory/                        # exact_memory.py, provision_memory.py
│   ├── retrieval/                     # bm25_retriever.py, query_analyzer.py, hybrid_fusion.py
│   ├── reranking/                     # cross_encoder.py
│   ├── generation/                    # prompt_builder.py
│   ├── postprocess/                   # extractive.py, article_stitcher.py, source_snap.py
│   ├── selector/                      # candidate_selector.py
│   ├── utils/                         # manifest.py (< 4B parameter budget validator)
│   └── pipeline.py                    # End-to-end LegalQAPipeline class
│
├── scripts/                           # Executable operational CLIs
│   ├── audit_artifacts.py             # Canonical artifact audit & manifest generator
│   ├── verify_download.py             # Atomic checksum-verifying downloader
│   ├── prepare_artifacts.py           # Builds canonical datasets & memory
│   ├── run_oof_validation.py          # 5-fold OOF cross-validation runner
│   ├── predict.py                     # Generates submission.json from test input
│   └── test_codabench_submission.py   # CodaBench format validator
│
├── docs/                              # Technical specifications and guides
│   └── artifact-collaboration.md      # Team collaboration & storage protocol
│
├── logs/                              # Stored official benchmark logs
│   ├── submission_v1_bm25/            # METEOR 0.2952, ROUGE-L 0.3957
│   └── submission_v2_stitcher/        # METEOR 0.3781, ROUGE-L 0.3718
│
└── tests/                             # Automated test suite (100 passing tests)
```

---

## 4. Quickstart Commands

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Audit Artifacts & Verify Integrity
```bash
python scripts/audit_artifacts.py --output artifacts/manifests/artifacts.json
```

### Step 3: Run Automated Test Suite
```bash
pytest tests/ -v
```

### Step 4: Generate Public Predictions
```bash
python scripts/predict.py \
  --input artifacts/raw/public-official.json \
  --output artifacts/submissions/submission.json \
  --train artifacts/raw/train.json \
  --warmup artifacts/raw/warmup.json \
  --chunks artifacts/chunks/legal_chunks.parquet
```

### Step 5: Validate Submission Format
```bash
python scripts/test_codabench_submission.py
```

---

## 5. Parameter Budget Governance

Under competition constraints, the sum of all learned parameters loaded at inference must be strictly less than $4{,}000{,}000{,}000$:

$$\text{Total Learned Parameters} = 595{,}776{,}512 + 595{,}776{,}512 + 2{,}031{,}739{,}904 = 3{,}223{,}292{,}928 < 4{,}000{,}000{,}000$$

- **Accounting Protocol**: Every loaded model checkpoint, adapter, and task head is counted additively.
- **Verification**: Enforced programmatically by `load_model_manifest()` in `src/utils/manifest.py`.
- **Team Collaboration**: Detailed storage and bootstrap guidelines are in [`docs/artifact-collaboration.md`](docs/artifact-collaboration.md).
