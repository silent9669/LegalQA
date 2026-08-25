# DSC 2026 Task 2 — LegalQA Maximum-Score Pipeline

This repository implements the end-to-end Legal Question Answering system for DSC 2026 Task 2, designed to maximize the official **METEOR** score under strict competition constraints ($< 4\text{B}$ total learned parameter budget, zero external data/APIs, offline self-contained inference).

---

## 1. Project Structure

```
LegalQA/
├── README.md
├── configs/
│   ├── pipeline.yaml          # Main pipeline configuration
│   ├── retrieval.yaml         # BM25 + Dense retrieval settings
│   ├── reranking.yaml         # Cross-encoder reranker settings
│   └── generation.yaml        # Generator & LoRA fine-tuning parameters
├── src/
│   ├── data/
│   │   ├── canonical.py       # Deduplicates train+warmup -> qa_unique.parquet & known_qa.json
│   │   ├── chunker.py         # Full legal hierarchy chunker -> legal_chunks.parquet
│   │   └── label_miner.py     # Parses citations & mines hard negatives -> retrieval_labels.parquet
│   ├── evaluation/
│   │   └── codabench_eval.py  # Exact CodaBench METEOR & ROUGE-L mirror
│   ├── memory/
│   │   ├── exact_memory.py    # Exact ID / normalized question memory lookup
│   │   └── provision_memory.py# In-context legal provision demonstration lookup
│   ├── retrieval/
│   │   ├── bm25_retriever.py  # Inverted index BM25 lexical retriever
│   │   └── hybrid_fusion.py   # Reciprocal Rank Fusion (RRF)
│   ├── reranking/
│   │   └── cross_encoder.py   # Cross-encoder relevance scoring
│   ├── generation/
│   │   └── prompt_builder.py  # Structured prompt constructor
│   ├── postprocess/
│   │   ├── extractive.py      # Deterministic legal template extractor
│   │   └── source_snap.py     # Precision number & amount alignment
│   ├── selector/
│   │   └── candidate_selector.py # Candidate selector
│   ├── utils/
│   │   └── manifest.py        # < 4B parameter budget auditor
│   └── pipeline.py            # End-to-end multi-stage pipeline
├── scripts/
│   ├── prepare_artifacts.py   # Builds canonical parquet files & memory
│   ├── run_oof_validation.py  # 5-fold OOF cross validation runner
│   ├── predict.py             # Generates submission.json from test input
│   └── test_codabench_submission.py # CodaBench format validator
├── artifacts/
│   ├── data/                  # qa_unique.parquet, known_qa.json
│   ├── chunks/                # legal_chunks.parquet (365,046 rows)
│   ├── labels/                # retrieval_labels.parquet
│   └── submissions/           # Generated submission files
├── logs/
│   └── scoring_result/        # Official CodaBench evaluation logs
├── tests/                     # 23 automated unit and integration tests
├── train.json                 # Raw training data (7,000 QA)
├── warmup.json                # Raw warmup data (500 QA)
├── public-official.json       # Public test data (1,000 queries)
└── selected-contexts/         # 8,532 raw legal documents
```

---

## 2. Quickstart Commands

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Prepare Canonical Datasets & Chunks
```bash
python scripts/prepare_artifacts.py
```

### Step 3: Run 5-Fold OOF Validation Benchmark
```bash
python scripts/run_oof_validation.py
```

### Step 4: Run Tests
```bash
pytest tests/ -v --cov=src
```

### Step 5: Generate Submission JSON
```bash
python scripts/predict.py --input public-official.json --output submission.json
```

### Step 6: Validate Submission for CodaBench
```bash
python scripts/test_codabench_submission.py
```

---

## 3. Key Baseline Benchmark Results

- **Official CodaBench Score**: `METEOR = 0.2952`, `ROUGE-L = 0.3957` (107.6s execution time).
- **Exact Memory Hits**: 41/1,000 public test questions (4.1%) answered with 100% precision.
- **Rule Compliance**: Parameter manifest total $\sim 2.9\text{B}$ parameters ($< 75\%$ of the 4.0B maximum limit).
# LegalQA
