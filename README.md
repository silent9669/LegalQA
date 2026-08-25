# DSC 2026 Task 2 — LegalQA Maximum-Score Pipeline

This repository implements the end-to-end Legal Question Answering system for DSC 2026 Task 2, designed to maximize the official **METEOR** score under strict competition constraints ($< 4\text{B}$ total learned parameter budget, zero external data/APIs, offline self-contained inference).

---

## 1. Project Structure

```
LegalQA/
├── README.md                      # Pipeline documentation & instructions
├── requirements.txt               # Dependencies
├── validation.py                  # Root CLI for 5-Fold OOF Validation
│
├── configs/                       # Independent YAML configurations
│   ├── pipeline.yaml              # Main pipeline configuration
│   ├── retrieval.yaml             # BM25 + Dense retrieval settings
│   ├── reranking.yaml             # Cross-encoder reranker settings
│   └── generation.yaml            # Generator & LoRA fine-tuning parameters
│
├── artifacts/                     # Central data, chunks, and intermediate representations
│   ├── raw/                       # Task 2 BTC files (train.json, warmup.json, public-official.json, selected-contexts/)
│   ├── chunks/                    # legal_chunks.parquet (365,046 rows, 409MB), chunks_output.jsonl
│   ├── data/                      # qa_unique.parquet (7,113 QA pairs), known_qa.json
│   ├── labels/                    # retrieval_labels.parquet (7,113 QA pairs)
│   └── submissions/               # submission.json, submission.json.zip
│
├── src/                           # Modular source code
│   ├── data/                      # canonical.py, chunker.py, label_miner.py
│   ├── evaluation/                # codabench_eval.py (Exact CodaBench METEOR & ROUGE-L mirror)
│   ├── memory/                    # exact_memory.py, provision_memory.py
│   ├── retrieval/                 # bm25_retriever.py, query_analyzer.py, hybrid_fusion.py
│   ├── reranking/                 # cross_encoder.py
│   ├── generation/                # prompt_builder.py
│   ├── postprocess/               # extractive.py, article_stitcher.py, source_snap.py
│   ├── selector/                  # candidate_selector.py
│   ├── utils/                     # manifest.py (< 4B parameter budget auditor)
│   └── pipeline.py                # End-to-end multi-stage pipeline
│
├── scripts/                       # Executable workflow scripts
│   ├── prepare_artifacts.py       # Builds canonical parquet files & memory
│   ├── run_oof_validation.py      # 5-fold OOF cross validation runner
│   ├── predict.py                 # Generates submission.json from test input
│   └── test_codabench_submission.py # CodaBench format validator
│
├── logs/                          # Verified CodaBench evaluation logs
│   ├── submission_v1_bm25/        # Baseline METEOR 0.2952, ROUGE-L 0.3957
│   └── submission_v2_stitcher/    # Upgraded METEOR 0.3781, ROUGE-L 0.3718
│
├── Scoring-Program-Task-LegalQA/  # Official CodaBench container scoring program
└── tests/                         # 27 automated unit and integration tests
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
python validation.py --samples 100 --splits 5
```

### Step 4: Run Tests
```bash
pytest tests/ -v --cov=src
```

### Step 5: Generate Submission JSON
```bash
python scripts/predict.py --input artifacts/raw/public-official.json --output artifacts/submissions/submission.json
```

### Step 6: Validate Submission for CodaBench
```bash
python scripts/test_codabench_submission.py
```

---

## 3. CodaBench Benchmark Progress

- **Submission v1 (Baseline)**: `METEOR = 0.2952`, `ROUGE-L = 0.3957`
- **Submission v2 (Article Stitcher + 3-Part Preamble)**: `METEOR = 0.3781`, `ROUGE-L = 0.3718` (**+28.1% relative gain, Top 26 Leaderboard**)
- **Rule Compliance**: Parameter manifest total $\sim 2.9\text{B}$ parameters ($< 75\%$ of the 4.0B maximum limit).
