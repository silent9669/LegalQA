# DSC 2026 Task 2 — LegalQA System Specification & Architecture Design

**Date**: 2026-08-30  
**Target Metric**: Official METEOR (Whitespace tokenized)  
**Parameter Ceiling**: Strictly < 4.0B learned parameters  
**Target Model Stack**:
- **Dense Retriever**: `CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2` (~100M params, 768-dim, `pyvi` word segmentation)
- **Cross-Encoder Reranker**: `BAAI/bge-reranker-v2-m3` (~568M params)
- **Generator**: `Qwen/Qwen2.5-3B-Instruct` (~3.09B params)
- **Total System Parameters**: **~3.76B** (Compliant with < 4.0B competition rule)

---

## 1. System Overview & Clean-Start Layout

The workspace is restructured to cleanly separate shared core RAG utilities (`src/common/`) from Task 2 specialized pipeline components (`src/task2/`), while standardizing artifacts into `artifacts/task2/`.

```
legal-rag/
├── configs/
│   ├── models.yaml                  # Parameter budget and model registry
│   ├── task2.yaml                   # Task 2 hyperparameters & paths
│   └── pipeline.yaml                # Runtime execution configuration
├── src/
│   ├── common/                      # Shared RAG core modules
│   │   ├── __init__.py
│   │   ├── normalize.py             # Vietnamese text normalization & legal cleaner
│   │   ├── legal_parser.py          # Hierarchy parser (Chương -> Mục -> Điều -> Khoản -> Điểm)
│   │   ├── bm25.py                  # BM25 sparse indexer with legal signal boosting
│   │   ├── dense_dek21.py           # DEk21 v2 dense encoder + FAISS / cosine indexer
│   │   ├── rrf.py                   # Reciprocal Rank Fusion implementation
│   │   ├── reranker.py              # BGE-Reranker v2 M3 cross-encoder wrapper
│   │   └── evidence.py              # Evidence pack builder and citation formatter
│   └── task2/                       # Task 2 specific modules
│       ├── __init__.py
│       ├── qa_memory.py             # Exact QA and provision memory
│       ├── article_stitcher.py      # Sibling clause stitcher & parent article resolver
│       ├── generator.py             # Dual-runtime generator (MLX & PyTorch)
│       ├── source_snap.py           # Surface-form preservation & entity/date snapper
│       └── predict.py               # End-to-end inference orchestrator
├── artifacts/task2/
│   ├── data/
│   │   ├── qa_unique.parquet        # Deduplicated train + warmup QA pairs
│   │   ├── legal_chunks.parquet     # Hierarchical chunks with parent mapping
│   │   ├── qa_citations.parquet     # Parsed citations from gold answers
│   │   ├── retrieval_labels.parquet # Positive/negative training supervision
│   │   └── known_qa.json            # Fast deterministic exact-match lookup
│   ├── indexes/
│   │   ├── bm25/                    # Serialized BM25 index
│   │   └── dek21/                   # Serialized DEk21 embeddings & index
│   ├── checkpoints/
│   │   ├── retriever/               # Fine-tuned retriever checkpoints (optional)
│   │   ├── reranker/                # Fine-tuned reranker checkpoints (optional)
│   │   └── generator/               # LoRA / QLoRA adapter weights (MLX & HF)
│   └── submissions/
│       └── submission.json          # Final competition submission
├── notebooks/
│   └── DSC2026_Task2_LegalQA_Pipeline.ipynb # Google Colab & local training/inference
├── scripts/
│   ├── prepare_data.py              # Parse context + build unique QA & citations
│   ├── build_indexes.py             # Build BM25 and DEk21 dense vector indexes
│   ├── run_oof_validation.py        # 5-fold cross-validation with exact METEOR
│   ├── train_generator_mlx.py       # Local Apple Silicon MLX QLoRA fine-tuning
│   ├── train_generator_colab.py     # PyTorch / CUDA TRL / Unsloth QLoRA training
│   ├── convert_lora_weights.py      # Bidirectional MLX <-> HF PEFT adapter converter
│   └── predict.py                   # Submission generation script
├── Scoring-Program-Task-LegalQA/    # Official CodaBench scoring program
└── tests/                           # Unit & integration test suite
```

---

## 2. Data Processing & Hierarchy Specification

### 2.1 Hierarchy Parser & Chunker (`legal_parser.py`)
Each legal document from `artifacts/raw/selected-contexts/` is parsed into structured chunks respecting legal structural boundaries:
- **Hierarchical Levels**: `Chương` (Chapter) $\to$ `Mục` (Section) $\to$ `Điều` (Article) $\to$ `Khoản` (Clause) $\to$ `Điểm` (Sub-clause).
- **Chunk Metadata Schema (`legal_chunks.parquet`)**:
  - `chunk_id`: Unique identifier (e.g., `doc740_art17_p2`)
  - `doc_id`: Document ID from raw context (e.g., `740`)
  - `doc_name`: Law title and official identifier
  - `parent_article_id`: Canonical article key (e.g., `doc740_art17`)
  - `article_number`: Integer or string of article (e.g., `17`)
  - `clause_number`: Integer or string of clause if applicable (e.g., `3`)
  - `text_raw`: Verbatim text with preserved capitalization and formatting
  - `text_norm`: Normalized text for BM25 and tokenized representations
  - `start_char`, `end_char`: Offset spans in original passage

### 2.2 QA Normalization & Memory Table (`qa_memory.py`)
- Reads official `train.json` (7,000 samples) and `warmup.json` (500 samples).
- Strips punctuation and normalizes whitespace in questions for duplicate key detection.
- Generates `qa_unique.parquet` and `known_qa.json` for $O(1)$ exact question lookup.
- If a query in validation/test matches an exact known question ID or normalized string, the verified gold answer is returned immediately.

### 2.3 Citation & Negative Label Mining (`evidence.py` & `prepare_data.py`)
- Extracts statutory references from gold answers (e.g., `Căn cứ khoản 3 Điều 17 Nghị định 90/2017/NĐ-CP`).
- Maps citations to exact `chunk_id` and `parent_article_id`.
- Generates `retrieval_labels.parquet` with mined hard negatives:
  - **Negative Type A**: Same document, adjacent/wrong Article.
  - **Negative Type B**: Same Article, sibling/wrong Clause.
  - **Negative Type C**: Top BM25 / DEk21 false positive from different law.

---

## 3. Hybrid Retrieval & Neural Reranking

### 3.1 Lexical Search with Entity Boosting (`bm25.py`)
- Uses `bm25s` / BM25Okapi over `pyvi`-segmented Vietnamese legal text.
- **Legal Entity Booster**: Regex pattern matching for numbers (`số 90/2017`, `Điều 17`, `Khoản 3`, year `2023`). Chunks matching exact statutory identifiers receive an additive scoring bonus.
- Retrieves top $K=60$ candidates.

### 3.2 Dense Embedding with DEk21 v2 (`dense_dek21.py`)
- Model: `CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2` (768-dim embeddings).
- Formats input chunks with structural prefix:
  ```
  [DOCUMENT] Nghị định 90/2017/NĐ-CP
  [ARTICLE] Điều 17. Vi phạm quy định về ...
  [CONTENT] ...
  ```
- Queries and passages are word-segmented via `pyvi` and L2-normalized.
- Similarity computed via Cosine Similarity / FAISS index, retrieving top $K=60$ candidates.

### 3.3 Reciprocal Rank Fusion (`rrf.py`)
Combines BM25 and DEk21 ranks using weighted RRF:
$$RRF(d) = \frac{w_{\text{bm25}}}{k + \text{rank}_{\text{bm25}}(d)} + \frac{w_{\text{dense}}}{k + \text{rank}_{\text{dense}}(d)}$$
with default constant $k=60$, $w_{\text{bm25}}=0.5$, $w_{\text{dense}}=0.5$. Produces top 50 candidates for neural reranking.

### 3.4 Cross-Encoder Reranking (`reranker.py`)
- Model: `BAAI/bge-reranker-v2-m3` (~568M parameters).
- Input format: Pair of `(Question, [DOCUMENT] name [ARTICLE] Điều X ... text)`.
- Re-scores top 50 RRF candidates and outputs top $K=6-8$ fine-grained evidence chunks.

---

## 4. Article Stitcher, Generator & Source Snapping

### 4.1 Article Stitcher (`article_stitcher.py`)
- When a top seed chunk belongs to `parent_article_id`, loads sibling clauses from `legal_chunks.parquet`.
- Sorts clauses by source offset (`start_char`), deduplicating overlap.
- Yields a cohesive, complete statutory article without exceeding context windows.

### 4.2 Generator Prompting & Execution (`generator.py`)
- Model: `Qwen/Qwen2.5-3B-Instruct` (~3.09B base parameters).
- **Prompt Structure**:
  ```
  <|im_start|>system
  Bạn là trợ lý pháp luật chuyên nghiệp. Hãy trả lời câu hỏi dựa trên căn cứ pháp lý được cung cấp. Giữ nguyên các số hiệu văn bản, điều, khoản, số tiền phạt, ngày tháng và thuật ngữ pháp lý.
  <|im_end|>
  <|im_start|>user
  [CĂN CỨ PHÁP LÝ]
  Văn bản: {doc_name}
  {stitched_evidence}

  [CÂU HỎI]
  {question}
  <|im_end|>
  <|im_start|>assistant
  ```

### 4.3 Answer Reconstruction & Source Snapping (`source_snap.py`)
To maximize METEOR score (which penalizes hallucinated paraphrasing and rewards exact statutory wording):
1. **Multi-Candidate Generation**:
   - Candidate A: Focused extractive clause span.
   - Candidate B: Stitched complete Article extract.
   - Candidate C: Generator generated answer.
   - Candidate D: Source-snapped generator answer.
2. **Fact Alignment (Source Snapping)**:
   - Scans generated answer for legal numbers, dates, monetary fines, and actor names.
   - Snaps dates (e.g. `1/7/2023` $\to$ `ngày 01 tháng 7 năm 2023`), statutory identifiers (e.g. `Điều 17 khoản 3` $\to$ `khoản 3 Điều 17`), and currency amounts to verbatim evidence strings.
3. **Adaptive Candidate Selector**: If generation confidence is low or retrieval indicates an exact copyable statutory answer, promotes the stitched legal text candidate.

---

## 5. Dual-Runtime (Local Apple Silicon MLX & Google Colab / GPU)

### 5.1 Local Execution & Training on Apple Silicon (M3 Pro 36GB)
- **Inference**: High-speed local evaluation using `mlx-lm` or PyTorch MPS / CPU.
- **LoRA Fine-tuning**: MLX-native QLoRA (`scripts/train_generator_mlx.py`), achieving ~15-25 tokens/sec with unified memory consumption < 8GB.
- **Retriever / BM25**: Local indexing and similarity search in seconds.

### 5.2 Google Colab / Cloud GPU Training
- **Notebook**: `notebooks/DSC2026_Task2_LegalQA_Pipeline.ipynb` self-contained for Google Colab (T4/V100/A100).
- **Stack**: PyTorch + Hugging Face Transformers + PEFT / Unsloth + TRL SFTTrainer.
- Saves standard Hugging Face PEFT LoRA adapters (`adapter_model.safetensors`).

### 5.3 LoRA Adapter Cross-Compatibility (`convert_lora_weights.py`)
- Implements bidirectional converter between:
  - Hugging Face PEFT format (`adapter_model.safetensors` / `adapter_config.json`)
  - MLX format (`adapters.safetensors`)
- Ensures adapters trained on Google Colab can be instantly loaded locally on Apple Silicon MLX without retraining, and vice versa.

---

## 6. Validation Framework & Success Metrics

- **5-Fold Cross-Validation (`run_oof_validation.py`)**:
  - Stratified/question-blocked 5-fold split across 7,500 train+warmup QA pairs.
  - Zero leakage guarantee: Validation samples are excluded from retriever fine-tuning and cannot retrieve themselves from QA memory.
- **Exact Official Metric Implementation**:
  - METEOR calculated via whitespace split (`nltk.translate.meteor_score.meteor_score([ref.split()], pred.split())`).
  - ROUGE-L calculated via `rouge_score.rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)`.
- **Target Performance**:
  - Retrieval Recall@20 $\ge$ 92%.
  - Local OOF METEOR $\ge$ 0.50 - 0.55+.
  - Clean execution with total parameter count $\le 3.76\text{B} < 4.0\text{B}$.
