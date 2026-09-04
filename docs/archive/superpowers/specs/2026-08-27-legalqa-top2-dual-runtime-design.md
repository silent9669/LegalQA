# DSC UIT 2026 Task 2 — LegalQA Top-2 Dual-Runtime (MLX & PyTorch/CUDA) Architecture Design

- **Date**: 2026-08-27
- **Target Competition**: DSC UIT 2026 — Task 2: Legal Question Answering (Vietnamese)
- **Measured Reference Result**: CodaBench METEOR **0.5487**, ROUGE-L **0.4884** (Top 2 Public Leaderboard)
- **Execution Environments**:
  - Local macOS (Apple Silicon M-series): Native **MLX** (`mlx-lm`) + MPS/CPU
  - Remote / GPU Server (Linux/Colab A100/L4): **PyTorch** + **CUDA** + FlashAttention/SDPA + TRL QLoRA

---

## 1. Executive Overview & Problem Context

The goal is to answer Vietnamese legal questions grounded in a provided corpus of 8,532 official documents (365,046 parsed statutory chunks) under a strict parameter limit (<4B learned parameters), with ranking based on the **METEOR** metric.

The winning baseline demonstrated that METEOR on Vietnamese degenerates to exact unigram matching with a 9:1 recall-to-precision weighting. Maximizing score requires:
1. High-precision sparse (BM25s with pyvi word segmentation) + dense (fine-tuned BGE-M3) retrieval fused via Reciprocal Rank Fusion (RRF $k=60$) and reranked with BGE-reranker-v2-m3.
2. A single-source-of-truth ChatML prompt enforcing verbatim clause quoting without summarization.
3. Fine-tuned Qwen2.5-3B-Instruct (3.09B parameters) using QLoRA (NF4 base, $r=16, \alpha=32$ across 7 projections) with completion-only loss.
4. **Strategy F** post-generation evidence appending: attaching the top-1 retrieved statutory chunk (`[:1500]` characters) under a standard citation header.
5. Exact QA memory lookup for public questions matching organizer training data.

This specification unifies the workspace, eliminates outdated exploratory code, consolidates canonical artifacts, and builds a dual-runtime architecture allowing full execution on Apple Silicon Mac (`mlx-lm`) and Linux GPU servers (`PyTorch/CUDA`).

---

## 2. Workspace & Artifact Governance

### 2.1 Canonical Directory Layout

All canonical datasets and pipeline artifacts are organized under `artifacts/`:

```
artifacts/
├── raw/                                 # Unaltered competition raw files
│   ├── train.json                       # 7,000 QA pairs
│   ├── warmup.json                      # 500 QA pairs
│   ├── public-official.json             # 1,000 test questions
│   └── selected-contexts/               # 8,532 context JSON documents
├── chunks/
│   └── legal_chunks.parquet             # 365,046 Điều-first structured chunks (18 fields)
├── data/
│   ├── qa_unique.parquet                # 7,113 deduplicated unique QA pairs
│   └── known_qa.json                    # Exact match QA memory dictionary
├── labels/
│   ├── retrieval_labels.parquet         # 7,113 queries with resolved chunk citation mappings
│   └── qa_citations.parquet             # Normalized citation metadata
├── splits/
│   ├── retrieval_val.parquet            # Frozen 400 resolvable queries for retriever Hit@K / MRR
│   ├── generation_val.parquet           # Frozen 300 held-out queries for local METEOR / ROUGE-L
│   └── train_split.parquet              # 6,413 queries for retriever and SFT training
├── adapters/
│   └── qwen25_3b_legal_lora/            # Fine-tuned LoRA adapter weights (PEFT / MLX)
└── submissions/
    ├── submission.json                  # Output JSON mapping qid -> {"answer": text}
    └── submission.json.zip              # Zip archive ready for CodaBench evaluation
```

### 2.2 Cleanup and Deprecation Actions

1. Archive `artifacts-task2/DSC2026_Task2_LegalQA_Pipeline.ipynb` into `notebooks/DSC2026_Task2_LegalQA_Pipeline.ipynb` as reference material.
2. Remove redundant directory `artifacts-task2/` once consolidated into `artifacts/`.
3. Purge legacy chunking archives (`artifacts/archive/trung-legacy/`).
4. Purge outdated exploratory bakeoff scripts and selector experiments that are obsolete in light of the 3-pillar Top-2 architecture.

---

## 3. End-to-End System Architecture

```
                       ┌────────────────────────────────────────┐
                       │     Input Query (qid, question)        │
                       └───────────────────┬────────────────────┘
                                           │
                       ┌───────────────────▼────────────────────┐
                       │ Exact QA Memory (`known_qa.json`)      ├─[Hit: 41 Qs]──► Return Answer
                       └───────────────────┬────────────────────┘
                                           │ Miss
                ┌──────────────────────────┴──────────────────────────┐
                │                                                     │
    ┌───────────▼───────────┐                             ┌───────────▼───────────┐
    │    Lexical BM25s      │                             │     Dense BGE-M3      │
    │  (pyvi segmented,     │                             │ (FAISS FlatIP cosine, │
    │   k1=0.9, b=0.4)      │                             │  cached shard matrix) │
    └───────────┬───────────┘                             └───────────┬───────────┘
                │ Top 50 Chunks                                       │ Top 50 Chunks
                └──────────────────────────┬──────────────────────────┘
                                           │
                               ┌───────────▼───────────┐
                               │  RRF Fusion (k=60)    │  (Top 30 Candidates)
                               └───────────┬───────────┘
                                           │
                               ┌───────────▼───────────┐
                               │ Cross-Encoder Rerank  │
                               │ (BGE-Reranker-v2-M3)  │
                               └───────────┬───────────┘
                                           │ Top-5 Evidence Chunks
                               ┌───────────▼───────────┐
                               │ Canonical Prompt Build│
                               │ (ChatML Quoting Rule) │
                               └───────────┬───────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    │                                             │
        ┌───────────▼───────────┐                     ┌───────────▼───────────┐
        │  macOS Apple Silicon  │                     │   Linux / GPU Server  │
        │    (mlx_generator)    │                     │   (torch_generator)   │
        │   mlx-lm Metal Engine │                     │  PyTorch BF16 / SDPA  │
        └───────────┬───────────┘                     └───────────┬───────────┘
                    │                                             │
                    └──────────────────────┬──────────────────────┘
                                           │ Raw Generated Prose
                               ┌───────────▼───────────┐
                               │ Strategy F Appender   │  (Appends Top-1 Chunk [:1500])
                               └───────────┬───────────┘
                                           │ Final Answer Text
                               ┌───────────▼───────────┐
                               │   submission.json     │
                               └───────────────────────┘
```

---

## 4. Component Specifications

### 4.1 Memory Module (`src/memory/known_qa.py`)
- **Function**: `lookup_exact_qa(qid: str, question: str) -> Optional[str]`
- **Logic**:
  1. Direct key lookup by `qid` in `artifacts/data/known_qa.json`.
  2. Normalized question lookup against cached mapping `norm_q(question) -> answer`.
- **Normalization**: Unicode NFC normalization, lowercasing, stripping leading/trailing whitespace, collapsing multiple whitespace characters.
- **Coverage**: Directly resolves ~41 public test questions matching organizer training data.

### 4.2 Sparse Retriever (`src/retrieval/bm25_retriever.py`)
- **Engine**: `bm25s` library with parameter configuration $k_1 = 0.9, b = 0.4$.
- **Preprocessing**: Vietnamese word boundary segmentation with `pyvi.ViTokenizer.tokenize`.
- **Index Field Composition**: `document_title` + `dieu` + `article_title` + `content`. This ensures exact matching of legal document numbers (e.g. `90/2017/NĐ-CP`) and article headers (`Điều 17`).
- **Output**: Top-$K$ chunk indices and BM25 scores (default $K=50$).

### 4.3 Dense Retriever (`src/retrieval/dense_retriever.py`)
- **Model**: `BAAI/bge-m3` (568M parameters, 1024 embedding dimension, 8192 token context).
- **Device Support**: Auto-routes to `mps` on Apple Silicon, `cuda` on Nvidia GPUs, or `cpu`.
- **Index**: FAISS `IndexFlatIP` over L2-normalized embeddings for exact inner-product (cosine) search.
- **Cache Persistence**: Sharded embedding matrices stored on disk to enable resumable indexing and fast startup.
- **Output**: Top-$K$ chunk indices and cosine scores (default $K=50$).

### 4.4 Fusion & Reranking (`src/retrieval/fusion.py` & `src/reranking/reranker.py`)
- **Reciprocal Rank Fusion**:
  $$score(d) = \sum_{m \in \{bm25, dense\}} \frac{1}{60 + rank_m(d)}$$
  Produces top-30 fused candidate chunks.
- **Cross-Encoder Reranker**:
  - Model: `BAAI/bge-reranker-v2-m3` (568M parameters, max sequence length 512).
  - Scores query-chunk pairs jointly: `reranker.predict([(query, chunk_text) for chunk in candidates])`.
  - Sorts candidates descending by score and outputs top-$K$ evidence chunks (default $K=5$).

### 4.5 Prompt Builder (`src/generation/prompt.py`)
- **ChatML Format**: Built using `tokenizer.apply_chat_template` across both training and inference.
- **System Prompt (Pillar 2)**:
  ```
  Bạn là trợ lý tư vấn pháp luật Việt Nam. Hãy trả lời câu hỏi dựa trên các trích đoạn văn bản pháp luật được cung cấp.
  Yêu cầu:
  - Trích dẫn ĐẦY ĐỦ, nguyên văn TẤT CẢ các khoản, điểm có liên quan đến câu hỏi (kể cả khi có nhiều khoản), giữ nguyên cách đánh số 1., 2., a), b) như trong văn bản. Không tóm tắt, không rút gọn nội dung điều luật. Sau khi trích dẫn đầy đủ mới đưa ra kết luận.
  - Nếu không có thông tin trong văn bản, hãy nói rõ không có thông tin.
  - Câu trả lời phải chính xác, rõ ràng và trung thực với văn bản pháp luật.
  ```
- **Context Format**:
  ```
  Các trích đoạn văn bản pháp luật:

  [Văn bản 1] {document_title}
  {dieu} - {article_title}
  {content}

  [Văn bản 2] ...

  Câu hỏi: {question}

  Trả lời:
  ```

### 4.6 Generation Engines (`src/generation/`)

#### MLX Backend (`src/generation/mlx_generator.py`)
- Targeted for macOS Apple Silicon.
- Uses `mlx-lm` for low-memory, zero-copy unified memory execution on Apple GPUs.
- Supports loading 4-bit/8-bit quantized weights or merged BF16 checkpoints.
- Applies greedy decoding (`temp=0.0`, `max_tokens=1400`, `repetition_penalty=1.0`).

#### PyTorch Backend (`src/generation/torch_generator.py`)
- Targeted for Linux / Colab / Cloud GPU nodes.
- Uses `AutoModelForCausalLM` with `torch_dtype=torch.bfloat16`, `attn_implementation="sdpa"`, `device_map={"": 0}`.
- Supports PEFT LoRA loading and `merge_and_unload()` deployment.
- Batched greedy generation with left-padding and left-truncation.

### 4.7 Post-Processing & Strategy F (`src/postprocess/strategy_f.py`)
- **Pillar 3**: Programmatic top-1 evidence appending:
  ```python
  def apply_strategy_f(answer: str, top1_content: str, max_chars: int = 1500) -> str:
      if not top1_content:
          return answer.strip()
      return answer.strip() + "\n\nTrích dẫn quy định:\n" + top1_content[:max_chars].strip()
  ```
- Maximizes unigram recall for reference statute terms while keeping the METEOR fragmentation penalty minimal.

---

## 5. Training Specifications

### 5.1 Retriever Fine-Tuning (`scripts/train_retriever_mnrl.py`)
- **Loss**: `MultipleNegativesRankingLoss`.
- **Triplets**:
  - `anchor`: Training question.
  - `positive`: Top BM25 chunk from the cited gold article.
  - `hard_negative`: Top BM25 chunk from an un-cited, completely different document.
- **Dataset**: Supervised on `artifacts/splits/train_split.parquet` (6,413 queries).

### 5.2 Generator QLoRA Training (`scripts/train_generator_qlora.py`)
- **Base Model**: `Qwen/Qwen2.5-3B-Instruct` (3.09B parameters).
- **Quantization**: 4-bit NF4 with double quantization, BF16 compute dtype.
- **LoRA Config**: $r=16, \alpha=32$, dropout 0.05, target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (29.9M trainable parameters, 0.97% of base).
- **SFT Trainer**: TRL `SFTTrainer` with `DataCollatorForCompletionOnlyLM` on `<|im_start|>assistant\n`.
- **Parameters**: `lr=1e-4`, cosine decay, 3% warmup, 1 epoch, effective batch size 16, `max_seq_length=3584`, gradient checkpointing (`use_reentrant=False`), paged 8-bit AdamW.

### 5.3 Generator MLX LoRA Training (`scripts/train_generator_mlx.py`)
- Script wrapping `mlx-lm.lora` for training directly on Apple Silicon local machines.

---

## 6. Evaluation & Inference Workflow

### 6.1 Validation Metrics (`src/evaluation/metrics.py`)
- **METEOR**: Exact unigram precision and recall ($F_{mean}$ at $\alpha=0.9$) with fragmentation penalty ($\gamma=0.5, \beta=3$).
- **ROUGE-L**: Longest Common Subsequence computed over whitespace-split Vietnamese words.
- **Validation Runner (`scripts/validate.py`)**: Runs evaluation over `artifacts/splits/generation_val.parquet` (300 queries) and produces a formatted markdown report.

### 6.2 Inference CLI (`scripts/predict.py`)
- Arguments:
  - `--input`: Path to input questions (default `artifacts/raw/public-official.json`).
  - `--output`: Path to output submission (default `artifacts/submissions/submission.json`).
  - `--runtime`: Execution engine (`auto`, `mlx`, `cuda`, `cpu`).
  - `--append-source / --no-append-source`: Enable/disable Strategy F (default enabled).
  - `--batch-size`: Batch size for generation (default 16 on CUDA, 4 on MLX).
- Flushes JSONL cache after each batch for interrupt-safe execution.
- Automatically generates and verifies `artifacts/submissions/submission.json.zip`.

---

## 7. Verification & Success Criteria

1. **Data Integrity**: `legal_chunks.parquet` (365,046 rows), `qa_unique.parquet` (7,113 rows), `retrieval_labels.parquet` (7,113 rows), `known_qa.json` (41 public hits) all validated with zero missing critical columns.
2. **Dual-Runtime Interface Parity**: Both `mlx_generator` and `torch_generator` accept the same prompt inputs and produce compatible greedy output token sequences.
3. **Pillar Verification**:
   - Greedy decoding matches reference parameters (`do_sample=False, repetition_penalty=1.0, max_new_tokens=1400`).
   - Strategy F appends top-1 chunk up to 1500 chars.
   - Exact QA lookup correctly matches all 41 known public queries.
4. **Test Suite**: Automated pytest coverage for BM25 retrieval, RRF fusion, prompt building, Strategy F, metric calculations, and submission packaging.
