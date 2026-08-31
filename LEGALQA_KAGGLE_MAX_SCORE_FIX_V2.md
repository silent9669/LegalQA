# LegalQA Kaggle Max-Score V2 — Coding Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use a disciplined plan/execution workflow (prefer test-driven development, isolated changes, and review checkpoints). Track every task below with checkboxes and do not declare completion until all acceptance gates pass.
>
> Repository: `https://github.com/silent9669/LegalQA`
>
> Current audited reference commit: `7bc753b52bd701c80aa929c4cbce9e916ae6141f` (2026-08-31). **Re-read HEAD before changing anything** because the repository may have moved after this prompt was written.

**Goal:** Turn `silent9669/LegalQA` into a clean, reproducible, correctly trained, Kaggle T4×2 LegalQA system whose architecture is selected by leakage-safe official-METEOR experiments rather than by README claims or intuition.

**Architecture:** Use an extractive-first, retrieval-dominant LegalQA system with exact/similar QA memory, hybrid sparse+dense retrieval, task-tuned reranking, structured evidence packing, optional QLoRA generation, multiple answer candidates, and a leakage-safe candidate gate. Treat Qwen as one candidate rather than automatically the final answer. Benchmark at least two parameter-compliant model stacks before freezing the final stack.

**Tech Stack:** Python, PyTorch, Transformers, PEFT/QLoRA, TRL or an equivalent completion-only SFT loop, SentenceTransformers, BM25S, NumPy/PyTorch exact inner-product retrieval, scikit-learn for the tiny candidate selector, Pandas/Parquet, Kaggle dual NVIDIA T4.

**Spec/source of truth:** official Task 2 rules and scoring code in this repository first; then actual organizer data schema; then executable code/configs; then README/docs.

---

# 0. Non-negotiable operating rules

You are the lead competition engineer. Do not merely review or produce suggestions. Inspect the full current repository, implement the changes, run tests/diagnostics that are possible in the environment, update the canonical Kaggle notebook, and leave a clean workspace.

The optimization target is **official Task 2 METEOR**. The scoring implementation in this repository uses whitespace-tokenized answers; optimize the actual scorer, not generic legal-answer elegance. Long, source-faithful statutory wording can outperform concise paraphrases.

Hard constraints:

- Use Task 2 organizer-provided task data only for task-specific training/memory/retrieval supervision.
- Pretrained open models are allowed only if the official rules permit them; verify licenses and rule compliance from the repository/competition materials.
- No external legal corpus, external QA corpus, synthetic QA from an external model, or external answer API.
- Total learned parameters loaded at final inference must be **strictly `< 4,000,000,000`**.
- Quantization does not reduce parameter count for the rule.
- Count adapters/LoRA/task heads/learned selector parameters explicitly.
- Kaggle target: `GPU T4 ×2`, about 16 GB VRAM per GPU.
- User has a Kaggle secret named exactly `HF_TOKEN`; never print, persist, upload, or commit it.
- Final output is `submission.json` plus `submission.json.zip`, exactly matching the public query IDs with no empty answers.
- Do not silently fall back from a neural component to a mock/fallback implementation in a final/full run.
- Any optimization that does not improve a correct leakage-safe validation signal must be rejected or kept behind an experimental flag.

A clean notebook run must fail loudly when required final artifacts are absent. “Continue with BM25 only” or “fall back to extractive generator” is acceptable only in explicitly named diagnostic modes, never in the final competition path.

---

# 1. Audit findings you must treat as open defects until verified fixed

Re-read the current HEAD and verify each item. Some may have changed after the audited commit; if already fixed, add a regression test instead of reimplementing it.

## 1.1 The canonical Kaggle notebook is still inference-only

At the audited commit, `kaggle_kernel/legalqa_gpu_pipeline.ipynb` loads base Qwen and directly performs retrieval/generation/submission. It does **not** execute QLoRA training, evaluate the trained adapter, select a checkpoint, reload it, and then infer.

The final canonical notebook must support this reproducible flow:

```text
preflight
  -> artifact/model validation
  -> retrieval index validation
  -> optional/task-selected reranker training
  -> generator QLoRA training
  -> checkpoint smoke evaluation
  -> load selected final checkpoints
  -> full inference pipeline
  -> strict submission validation
```

Provide explicit notebook flags:

```python
RUN_RERANKER_TRAINING = True
RUN_GENERATOR_TRAINING = True
RUN_DEV_EVALUATION = True
RUN_PUBLIC_INFERENCE = True
REUSE_EXISTING_CHECKPOINTS = False
```

The default competition-training notebook should actually train the selected learned components. A separate `INFERENCE_ONLY=True` path is allowed only for later reruns after checkpoints exist.

## 1.2 The current README command does not make QLoRA dual-GPU

At the audited commit, `train_generator_qlora.py` reads `LOCAL_RANK`, but running it with ordinary `python ...` creates only one process. Do not claim “dual-T4 training” unless it is actually launched and tested as distributed training.

Recommended default: **do not force DDP for a 3B QLoRA model**. A 4-bit 3B QLoRA model fits on one T4; stable single-GPU training is preferable to fragile DDP. Use GPU 0 for QLoRA and GPU 1 for retrieval/reranker development phases. Only enable `torchrun --nproc_per_node=2` after a short real smoke test proves the exact trainer/config works without device-map conflicts.

The notebook and README must state the truth about device usage.

## 1.3 OOF memory isolation is currently incorrect for sampled folds

At the audited commit, `run_oof_validation.py` samples a subset of a validation fold and removes only those sampled rows from QA memory. The rest of the same validation fold can remain in fuzzy memory.

Correct rule:

```text
fold k is validation
ALL records assigned to fold k are excluded from train/memory/model fitting
then an optional sample may be drawn from fold k for cheap evaluation
```

Never:

```text
sample validation rows first
then exclude only sampled IDs
```

Also strengthen fold grouping beyond exact `question_norm`; near-duplicate paraphrases should be grouped when practical.

## 1.4 Dense search currently runs the corpus dot product on NumPy/CPU

At the audited commit, `DEk21Retriever.search()` calls `np.dot(corpus_embeddings, q_emb)` followed by a full `np.argsort`. With ~801k × 768 embeddings this wastes GPU 1 and is unnecessarily slow.

Implement exact batched GPU inner-product search first:

- store corpus embeddings in FP16 after verifying Recall@K parity against FP32;
- memory-map from disk on CPU when loading;
- move one persistent FP16 corpus tensor to `cuda:1` when VRAM permits;
- encode query batches on `cuda:1`;
- use `torch.matmul(query_batch, corpus_matrix.T)` and `torch.topk`;
- process query batches so a `1000 × 801k` full similarity matrix is never materialized;
- fall back to chunked matrix multiplication if full corpus tensor cannot remain on GPU;
- keep exact-search mode as the reference before considering ANN/FAISS.

Approximate FP16 corpus footprint:

```text
801,863 × 768 × 2 bytes ≈ 1.15 GiB
```

That is reasonable on a 16 GB T4 together with the DEk21 query encoder and reranker if memory is managed correctly.

## 1.5 Missing DEk21 embeddings can silently degrade final inference to BM25-only

At the audited commit, the notebook initializes a dense retriever without actually fitting corpus embeddings when the packaged dense index is absent, then skips dense results because `corpus_embeddings is None`.

Final mode must instead raise a clear fatal error such as:

```text
FINAL_PIPELINE_ERROR: DEk21/BGE-M3 dense corpus index is missing or invalid.
Rebuild/package the dense index before final inference.
```

A slow rebuild path may exist behind `ALLOW_INDEX_REBUILD=True`, default `False`.

## 1.6 Kaggle source-code availability is not deterministic

The notebook imports `src.*`, while the Kaggle dataset packager at the audited commit primarily stages data/index artifacts. GitHub notebook linkage must not be assumed to magically place the repo at a particular filesystem path.

Choose one deterministic solution and implement it end-to-end. Preferred solution:

```text
kaggle_dataset/staged/
  data artifacts
  indexes/
  code/LegalQA/src/
  code/LegalQA/configs/
  code/LegalQA/pyproject.toml or dependency manifest
  code_manifest.json with git SHA + hashes
```

The notebook resolves exactly one packaged code root, validates the expected git/code manifest, then imports modules from it. A pinned clone is an acceptable development fallback with Internet ON, but the final reproducible path should not depend on whatever `main` contains at execution time.

## 1.7 Current config paths disagree with operational scripts

At the audited commit, `configs/pipeline.yaml` contains paths under locations such as `artifacts/data` / `artifacts/chunks`, while current scripts commonly use `artifacts/task2/data` and `artifacts/task2/indexes`.

Create exactly one canonical artifact layout and make all scripts/configs/notebook/tests agree.

Recommended canonical layout:

```text
artifacts/task2/
├── data/
│   ├── legal_chunks.parquet
│   ├── qa_unique.parquet
│   ├── known_qa.json
│   ├── qa_citations.parquet
│   ├── retrieval_labels.parquet
│   └── fold_assignments.parquet
├── indexes/
│   ├── bm25/
│   ├── dek21/
│   └── bge_m3/                 # only when that stack is evaluated
├── checkpoints/
│   ├── reranker/
│   ├── generator/
│   └── selector/
├── evaluations/
└── submissions/
```

## 1.8 BM25 load path rebuilds unnecessary Python postings

At the audited commit, `BM25Retriever.load()` can load a memory-mapped BM25S index and then still loop over the whole corpus rebuilding Python `df`, `postings`, and lengths for fallback compatibility. On ~801k chunks this defeats much of the benefit of precomputation.

Refactor so final mode uses one sparse index implementation, preferably BM25S mmap. Do not duplicate an 801k-corpus inverted index in Python unless BM25S is unavailable in an explicitly diagnostic environment.

Required behavior:

```text
BM25S available + valid index -> mmap/load only + lightweight corpus metadata
BM25S unavailable in fast tests -> Python exact fallback allowed
final Kaggle mode + invalid/missing BM25 index -> fail unless ALLOW_INDEX_REBUILD=True
```

## 1.9 The current candidate selector underperforms its own best candidate

The current README reports approximately:

```text
generated          0.0749
strategy_f_1000    0.2009
strategy_f_1500    0.2339
focused_extract    0.2026
stitched_extract   0.3051
selected           0.2009
oracle_best        0.3112
```

Treat these numbers as **untrusted provenance until reproduced under corrected full validation**, but they expose a serious design smell: a selector must never be shipped when “always choose stitched extract” scores materially higher.

Mandatory rule:

```text
final selector score >= strongest simple fixed candidate baseline
```

on a selector-validation split that the selector did not train on.

If no learned/rule selector beats `stitched_extract`, use `stitched_extract` as final until a better candidate family exists.

## 1.10 QLoRA training evidence does not match inference evidence

At the audited commit, `build_training_examples()` maps each QA to one `positive_chunk_id`; multiple resolved citations can overwrite each other. Inference uses a stitched multi-seed evidence pack.

Fix this distribution mismatch. Build generator training examples from structured multi-positive evidence, not a single last positive chunk.

For each training QA:

1. gather **all** resolved positive article/clause IDs;
2. reconstruct the corresponding legal evidence in source order;
3. pack evidence with the same header/format rules used at inference;
4. optionally create a second noisy-retrieval version using the normal retriever/reranker so Qwen sees realistic inference noise;
5. never use validation-fold labels to construct training evidence in fold-specific generator experiments.

## 1.11 Exact QA ID reuse needs question consistency

Do not blindly trust an ID collision. Store the normalized question associated with each known QA ID and require ID/question consistency unless official data semantics explicitly guarantee global stable IDs and a dataset audit proves all overlaps are identical.

Report separately:

```text
public ID overlap
public normalized-question overlap
public overlap where both ID and question match
ID-only conflicts
question-only matches
```

Only deterministic safe matches may bypass retrieval.

## 1.12 The current “source snap” implementation does less than its name claims

At the audited commit, snapping mainly normalizes short dates; it does not robustly snap all money/legal-document/entity facts described in comments/docs.

Either implement contextual fact snapping with tests or narrow the claims. Never replace a generated number merely because another number appears somewhere in evidence.

## 1.13 Article stitching is too unconditional

The current stitcher starts with all sibling clauses from the primary article and then packs secondary articles until a character budget is exhausted. This can inject irrelevant legal text and hurt METEOR precision.

Replace “one stitched context” with an evidence-candidate family:

- top clause only;
- primary article focused siblings;
- primary full article;
- relevance-selected clauses across top 2 articles;
- multi-seed pack with configurable budget.

Use OOF to decide which evidence candidate is best for extractive answers and which evidence should feed the generator.

---

# 2. Architecture bake-off: do not assume the current model stack is globally best

The current stack is sensible but not automatically optimal. Because current evidence suggests extractive answers can dominate generated answers, stronger retrieval may be worth more than a 3B generator.

Benchmark at least these two **mutually exclusive final stacks**. Never load both final dense encoders at once for parameter accounting.

## Stack A — generator-capacity oriented

```text
BM25 (0 learned params)
+ DEk21 v2 dense (~100M)
+ BGE-reranker-v2-m3 (~568M)
+ Qwen2.5-3B-Instruct (~3.09B)
+ small adapters/selector
≈ 3.758B before adapters
```

Use when QLoRA generation produces a reliable end-to-end METEOR gain.

## Stack B — retrieval-capacity oriented

```text
BM25
+ BAAI/bge-m3 dense (~568M; verify exact count)
+ BGE-reranker-v2-m3 (~568M)
+ Qwen2.5-1.5B-Instruct (~1.5B; verify exact count)
+ small adapters/selector
```

This remains comfortably below 4B if exact counts confirm it. It may be superior when final answers are mostly extractive because stronger retrieval can matter more than generator size.

For Stack B, benchmark BGE-M3 only after verifying:

- official model/license allowance;
- exact parameter count;
- corpus indexing runtime/storage;
- retrieval Recall@20/50;
- final end-to-end METEOR using the same folds/candidate logic.

## Optional diagnostic Stack C — retrieval/extractive only

```text
BM25 + best compliant dense retriever + task-tuned reranker + no LLM generator
```

This is not necessarily the final system, but it establishes how much Qwen actually adds. If generator candidates never beat the extractive system reliably, do not let generation reduce the final score merely because the project is called RAG.

### Stack promotion rule

Choose the final stack by the following ordered evidence:

1. valid official-METEOR dev/full OOF;
2. retrieval positive-article Recall@K and MRR;
3. stability across folds/query types;
4. runtime feasibility on T4×2;
5. parameter compliance.

Do not choose based on embedding benchmark reputation.

---

# 3. Target max-score inference architecture

Implement this as the conceptual target, but keep components only when empirical gates pass.

```text
Question
  |
  +--> Safe Exact QA Memory ------------------------------+
  |                                                       |
  +--> Similar-QA Retrieval --------------------+          |
  |                                             |          |
  +--> True BM25 -----------------------+       |          |
  |                                     |       |          |
  +--> Selected Dense Retriever --------+--> Fusion        |
                                             |             |
                                             v             |
                                      Candidate pool       |
                                             |             |
                                             v             |
                                  Task-tuned BGE reranker  |
                                             |             |
                                             v             |
                                  Evidence candidate packer|
                                             |             |
                    +------------------------+-------------+
                    |                        |             |
                    v                        v             v
             focused extract         relevance extract   QA-memory candidate
                    |                        |             |
                    +------------+-----------+-------------+
                                 |
                                 +--> base/QLoRA Qwen candidates
                                 |
                                 +--> source-safe candidate variants
                                 |
                                 v
                         leakage-safe selector
                                 |
                                 v
                            final answer
```

Qwen is a **candidate producer**, not a mandatory finalizer.

---

# 4. Clean repository structure

Do not perform a gratuitous rename of every module. Keep the existing `src/common` + `src/task2` pattern, but make responsibilities unambiguous and remove duplicate logic.

Target structure:

```text
LegalQA/
├── README.md
├── task.md
├── pyproject.toml or requirements.txt
├── requirements-kaggle.txt
├── configs/
│   ├── pipeline.yaml
│   ├── models.yaml
│   └── experiments.yaml
├── src/
│   ├── common/
│   │   ├── normalize.py
│   │   ├── legal_parser.py
│   │   ├── bm25.py
│   │   ├── dense.py
│   │   ├── rrf.py
│   │   ├── reranker.py
│   │   ├── evidence.py
│   │   └── security.py
│   └── task2/
│       ├── qa_memory.py
│       ├── evidence_packer.py
│       ├── generator.py
│       ├── candidates.py
│       ├── selector.py
│       ├── predict.py
│       └── training/
│           ├── train_reranker.py
│           └── train_generator.py
├── scripts/
│   ├── prepare_data.py
│   ├── build_indexes.py
│   ├── mine_retrieval_negatives.py
│   ├── run_oof_validation.py
│   ├── train_reranker.py
│   ├── train_generator_qlora.py
│   ├── audit_parameters.py
│   ├── package_kaggle_dataset.py
│   └── preflight_kaggle.py
├── tests/
├── kaggle_kernel/
│   ├── legalqa_gpu_pipeline.ipynb
│   └── kernel-metadata.json
├── kaggle_dataset/
│   ├── dataset-metadata.json
│   └── staged/              # generated/ignored
├── artifacts/               # generated/ignored except intentionally tiny manifests
└── docs/
    ├── architecture.md
    └── archive/             # superseded docs/specs if worth preserving
```

Guidelines:

- `scripts/*` should be thin CLI wrappers around importable logic where practical.
- Do not keep notebook-only copies of BM25/dense/reranker/generation code.
- `source_snap.py` may be split into `candidates.py` and focused fact-normalization utilities if that improves clarity.
- Move superseded architecture docs to `docs/archive/` or clearly mark them superseded. Do not leave multiple “canonical” pipelines.
- Delete dead MLX/cloud runner stubs and stale duplicate notebooks only after verifying they are not referenced by the current workflow.
- Keep generated indexes/checkpoints/submissions out of Git.

---

# 5. Task 1 — Establish one canonical configuration and preflight

**Files:**

- Modify: `configs/pipeline.yaml`
- Modify: `configs/models.yaml`
- Create: `configs/experiments.yaml`
- Create: `scripts/preflight_kaggle.py`
- Test: `tests/test_config_and_preflight.py`

Required canonical `pipeline.yaml` fields:

```yaml
seed: 42

paths:
  data_dir: artifacts/task2/data
  index_dir: artifacts/task2/indexes
  checkpoint_dir: artifacts/task2/checkpoints
  evaluation_dir: artifacts/task2/evaluations
  submission_dir: artifacts/task2/submissions

retrieval:
  sparse:
    method: bm25s
    top_k: 50
    k1: 1.5
    b: 0.75
  dense:
    stack_a_model: CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2
    stack_b_model: BAAI/bge-m3
    top_k: 50
    dtype: float16
    device: cuda:1
  fusion:
    method: rrf
    rrf_k: 60

reranker:
  model: BAAI/bge-reranker-v2-m3
  candidate_k: 50
  output_k: 8
  device: cuda:1

generation:
  stack_a_model: Qwen/Qwen2.5-3B-Instruct
  stack_b_model: Qwen/Qwen2.5-1.5B-Instruct
  device: cuda:0
  do_sample: false
  max_new_tokens: 384

validation:
  folds: 5
  primary_metric: meteor

final:
  fail_on_missing_index: true
  fail_on_model_fallback: true
```

Do not hard-code an experimental stack as “approved” until the bake-off is complete. `models.yaml` should list exact model identities, roles, measured/verified parameter counts, license, and whether loaded in each candidate stack.

`preflight_kaggle.py` must check:

- both CUDA devices when final config expects two;
- required dataset files;
- code manifest;
- BM25 index manifest;
- dense index manifest and row count;
- checkpoint presence if `REUSE_EXISTING_CHECKPOINTS=True`;
- model identity/path validation;
- no secret strings in workspace;
- parameter budget;
- public ID count/schema.

Final preflight must exit nonzero on any critical mismatch.

---

# 6. Task 2 — Fix fold construction and leakage-safe evaluation

**Files:**

- Modify: `scripts/prepare_data.py`
- Modify: `scripts/run_oof_validation.py`
- Create or modify: `tests/test_oof_isolation.py`

Implement near-duplicate-aware fold groups.

Recommended practical approach for ~7.5k QA:

1. normalize questions;
2. char `TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=2)`;
3. use `NearestNeighbors(metric="cosine")` to retrieve a small neighbor set;
4. union questions whose similarity is above a conservative threshold such as `0.92` **and** whose important legal-number signals do not conflict;
5. assign entire connected components to folds deterministically while roughly balancing fold size.

Tests must prove:

```python
assert set(train_ids).isdisjoint(set(val_ids))
assert set(train_question_norms).isdisjoint(set(val_question_norms))
```

and for sampled validation:

```python
all_fold_ids = set(df[df.fold_id == k].qa_id)
assert all_fold_ids.isdisjoint(set(memory.id_to_answer))
```

Do not fit the fuzzy-memory index, selector, retriever adaptation, reranker adaptation, or generator adapter on the held-out fold for any fold-specific evaluation.

Every evaluation output must persist provenance:

```json
{
  "git_sha": "...",
  "mode": "full",
  "folds": 5,
  "evaluated_ids_sha256": "...",
  "dense_model": "...",
  "dense_index_sha256": "...",
  "reranker_model": "...",
  "reranker_checkpoint": "...",
  "generator_model": "...",
  "generator_adapter": "...",
  "selector_checkpoint": "...",
  "max_new_tokens": 384,
  "timestamp_utc": "..."
}
```

---

# 7. Task 3 — Make exact QA memory safe and fuzzy memory useful rather than dangerous

**Files:**

- Modify: `src/task2/qa_memory.py`
- Test: `tests/test_task2_qa_memory.py`

Store, at minimum:

```text
qa_id
question_raw
question_norm
answer_raw
doc_numbers
articles
clauses
points
numbers/dates when useful
answer_len_words
source_split
fold_id
```

Exact lookup rule:

- exact normalized-question match with a unique answer is safe;
- ID match is safe only when the stored normalized question also matches, unless an official-data audit proves IDs are globally stable and identical;
- conflicts never directly return an answer.

Fuzzy direct-reuse rule must be stricter than the audited `0.96 trigram` heuristic. Start with direct reuse disabled; treat the fuzzy answer as a candidate/exemplar. Enable direct reuse only if held-out OOF proves a threshold with extremely high precision and no key legal-entity conflicts.

Add features for later selection:

```text
fuzzy_char_similarity
fuzzy_dense_similarity (if available)
same_doc_number
same_article
same_clause
conflicting_doc_number
conflicting_article
question_length_ratio
nearest_answer_length
```

Brute-force character similarity over 7.5k QAs is acceptable for 1k public queries, but vectorized TF-IDF nearest-neighbor search is cleaner and faster.

---

# 8. Task 4 — Sparse retrieval must be real, memory-efficient BM25

**Files:**

- Modify: `src/common/bm25.py`
- Modify: `scripts/build_indexes.py`
- Test: `tests/test_common_bm25.py`

Requirements:

- no posting truncation;
- proper BM25 term-frequency saturation and length normalization;
- BM25S mmap/load path for final mode;
- no full Python postings rebuild when BM25S is valid;
- consistent query/document normalization;
- preserve original corpus row/chunk alignment;
- legal entity boosts applied only to a bounded BM25 candidate pool;
- boosts must not mutate the underlying raw BM25 score field.

Return separate fields:

```python
{
  "bm25_raw_score": ...,
  "legal_boost": ...,
  "bm25_final_score": ...,
  "rank": ...
}
```

Tune only a compact staged set:

```text
k1: 1.2 / 1.5 / 1.8
b: 0.55 / 0.75 / 0.9
BM25 K: 30 / 50 / 80
```

Do not run a huge Cartesian grid. First tune retrieval recall on a fixed fold subset, then confirm the best 2–3 configurations end-to-end.

---

# 9. Task 5 — GPU exact dense retrieval with model-stack abstraction

**Files:**

- Refactor/replace: `src/common/dense_dek21.py` -> preferably `src/common/dense.py` with a generic `DenseRetriever`
- Modify: `scripts/build_indexes.py`
- Test: `tests/test_common_dense_and_rrf.py`

Required interface:

```python
class DenseRetriever:
    def build_index(...): ...
    def load_index(...): ...
    def search(query: str, top_k: int) -> list[dict]: ...
    def search_batch(queries: list[str], top_k: int, batch_size: int) -> list[list[dict]]: ...
```

Support both evaluated dense models by configuration, never simultaneously in final inference.

Index manifest must include:

```json
{
  "model_id": "...",
  "revision": "...",
  "dim": 768,
  "dtype": "float16",
  "normalized": true,
  "corpus_rows": 801863,
  "chunk_ids_sha256": "...",
  "embeddings_sha256": "..."
}
```

On load, a chunk-ID hash mismatch is fatal in final mode.

Implement exact GPU search using `torch.topk`. Add a CPU/NumPy exact reference only for tests. Compare FP16 vs FP32 top-K recall on a representative validation sample; use FP16 only if differences are negligible for end-to-end retrieval.

Do not use `np.argsort` over all ~801k scores per query in final mode.

If BGE-M3 requires a specific query/document instruction format, implement it from its model documentation and use exactly the same format in index build and query encoding.

---

# 10. Task 6 — Improve retrieval supervision and mine true hard negatives

**Files:**

- Modify: `src/common/evidence.py`
- Create: `scripts/mine_retrieval_negatives.py`
- Test: `tests/test_common_evidence.py`

The current “different document” negatives based on early corpus rows are not hard enough. Build negatives from actual retrieval results.

For each training QA with resolved positives:

1. retrieve top candidates from BM25 + selected dense model;
2. exclude all resolved positive chunk/article IDs;
3. mark candidates from the same document but wrong article as high-value hard negatives;
4. mark same article but wrong clause only when citation semantics prove the clause is not also valid;
5. include top cross-document false positives;
6. persist retrieval score/rank/source for each negative.

Produce a normalized pair/triple table suitable for reranker training:

```text
qa_id
question
positive_chunk_id
negative_chunk_id
negative_type
bm25_rank
dense_rank
fused_rank
fold_id
```

Add a supervision quality report:

```text
citation parse coverage
citation resolution coverage
article exact-match checks
clause exact-match checks
sampled suspicious-label count
```

Do not train the reranker until the label audit indicates usable supervision.

---

# 11. Task 7 — Fine-tune the reranker before spending quota on generator tuning

**Files:**

- Create importable logic: `src/task2/training/train_reranker.py`
- Create/update thin CLI: `scripts/train_reranker.py`
- Test: `tests/test_reranker_training_data.py`

Use `BAAI/bge-reranker-v2-m3` as the initial reranker base. Train on organizer Task 2 positives + mined hard negatives.

Prefer replacing/fine-tuning the same 568M checkpoint rather than adding another inference model. A LoRA experiment is allowed for training efficiency, but parameter audit must count/merge it correctly.

Screen with one held-out fold first:

```text
train folds 1–4
validate fold 0
```

Metrics:

- positive Article Recall@1/5/8 after reranking;
- MRR;
- final best-extractive-candidate METEOR;
- final selected-system METEOR.

Promotion condition:

```text
reranker tuned checkpoint must improve retrieval ranking and not reduce end-to-end METEOR
```

If a gain is clear, retrain the final reranker on all allowed labeled QA after architecture selection.

---

# 12. Task 8 — Replace one unconditional stitch with evidence candidate packing

**Files:**

- Refactor: `src/task2/article_stitcher.py` -> `src/task2/evidence_packer.py` (or preserve filename with a clearer API)
- Test: `tests/test_task2_evidence_packer.py`

Required candidate evidence packs:

```text
focused_clause
primary_article_relevant_siblings
primary_full_article
relevance_selected_top2_articles
multi_seed_2500_chars
multi_seed_4000_chars
```

For relevance-selected evidence:

1. split retrieved articles into clauses/points;
2. score units with the existing reranker or a cheap lexical+rerank combination;
3. retain the most relevant units under a budget;
4. restore original legal source order before presenting to generator/extractive candidate;
5. preserve parent Article/Clause headers and legal document identity.

Do not slice statutory text blindly in the middle of a token/sentence when a structured boundary is available.

The packer returns metadata:

```python
{
  "text": str,
  "pack_type": str,
  "doc_ids": list[str],
  "article_ids": list[str],
  "clause_ids": list[str],
  "chars": int,
  "rerank_score_top1": float,
  "rerank_margin_1_2": float,
}
```

This metadata becomes selector features.

---

# 13. Task 9 — Fix generator prompt parity and answer-preserving SFT data

**Files:**

- Modify: `src/task2/generator.py`
- Refactor training logic: `src/task2/training/train_generator.py`
- Modify thin CLI: `scripts/train_generator_qlora.py`
- Test: `tests/test_task2_generator.py`
- Test: `tests/test_generator_training_data.py`

Do not manually approximate the Qwen chat template in one place and tokenize differently elsewhere.

Use tokenizer-native chat rendering for both training and inference:

```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_content},
]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

Training examples must render the same prompt, append the gold assistant response with the correct chat-template semantics, and compute loss **only on assistant answer tokens**.

Before selecting sequence length, compute tokenized truncation diagnostics at:

```text
2048
3072
4096
```

For each, report:

```text
fraction of examples where evidence truncates
fraction where gold answer truncates
p50/p90 total tokens
estimated T4 memory
```

Prefer an **answer-preserving truncation policy**:

1. preserve system/question;
2. preserve the full gold answer when it fits the chosen maximum;
3. truncate evidence first at structured clause boundaries;
4. log the rare examples whose answer itself exceeds the maximum.

Do not let default left/right truncation silently cut the answer target tail.

Generator training evidence should include all resolved positives and match inference pack formatting. Optionally mix a controlled fraction of retrieval-noisy evidence so the model is robust to inference errors. Keep that mixture deterministic and record it in the manifest.

---

# 14. Task 10 — QLoRA experiment design that respects Kaggle quota

Use staged validation; do not immediately train five 3B adapters.

## Stage G0 — training smoke test

Train 50–100 optimizer steps on a small subset and verify:

- loss decreases;
- no NaNs;
- adapter saves;
- base+adapter reloads in a fresh process;
- generation works;
- parameter audit includes adapter;
- one T4 stays within memory.

## Stage G1 — one-fold screen

Train on folds 1–4 and evaluate on fold 0 using the actual retrieval/evidence path.

Search only a small high-value set:

```text
LoRA r: 8 or 16
learning rate: 1e-4 or 2e-4
epochs: 1 first; 2 only if epoch-1 validation still improving
max sequence length: value chosen by truncation/memory diagnostics
```

Do not Cartesian-search all combinations. Start with `r=16, lr=1e-4, 1 epoch`; compare one alternative only if needed.

## Stage G2 — confirm if gain exists

If QLoRA beats base generation/end-to-end METEOR by a meaningful, stable margin, confirm on another held-out fold or a second independent grouped split before final training.

A practical promotion signal is:

```text
bootstrap confidence mostly above zero
and absolute end-to-end METEOR improvement >= ~0.005
```

Treat `0.005` as an engineering screening threshold, not an official rule. If variance is high, require more validation rather than claiming a gain.

## Stage G3 — final adapter

After architecture/hyperparameters are frozen, train one final adapter on all allowed training+warmup Task 2 QA using the chosen configuration.

Do not train on public-test answers.

---

# 15. Task 11 — Candidate generation should exploit the metric without adding random text

**Files:**

- Refactor/create: `src/task2/candidates.py`
- Refactor narrow snapping utilities out of `source_snap.py`
- Test: `tests/test_task2_candidates.py`

At minimum produce:

```text
safe_exact_memory
fuzzy_memory_candidate
focused_clause_extract
relevance_extract
primary_article_extract
multi_seed_extract
generated_base
generated_qlora
source_safe_generated
strategy_f_300
strategy_f_600
strategy_f_1000
strategy_f_1500
```

Not every run needs both base and QLoRA in final inference; the model bake-off may generate them offline during validation. The final public inference path should load only the selected generator stack and parameter-compliant components.

Improve extractive candidates beyond `clean_ev[:1500]`:

- preserve complete clause/point boundaries;
- remove website/image/noise captions;
- preserve exact legal citation surface forms;
- include canonical “Căn cứ … quy định” header only when OOF proves it helps;
- avoid duplicate clauses;
- produce length variants based on structured boundaries rather than arbitrary character slicing.

Strategy F must remain an experimental candidate family. Never append 1500 characters unconditionally.

---

# 16. Task 12 — Train a tiny cross-fitted candidate selector, but keep the strongest fixed baseline as a guardrail

**Files:**

- Create: `src/task2/selector.py`
- Modify: `scripts/run_oof_validation.py`
- Test: `tests/test_task2_selector.py`

Use only inference-available features. Examples:

```text
candidate family
candidate word count
candidate/evidence lexical overlap
question length
question intent flags: amount/date/deadline/conditions/procedure/sanction/definition/scenario
BM25 top1 score and margin
Dense top1 score and margin
RRF overlap count
Reranker top1 score
Reranker 1-vs-2 margin
number of unique docs/articles in evidence
fuzzy QA similarity
fuzzy entity-consistency flags
nearest train-answer length
candidate legal-number coverage
candidate date/money coverage
```

Create one training row per `(qa_id, candidate_family)` with target = that candidate’s OOF METEOR. A small scikit-learn regressor such as `HistGradientBoostingRegressor` is adequate; do not introduce a large neural selector.

**Cross-fit the selector itself.** Do not train on all OOF candidate scores and report performance on those same rows.

Example:

```text
OOF candidate table
 -> meta folds by qa_id
 -> train selector on meta folds A-D
 -> select candidates on meta fold E
 -> combine meta-OOF selector predictions
```

Final selector training happens only after its meta-OOF performance is established.

Mandatory guardrail:

```python
if selector_meta_oof_meteor < best_fixed_candidate_meta_oof_meteor:
    final_policy = best_fixed_candidate
else:
    final_policy = learned_selector
```

This prevents repetition of the current “0.2009 selector vs 0.3051 stitched baseline” failure mode.

Report oracle candidate METEOR as an upper bound but never confuse oracle with deployable performance.

---

# 17. Task 13 — Full validation modes and staged experiment matrix

**Files:**

- Modify: `scripts/run_oof_validation.py`
- Create: `configs/experiments.yaml`

Validation modes:

### `--mode unit`

Mocks only. Used by tests. Never reported as competition quality.

### `--mode fast`

Real BM25, cached real retrieval outputs where available, cheap/fallback generation optionally disabled. Used for rapid candidate/selector diagnostics. Every output must be labeled `fast`.

### `--mode full`

Exact intended final components. No mocks, no silent fallback, real dense retriever, real reranker checkpoint, real selected generator/adapter, actual evidence packer, actual candidate policy.

The staged experiment sequence should be:

```text
E0  corrected sparse/extractive baseline
E1  + dense retriever A
E2  + pretrained reranker
E3  + task-tuned reranker
E4  evidence-pack candidate ablation
E5  Stack A base generator
E6  Stack A QLoRA
E7  Stack B retrieval + smaller base generator
E8  Stack B QLoRA if justified
E9  selector meta-OOF
E10 final chosen stack confirmation
```

For every run record:

```text
run_id
git_sha
fold/sample IDs
stack name
retrieval Recall@20/50
reranker Recall@1/5/8 + MRR
candidate family METEORs
selected METEOR
best fixed candidate METEOR
oracle candidate METEOR
mean answer length
runtime
peak GPU memory
```

Do not overwrite evaluation files from different runs; use run-specific directories/manifests.

---

# 18. Task 14 — Make the Kaggle dataset self-contained and verifiable

**Files:**

- Modify: `scripts/package_kaggle_dataset.py`
- Modify: `kaggle_dataset/dataset-metadata.json`
- Test: `tests/test_kaggle_packaging.py`

Package:

```text
legal_chunks.parquet
qa_unique.parquet
known_qa.json
qa_citations.parquet
retrieval_labels.parquet
fold_assignments.parquet
public-official.json
indexes/bm25/**
indexes/<selected_dense>/**
code/LegalQA/src/**
code/LegalQA/configs/**
code/LegalQA/requirements-kaggle.txt
code_manifest.json
dataset_manifest.json
```

The dataset manifest must hash critical files and include corpus row count. Do not package raw secrets, local caches, `.git`, notebooks with outputs containing tokens, or training checkpoints unless intentionally creating a separate checkpoint dataset.

Use separate Kaggle datasets if necessary:

```text
legalqa-task2-runtime-data   # corpus/data/index/code
legalqa-task2-checkpoints    # final trained adapter/reranker when reusing
```

The notebook must verify manifests before inference.

---

# 19. Task 15 — Rebuild the canonical Kaggle notebook as TRAIN -> VALIDATE -> INFER

**File:** `kaggle_kernel/legalqa_gpu_pipeline.ipynb`

The notebook should be short and orchestration-focused. Do not paste duplicate implementations from `src/`.

Required cell structure:

## Cell 1 — Configuration

```python
SEED = 42
RUN_RERANKER_TRAINING = True
RUN_GENERATOR_TRAINING = True
RUN_DEV_EVALUATION = True
RUN_PUBLIC_INFERENCE = True
REUSE_EXISTING_CHECKPOINTS = False
ALLOW_INDEX_REBUILD = False
FINAL_STACK = "auto"  # resolved from validated experiment manifest; not guessed
```

## Cell 2 — Environment + secrets

- print Python/PyTorch/CUDA versions;
- detect both T4s;
- set deterministic seeds;
- retrieve `HF_TOKEN` without printing it;
- install only missing compatible dependencies;
- do not blindly upgrade Kaggle’s CUDA/PyTorch stack.

Create `requirements-kaggle.txt` for the non-PyTorch packages the project actually needs. Before pinning a version, verify it installs under the Kaggle environment; do not pin a known-incompatible stack merely to satisfy documentation.

## Cell 3 — Resolve packaged source/data/models

- locate exactly one runtime dataset root;
- validate code/data manifests;
- add packaged `code/LegalQA` root to `sys.path`;
- resolve mounted Qwen model explicitly by expected identity;
- download/mount dense/reranker models only as permitted;
- fail on ambiguous model path.

## Cell 4 — Preflight

Call `scripts/preflight_kaggle.py` or import its logic. Final mode must stop before expensive training if required artifacts are invalid.

## Cell 5 — Load data/index metadata

Do not load unused columns. Keep the 801k corpus memory footprint controlled.

## Cell 6 — Optional reranker training

If enabled, train the selected task-tuned reranker using the fixed training split/config. Save checkpoint + manifest under `/kaggle/working/checkpoints/reranker`.

## Cell 7 — QLoRA training

If enabled, train the selected Qwen stack on the final training data/config. Save adapter + tokenizer + manifest under `/kaggle/working/checkpoints/generator`.

A real smoke reload must follow training:

```text
release trainer/model objects
clear GPU cache
reload base + adapter in inference mode
run 2 known validation questions
generate non-empty outputs
```

## Cell 8 — Development evaluation

Run a bounded real validation sample/fold sufficient to catch catastrophic checkpoint regressions. If the newly trained model is worse than the validated base policy beyond the configured promotion rule, do not automatically use it for public inference.

This notebook-time check is not a substitute for offline architecture OOF; it is a safety check.

## Cell 9 — Load final inference pipeline

GPU layout:

```text
GPU 0 -> selected Qwen generator only
GPU 1 -> selected dense query encoder + corpus tensor/search + BGE reranker
CPU   -> BM25 mmap metadata, QA memory, evidence structures
```

If Stack B uses Qwen 1.5B and BGE-M3, load exactly those models. If Stack A wins, load DEk21 + Qwen3B. Never accidentally load both dense encoders.

## Cell 10 — Batch retrieval

Use batch operations:

- BM25 search may loop if BM25S API requires it but avoid rebuilding indexes;
- encode all unseen queries in batches;
- exact dense GPU top-K in batches;
- fusion;
- flatten reranker pairs across many queries and score in batches where practical;
- build evidence packs.

## Cell 11 — Batched generation

Use length-aware batches. Start with safe batch sizes such as 2–4 for longer contexts/outputs on T4 and increase only after measuring memory. Do not assume batch 8 with 384+ tokens is always safe.

## Cell 12 — Candidate selection

Use the frozen best fixed candidate or validated selector artifact. Print candidate-family selection counts.

## Cell 13 — Strict submission validation

Verify:

```text
exactly 1000 IDs
exact ID-set equality with public input
no missing/None/empty answer
all answers strings
no internal [DOCUMENT]/[ARTICLE]/chat/debug tokens
UTF-8 ensure_ascii=False
no NaN
zip contains exactly submission.json at archive root
```

Print diagnostics:

```text
exact-memory count
fuzzy-memory selected count
focused/relevance/full-article extract counts
generated/Q-LoRA counts
mean/median/p90 answer words
min/max words
retrieval confidence summary
runtime per stage
peak VRAM per GPU
```

## Cell 14 — Save artifacts

Create:

```text
/kaggle/working/submission.json
/kaggle/working/submission.json.zip
/kaggle/working/run_manifest.json
/kaggle/working/checkpoints/... (if trained)
```

---

# 20. Task 16 — Parameter audit must use real counts, not only approximate constants

**Files:**

- Modify: `scripts/audit_parameters.py`
- Test: `tests/test_validation_and_audit.py`

Keep a fast manifest audit, but add an exact/verified path that counts:

- model parameters from the actual selected checkpoints/configs;
- PEFT adapter trainable parameter count;
- extra classification/regression heads;
- learned selector parameters if applicable.

For LoRA merged into base weights, document how the competition counts learned parameters and follow the official interpretation. Never claim the 242M margin without auditing the chosen final stack.

The final notebook should print something like:

```text
FINAL STACK PARAMETER AUDIT
Dense retriever:  ...
Reranker:         ...
Generator:        ...
Adapter/head:     ...
Selector:         ...
TOTAL:            ...
LIMIT:            4,000,000,000 exclusive
STATUS:           COMPLIANT
```

and abort if non-compliant.

---

# 21. Task 17 — Tests that must exist before final Kaggle training

The repository currently has a useful test suite, but many tests are mock-level. Add real invariants.

Required tests:

1. BM25 hand-computable ranking.
2. BM25 no corpus-order/posting truncation bias.
3. BM25 mmap load does not rebuild full Python postings in final mode.
4. Dense index row-count mismatch fails.
5. Dense chunk-ID hash mismatch fails.
6. FP16 exact top-K closely matches FP32 reference on a small deterministic matrix.
7. Batch dense search equals single-query search.
8. RRF deterministic and preserves unique chunk IDs.
9. Exact-memory ID collision with different question does not reuse answer.
10. Fuzzy memory validation fold cannot retrieve any record from the held-out fold.
11. Sampled OOF still excludes the entire held-out fold from memory.
12. Near-duplicate fold grouping keeps known duplicates together.
13. Retrieval hard-negative miner excludes all resolved positives.
14. Evidence pack preserves source order and structured boundaries.
15. Evidence candidate pack does not duplicate clauses.
16. Generator train/inference chat template parity.
17. Completion loss mask contains no system/user tokens.
18. Answer-preserving truncation truncates evidence before gold answer.
19. Base generator loads without adapter.
20. Adapter saves/reloads and changes trainable parameter audit.
21. Candidate selector never uses reference/gold-only features at inference.
22. Meta-selector split is grouped by `qa_id`.
23. Guardrail falls back to best fixed candidate when selector underperforms.
24. Submission ID equality/schema/zip root.
25. Final-mode missing dense index fails loudly.
26. Final-mode neural model load failure fails loudly instead of falling back.
27. Packaged Kaggle runtime contains importable `src` + configs + manifests.
28. Config/model stack consistency for both Stack A and Stack B.

Mock tests are fine for unit behavior; final readiness also requires at least one real Kaggle smoke run.

---

# 22. Experiment/promotion policy

Do not optimize against one lucky 100-example run.

Use this staged approach to conserve quota:

## Phase A — CPU/cheap structural correctness

- unit tests;
- fold isolation;
- candidate extraction;
- manifest packaging;
- parameter/config audit.

## Phase B — retrieval bake-off

On a fixed held-out fold or representative 500–1000 QA sample:

```text
BM25 only
BM25 + DEk21
BM25 + BGE-M3
+ pretrained reranker
+ tuned reranker
```

Compare Article/Chunk Recall@20/50, reranker Recall@8/MRR, and extractive METEOR.

## Phase C — evidence/candidate bake-off

Use real best retrieval outputs. Compare:

```text
focused clause
relevance-selected extract
primary article
multi-seed pack
Strategy-F variants
```

Freeze the best extractive baseline before generator training.

## Phase D — generator bake-off

Compare:

```text
Stack A base Qwen3B
Stack A QLoRA Qwen3B
Stack B base Qwen1.5B
Stack B QLoRA Qwen1.5B (only if Stack B retrieval is competitive)
```

Evaluate both generator-only and end-to-end candidate policy.

## Phase E — selector

Train/meta-validate the selector. It must beat the strongest fixed candidate or be rejected.

## Phase F — final training

Once the stack is frozen:

- rebuild/freeze selected dense index;
- train final reranker on all allowed labels;
- train final generator adapter on all allowed QA if QLoRA was promoted;
- train final selector on all valid OOF candidate rows if it was promoted;
- package/checkpoint manifests;
- run Kaggle public inference.

---

# 23. What “maximize score” means in this project

Do not promise an unknowable hidden/private score. Maximize the **quality of the selection process**:

- exact official scorer;
- leakage-safe grouped validation;
- retrieval layer metrics;
- candidate ablation;
- full-pipeline confirmation;
- parameter compliance;
- runtime feasibility.

The architecture should be allowed to simplify if the metric says simpler is better. For example, if task-tuned retrieval + relevance extract consistently beats Qwen, use extractive answers for those query types. If QLoRA wins on scenario/application questions, gate it there.

A technically sophisticated component is not valuable if it lowers METEOR.

---

# 24. Definition of Done — do not stop early

Do **not** say the project is “ready to train” until all of the following are true:

- [ ] repository paths/configs are canonical and consistent;
- [ ] stale duplicate pipeline logic is removed or archived;
- [ ] notebook imports packaged canonical source code deterministically;
- [ ] Kaggle dependency setup is reproducible;
- [ ] BM25 final path is mmap/efficient and has no posting truncation;
- [ ] selected dense index is packaged, hashed, row-aligned, and GPU-searchable;
- [ ] final mode cannot silently become BM25-only;
- [ ] reranker training data uses real hard negatives and excludes positives;
- [ ] reranker dev training/evaluation works;
- [ ] generator training uses structured multi-positive evidence;
- [ ] generator chat-template parity is tested;
- [ ] gold-answer truncation is measured and controlled;
- [ ] a real QLoRA smoke training/save/reload passes on T4;
- [ ] sampled OOF excludes the entire validation fold;
- [ ] near duplicates are grouped sufficiently to prevent obvious memory leakage;
- [ ] candidate selector meta-validation is leakage-safe;
- [ ] selected policy is at least as good as the strongest fixed candidate baseline;
- [ ] Stack A vs Stack B is evaluated under the same protocol;
- [ ] the chosen final model stack passes exact parameter audit `<4B`;
- [ ] `pytest` passes;
- [ ] a clean Kaggle `Restart Session + Run All` actually trains the selected model/checkpoints when training flags are enabled;
- [ ] the notebook reloads trained checkpoints before public inference;
- [ ] final inference uses both T4s intentionally according to the selected stack;
- [ ] `submission.json.zip` contains exactly one root `submission.json` with exactly all 1,000 public IDs;
- [ ] README and architecture docs report measured results with provenance and clearly distinguish fast/mock vs full runs.

---

# 25. Required final report from the coding agent

When implementation is complete, do not respond with only “done.” Produce a concise but evidence-rich report containing:

## A. Repository cleanup

```text
files removed/archived
files created
canonical config paths
canonical notebook path
```

## B. Test results

```text
pytest command
number passed/failed/skipped
```

## C. Data/index validation

```text
QA count
public count
legal chunk count
resolved labels
BM25 index status
dense index model + rows + dtype + hash status
```

## D. Leakage checks

```text
exact duplicate groups
near-duplicate groups
fold sizes
proof sampled validation excludes full held-out fold
```

## E. Retrieval bake-off

A table with BM25, DEk21, BGE-M3, pretrained/tuned reranker results.

## F. Candidate/answer bake-off

A table including at least:

```text
focused extract
relevance extract
article extract
best Strategy F
base generation
QLoRA generation
best fixed candidate
selector
oracle candidate
```

## G. Generator training

```text
base model
LoRA r/alpha/dropout
LR
epochs
max sequence length
evidence mixture
examples
gold-answer truncation rate
training runtime
peak VRAM
adapter params
reload smoke test
```

## H. Final stack

```text
selected dense model
selected reranker checkpoint
selected generator + adapter
selected candidate policy
exact total learned params
validation METEOR
runtime estimate
reason it beat alternatives
```

## I. Kaggle run instructions

Give exact click/run sequence for the user’s existing Kaggle notebook setup:

```text
GPU T4 ×2
Internet ON
LegalQA runtime dataset mounted
Qwen model mounted if selected stack uses it
HF_TOKEN secret enabled
Restart Session
Run All
expected output paths
```

## J. Remaining uncertainty

Explicitly state what is not known until a real CodaBench submission is scored. Never invent a leaderboard score.

---

# 26. Priority order if time/GPU quota becomes limited

If quota is constrained, spend effort in this order:

1. **Fix OOF leakage and final-mode silent fallbacks.**
2. **Establish strongest extractive baseline.**
3. **GPU exact dense search + retrieval stack bake-off.**
4. **Task-tune reranker with real hard negatives.**
5. **Evidence relevance extraction.**
6. **One-fold QLoRA screen.**
7. **Second-fold QLoRA confirmation only if it shows gain.**
8. **Meta-selector only after candidate families are strong.**
9. Expensive full 5-fold neural re-training only if time remains.

This ordering is deliberate: the current repository evidence suggests retrieval/extractive quality has more upside than blindly spending the remaining GPU quota on a generator whose candidate may not be selected.

---

# 27. Final instruction

Inspect the actual current repository before acting. Implement the smallest coherent version of the architecture above that wins under correct validation. Do not preserve a component merely because this prompt suggested it.

The final Kaggle notebook must be a **real training-and-submission notebook**, not an inference notebook wearing a training title. It must train the selected reranker/generator checkpoints when the training flags are enabled, validate/reload them, follow the exact same canonical retrieval/evidence/candidate pipeline used in evaluation, and create a strictly valid submission.

The final objective is not “most complicated pipeline.” The final objective is **highest defensible official METEOR under the competition constraints**.
