# LegalQA Neural Model Bake-Off Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare compliant dense retrievers, neural rerankers, and small generators under identical verified folds, then promote only statistically supported gains that fit the offline runtime and strict parameter budget.

**Architecture:** Each neural component implements a narrow interface and writes immutable cached outputs keyed by model revision, artifact hash, fold, and configuration. Retriever, reranker, and generator experiments are promoted independently before end-to-end assembly.

**Tech Stack:** Python 3, PyTorch, Transformers, sentence-transformers/FlagEmbedding where appropriate, FAISS or equivalent local ANN index, PEFT/LoRA, pandas/pyarrow, pytest.

**Spec:** `docs/superpowers/specs/2026-08-25-legalqa-artifacts-validation-model-design.md`

## Global Constraints

- Do not start this plan until the artifact-governance and full-validation plans pass.
- Use only organizer Task 2 QA/context data for task adaptation and evidence.
- No external inference or training APIs.
- Pin model revisions and licenses in `configs/models.yaml`.
- Treat organizer permission to use pretrained checkpoints as **unverified** until the written competition rules or organizer clarify it; do not promote or submit a neural stack while this remains unresolved.
- Conservative total learned parameters must remain below 4,000,000,000, counting every independently loaded checkpoint, adapter, and learned selector/head.
- All candidates use identical folds, chunk corpus, top-k values, answer constructors, and promotion tests.

---

### Task 1: Add common retrieval interfaces

**Files:**
- Create: `src/retrieval/base.py`
- Modify: `src/retrieval/bm25_retriever.py`
- Create: `tests/test_retriever_interface.py`

**Interfaces:**
- Produces: `RetrievalHit(chunk_id: str, score: float, rank: int, source: str)`
- Produces: protocol `Retriever.search(query: str, top_k: int) -> list[RetrievalHit]`

- [ ] **Step 1: Write interface tests against BM25**

Assert stable rank ordering, source name `bm25`, unique chunk IDs, and exactly `min(top_k, corpus_size)` hits.

- [ ] **Step 2: Adapt BM25 without changing scores**

Keep current query analyzer and entity boosts. Add a compatibility conversion from tuple outputs to `RetrievalHit`.

- [ ] **Step 3: Run regression tests and commit**

```bash
pytest tests/test_retrieval.py tests/test_retriever_interface.py -v
git add src/retrieval tests/test_retriever_interface.py
git commit -m "refactor(retrieval): define common retrieval hits"
```

### Task 2: Implement offline dense indexing

**Files:**
- Create: `src/retrieval/dense_retriever.py`
- Create: `scripts/build_dense_index.py`
- Create: `tests/test_dense_retriever.py`

**Interfaces:**
- Produces: `DenseRetriever(model_path, index_path, metadata_path)`
- Produces index metadata with model revision, embedding dimension, pooling, normalization, chunk-manifest hash, dtype, and index parameters.

- [ ] **Step 1: Write tests with a tiny local fake encoder**

Tests must not download models. Inject an encoder callable returning deterministic vectors and verify cosine ranking and manifest-hash rejection.

- [ ] **Step 2: Implement batched corpus encoding and index persistence**

Support 512, 768, and 1,024 dimensions when the model supports Matryoshka embeddings. Store vectors as float16 where retrieval accuracy remains unchanged in fixtures.

- [ ] **Step 3: Add model adapters**

Implement pinned local-path loading for:

- Qwen3-Embedding-0.6B
- BGE-M3 dense mode
- multilingual-E5-large-instruct

Network access must be disabled during runtime.

- [ ] **Step 4: Run tests and commit**

```bash
pytest tests/test_dense_retriever.py -v
git add src/retrieval/dense_retriever.py scripts/build_dense_index.py tests/test_dense_retriever.py
git commit -m "feat(retrieval): add offline dense indexes"
```

### Task 3: Integrate deterministic hybrid fusion

**Files:**
- Modify: `src/retrieval/hybrid_fusion.py`
- Create: `src/retrieval/hybrid_retriever.py`
- Modify: `tests/test_retrieval.py`

**Interfaces:**
- Produces: `HybridRetriever(lexical: Retriever, dense: Retriever, rrf_k: int = 60)`

- [ ] **Step 1: Write tests for RRF tie-breaking and deduplication**

Tie-break equal fused scores by best constituent rank, then `chunk_id` for deterministic output.

- [ ] **Step 2: Implement fusion and trace constituent ranks**

Each fused hit retains lexical rank/score and dense rank/score for diagnostics.

- [ ] **Step 3: Run tests and commit**

```bash
pytest tests/test_retrieval.py -v
git add src/retrieval/hybrid_fusion.py src/retrieval/hybrid_retriever.py tests/test_retrieval.py
git commit -m "feat(retrieval): integrate BM25 and dense RRF"
```

### Task 4: Run the retriever bake-off

**Files:**
- Create: `scripts/run_retriever_bakeoff.py`
- Create: `configs/experiments/retrievers.yaml`

**Interfaces:**
- Consumes verified retrieval labels and fold manifests.
- Produces one validation run per model/configuration.

- [ ] **Step 1: Define fixed experiments**

Include:

- BM25 only.
- Qwen3 dense at 512/768/1024 dimensions.
- BGE-M3 dense.
- BGE-M3 dense+sparse if supported locally.
- multilingual-E5 dense.
- BM25 plus each dense model through RRF.

Use fixed top-k values `20`, `50`, and `100`.

- [ ] **Step 2: Build or reuse indexes by manifest hash**

Never reuse an index when chunk hash, model revision, dimension, pooling, or normalization changes.

- [ ] **Step 3: Run full question-blocked retrieval evaluation**

Promotion requires non-regression in document/article Recall@10 and statistically supported end-to-end METEOR after unchanged extractive reconstruction.

- [ ] **Step 4: Record resource measurements**

Save encoding time, index bytes, peak RAM/VRAM, and query latency p50/p95.

- [ ] **Step 5: Commit experiment definitions and reports, not indexes**

```bash
git add scripts/run_retriever_bakeoff.py configs/experiments/retrievers.yaml artifacts/manifests
git commit -m "exp(retrieval): compare multilingual legal retrievers"
```

### Task 5: Implement the neural reranker interface

**Files:**
- Modify: `src/reranking/cross_encoder.py`
- Create: `src/reranking/neural_reranker.py`
- Create: `tests/test_neural_reranker.py`

**Interfaces:**
- Produces: `RerankHit(chunk_id, retrieval_rank, rerank_score, rerank_rank)`
- Produces: `NeuralReranker.rank(query: str, candidates: list[dict], top_k: int) -> list[RerankHit]`

- [ ] **Step 1: Preserve lexical reranker tests**

Rename documentation only; do not change its measured behavior.

- [ ] **Step 2: Test deterministic batching with a fake scorer**

Verify stable tie-breaking, truncation, and candidate-set preservation.

- [ ] **Step 3: Add local checkpoint adapters**

Support:

- Qwen3-Reranker-0.6B
- BGE-reranker-v2-m3

Start with 512- and 1,024-token limits and candidate sets of 20/50/100.

- [ ] **Step 4: Run tests and commit**

```bash
pytest tests/test_reranker.py tests/test_neural_reranker.py -v
git add src/reranking tests/test_neural_reranker.py
git commit -m "feat(reranking): add offline neural rerankers"
```

### Task 6: Run the reranker bake-off

**Files:**
- Create: `scripts/run_reranker_bakeoff.py`
- Create: `configs/experiments/rerankers.yaml`

**Interfaces:**
- Consumes the exact same cached candidate lists from the promoted retriever.

- [ ] **Step 1: Define fixed candidate/input grids**

Candidate counts: 20, 50, 100. Input limits: 512, 1,024. Evidence outputs: 4, 6, 8.

- [ ] **Step 2: Evaluate pre/post rank metrics and final METEOR**

Reject any reranker that increases MRR while reducing document/article Recall@10 or final answer METEOR beyond promotion limits.

- [ ] **Step 3: Record latency and memory**

Save p50/p95 pair-scoring latency, peak VRAM, batch size, and tokens processed.

- [ ] **Step 4: Commit definitions and reports**

```bash
git add scripts/run_reranker_bakeoff.py configs/experiments/rerankers.yaml artifacts/manifests
git commit -m "exp(reranking): compare multilingual legal rerankers"
```

### Task 7: Add generator candidates without replacing extraction

**Files:**
- Create: `src/generation/generator.py`
- Modify: `src/generation/prompt_builder.py`
- Modify: `src/pipeline.py`
- Create: `tests/test_generator.py`

**Interfaces:**
- Produces: `Generator.generate(question: str, evidence: list[dict], examples: list[dict] | None = None) -> str`
- Adds `candidate_generate` while retaining extractive and snapped candidates.

- [ ] **Step 1: Write tests with a fake deterministic generator**

Verify prompt evidence uses `raw_text`, no validation answer appears in the prompt, and pipeline candidate fallback remains extractive when generation fails.

- [ ] **Step 2: Add local checkpoint adapters**

Support Qwen3-1.7B non-thinking and Qwen2.5-1.5B-Instruct. Use deterministic decoding: temperature 0, no sampling, fixed maximum tokens.

- [ ] **Step 3: Add gold-evidence and retrieved-evidence modes**

Keep the same answer constructor and source-alignment checks to measure retrieval loss separately from generation loss.

- [ ] **Step 4: Run tests and commit**

```bash
pytest tests/test_generator.py tests/test_prompt.py tests/test_pipeline.py -v
git add src/generation src/pipeline.py tests/test_generator.py
git commit -m "feat(generation): add grounded generator candidates"
```

### Task 8: Run generator and LoRA experiments

**Files:**
- Create: `scripts/run_generator_bakeoff.py`
- Create: `scripts/train_generator_lora.py`
- Create: `configs/experiments/generators.yaml`
- Modify: `artifacts/manifests/models.json`

**Interfaces:**
- Produces fold-isolated adapters and validation traces.

- [ ] **Step 1: Compare untuned generators and extractive baseline**

Conditions:

- Gold evidence + extractive.
- Gold evidence + each generator.
- Retrieved evidence + extractive.
- Retrieved evidence + each generator.

- [ ] **Step 2: Train fold-isolated LoRA only on Task 2 QA**

No validation-fold answers, provision examples, or retrieved labels may enter training. Record actual trainable parameter counts from the state dict.

- [ ] **Step 3: Run full validation and promotion gates**

The generator is promoted only when it improves overall and unseen-only METEOR without degrading citation/number fidelity.

- [ ] **Step 4: Store adapters externally**

Upload adapters/checkpoints to the approved private model store. Commit only manifests, hashes, configs, and reports.

- [ ] **Step 5: Commit experiment code and manifests**

```bash
git add scripts/run_generator_bakeoff.py scripts/train_generator_lora.py configs/experiments/generators.yaml artifacts/manifests/models.json
git commit -m "exp(generation): evaluate Task 2 grounded generators"
```

### Task 9: Gate provision memory and learned selection

**Files:**
- Modify: `src/memory/provision_memory.py`
- Modify: `src/selector/candidate_selector.py`
- Create: `tests/test_provision_memory_leakage.py`
- Create: `tests/test_selector_training.py`

**Interfaces:**
- Provision examples carry source QA ID, fold, resolved document/article, and conflict status.
- Learned selector consumes only OOF candidate features.

- [ ] **Step 1: Write leakage tests**

Reject examples from the same validation QA group or held-out document supervision. Exclude ambiguous normalized-question groups.

- [ ] **Step 2: Add provision-memory ablation**

Compare zero versus one same-provision example. Do not exceed one example until a measured gain exists.

- [ ] **Step 3: Train a minimal selector only if rule selection leaves measurable oracle gap**

Start with logistic regression or a shallow tree using OOF features; do not introduce a larger model.

- [ ] **Step 4: Run promotion gates and commit**

```bash
pytest tests/test_provision_memory_leakage.py tests/test_selector_training.py -v
git add src/memory/provision_memory.py src/selector/candidate_selector.py tests
git commit -m "feat(selection): gate provision memory and learned selection"
```

### Task 10: Audit and promote the assembled stack

**Files:**
- Create: `scripts/audit_runtime_stack.py`
- Create: `configs/experiments/promoted-stack.yaml`
- Create: `tests/test_runtime_stack.py`
- Modify: `artifacts/manifests/models.json`

**Interfaces:**
- Produces: `audit_runtime_stack(config_path: Path, model_manifest_path: Path) -> dict`
- Produces: a report containing loaded model IDs/revisions, per-component and total learned parameters, adapter/head counts, licenses, local availability, offline-load result, peak RAM/VRAM, and promotion evidence.

- [ ] **Step 1: Write failing tests for conservative counting**

Use fixtures that prove shared architecture names do not deduplicate independently loaded checkpoints, adapters add to their base model, and a total of exactly 4,000,000,000 is rejected.

- [ ] **Step 2: Implement runtime-stack validation**

Resolve every configured neural component through `artifacts/manifests/models.json`. Reject unpinned revisions, missing licenses, missing local files, remote API endpoints, parameter totals greater than or equal to 4,000,000,000, or components without a passing validation-run reference.

- [ ] **Step 3: Assemble the best independently promoted components**

Start from the preferred candidate:

```text
BM25
+ Qwen3-Embedding-0.6B
+ Qwen3-Reranker-0.6B
+ Qwen3-1.7B non-thinking
+ deterministic extraction/stitch/source alignment
```

If a control wins its bake-off, substitute only that component and rerun the full end-to-end comparison.

- [ ] **Step 4: Run offline inference and resource audit**

Disable network access, load from local paths only, predict the smoke manifest twice, and require byte-identical answers. Record total learned parameters, index size, latency p50/p95, and peak RAM/VRAM.

- [ ] **Step 5: Run both full validation split families**

Run question-blocked and document-held-out validation for the complete 7,113-row canonical set. Compare against the promoted lexical/extractive baseline with grouped paired bootstrap reports.

- [ ] **Step 6: Promote only if every gate passes**

Write `configs/experiments/promoted-stack.yaml` only when scorer parity, leakage checks, METEOR delta, confidence interval, retrieval non-regression, subgroup limits, determinism, resource reporting, model license review, rule clarification, and the parameter budget all pass.

- [ ] **Step 7: Run tests and commit**

```bash
pytest tests/test_runtime_stack.py tests/test_model_manifest.py tests/test_compare_validation_runs.py -v
git add scripts/audit_runtime_stack.py configs/experiments/promoted-stack.yaml tests/test_runtime_stack.py artifacts/manifests/models.json
git commit -m "feat(pipeline): audit and promote compliant neural stack"
```
