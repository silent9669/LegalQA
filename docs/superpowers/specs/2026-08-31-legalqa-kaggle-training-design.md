# LegalQA Metric-Gated Kaggle Training Design

**Date:** 2026-08-31  
**Status:** Approved design; implementation against this specification not started  
**Primary metric:** Official whitespace-tokenized METEOR  
**Parameter ceiling:** Strictly below 4.0B learned parameters  
**Target cloud runtime:** Kaggle dual NVIDIA T4, with a Google Colab-compatible launcher  
**Reference architecture:** [DSC 2026 Task 2 — LegalQA Wiki & Pipeline](https://dangphuc.notion.site/task-2?source=copy_link)

## 1. Purpose

Build one reproducible LegalQA pipeline from prepared competition data through retrieval, reranking, grounded QLoRA training, out-of-fold evaluation, and final submission generation.

The design follows the Notion LegalQA architecture where it is measurable, but it does not treat planned components or historical scores as implemented facts. Each optional learned layer is promoted only when it improves its intended metric and does not reduce final METEOR.

The Kaggle and Colab notebooks become thin launchers. Training, retrieval, prompt construction, reconstruction, checkpointing, evaluation, and submission validation live in canonical Python modules and scripts.

### 1.1 Competition constraints

- Use only the supplied Task 2/BTC QA and legal-context data for training, retrieval, memory, evaluation, and reconstruction.
- Do not add external legal corpora, synthetic QA, or external answer-generation APIs.
- Pretrained model weights are allowed only as part of the audited model stack; they do not authorize retrieval from external text sources.
- Keep the exact audited learned-parameter total strictly below 4.0B.
- Score model and candidate decisions with the official whitespace-tokenized METEOR implementation.

## 2. Verified Starting State

The design is based on a read-only audit of the repository and the actual remote Kaggle kernel.

### 2.1 Data and artifacts

Current canonical raw inputs are:

- `artifacts/raw/train.json`: 7,000 QA records.
- `artifacts/raw/warmup.json`: 500 QA records.
- `artifacts/raw/public-official.json`: 1,000 public questions.
- `artifacts/raw/selected-contexts/`: 8,532 legal context documents.

Current prepared artifacts include:

- `artifacts/task2/data/legal_chunks.parquet`: 801,863 rows.
- `artifacts/task2/data/qa_unique.parquet`: 7,096 normalized unique questions.
- `artifacts/task2/data/qa_citations.parquet`: 13,476 citation rows.
- `artifacts/task2/data/known_qa.json`.
- `artifacts/task2/indexes/bm25/`.

The intended `retrieval_labels.parquet` and persistent DEk21 index do not exist.

### 2.2 Training and inference

- Both tracked notebooks are inference/submission notebooks, not training notebooks.
- `scripts/train_generator_qlora.py` is incompatible with current TRL APIs and uses BF16 plus `device_map="auto"`, which is not a safe dual-T4 DDP design.
- `scripts/train_retriever_mnrl.py` joins citation `doc_number` values against chunk `doc_name` values and likely resolves zero or near-zero positives.
- `scripts/train_generator_mlx.py` prints a command but does not execute training.
- The real inference path loads an unfitted mock dense retriever.
- The Kaggle notebook builds a truncated custom sparse index over only the first 400,000 chunks and returns the first chunk when no term matches.

### 2.3 Remote Kaggle state

Verified with project-local Kaggle CLI 2.2.4:

- Dataset ref: `phucdangg/legalqa-task2-clean-data`.
- Current dataset title: `LegalQA Task 2 Clean Artifacts`.
- Kernel ref: `phucdangg/legalqa-top-2-training-gpu`.
- Current kernel title: `legalqa-top-2-training-gpu`.
- Stable kernel `id_no`: `132539626`.
- Remote notebook source matches `kaggle_kernel/legalqa_gpu_pipeline.ipynb`.
- The last remote run failed on a Tesla P100 because the installed PyTorch binary did not contain an `sm_60` kernel image.
- No output artifacts were produced.

### 2.4 Security

A real-token-shaped Hugging Face credential is committed in:

- `kaggle_kernel/legalqa_gpu_pipeline.ipynb`, cell `cell-4`.
- `kaggle_kernel/legalqa_train_gpu.py:96`.
- Git history.

The credential value must never be reproduced. It must be revoked or rotated outside this implementation. Working-tree removal is in scope; Git-history rewriting is destructive and remains a separately approved operation.

## 3. Goals

1. Maximize final METEOR under the competition's strict learned-parameter ceiling.
2. Preserve exact legal wording, numbers, dates, actors, Article/Clause identifiers, and source spans.
3. Implement the Notion pipeline as a sequence of independently measurable stages.
4. Use Kaggle dual T4 GPUs correctly for QLoRA training.
5. Make the same canonical training path runnable from Google Colab.
6. Fail before expensive GPU work when data, dependencies, hardware, labels, folds, or parameter counts are invalid.
7. Resume interrupted training and generation without redoing verified work.
8. Keep Kaggle notebooks minimal and remove duplicated pipeline logic.
9. Rename the Kaggle dataset display title to `LegalQA` and the kernel to `LegalQA training` through the Kaggle CLI after local verification.

## 4. Non-Goals

- Do not implement every speculative Notion component before measurement.
- Do not train DEk21 or BGE merely because training scripts exist.
- Do not preserve mock retrieval, arbitrary evidence fallback, print-only training wrappers, or duplicated notebook inference code.
- Do not rewrite unrelated repository history, delete unrelated artifacts, or broadly refactor documentation.
- Do not rewrite the official competition scorer. Wrap and validate its exact metric behavior instead.
- Do not rely on quantization or LoRA to reduce the official parameter count.
- Do not introduce an additional learned candidate selector unless an exact parameter audit and OOF gain justify it. The initial selector is deterministic.

## 5. Design Principles

### 5.1 One canonical implementation

Notebooks call Python entry points. They do not contain retrieval, prompt, model-loading, reconstruction, training, or submission logic.

### 5.2 Metric-gated promotion

Each optional component has a baseline, an intended layer metric, and a final METEOR gate. A component is retained only if it improves its intended layer and does not regress mean OOF METEOR.

### 5.3 Raw text is authoritative

Normalization is used for search and joining. Answers and source reconstruction preserve `text_raw` and `answer_raw`.

### 5.4 Explicit fallback only

A missing dense index or reranker may fall back to a named, measured baseline. It must not silently instantiate a mock or fabricate evidence.

### 5.5 Configuration and artifact provenance

`configs/task2.yaml` becomes the authoritative configuration. Every generated artifact records the configuration hash, input hashes, code revision when available, model revisions, dependency versions, and creation timestamp.

## 6. End-to-End Architecture

```text
official Task 2 data
  -> validated legal parsing and canonical legal identifiers
  -> QA/citation deduplication
  -> citation-to-chunk labels and guarded hard negatives
  -> group-aware folds
  -> exact-memory + corrected BM25 baseline
  -> persistent DEk21 embeddings/index
  -> RRF fusion
  -> pretrained or promoted BGE reranker
  -> selective Article/Clause stitching
  -> optional fold-safe same-provision memory
  -> evidence-conditioned Qwen2.5-3B QLoRA
  -> focused/stitched/generated/source-snapped candidates
  -> deterministic OOF-tuned candidate selection
  -> validated submission JSON and ZIP
```

## 7. Canonical Data Model

### 7.1 Legal chunks

`legal_chunks.parquet` remains the single canonical corpus table. It must not be duplicated as a second full metadata parquet inside the BM25 index.

Required fields:

- `chunk_id`
- `doc_id`
- `doc_name`
- `legal_number`
- `year`
- `chapter_number`
- `section_number`
- `article_number`
- `clause_number`
- `point_label`
- `parent_article_id`
- `parent_clause_id`
- `text_raw`
- `text_norm`
- `start_char`
- `end_char`

Nullable hierarchy fields are allowed. `start_char` and `end_char` must reference the actual chunk span, not the full parent Article for every child Clause.

### 7.2 Canonical legal identifiers

The identifier parser must preserve full suffixes such as `QH14`, `NĐ-CP`, and similar official components. Citation and corpus identifiers are normalized into the same canonical representation before joining.

Document matching may use normalized aliases derived from `doc_name`, but the resolved canonical `legal_number` is stored explicitly.

### 7.3 QA table and memory policy

`qa_unique.parquet` preserves:

- source QA IDs
- `question_raw`
- `question_norm`
- `answer_raw`
- source split
- duplicate group ID
- conflict flag
- fold ID

Memory policy:

- Exact QA ID matches may return the corresponding verified answer.
- A normalized-question match is eligible only when every record in the duplicate group agrees on the answer.
- Conflicting groups are excluded from normalized exact memory and same-provision demonstrations.
- Validation records are excluded from every memory structure used by their fold.

### 7.4 Citations and retrieval labels

`qa_citations.parquet` stores parsed citation components and their source QA ID.

`retrieval_labels.parquet` stores:

- `qa_id`
- `question`
- `positive_chunk_id`
- `positive_article_id`
- `negative_chunk_ids`
- `negative_types`
- `fold_id`
- resolution status and reason

A label may be Article-level when the gold citation is broad. Training code must distinguish broad Article supervision from exact Clause supervision.

Hard-negative types:

1. Same document, wrong Article.
2. Same Article, wrong Clause or Point.
3. High-ranked BM25/DEk21 false positive.
4. Same legal topic, different document or provision.

False-negative guard: a chunk from the same resolved positive provision cannot be used as a negative unless the label explicitly targets a finer hierarchy level and the distinction is unambiguous.

### 7.5 Group-aware folds

Five folds are assigned deterministically from a configured seed.

The grouping key keeps exact and normalized duplicate questions together, including every conflicting-answer group. Sharing a citation alone does not force unrelated questions into one fold; same-provision demonstrations are made leakage-safe by building them from the training partition only. If a later near-duplicate detector is proposed, it must be evaluated and documented before it changes fold assignments.

For each fold, validation records are excluded from:

- QA memory
- provision demonstrations
- retriever tuning
- reranker tuning
- hard-negative construction from gold labels
- generator training
- threshold selection for that fold

## 8. Artifact Layout

```text
artifacts/task2/
├── data/
│   ├── legal_chunks.parquet
│   ├── qa_unique.parquet
│   ├── qa_citations.parquet
│   ├── retrieval_labels.parquet
│   ├── fold_assignments.parquet
│   └── generator_examples.parquet
├── indexes/
│   ├── bm25/
│   └── dek21/
├── candidates/
│   ├── retrieval/
│   ├── reranked/
│   └── generator/
├── checkpoints/
│   ├── retriever/
│   ├── reranker/
│   └── generator/
├── evaluations/
│   ├── folds/
│   └── run_manifest.json
└── submissions/
    ├── submission.json
    └── submission.json.zip
```

Every directory containing generated outputs includes a manifest or embeds equivalent provenance in its primary artifact.

## 9. Pipeline Stages and Promotion Gates

### 9.1 Stage A: data preparation and preflight

Required fixes:

- Full legal-number suffix parsing.
- Citation-to-document canonical joins.
- Hierarchy-aware chunk offsets.
- Retrieval-label generation.
- Conflicting QA detection.
- Parser errors reported with document IDs instead of silently returning empty chunks.
- Recursive or explicitly complete context discovery.

Data preflight must report:

- input and output row counts
- parse failures
- unresolved citation counts and reasons
- positive-label coverage
- negative counts by type
- duplicate/conflict counts
- fold sizes and leakage checks

Training must not start if required schemas are missing, ID sets are inconsistent, no positives are resolved, or fold isolation fails.

### 9.2 Stage B: exact memory and BM25 baseline

BM25 corrections include:

- reset all mutable state on `fit()`
- use one canonical corpus/query normalization path
- handle both raw and normalized fields consistently
- persist and restore all fallback state
- preserve the full corpus
- return no evidence on no match instead of the first chunk

The baseline produces focused extractive and selectively stitched candidates and records official METEOR.

### 9.3 Stage C: pretrained DEk21 and RRF

DEk21 requirements:

- pinned model revision
- batched encoding that honors configured batch size
- normalized 768-dimensional embeddings
- persistent vector/index storage
- resumable/sharded corpus encoding for 801,863 chunks
- query and corpus segmentation through the same canonical normalizer

RRF initially uses equal weights and `k=60`, with branch candidate counts defined in `configs/task2.yaml`.

Promotion gate:

- Hybrid Article Recall@20 must exceed the corrected BM25 baseline.
- Dense retrieval must contribute a measurable unique-positive gain.
- Final OOF METEOR must not regress.

DEk21 fine-tuning is deferred unless pretrained hybrid retrieval remains the measured bottleneck.

### 9.4 Stage D: BGE reranking

The pretrained `BAAI/bge-reranker-v2-m3` reranks evidence packs containing document, Article, Clause, and raw text.

Initial flow:

- BM25 and DEk21 each retrieve configured candidates.
- RRF fuses candidates.
- BGE reranks the fused set.
- Top 6-8 candidates are eligible for stitching.

BGE tuning uses only fold-safe positives and guarded hard negatives.

Promotion gate:

- Top-1 or Top-8 evidence accuracy improves.
- Final OOF METEOR does not regress.
- Runtime remains within the configured budget.

### 9.5 Stage E: selective Article/Clause stitching and provision memory

Stitching occurs after reranking, never before.

It reconstructs sibling chunks by source offsets and produces both:

- focused evidence
- broader stitched evidence

The pipeline retains both candidates because unconditional Article expansion can reduce METEOR precision.

Provision memory is optional and Task 2-only. At most one fold-safe, high-confidence example from the same resolved provision may be inserted. It cannot override retrieval or be used for conflicting QA groups.

### 9.6 Stage F: evidence-conditioned QLoRA generator

Base model: `Qwen/Qwen2.5-3B-Instruct`, pinned to an exact revision.

Training and inference share:

- tokenizer revision
- chat template
- system instruction
- evidence formatting
- deliberate evidence truncation policy
- maximum sequence length
- optional provision-memory format

Training row shape:

```text
[QUESTION]
...

[LEGAL EVIDENCE]
...

[OPTIONAL SAME-PROVISION QA]
0 or 1 fold-safe example

[TARGET]
verbatim gold answer
```

The implementation uses current pinned TRL APIs. The configuration uses `SFTConfig(max_length=...)` or the exact equivalent required by the pinned version, not an unverified historical signature.

Promotion gate:

- One-batch smoke training succeeds.
- A pilot fold trains, evaluates, checkpoints, and resumes.
- Generated-answer METEOR improves over untuned generation.
- Final selected-answer METEOR improves over the best extractive baseline.

### 9.7 Stage G: reconstruction and candidate selection

Candidates:

1. Focused extractive evidence.
2. Selectively stitched extractive evidence.
3. Raw generated answer.
4. Safely source-snapped generated answer.

Source snapping aligns only evidence-supported values and spans:

- document, Article, Clause, and Point identifiers
- dates
- monetary amounts
- named legal actors when unambiguous

It must not globally replace every matching token with the first evidence value.

The initial selector is deterministic and uses features such as retrieval confidence, evidence coverage, candidate length, citation consistency, and source overlap. For outer fold `f`, thresholds are selected without using fold `f` labels, using only training-partition predictions or already completed non-`f` folds. The unbiased OOF report is frozen before final thresholds are fitted on all OOF predictions for public-test inference. The selector does not automatically append a fixed 1,500-character evidence suffix.

## 10. Evaluation Strategy

### 10.1 Metrics

Primary:

- Official METEOR using whitespace-tokenized reference and prediction text.

Secondary:

- ROUGE-L without stemming
- document Recall@K
- Article Recall@K
- chunk Recall@K
- dense unique-positive gain
- reranker Top-1 and Top-8 evidence hit
- answer length
- latency
- peak GPU memory

### 10.2 Evaluation cadence

1. Run schema, label, and leakage tests on CPU.
2. Establish corrected memory/BM25/extractive baselines.
3. Evaluate retrieval and reranking across all five folds.
4. Run one QLoRA smoke batch.
5. Train and resume one pilot fold.
6. Run full five-fold QLoRA only for the promoted configuration.
7. Confirm the final stack against the public evaluation path.

A historical or Notion score is not accepted as a baseline unless its source revision, artifacts, configuration, and predictions can be reproduced.

### 10.3 Promotion rule

A component is promoted when:

- its intended layer metric improves
- mean OOF METEOR does not regress
- gains are not isolated to one anomalous fold
- parameter, runtime, and memory constraints remain valid

The run table records rejected as well as promoted configurations.

## 11. Parameter Compliance

The target approximate stack is:

- DEk21 embedding v2: approximately 0.10B.
- BGE reranker v2 M3: approximately 0.568B.
- Qwen2.5-3B-Instruct: approximately 3.09B.

Approximate total: 3.76B before any separately counted adapter tensors.

Before training or submission, the parameter audit loads pinned model configurations and counts:

- all base-model learned parameters included in the stack
- all LoRA/QLoRA adapter tensors
- any retriever or reranker adapters
- any other learned selector or memory model

The run fails unless the exact audited total is strictly below 4.0B. Quantization changes memory, not the official learned-parameter count.

## 12. Kaggle and Colab Runtime Design

### 12.1 Thin notebook contract

Each notebook contains only:

1. Runtime/path selection.
2. Dependency, model, secret, storage, and GPU diagnostics.
3. A small data/model smoke test.
4. Calls to canonical scripts for the selected stage.
5. Metrics, checkpoints, adapter export, and optional submission packaging.

Notebook cells must not duplicate:

- sparse or dense retrieval
- prompt construction
- tokenizer/model loading
- training loops
- reconstruction
- submission caching
- artifact schemas

### 12.2 Path handling

The launcher detects or accepts explicit roots:

- Kaggle inputs: `/kaggle/input`
- Kaggle outputs: `/kaggle/working`
- Colab working root: `/content`
- Colab persistent root: configured Google Drive path
- Local: explicit repository/artifact paths

All canonical Python functions receive paths through configuration or arguments, not hard-coded notebook constants.

### 12.3 Reproducible dependencies

A pinned CUDA constraint set covers at minimum:

- Python
- PyTorch and compatible CUDA wheel
- Transformers
- TRL
- PEFT
- bitsandbytes
- Accelerate
- Sentence Transformers
- PyArrow
- Vietnamese segmentation dependencies

The runtime verifies installed versions before model loading. Kaggle and Colab use the same tested compatibility matrix.

### 12.4 Hardware preflight

Before expensive work, report and validate:

- GPU count and names
- compute capabilities
- installed Torch CUDA version
- architectures supported by the Torch binary
- free and total memory
- NCCL/distributed availability
- writable checkpoint/output paths

Dual mode requires two supported GPUs. A single-GPU fallback must be explicit. An unsupported P100/Torch combination stops before model loading or generation.

### 12.5 Dual-T4 QLoRA

- Launch through Accelerate or `torchrun` with one process per GPU.
- Each rank loads one 4-bit model replica on its local CUDA device.
- Do not use `device_map="auto"` with DDP.
- Use NF4, double quantization, and FP16 compute on T4.
- Enable BF16 only after `torch.cuda.is_bf16_supported()` succeeds.
- Set `use_cache=False` during training.
- Enable gradient checkpointing.
- Bucket examples by length.
- Truncate evidence intentionally before applying the chat template.
- Begin with microbatch 1 per GPU and gradient accumulation 8, for global batch 16.
- Promote microbatch 2 only after a measured smoke test leaves safe memory headroom.
- Only rank 0 writes checkpoints and manifests.

### 12.6 Checkpointing and resume

Checkpoints include:

- adapter weights
- optimizer and scheduler state
- trainer/global step
- RNG states
- fold ID
- config and data hashes
- model revisions
- dependency versions

Resume selects the latest compatible checkpoint. A mismatched checkpoint fails with a clear diagnostic rather than silently restarting or mixing runs.

Kaggle writes active state to `/kaggle/working`. Colab may mirror promoted checkpoints to Drive. Final promoted adapters can be published as a separate versioned Kaggle output dataset after validation.

### 12.7 Secrets

- Remove all literal tokens and fallback tokens.
- Mounted public models are loaded without a token.
- Gated access uses Kaggle Secrets, Colab userdata, or environment variables.
- Secret-shaped strings are checked before push.

## 13. Kaggle Training Dataset Package

The Kaggle dataset titled `LegalQA` contains only training-ready, reproducible inputs:

- canonical data parquets
- fold assignments
- BM25 index without a duplicated full corpus parquet
- persistent DEk21 embeddings/index when promoted
- cached retrieval/reranking candidates keyed by manifest hash
- configuration and provenance manifests

It excludes:

- duplicated raw ZIP and extracted copies when canonical parquets suffice
- Python caches and OS metadata
- smoke-test adapters
- stale 100-row OOF samples
- redundant intermediate checkpoints
- dated score logs without reproducible provenance

Dataset packaging validates file hashes and schemas before a new Kaggle dataset version is created.

## 14. Kaggle Metadata Changes

### 14.1 Dataset

Target display title: `LegalQA`.

The installed Kaggle CLI supports updating the dataset title through metadata update. The existing ref remains:

- `phucdangg/legalqa-task2-clean-data`

The CLI does not expose a safe in-place dataset slug rename. Creating a replacement dataset solely to change the slug is out of scope unless separately approved.

### 14.2 Kernel

Target title: `LegalQA training`.  
Target ref: `phucdangg/legalqa-training`.  
Stable `id_no`: `132539626`.

Before push:

- pull current metadata into a scratch directory
- preserve `id_no`
- update title and versionless ID
- align `code_file` with the canonical notebook filename
- run local notebook/static preflight
- verify no credentials are present

Pushing starts a kernel run. Push occurs only after local verification, followed by status, log, and output validation.

## 15. Failure Handling

The pipeline fails before expensive work for:

- secret-shaped credentials in source
- unsupported GPU architecture or wrong GPU count
- incompatible dependency versions
- missing or invalid artifacts
- mismatched question/reference IDs
- empty positive labels or collapsed citation joins
- fold leakage
- parameter count at or above 4.0B
- incompatible checkpoint hashes
- non-writable output/checkpoint paths

Operational rules:

- parser errors include the document ID and reason
- subprocess return codes are checked
- cached predictions require matching input/config/model hashes
- malformed cache entries are rejected
- generation writes incremental state
- submission packaging verifies exact IDs, answer schema, non-empty answers, and ZIP member name
- failures never reuse an unverified stale submission

## 16. Test Strategy

### 16.1 Unit tests

- Full legal-number suffix parsing, including `QH14`-style identifiers.
- Canonical document alias matching.
- Hierarchy parsing and child offsets.
- BM25 state reset, case normalization, fallback fields, save/load, and no-match behavior.
- Duplicate/conflicting QA policy.
- Hard-negative false-negative guards.
- Multi-date, monetary, legal-identifier, and actor snapping without global corruption.
- Candidate selection rules.

### 16.2 Integration tests

- Real artifact schema and row-count sanity checks.
- Citation-to-corpus join coverage and non-zero positives.
- Retrieval-label and fold generation.
- Persistent dense index build, save, load, and non-mock query.
- BM25 + DEk21 + RRF + BGE flow on a representative sample.
- Fold isolation across memory, labels, negatives, demonstrations, and training rows.
- Exact official METEOR wrapper and ID alignment.
- Submission JSON and ZIP validation.

### 16.3 Training/runtime tests

- Pinned TRL/Transformers/PEFT API smoke test.
- One-batch QLoRA forward/backward/save/load.
- T4 FP16 selection and BF16 rejection.
- One-process-per-GPU placement.
- Two-rank launch.
- Checkpoint/resume compatibility.
- Peak-memory reporting.

### 16.4 Notebook and Kaggle tests

- No literal secret patterns.
- No embedded pipeline implementation.
- Colab launcher contains no unconditional Kaggle paths.
- Kaggle launcher reports both GPUs.
- Metadata `code_file`, title, ID, and sources are consistent.
- Remote run reaches success and exposes expected outputs.

## 17. Implementation Boundary

High-confidence files in scope include:

- `configs/task2.yaml`
- `src/common/normalize.py`
- `src/common/legal_parser.py`
- `src/common/evidence.py`
- `src/common/bm25.py`
- `src/common/dense_dek21.py`
- `src/common/reranker.py`
- `src/task2/qa_memory.py`
- `src/task2/article_stitcher.py`
- `src/task2/generator.py`
- `src/task2/source_snap.py`
- `src/task2/predict.py`
- `scripts/prepare_data.py`
- `scripts/build_indexes.py`
- `scripts/train_retriever_mnrl.py`
- `scripts/train_generator_qlora.py`
- `scripts/run_oof_validation.py`
- `scripts/predict.py`
- `scripts/fetch_kaggle_submission.py`
- `scripts/monitor_kaggle.py`
- `kaggle_kernel/kernel-metadata.json`
- `kaggle_kernel/legalqa_gpu_pipeline.ipynb`
- `notebooks/DSC2026_Task2_LegalQA_Pipeline.ipynb`
- focused tests for the approved flow

`kaggle_kernel/legalqa_train_gpu.py` is removed after its required behavior is represented by canonical modules and the notebook launcher.

Stale configs and docs are changed only when they would misdirect this workflow. Unrelated artifacts and dead code are not deleted automatically.

## 18. Delivery Sequence

1. Remove working-tree credentials and add secret preflight.
2. Repair normalization, legal identifiers, parser reporting, QA conflicts, citation joins, and retrieval-label generation.
3. Correct BM25 and establish the extractive baseline.
4. Implement and persist batched DEk21 retrieval and RRF.
5. Evaluate pretrained BGE and add tuning only if promoted.
6. Repair selective stitching, provision memory, source snapping, and candidate selection.
7. Pin CUDA dependencies and repair evidence-conditioned QLoRA.
8. Implement fold-safe evaluation, checkpointing, resume, and run manifests.
9. Replace Kaggle and Colab notebooks with thin launchers.
10. Package the training-ready Kaggle dataset.
11. Run local/CPU tests, one-batch training smoke, one-fold pilot, and checkpoint resume.
12. Run full promoted evaluation and dual-T4 training.
13. Update Kaggle titles/metadata, push the verified kernel, and monitor it to completion.
14. Validate and retrieve remote outputs.

## 19. Completion Criteria

The work is complete when:

- the exposed credential is absent from the working tree and the user has been told to revoke or rotate it
- all required data schemas and IDs validate
- retrieval labels contain verified positives and guarded negatives
- corrected BM25 and extractive baselines are recorded
- the real DEk21 index persists and returns non-mock results
- every optional component has a recorded promotion or rejection decision
- the exact learned-parameter audit is strictly below 4.0B
- current tests and new targeted tests pass
- one QLoRA smoke batch and one-fold pilot complete
- a compatible checkpoint resumes successfully
- dual-T4 training uses two ranks
- five-fold metrics and final promotion decisions are recorded
- submission JSON and ZIP pass strict validation
- Kaggle dataset display title is `LegalQA`
- Kaggle kernel title is `LegalQA training`
- the pushed Kaggle kernel finishes successfully and expected outputs can be retrieved
- the Colab launcher uses the same canonical code and configuration without Kaggle-only assumptions

## 20. Deferred Decisions

These are decided by measured evidence during implementation, not assumed in advance:

- whether DEk21 needs Task 2-specific fine-tuning
- whether BGE reranker tuning improves final METEOR
- exact retrieval and reranking candidate counts
- selective stitching thresholds
- provision-memory eligibility threshold
- QLoRA learning rate, epochs, and promoted microbatch
- deterministic candidate-selection thresholds
- whether a replacement Kaggle dataset is worth creating solely for a cleaner slug

All deferred values must be resolved through the promotion gates and recorded in run manifests before final delivery.
