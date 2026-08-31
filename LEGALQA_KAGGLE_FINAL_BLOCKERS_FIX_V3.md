# LegalQA Kaggle V3 — Final Blocker Fix Plan Before `Run All`

> Repository: `https://github.com/silent9669/LegalQA`
>
> Audited HEAD: `17bfe90fffb69df7f93334516fdd35293c42f712`
>
> This document is a **targeted remediation plan**, not another architecture rewrite. V2 moved the repository in the correct direction, but the current canonical Kaggle notebook is still **not safe to run as a real train → validate → infer submission notebook**. Fix every P0 item below before consuming Kaggle GPU quota.

## Goal

Produce one clean, reproducible **Stack A production pipeline** that actually:

```text
mounts verified runtime data/code/indexes
→ installs/validates dependencies
→ performs preflight
→ trains the Task-2 reranker when requested
→ trains Qwen2.5-3B QLoRA when requested
→ fails if requested training is skipped or broken
→ reloads both trained checkpoints
→ evaluates those exact checkpoints on held-out data
→ promotes/rejects them using real METEOR/retrieval metrics
→ loads the exact promoted pipeline
→ runs efficient dual-T4 inference
→ creates a strictly valid submission
```

Do **not** keep fake Stack-B support in the production notebook. Keep Stack B as an offline experiment until it has a complete index/model path and measured results. Production readiness is more important than pretending both stacks are implemented.

---

# 0. Current verdict

**DO NOT RUN THE CURRENT NOTEBOOK FOR THE FINAL TRAINING/SUBMISSION YET.**

The latest push correctly added many V2 components, including:

- canonical config files;
- near-duplicate fold grouping;
- whole-fold QA-memory exclusion in sampled OOF;
- BM25S mmap loading;
- a generic GPU dense retriever;
- reranker training code;
- QLoRA training code;
- evidence packing;
- candidate/selector modules;
- a 14-cell Kaggle notebook;
- runtime code packaging.

However, the current repository still has hard blockers. The most important ones are:

1. **The notebook defaults to not training anything.**
2. **The packaged Kaggle runtime does not include `scripts/`, yet the notebook imports `scripts.*`.**
3. **The reranker training pairs are not included in the packaged runtime, so reranker training can silently skip.**
4. **Requested training failures/skips do not fail the notebook; it silently falls back to the base model.**
5. **The “dev evaluation” uses `mode="fast"`, which deliberately uses mock dense/reranker/fallback generation; therefore it does not evaluate the adapter/reranker just trained.**
6. **The final notebook hardcodes Stack A internals while exposing a fake `FINAL_STACK="stack_b"` option.**
7. **Dense index integrity is not actually enforced; chunk-ID mismatch is only a warning and embedding hashes are not checked.**
8. **Dense FP16 files are loaded then immediately converted to FP32, defeating much of the memory optimization.**
9. **The notebook inference loop calls `predict_single()` per question, so the advertised batched dual-T4 inference is largely not batched.**
10. **The current reranker trainer never evaluates its validation fold or saves the best checkpoint.**
11. **The current QLoRA “answer-preserving truncation” is character-based and does not use the answer/token budget at all.**
12. **Training/inference do not actually use tokenizer-native chat templates despite claiming they do.**
13. **A failed QLoRA reload smoke test only prints a warning but still reports training as `completed`.**
14. **The current parameter audit does not include the actual trained adapter parameter count.**
15. **The README is still the pre-V2 architecture/results and must not be trusted as current provenance.**

Fix these before the user presses `Run All`.

---

# 1. P0 — Make the Kaggle runtime actually importable

## Files

- Modify: `scripts/package_kaggle_dataset.py`
- Modify: `tests/test_kaggle_packaging.py`
- Modify: `kaggle_kernel/legalqa_gpu_pipeline.ipynb`

## Problem

The current package stages:

```text
code/LegalQA/src/
code/LegalQA/configs/
code/LegalQA/requirements-kaggle.txt
```

but the notebook imports:

```python
from scripts.preflight_kaggle import run_preflight_checks
from scripts.audit_parameters import audit_parameter_budget
from scripts.run_oof_validation import run_oof_validation
```

Therefore a genuinely self-contained Kaggle runtime dataset is incomplete.

## Required fix

Package these directories/files:

```text
code/LegalQA/
├── src/**
├── scripts/**
├── configs/**
├── requirements-kaggle.txt
└── code_manifest.json
```

Exclude caches and generated outputs:

```python
ignore=shutil.ignore_patterns(
    "__pycache__",
    "*.pyc",
    ".DS_Store",
    "artifacts",
    "kaggle_dataset",
)
```

Update the code manifest to hash **both `src/` and `scripts/`**.

## Required test

```python
def test_packaged_runtime_imports_notebook_dependencies(tmp_path):
    # package fixture
    # add staged code/LegalQA to sys.path
    from scripts.preflight_kaggle import run_preflight_checks
    from scripts.audit_parameters import audit_parameter_budget
    from scripts.run_oof_validation import run_oof_validation
    from src.task2.predict import LegalQAPipeline
```

This must import using the staged runtime path, not the developer checkout.

---

# 2. P0 — Package reranker training data or generate it before training

## Files

- Modify: `scripts/package_kaggle_dataset.py`
- Modify: `scripts/preflight_kaggle.py`
- Modify: `kaggle_kernel/legalqa_gpu_pipeline.ipynb`

## Problem

The notebook expects:

```text
DATA_DIR/reranker_training_pairs.parquet
```

when `RUN_RERANKER_TRAINING=True`.

The current packager does not include that file in `REQUIRED_FILES`, and the notebook does not mine negatives before reranker training.

## Preferred production fix

Generate hard negatives **before uploading the Kaggle runtime dataset**:

```bash
python scripts/mine_retrieval_negatives.py \
  --data_dir artifacts/task2/data \
  --bm25_dir artifacts/task2/indexes/bm25 \
  --dense_dir artifacts/task2/indexes/dek21 \
  --output artifacts/task2/data/reranker_training_pairs.parquet
```

Then package it as required whenever reranker training is enabled:

```python
REQUIRED_TRAINING_FILES = [
    "data/reranker_training_pairs.parquet",
]
```

Add a packaging flag if needed:

```text
--training-runtime
```

but the canonical training notebook must mount a dataset that contains the pairs.

## Preflight

When `RUN_RERANKER_TRAINING=True`:

```python
assert os.path.exists(RERANKER_PAIRS_PATH)
assert len(pd.read_parquet(RERANKER_PAIRS_PATH, columns=["qa_id"])) > 0
```

A missing file is fatal, not a warning.

---

# 3. P0 — The default notebook must truly be a training notebook

## File

- Modify: `kaggle_kernel/legalqa_gpu_pipeline.ipynb`

## Problem

The current default is:

```python
RUN_RERANKER_TRAINING = False
RUN_GENERATOR_TRAINING = False
```

So clicking `Run All` produces an inference run, not a training run.

## Fix

Use one explicit execution profile instead of independent ambiguous booleans:

```python
EXECUTION_PROFILE = "train_and_submit"
# choices:
#   "train_and_submit"
#   "reuse_checkpoints_and_submit"
#   "smoke_only"

if EXECUTION_PROFILE == "train_and_submit":
    RUN_RERANKER_TRAINING = True
    RUN_GENERATOR_TRAINING = True
    RUN_DEV_EVALUATION = True
    RUN_PUBLIC_INFERENCE = True
    REUSE_EXISTING_CHECKPOINTS = False
elif EXECUTION_PROFILE == "reuse_checkpoints_and_submit":
    RUN_RERANKER_TRAINING = False
    RUN_GENERATOR_TRAINING = False
    RUN_DEV_EVALUATION = True
    RUN_PUBLIC_INFERENCE = True
    REUSE_EXISTING_CHECKPOINTS = True
elif EXECUTION_PROFILE == "smoke_only":
    ...
else:
    raise ValueError(...)
```

Canonical committed default:

```python
EXECUTION_PROFILE = "train_and_submit"
```

The first cell must print a large summary before any expensive work.

---

# 4. P0 — Requested training must never silently skip

## Files

- Modify: `kaggle_kernel/legalqa_gpu_pipeline.ipynb`
- Modify: `src/task2/training/train_generator.py`
- Modify: `src/task2/training/train_reranker.py`

## Current dangerous behavior

Training functions can return:

```python
{"status": "skipped", "reason": ...}
```

and the notebook simply continues using the base checkpoint.

That means the user can enable training, wait for a run, and still submit an untrained model.

## Required behavior

Notebook:

```python
if RUN_GENERATOR_TRAINING:
    result = run_qlora_training(...)
    if result.get("status") != "completed":
        raise RuntimeError(
            f"Generator training was requested but did not complete: {result}"
        )

if RUN_RERANKER_TRAINING:
    result = train_bge_reranker(...)
    if result.get("status") != "completed":
        raise RuntimeError(
            f"Reranker training was requested but did not complete: {result}"
        )
```

Training code should raise exceptions for fatal production errors rather than convert every failure into `status="skipped"`.

Diagnostic/smoke modes may catch them explicitly.

---

# 5. P0 — Install/validate Kaggle dependencies before importing project modules

## Files

- Modify: `requirements-kaggle.txt`
- Modify: notebook Cell 2/3
- Create test/helper if useful: `scripts/check_runtime_dependencies.py`

## Problem

The notebook currently does not install `requirements-kaggle.txt`.

A clean Kaggle image may not contain compatible versions of:

```text
bm25s
sentence-transformers
peft
trl
bitsandbytes
pyvi
```

and training can skip/fail.

## Fix

Do **not** blindly upgrade torch/CUDA packages.

After resolving the packaged code root, inspect required imports and install only missing project packages:

```python
REQ = os.path.join(resolved_code_root, "requirements-kaggle.txt")
subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "-r",
    REQ,
])
```

If installing the whole requirements file risks upgrading already-compatible heavy packages, create a Kaggle-safe dependency bootstrap that checks import/version first.

`requirements-kaggle.txt` must include `pyvi` because normalization imports it optionally and the canonical BM25 index/query tokenizer must use the same preprocessing environment.

After installation, print versions of:

```text
torch
transformers
peft
trl
bitsandbytes
sentence_transformers
bm25s
```

Do not log the HF token.

---

# 6. P0 — Replace fake “dev validation” with evaluation of the exact trained checkpoints

## Files

- Major modify: `scripts/run_oof_validation.py`
- Modify: notebook evaluation cell

## Current bug

The notebook calls:

```python
run_oof_validation(..., mode="fast", adapter_path=ADAPTER_PATH)
```

But `mode="fast"` deliberately uses:

```text
mock dense retriever
mock/lexical reranker
fallback extractive generator
```

Therefore the returned METEOR **does not evaluate the QLoRA adapter or trained reranker**.

This is a P0 correctness failure.

## Required architecture

Separate two concepts:

### A. Architecture OOF

Offline / expensive experiment script.

```text
run_oof_validation --mode full
```

must use:

- `DenseRetriever`, not the legacy conceptual path;
- actual dense index;
- explicitly supplied reranker checkpoint;
- explicitly supplied Qwen base + adapter;
- current `EvidencePacker`;
- current candidates;
- current selector/fixed policy;
- no mocks/fallbacks.

Add arguments:

```text
--dense_model
--dense_dir
--reranker_checkpoint
--generator_model
--adapter
--selector_checkpoint
--fail_on_fallback
```

### B. Notebook post-training holdout sanity evaluation

This does not need full 5-fold retraining.

Build the exact pipeline using:

```python
reranker_checkpoint=RERANKER_CHECKPOINT
generator_adapter=ADAPTER_PATH
```

and evaluate a held-out fold/sample that was excluded from the screen-training configuration.

If the final production reranker/generator was trained on **all data**, this sanity check cannot be called OOF. Call it `training_smoke_eval` only.

For architecture promotion, use previously generated leakage-safe OOF results.

## Absolutely forbidden

Do not print:

```text
"trained adapter METEOR = ..."
```

from a fast/mock run.

---

# 7. P0 — Reranker trainer must validate and save best checkpoint

## File

- Modify: `src/task2/training/train_reranker.py`

## Current issue

`prepare_reranker_dataset()` returns validation examples, but training never evaluates them.

The code always saves the final epoch.

## Required minimum

Use a validation DataLoader and report:

```text
val binary loss
pairwise accuracy: score(pos) > score(neg)
MRR/Recall@K on a retrieval validation sample if available
```

Save best validation checkpoint, not just last epoch.

### T4 memory safety

Do not assume full fine-tuning with batch size 8 and 512 tokens fits comfortably on a 16 GB T4.

Preferred first production configuration:

```yaml
batch_size: 2
gradient_accumulation_steps: 4
max_length: 384 or 512 after measurement
fp16: true
```

Better option if full fine-tuning is unstable: **LoRA-tune the reranker** rather than OOMing.

If using LoRA, ensure the final inference checkpoint is loadable through the normal reranker wrapper and parameter accounting is correct.

### Required smoke test

Train 20–50 optimizer steps on a tiny fixture/real subset on T4 and prove:

```text
loss finite
checkpoint saves
checkpoint reloads
scores a pair
GPU peak memory printed
```

---

# 8. P0 — Fix QLoRA prompt construction and token-level truncation

## Files

- Modify: `src/task2/generator.py`
- Modify: `src/task2/training/train_generator.py`
- Modify tests: `tests/test_generator_training_data.py`, `tests/test_task2_generator.py`

## Current problems

### Problem A — “native chat template” is not actually used

`format_qwen_chat_prompt(..., tokenizer=None)` uses a manual ChatML fallback.

Training calls it without the tokenizer.

Inference also calls `self.format_prompt(...)` without passing `self.tokenizer`.

Therefore the native tokenizer template path is effectively unused.

### Required fix

`QwenGenerator` must render prompts with its tokenizer:

```python
def format_prompt(self, question: str, evidence: str) -> str:
    return format_qwen_chat_prompt(
        question,
        evidence,
        tokenizer=self.tokenizer,
    )
```

Training should load the tokenizer **before building examples**, then pass it into the training-example builder.

Use the same helper for train and inference.

### Problem B — “answer-preserving truncation” is not answer/token aware

Current helper accepts `answer` but never uses it. It simply trims evidence by character count.

Replace with tokenizer-aware packing.

Pseudo-interface:

```python
def build_sft_example(
    tokenizer,
    question: str,
    evidence_units: list[str],
    answer: str,
    max_seq_len: int,
) -> dict:
    ...
```

Algorithm:

1. tokenize answer + assistant framing first;
2. reserve those tokens;
3. tokenize system/question framing;
4. remaining token budget belongs to evidence;
5. add complete evidence units until the budget is exhausted;
6. only truncate the final evidence unit as a last resort;
7. verify the final answer tokens remain present;
8. record whether evidence/answer truncation occurred.

Output manifest diagnostics:

```text
p50/p90 total tokens
% evidence truncated
% answer truncated
max answer tokens
```

Target `% answer truncated` should be near zero. If not, evaluate `max_seq_len=3072`/`4096` versus T4 memory before accepting 2048.

---

# 9. P0 — A failed adapter reload is a failed training run

## File

- Modify: `src/task2/training/train_generator.py`

## Current bug

Reload smoke test exceptions are caught and only printed:

```python
except Exception as e:
    print("Warning ...")

return {"status": "completed"}
```

## Fix

```python
try:
    reload_gen = ...
    test_out = reload_gen.generate(...)
    if not test_out.strip():
        raise RuntimeError("empty smoke output")
except Exception as e:
    raise RuntimeError(
        f"QLoRA checkpoint saved but failed reload smoke test: {e}"
    ) from e
```

Also explicitly free `reload_gen` after smoke verification before loading final inference components.

---

# 10. P0 — Production notebook must support one real stack, not fake Stack B

## Files

- Modify: notebook
- Modify: configs/models.yaml
- Modify: configs/experiments.yaml / docs

## Current inconsistency

The notebook exposes:

```python
FINAL_STACK = "stack_a" or "stack_b"
```

but still resolves:

```text
DEK21_DIR
Qwen 3B mounted model path
```

for both.

So Stack B is not truly executable.

## Fix for final readiness

For the canonical notebook, use:

```python
PRODUCTION_STACK = "stack_a"
```

and remove Stack B from runtime switching until all of the following exist:

```text
BGE-M3 corpus index
BGE-M3 model resolution
Qwen1.5B model resolution
correct parameter audit
measured retrieval metrics
measured official-METEOR result
```

Keep Stack B in `configs/experiments.yaml` as an **offline experimental candidate**.

Do not delay a stable Stack-A training run merely to preserve a non-functional toggle.

---

# 11. P0 — Dense index integrity and memory behavior

## File

- Modify: `src/common/dense.py`
- Modify tests: `tests/test_common_dense_and_rrf.py`
- Modify preflight

## Problems

Current load path does:

```python
np.load(emb_path).astype(np.float32)
```

so a saved FP16 index is expanded to FP32 in RAM before being copied to FP16 GPU memory.

Also:

- saved `chunk_ids_sha256` is not checked;
- `embeddings_sha256` is not checked;
- mismatched chunk IDs only print a warning;
- GPU corpus allocation/search failure silently falls back to CPU.

## Required fix

Load with mmap and preserve stored dtype:

```python
arr = np.load(emb_path, mmap_mode="r")
```

Do not `.astype(np.float32)` unless explicitly needed for a CPU reference path.

Validate:

```python
assert arr.ndim == 2
assert arr.shape[0] == corpus_rows
assert arr.shape[1] == manifest_dim
assert str(arr.dtype) == manifest_dtype
```

Compute current corpus chunk-ID hash without storing/reading a gigantic full `doc_ids` list in the manifest if possible:

```python
hash.update(chunk_id.encode())
hash.update(b"\n")
```

Mismatch in final mode:

```python
raise ValueError("FINAL_PIPELINE_ERROR: dense/corpus chunk-id hash mismatch")
```

GPU allocation/search failure in final mode must also fail loudly:

```python
DenseRetriever(..., final_mode=True)
```

Diagnostic mode may permit CPU fallback.

### Model-specific preprocessing

Do not unconditionally run PyVi tokenization for every SentenceTransformer model.

Add model-specific preprocessing:

```python
if model_family == "dek21":
    text = verified_dek21_preprocess(text)
elif model_family == "bge_m3":
    text = raw_normalized_text
```

Verify the expected DEk21 preprocessing from the model documentation/config rather than assuming PyVi is optimal.

---

# 12. P0 — Fix BM25 failure path

## File

- Modify: `src/common/bm25.py`

The file calls:

```python
print(..., file=sys.stderr)
```

but does not currently import `sys`.

Add:

```python
import sys
```

Also expose `final_mode`/`fail_on_missing_index` so a final Kaggle run cannot silently rebuild an 801k BM25 index when the packaged index is broken.

Notebook Cell 5 should obey `ALLOW_INDEX_REBUILD=False` for BM25 **and** dense indexes.

---

# 13. P1 — Fix reranker metadata used by the candidate selector

## File

- Modify: `src/task2/predict.py`

Current code derives:

```python
"rerank_top1": top_seeds[0].get("score", 0.0)
```

but the reranker stores the neural value in `rerank_score`.

Use:

```python
r1 = float(top_seeds[0].get("rerank_score", top_seeds[0].get("score", 0.0)))
r2 = float(top_seeds[1].get("rerank_score", top_seeds[1].get("score", 0.0))) if len(top_seeds) > 1 else r1
```

then:

```python
"rerank_top1": r1
"rerank_margin": r1 - r2
```

Add a regression test.

---

# 14. P1 — Fix selector training/inference feature distribution shift

## File

- Modify: `src/task2/selector.py`
- Modify OOF candidate table generation

## Current bug

During `fit_meta_oof()`, `extract_candidate_features()` is called without per-query retrieval metadata, so:

```text
rerank_top1
rerank_margin
bm25_top1
dense_top1
fuzzy_sim
```

are all zeros in selector training.

At inference they may be non-zero.

That is a feature-distribution mismatch.

## Required fix

Persist retrieval metadata into every OOF candidate row:

```text
qa_id
fold_id
question
cand_name
cand_text
meteor
evidence
bm25_top1
dense_top1
rerank_top1
rerank_margin
fuzzy_sim
```

Pass it into feature extraction during selector fit.

If retrieval metadata is not available, remove those features entirely rather than train on zeros and infer on nonzeros.

### Leakage fallback

If `fold_id` is missing, do **not** assign random folds per candidate row.

Generate deterministic meta folds by **unique qa_id**, ensuring all candidates for the same QA stay together.

---

# 15. P1 — Do not promote a learned selector until corrected full OOF exists

## Production rule for this release

Until a correct full OOF candidate table is available after the V3 fixes:

```python
selector = CandidateSelector(
    policy="fixed_baseline",
    best_fixed_candidate=<best validated fixed family>
)
```

Do not automatically assume old `stitched_extract=0.3051` is still the winner because the new EvidencePacker changed the pipeline.

Run corrected evaluation on a representative fold/sample and compare at least:

```text
focused_extract
stitched_extract
pack_focused
pack_full_article
pack_top2_relevance
Strategy F variants
generated base
generated QLoRA
```

If no new selector has valid meta-OOF proof, ship the best fixed family.

---

# 16. P1 — Make public inference actually batched

## Files

- Modify: `src/task2/predict.py`
- Modify: notebook inference cell

## Current issue

The notebook chunks queries into batches but then calls:

```python
pipeline.predict_single(...)
```

inside the batch.

`predict_batch()` also loops over `predict_single()`.

So dense retrieval, reranking, and generation remain mostly serial.

## Target implementation

Create a true batch method:

```python
def predict_batch(
    items: list[dict],
    max_new_tokens: int = 384,
    retrieval_batch_size: int = 32,
    rerank_batch_size: int = 32,
    generation_batch_size: int = 2,
) -> dict:
    ...
```

Flow:

```text
exact-memory prepass
→ BM25 candidates
→ DenseRetriever.search_batch(all unseen questions)
→ fuse each query
→ batch cross-encoder pairs if wrapper supports it
→ pack evidence
→ QwenGenerator.generate_batch()
→ candidates
→ selector
```

Do not optimize batching at the cost of correctness. Verify batch results equal single-query results on a deterministic small fixture.

Given long legal contexts, start Qwen generation at batch size 2 on T4 and increase only after real VRAM measurement.

---

# 17. P1 — Strengthen preflight so it checks what matters

## File

- Modify: `scripts/preflight_kaggle.py`
- Modify notebook Cell 4

Current notebook sets:

```python
check_dataset_files=False
```

and preflight does not validate retrieval index manifests.

For `train_and_submit`, require:

```text
2 CUDA GPUs
all canonical data files
reranker pairs
BM25 manifest + BM25S files
dense manifest + embeddings.npy
corpus row count
dense model ID == production stack model
chunk ID hash
code_manifest
runtime code root
training dependencies
public test count == 1000
parameter budget base stack
```

Pass the actual resolved Kaggle paths into preflight rather than default local `artifacts/...` paths.

A fatal preflight error must stop before model loading/training.

---

# 18. P1 — Exact adapter parameter audit

## Files

- Modify: `src/task2/training/train_generator.py`
- Modify: `scripts/audit_parameters.py`
- Modify notebook after training

During QLoRA training record:

```python
adapter_trainable_params = sum(
    p.numel() for p in trainer.model.parameters() if p.requires_grad
)
```

Persist in `training_manifest.json`.

After training:

```python
audit_parameter_budget(
    stack="stack_a",
    extra_adapter_params=adapter_trainable_params,
    adapter_name="qwen_qlora"
)
```

Abort if total is `>= 4_000_000_000`.

Also include any learned reranker adapter if reranker uses LoRA and any learned selector parameters if competition accounting requires them.

---

# 19. P1 — Do not call the current README results current

## File

- Rewrite relevant sections of `README.md`

The current README still documents the old article-stitcher/source-snap architecture and stale OOF table.

After V3:

- document the real production Stack A;
- distinguish “historical/legacy result” from new full validation;
- give exact Kaggle execution profile;
- state whether reranker/QLoRA are trained in notebook;
- state exact parameter total including adapter;
- include the latest validated fixed/selector METEOR only if produced by the corrected pipeline;
- do not claim 48 tests if the current count differs;
- do not claim dual-GPU training if QLoRA itself uses one GPU.

The correct hardware statement is likely:

```text
Training:
  GPU 0 -> Qwen QLoRA
  GPU 1 -> reranker training in a separate sequential stage

Inference:
  GPU 0 -> Qwen generator
  GPU 1 -> dense retrieval + reranker
```

unless real concurrent behavior is implemented and measured.

---

# 20. QLoRA configuration for the first real Kaggle run

Do not perform a large hyperparameter sweep in the canonical notebook.

After a successful smoke test, use one stable production candidate:

```yaml
base_model: Qwen2.5-3B-Instruct
quantization: 4-bit NF4 + double quant
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
learning_rate: 1.0e-4
num_epochs: 1
batch_size: 1
grad_accum: 8
gradient_checkpointing: true
optimizer: paged_adamw_8bit
sampling: deterministic at inference
```

The current `2e-4` may work, but `1e-4` is the safer first full run unless prior correct validation proved `2e-4` better.

Do not blindly use `max_seq_len=2048`; choose the value from the tokenized truncation diagnostic. If 2048 truncates a meaningful share of gold answers, test 3072 with batch 1 before deciding.

---

# 21. Reranker training configuration for the first real Kaggle run

Prefer a stable memory-safe configuration:

```yaml
base_model: BAAI/bge-reranker-v2-m3
val_fold: 0
max_length: 384  # 512 only if memory allows and validation improves
batch_size: 2
grad_accum: 4
learning_rate: 2e-5
epochs: 1 first
fp16: true
best_checkpoint_metric: validation pairwise accuracy / loss
```

Run an explicit 50-step smoke training before the full stage.

If full fine-tuning OOMs, switch to PEFT/LoRA instead of reducing correctness checks.

---

# 22. Correct final notebook order

The canonical notebook should end up with roughly this structure:

```text
Cell 1  Execution profile / production Stack A config
Cell 2  Base environment + secrets + device diagnostics
Cell 3  Resolve packaged runtime code/data
Cell 4  Install/check project dependencies
Cell 5  Full production preflight
Cell 6  Load QA/corpus/index metadata
Cell 7  Reranker training smoke -> full train -> best checkpoint reload
Cell 8  QLoRA token diagnostics -> smoke -> full train -> adapter reload
Cell 9  Exact parameter audit including adapter
Cell 10 Real held-out checkpoint regression evaluation
Cell 11 Release training objects / clear VRAM / load final promoted pipeline
Cell 12 True batched public inference
Cell 13 Strict submission checks + runtime diagnostics
Cell 14 Save submission + run/training manifests + checkpoints
```

Do not run public inference if either requested training stage did not complete and pass reload/evaluation gates.

---

# 23. Mandatory acceptance tests

Add/repair tests so these behaviors are executable, not comments.

## Runtime packaging

- [ ] staged runtime contains `src/`
- [ ] staged runtime contains `scripts/`
- [ ] staged runtime contains configs
- [ ] staged runtime contains requirements
- [ ] training runtime contains `reranker_training_pairs.parquet`
- [ ] imports work using only staged code root

## Notebook/profile

- [ ] committed default profile is `train_and_submit`
- [ ] requested skipped generator training raises
- [ ] requested skipped reranker training raises
- [ ] public inference cannot execute before training promotion state is resolved

## BM25

- [ ] `sys` import/fallback path tested
- [ ] valid BM25S mmap does not build Python postings
- [ ] missing final BM25 index fails when rebuild is disabled

## Dense

- [ ] FP16 index remains FP16 after mmap load
- [ ] dense row mismatch raises
- [ ] chunk-ID hash mismatch raises
- [ ] wrong dense model ID raises
- [ ] batch top-K matches single-query top-K
- [ ] final-mode GPU search failure raises instead of CPU fallback

## Reranker

- [ ] held-out fold excluded from training
- [ ] validation loop actually runs
- [ ] best checkpoint selection tested
- [ ] checkpoint reload scores pairs
- [ ] training pair file missing causes requested-training failure

## Generator

- [ ] tokenizer-native prompt rendering used in both train and infer
- [ ] prompt bytes/text are identical for same tokenizer/question/evidence
- [ ] completion loss excludes system/user tokens
- [ ] token-level truncation trims evidence before answer
- [ ] answer truncation diagnostic exists
- [ ] adapter reload smoke failure raises
- [ ] adapter parameter count written to manifest

## Evaluation

- [ ] fast mode clearly identifies mocks
- [ ] notebook regression evaluation does not call fast mode
- [ ] full/checkpoint eval accepts reranker checkpoint + adapter
- [ ] actual generator runtime is `torch`, not fallback
- [ ] actual dense model/index identity recorded

## Selector

- [ ] `rerank_score` is used, not stale fused `score`
- [ ] selector train features and inference features have same schema/distribution
- [ ] fallback meta-fold grouping is by `qa_id`
- [ ] learned selector cannot be promoted below best fixed candidate

## Submission

- [ ] exactly 1000 IDs
- [ ] exact public ID set equality
- [ ] no empty answers
- [ ] no internal tags/chat tokens
- [ ] zip contains exactly root `submission.json`

---

# 24. Real Kaggle smoke gate — must happen before full training

After code changes and runtime dataset upload, create one temporary smoke profile:

```python
EXECUTION_PROFILE = "smoke_only"
```

It must do:

```text
load packaged code/scripts
preflight
load BM25 + dense index
train reranker for ~20–50 steps on small subset
save/reload reranker
train QLoRA for ~20–50 steps on small subset
save/reload adapter
build exact pipeline
answer 2–5 held-out questions
validate non-empty candidate set
print GPU peak memory
```

Only after this succeeds from a clean **Restart Session + Run All** should the canonical profile be changed/run as:

```python
EXECUTION_PROFILE = "train_and_submit"
```

---

# 25. What not to spend time on before the first stable run

Do not add more models/features until the production path is verified.

Specifically postpone:

- Stack B runtime switching;
- additional dense models;
- complex neural candidate selectors;
- multi-stage generator ensembles;
- ANN search;
- DDP QLoRA;
- multi-epoch generator sweeps.

The repository currently has enough modeling complexity. The highest-value work is making the existing strong stack **actually train, validate, reload, and infer without silent substitutions**.

---

# 26. After the first stable training run: score optimization order

Once the pipeline is technically trustworthy, optimize in this order:

```text
1. Verify retrieval Recall@20/50 on resolved citation labels
2. Compare pretrained vs task-tuned reranker
3. Compare evidence packs / extractive candidates
4. Compare base Qwen vs QLoRA
5. Tune QLoRA LR/sequence length only if QLoRA adds value
6. Rebuild meta-OOF selector only after candidate families stabilize
7. Optional Stack B bake-off
```

For METEOR, do not assume generation is superior. The fixed extractive candidate remains a legitimate final answer policy if it wins corrected validation.

---

# 27. Required final coding-agent report

Do not say “fixed” without evidence. Return:

## A. Git state

```text
HEAD SHA
changed files
removed/archived stale files
```

## B. Tests

```text
pytest command
passed / failed / skipped
```

## C. Runtime packaging proof

```text
staged code root
scripts present: yes/no
reranker pairs present: yes/no
BM25 index present: yes/no
dense index present: yes/no
code manifest SHA
```

## D. Dense validation

```text
model ID
dtype on disk
dtype after load
shape
chunk hash match
GPU tensor dtype/device
FP16 vs FP32 top-K parity sample
```

## E. Training smoke

```text
reranker smoke steps / peak VRAM / reload result
QLoRA smoke steps / peak VRAM / adapter reload result
adapter parameter count
```

## F. Real checkpoint evaluation

```text
validation IDs/fold
reranker checkpoint
adapter checkpoint
dense index
no mocks: yes
no fallbacks: yes
METEOR candidate table
best fixed candidate
promoted policy
```

## G. Parameter audit

```text
Dense
Reranker
Generator base
Generator adapter
Selector/head if applicable
TOTAL
<4B PASS/FAIL
```

## H. Clean Kaggle instructions

Exact user steps:

```text
1. Upload/version the newly packaged runtime dataset
2. Open canonical notebook
3. Mount LegalQA runtime dataset
4. Mount Qwen2.5-3B Kaggle model
5. Enable HF_TOKEN secret
6. Select GPU T4 x2
7. Internet ON
8. Restart Session
9. Run All
10. Verify expected checkpoint/evaluation/submission artifacts
```

---

# 28. Definition of “safe to run on Kaggle”

The coding agent may declare **SAFE TO RUN** only when all are true:

- [ ] clean runtime dataset includes every imported module/script;
- [ ] clean runtime includes every training artifact required by enabled stages;
- [ ] training profile enables real training by default;
- [ ] requested training cannot silently skip;
- [ ] reranker checkpoint validation/reload succeeds;
- [ ] QLoRA adapter validation/reload succeeds;
- [ ] post-training evaluation uses those exact checkpoints and no mocks;
- [ ] final dense/BM25 indexes are verified and cannot silently degrade;
- [ ] final production stack is a real implemented stack, not a fake toggle;
- [ ] final parameter count including adapter is `<4B`;
- [ ] clean Kaggle smoke `Restart Session + Run All` passes;
- [ ] final notebook writes valid `submission.json.zip`.

If any item is false, report **NOT SAFE TO RUN** and explain the remaining blocker.

---

# 29. Final instruction

Do not redesign LegalQA again. Repair the production execution path.

The current architecture is already sophisticated enough to score well if retrieval quality is strong. The immediate risk is not lack of another model; it is **silent training skips, mock validation, incomplete runtime packaging, and checkpoint/runtime mismatch**.

Fix those first. Then the user can spend Kaggle GPU quota on a run whose output is scientifically interpretable and actually corresponds to the trained pipeline.
