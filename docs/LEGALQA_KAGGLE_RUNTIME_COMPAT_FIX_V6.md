# LegalQA Kaggle V6 — Runtime Compatibility & Promotion Correctness

**Repository:** https://github.com/silent9669/LegalQA  
**Audited HEAD:** `92ed0f5ecd4207a468da3f8afbaa9cf86d7aa130`  
**Previous spec:** `docs/LEGALQA_KAGGLE_LAST_MILE_FIX_V5.md`

## Verdict

**NOT YET SAFE FOR THE REAL KAGGLE SMOKE/SCREEN/FINAL SEQUENCE.**

V5 fixed most architecture and execution-structure problems. Do not redesign the model stack. V6 is only about runtime compatibility, strict environment/index behavior, and correct screening/promotion.

## What V5 already fixed

Keep the current `smoke_only` default, `UNVALIDATED -> PROMOTED` production gate, all-data final training, bounded smoke datasets/steps, optional generator, strict QLoRA adapter load, fail-loud generation, answer-preserving SFT packing, checkpoint manifests, batched reranking, deterministic held-out sampling, canonical METEOR helper, stronger index checks, and GitHub CI.

Latest CI is useful: 78 tests passed and 1 was skipped. However, it is a CPU test run and does not prove Kaggle QLoRA/T4 compatibility.

---

# P0-1 — Fix the TRL API before Kaggle

`requirements-kaggle.txt` currently has broad versions such as:

```text
trl>=0.8.0
peft>=0.10.0
sentence-transformers>=3.0.0
```

while `train_generator.py` imports:

```python
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
```

Current TRL removed `DataCollatorForCompletionOnlyLM`.

### Required implementation

Prefer the current prompt-completion API:

```python
from trl import SFTConfig, SFTTrainer
```

Build each example as:

```python
{
    "prompt": prompt_without_gold_answer,
    "completion": gold_answer_plus_eos,
}
```

and use:

```python
SFTConfig(
    ...,
    completion_only_loss=True,
)
```

Remove the obsolete collator import.

Add a CPU compatibility test that catches TRL API drift without downloading the 3B model.

---

# P0-2 — Do not let pip replace Kaggle Torch/CUDA

The notebook currently runs a general:

```python
pip install -r requirements-kaggle.txt
```

Broad dependencies can resolve to a new Torch/Transformers/CUDA stack.

Create:

```text
scripts/bootstrap_kaggle_env.py
```

Rules:

1. Print preinstalled versions first.
2. Never upgrade/downgrade `torch`, CUDA runtime wheels, cuDNN, NCCL, Triton, or torchvision automatically.
3. Install only missing compatible user-space packages.
4. Run `pip check`.
5. Import:
   `transformers`, `trl`, `peft`, `bitsandbytes`, `sentence_transformers`.
6. Verify CUDA remains available.
7. After a successful real Kaggle smoke, freeze the tested compatibility tuple.

Notebook must call the bootstrap instead of blindly installing all lower-bounded packages.

---

# P0-3 — Hard fail when CUDA is absent

Current notebook only raises inside `if torch.cuda.is_available()`, so zero-CUDA can fall through to CPU.

For canonical Kaggle profiles:

```python
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required.")

if torch.cuda.device_count() < 2 and not ALLOW_SINGLE_GPU_SMOKE:
    raise RuntimeError("Dual-T4 execution requires >=2 CUDA GPUs.")
```

Preflight must receive:

```python
require_cuda=True
expected_gpu_count=2
```

Do not derive `require_cuda` from the environment.

---

# P0-4 — Missing BM25/DEk21 directories must fail preflight

Current preflight validates an index only when its directory already exists.

Add:

```python
if not os.path.isdir(bm25_dir):
    errors.append(...)
if not os.path.isdir(dek21_dir):
    errors.append(...)
```

In strict Kaggle mode require:

```text
BM25/bm25_manifest.json
BM25/bm25s_index/params.index.json
DEk21/dense_manifest.json or dek21_manifest.json
DEk21/embeddings.npy
```

Missing BM25 manifest/index is an error, not a warning.

---

# P0-5 — Stream dense SHA256

Do not do:

```python
hashlib.sha256(f.read()).hexdigest()
```

on a ~GB file.

Use chunked hashing:

```python
def sha256_file(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()
```

Use the same helper for package/preflight integrity checks.

---

# P0-6 — Fail on ambiguous runtime/model paths

`path_resolver.py` still selects the first runtime root/model candidate when several exist.

Required:

```python
if len(valid_runtime_roots) != 1:
    raise RuntimeError(...)
```

For Qwen:

- inspect candidate `config.json`;
- require the intended Qwen2/Qwen2.5 ~3B causal LM;
- if multiple valid candidates remain, raise.

Never return `roots[0]` or `candidate_dirs[0]`.

Remove the notebook's first-match recursive fallback for `public-official.json`; it must come from the validated runtime dataset or an explicit configured path.

---

# P0-7 — Fix retrieval metrics in `evaluate_checkpoint`

Current evaluator receives:

```python
selected, candidates, primary_evidence
```

from `predict_single`, where the third object is a **string**.

It then tries to compute retrieval metrics as if that value contained reranked dictionaries. It does not.

Also, gold positives are in `retrieval_labels.parquet`, not reliably in `qa_unique.parquet`.

### Required trace API

Add either:

```python
pipeline.retrieve_and_rerank(question)
```

or:

```python
predict_single(..., return_trace=True)
```

with:

```python
{
  "bm25_results": ...,
  "dense_results": ...,
  "fused_results": ...,
  "reranked_results": ...,
  "primary_evidence": ...,
  "retrieval_meta": ...
}
```

Evaluator loads `retrieval_labels.parquet`, groups gold:

```text
positive_chunk_id
positive_article_id
```

by `qa_id`, then computes real:

```text
Chunk Recall@1/5/8
Article Recall@1/5/8
MRR
num_queries_with_retrieval_labels
```

If screen supervision is unavailable, fail instead of silently reporting zeros.

---

# P0-8 — Reranker promotion must require real retrieval improvement

Do not convert missing MRR to `0.0` and allow equality to pass.

Use:

```python
if s0_mrr is None or s1_mrr is None:
    raise RuntimeError("Cannot promote reranker without retrieval metrics.")
```

Then:

```python
retrieval_improved = (
    s1_mrr > s0_mrr + tol
    or s1_recall8 > s0_recall8 + tol
)

downstream_ok = (
    s1_fixed_meteor >= s0_fixed_meteor - meteor_tol
)

promote_reranker = retrieval_improved and downstream_ok
```

Persist exact tolerances.

---

# P0-9 — QLoRA promotion must compare every generator-derived family

Do not compare only `generated`.

Use:

```python
GENERATOR_DEPENDENT_FAMILIES = {
    "generated",
    "snapped",
    "strategy_f_300",
    "strategy_f_600",
    "strategy_f_1000",
    "strategy_f_1500",
}
```

Compare S1 vs S2 for all of them.

Report:

```text
best non-generator candidate
best base-generator candidate
best QLoRA-derived candidate
overall deployable winner
```

Promote QLoRA only if its best deployable policy actually improves the best no-QLoRA policy.

---

# P0-10 — Fix promoted policy encoding

Do **not** write:

```yaml
candidate_policy:
  type: generated
```

`CandidateSelector` does not interpret that as "always choose generated".

If generated wins, write:

```yaml
candidate_policy:
  type: fixed_baseline
  best_fixed_candidate: generated
```

If Strategy F wins:

```yaml
candidate_policy:
  type: fixed_baseline
  best_fixed_candidate: strategy_f_1000
```

Alternatively add one tested `direct_candidate` policy.

Candidate names must not be overloaded as policy types.

---

# P0-11 — `learned_model` must require a generator

Update:

```python
policy_requires_generator()
```

so these are generator-requiring:

```text
learned
learned_model
meta_selector
```

Add regression test.

---

# P0-12 — Validate freshly trained final checkpoints too

Reuse profile validates manifests, but fresh final training should also call:

```python
assert_final_checkpoint(...)
```

after training and before inference.

Do this for:

- tuned reranker when promoted;
- QLoRA adapter when promoted.

Block public inference on failure.

---

# P0-13 — Fix checkpoint manifest fold key

Validator checks `val_fold`, trainers write `val_fold_excluded`.

Use:

```python
excluded_fold = manifest.get(
    "val_fold_excluded",
    manifest.get("val_fold"),
)
```

Final checkpoint must have no excluded fold.

---

# P1-1 — Make QLoRA smoke preprocessing genuinely small

Even with 128 QA examples, the builder currently loads the whole legal corpus into a Python chunk map.

For smoke:

1. sample QA first;
2. filter labels to selected QA IDs;
3. collect only needed positive chunk IDs;
4. read only `chunk_id,text_raw`;
5. keep only needed rows.

At minimum use Parquet column projection.

---

# P1-2 — Make sequence-length diagnostics actionable

Before generator screen, compare 2048 and 3072 and report:

```text
kept count
dropped count / %
evidence truncated %
P50/P90/P95/P99/max tokens
```

Use 2048 unless 3072 provides meaningful preservation gains and smoke VRAM is safe.

---

# P1-3 — Record T4 peak VRAM

For both trainers:

```python
torch.cuda.reset_peak_memory_stats(device)
...
peak = torch.cuda.max_memory_allocated(device)
```

Store peak VRAM in manifests.

If reranker full fine-tuning is near T4 limits, use an explicitly tested FP16/gradient-checkpointing configuration.

---

# P1-4 — Harden full OOF provenance

For `fold_checkpoint_map`, validate that each fold-specific checkpoint manifest has:

```text
smoke_only=false
val_fold_excluded == current_fold
base model matches
```

Full mode must also use:

```python
BM25Retriever.load(..., fail_on_missing_index=True)
DenseRetriever.load_index(
    expected_model_name=DEK21_MODEL,
    expected_dtype="float16",
    final_mode=True,
)
```

No rebuild or fallback.

---

# P1-5 — Audit actual loaded parameters

Keep the conservative Stack-A budget audit, but also report an exact **actual loaded inference** count.

If generator is omitted by the promoted extractive policy, actual loaded count should not include Qwen/adapter.

Report both:

```text
approved maximum stack count
actual deployed learned parameter count
```

Both must remain <4B.

---

# P1-6 — Add deterministic promotion script

Create:

```text
scripts/promote_production_selection.py
```

Input:

```text
promotion_report.json
```

Output:

```yaml
status: PROMOTED
source_screen_manifest: ...
source_screen_sha256: ...
reranker:
  use_task_tuned: <measured>
generator:
  use_qlora: <measured>
candidate_policy:
  type: fixed_baseline
  best_fixed_candidate: <measured winner>
```

It must compute the report SHA and must never invent metrics.

---

# CI requirements

Current CI is green, but it installs the CPU `requirements.txt`, not the Kaggle QLoRA stack.

Add a lightweight compatibility job that:

- does not replace Torch/CUDA;
- installs the selected TRL/PEFT/Transformers user-space compatibility set;
- imports all training modules;
- verifies prompt-completion SFT plumbing;
- ensures `DataCollatorForCompletionOnlyLM` is no longer required.

GPU correctness remains a Kaggle smoke gate.

---

# Required regression tests

Add tests for:

- zero CUDA fails canonical profiles;
- missing BM25 directory fails;
- missing dense directory fails;
- missing BM25 manifest/index fails;
- streaming SHA parity;
- multiple runtime roots raise;
- multiple valid Qwen roots raise;
- retrieval metrics use actual reranked results;
- retrieval metrics use `retrieval_labels`;
- missing retrieval metrics cannot promote reranker;
- reranker requires actual retrieval improvement;
- QLoRA promotion covers all generator-dependent families;
- generated winner maps to `fixed_baseline + generated`;
- `learned_model` requires generator;
- new final reranker manifest is validated;
- new final QLoRA manifest is validated;
- validator reads `val_fold_excluded`;
- no obsolete TRL collator import remains;
- prompt/completion completion-only SFT contract;
- smoke evidence loading is bounded.

---

# Required execution after V6

## 1. `smoke_only`

Must prove:

```text
Kaggle Torch/CUDA preserved
T4 x2
strict BM25
strict DEk21
30-step reranker
reranker reload PASS
30-step QLoRA
PEFT reload PASS
5 real held-out predictions
no mock/fallback
GPU0/GPU1 peak VRAM
```

## 2. `screen_fold0`

Must produce on identical deterministic IDs:

```text
S0 base
S1 tuned reranker
S2 QLoRA
real retrieval Recall/MRR
all candidate-family METEOR
promotion_report.json
```

## 3. Promotion

Generate a `PROMOTED` config only from the measured report.

## 4. `final_train_and_submit`

Train only promoted components with:

```text
val_fold=None
```

Then validate final manifests, strict-reload production stack, audit parameters, infer 1000 public IDs, and create valid `submission.json.zip`.

---

# Definition of SAFE FOR KAGGLE SMOKE

Only output:

```text
SAFE FOR KAGGLE SMOKE
```

when:

- current TRL-compatible SFT path works;
- bootstrap cannot replace Kaggle Torch/CUDA;
- no-CUDA fails immediately;
- missing indexes fail strict preflight;
- path ambiguity fails;
- real retrieval metrics are implemented;
- promotion cannot pass on missing metrics;
- QLoRA winner policy is encoded correctly;
- fresh final manifests are validated;
- V6 CPU/compatibility CI passes.

---

# Definition of SAFE TO RUN FINAL KAGGLE TRAINING

Only output this **after actual Kaggle evidence**:

```text
SAFE TO RUN FINAL KAGGLE TRAINING
```

Required real evidence:

```text
clean Restart Session -> Run All smoke passes
recorded package tuple
T4 x2
safe VRAM
no fallback
reranker reload PASS
QLoRA PEFT reload PASS if applicable
screen_fold0 completed
valid retrieval metrics
valid candidate METEOR table
PROMOTED config frozen from measured report
```

GitHub CPU tests alone are not enough for this claim.

---

# Final instruction to coding agent

Do not add another model or redesign the architecture.

The current bottleneck is:

```text
dependency compatibility
strict environment enforcement
real retrieval metrics
correct promotion policy encoding
```

Fix those, test them, and prepare the repo for the first trustworthy dual-T4 smoke.
