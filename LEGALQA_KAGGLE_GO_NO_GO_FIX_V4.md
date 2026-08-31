# LegalQA Kaggle V4 — Final Go/No-Go Fix Before Expensive Training

> Repository: `https://github.com/silent9669/LegalQA`
>
> Audited HEAD: `78063bd085589d7ea86b335d40e8c794114abf88`
>
> Previous reference: `LEGALQA_KAGGLE_FINAL_BLOCKERS_FIX_V3.md`
>
> **Current verdict: NOT SAFE FOR THE FINAL EXPENSIVE KAGGLE `train_and_submit` RUN YET.**
>
> V3 fixed a large portion of the execution wiring, but the remaining defects are now concentrated in **training semantics, checkpoint evaluation, production model selection, smoke-run behavior, and reproducibility**. Do not redesign the whole architecture. Fix the items below, prove them with tests/smoke evidence, then stop changing architecture and run Kaggle.

---

# 1. What V3 successfully fixed

Do not undo these improvements:

- canonical notebook now defaults to `EXECUTION_PROFILE = "train_and_submit"`;
- packaged runtime now stages `src/`, `scripts/`, `configs/`, and `requirements-kaggle.txt`;
- `reranker_training_pairs.parquet` can be staged;
- requested training can be configured to fail loudly;
- dense embeddings are loaded with mmap instead of unconditional FP32 expansion;
- dense final mode can fail on GPU allocation/search errors;
- Qwen inference now uses the tokenizer instance for native chat-template formatting;
- generator training records adapter trainable parameter count;
- reranker training now contains a validation loop;
- exact-memory question/ID consistency is safer;
- `predict_batch()` now batches dense retrieval and Qwen generation;
- candidate selector receives `rerank_score` rather than only stale fused score;
- the production stack is now Stack A only.

These are good changes. V4 is not an architecture reset.

---

# 2. Executive summary of remaining blockers

The current commit still has the following critical problems.

## P0-1 — Final submission models are trained on only ~80% of allowed QA

The notebook calls both trainers with:

```python
val_fold=0
```

for the reranker and QLoRA generator.

That means fold 0 is excluded from training, and the notebook then uses those fold-0-excluded checkpoints for public inference.

This is appropriate for a **validation screen**, but not for the **final competition model**.

The final submission should use all allowed TRAIN + WARMUP records after hyperparameters and architecture are frozen.

---

## P0-2 — The notebook's post-training evaluation is still fake for checkpoint quality

The notebook currently calls:

```python
run_oof_validation(
    ...
    mode="fast",
    adapter_path=ADAPTER_PATH,
)
```

But `run_oof_validation.py` explicitly uses in `fast` mode:

```text
mock dense retriever
mock/lexical reranker
fallback extractive generator
```

So the printed METEOR is **not the METEOR of the reranker/QLoRA that were just trained**.

This must be removed before calling the notebook scientifically valid.

---

## P0-3 — `run_oof_validation.py` still cannot evaluate the trained reranker

Even in `mode="full"`, it currently hard-codes:

```python
BGEReranker(model_name="BAAI/bge-reranker-v2-m3")
```

There is no explicit `reranker_checkpoint` argument.

Therefore the OOF script cannot evaluate:

```text
/kaggle/working/checkpoints/reranker/best
```

produced by the notebook.

---

## P0-4 — The current OOF script is not true fold-specific neural OOF

`run_oof_validation.py` loads one generator/adapter and one reranker once, then evaluates all five folds.

A single adapter trained with `val_fold=0` has already seen folds 1–4. Evaluating it on folds 1–4 is not OOF.

Correct neural 5-fold OOF requires one fold-specific checkpoint per fold:

```text
fold 0 checkpoint trained on folds 1–4
fold 1 checkpoint trained on folds 0,2,3,4
...
```

or the evaluation must be honestly labeled as a **single held-out fold screen**, not 5-fold OOF.

---

## P0-5 — QLoRA training is mostly wasted by the current final selector

The notebook constructs:

```python
selector = CandidateSelector(
    policy="fixed_baseline",
    best_fixed_candidate="stitched_extract",
)
```

Therefore, for ordinary unseen queries, the final answer is normally `stitched_extract`.

The trained QLoRA output is generated as a candidate but is generally discarded.

Do not spend hours training QLoRA unless:

1. corrected validation shows a generator/QLoRA candidate improves the deployable policy; or
2. a validated selector/gate chooses QLoRA for specific query types.

---

## P0-6 — `smoke_only` is not actually a smoke run

Current profile:

```python
EXECUTION_PROFILE = "smoke_only"
```

still enables:

```text
RUN_RERANKER_TRAINING=True
RUN_GENERATOR_TRAINING=True
```

but no max-step/subset limits are passed to either trainer.

So `smoke_only` can still execute the full epoch over the full training dataset.

A smoke profile must run only ~20–50 optimizer steps or a tiny deterministic subset.

---

## P0-7 — QLoRA reload smoke can still falsely pass through generator fallback

`run_qlora_training()` reloads using:

```python
QwenGenerator.load(
    model_path=model_name,
    adapter_path=output_dir,
    device=dev,
    runtime="torch",
)
```

but does **not** pass:

```python
fail_on_fallback=True
```

If neural load fails, `QwenGenerator.load()` can return the fallback generator. The smoke call then produces a non-empty extractive string and the adapter can be incorrectly considered valid.

The reload smoke must assert:

```python
reload_gen.runtime == "torch"
reload_gen.model is not None
reload_gen.tokenizer is not None
adapter actually loaded
```

and use `fail_on_fallback=True`.

---

## P0-8 — Single-query generation can silently fall back after runtime errors

`QwenGenerator.generate()` catches PyTorch generation exceptions and then returns the extractive fallback.

That violates the final-mode fail-loud policy.

Even if batched public inference currently uses `generate_batch()`, this behavior contaminates:

- smoke tests;
- `predict_single()`;
- evaluation code;
- debugging;
- any future batch fallback.

Add an instance-level `fail_on_generation_error` / `final_mode` policy.

Final mode:

```python
except Exception as e:
    raise RuntimeError("FINAL_PIPELINE_ERROR: generation failed ...") from e
```

Diagnostic mode may use the extractive fallback.

---

## P0-9 — Preflight still does not verify the indexes it receives

`run_preflight_checks()` now accepts:

```text
bm25_dir
dek21_dir
```

but it does not actually validate:

```text
BM25 index files/manifest
DEk21 embeddings
DEk21 manifest
dense row count
dense model ID
dense dtype
chunk-ID hash
```

This means the notebook can print `Production preflight completely PASSED!` before the core retrieval indexes have been verified.

---

## P0-10 — Preflight does not actually require dual T4

Notebook currently calls:

```python
require_cuda=torch.cuda.is_available()
expected_gpu_count=2 if torch.cuda.device_count() >= 2 else 1
```

If Kaggle accidentally starts with one GPU, the notebook lowers the expectation to one and passes.

For production:

```python
require_cuda=True
expected_gpu_count=2
```

must be unconditional for `train_and_submit`.

`smoke_only` may optionally allow one GPU only if explicitly configured.

---

## P0-11 — Dense model identity is not enforced

`DenseRetriever.load_index()` reads the model ID from the index manifest and overwrites `model_name`.

It does not verify:

```text
expected production dense model == index dense model
```

A BGE-M3 or stale index could therefore cause runtime model drift and potentially invalidate the parameter budget.

Add:

```python
expected_model_name: Optional[str]
```

and in final mode:

```python
if manifest_model != expected_model_name:
    raise ValueError(...)
```

Also verify embedding dtype and embedding file checksum.

---

## P0-12 — `embeddings_sha256` is saved but never checked

The dense manifest stores:

```json
"embeddings_sha256": "..."
```

but `load_index()` never compares the actual file hash to it.

Either:

- verify the SHA256 during production preflight; or
- use a faster robust file fingerprint if full 1.15GB hashing is too slow and document the tradeoff.

At minimum, final preflight should verify:

```text
file size
.npy shape
dtype
chunk-id hash
model ID
```

and optionally full SHA once per packaged dataset build.

---

## P0-13 — Dependency installation is still unbounded and not reproducible

`requirements-kaggle.txt` says "pinned", but contains broad lower bounds such as:

```text
trl>=0.8.0
peft>=0.10.0
sentence-transformers>=3.0.0
bitsandbytes>=0.43.0
```

The notebook executes:

```bash
pip install -r requirements-kaggle.txt
```

on every run.

A future/latest TRL can remove or change:

```text
DataCollatorForCompletionOnlyLM
SFTTrainer
SFTConfig
processing_class
dataset_text_field
max_length / max_seq_length
```

and can also pull a different Transformers stack.

Do not guess versions blindly.

Perform one real Kaggle smoke, record the exact working package tuple, then freeze or constrain those versions.

Do **not** upgrade/downgrade `torch` or CUDA packages unless absolutely required.

---

# 3. Correct architecture decision: separate validation training from final training

The current notebook mixes two different jobs:

```text
A. learn whether a tuned model is good
B. train the final model used for submission
```

These must be separated.

## Execution profiles

Use four explicit profiles:

```python
EXECUTION_PROFILE = "smoke_only"
EXECUTION_PROFILE = "screen_fold0"
EXECUTION_PROFILE = "final_train_and_submit"
EXECUTION_PROFILE = "reuse_final_checkpoints_and_submit"
```

### `smoke_only`

Purpose: prove code/hardware/checkpoint save-reload.

Requirements:

```text
20–50 optimizer steps
tiny deterministic QA subset
tiny deterministic reranker-pair subset
no public inference
no quality claims
```

### `screen_fold0`

Purpose: decide whether tuned reranker/QLoRA improve held-out performance.

Training:

```text
train on folds 1–4
validate on fold 0
```

Outputs:

```text
screen reranker checkpoint
screen QLoRA adapter
real fold-0 candidate table
real fold-0 METEOR
```

No public submission should be generated from this profile.

### `final_train_and_submit`

Purpose: final competition run.

Rules:

```text
hyperparameters already frozen
train reranker on ALL allowed reranker training records
train QLoRA on ALL allowed QA if QLoRA was promoted
train/load final selector if selector was promoted
public inference
```

Use:

```python
val_fold=None
```

for final component training.

### `reuse_final_checkpoints_and_submit`

Requires checkpoint manifests explicitly saying:

```json
{
  "training_scope": "all_allowed_task2_data",
  "final_checkpoint": true
}
```

Do not reuse a fold-specific screen checkpoint as a final checkpoint by accident.

---

# 4. Add checkpoint provenance and prevent screen/final confusion

Every trained checkpoint must have a manifest.

## Reranker manifest

```json
{
  "base_model": "BAAI/bge-reranker-v2-m3",
  "training_scope": "folds_1_2_3_4",
  "excluded_fold": 0,
  "is_final_checkpoint": false,
  "num_unique_qa": 0,
  "num_pair_rows": 0,
  "git_sha": "...",
  "data_manifest_sha256": "...",
  "hyperparameters": {}
}
```

Final:

```json
{
  "training_scope": "all_allowed_task2_data",
  "excluded_fold": null,
  "is_final_checkpoint": true
}
```

## QLoRA manifest

Same semantics:

```text
screen adapter != final adapter
```

The notebook should refuse to submit with a manifest that says:

```json
"is_final_checkpoint": false
```

unless an explicit development override is enabled.

---

# 5. Fix real checkpoint evaluation

Create an evaluation function for **one explicitly held-out fold with explicit checkpoints**.

Preferred new module:

```text
src/task2/evaluation.py
```

or a clearly named script:

```text
scripts/evaluate_checkpoint.py
```

Interface:

```python
evaluate_checkpoint(
    qa_path=...,
    fold_path=...,
    held_out_fold=0,
    bm25_dir=...,
    dense_dir=...,
    dense_model="CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2",
    reranker_checkpoint=...,
    generator_model=...,
    adapter_path=...,
    selector_checkpoint=None,
    candidate_policy="fixed_or_loaded",
    final_mode=True,
    sample_size=...,
)
```

It must use exactly:

```text
BM25Retriever.load(... fail_on_missing_index=True)
DenseRetriever.load_index(... final_mode=True, expected_model_name=...)
BGEReranker(model_name=reranker_checkpoint)
EvidencePacker
QwenGenerator.load(... fail_on_fallback=True)
generate_candidate_ensemble
CandidateSelector
```

No mock components.

Output provenance:

```json
{
  "evaluation_type": "held_out_fold_screen",
  "held_out_fold": 0,
  "git_sha": "...",
  "bm25_index": "...",
  "dense_model": "...",
  "dense_manifest_sha": "...",
  "reranker_checkpoint": "...",
  "generator_model": "...",
  "adapter_checkpoint": "...",
  "no_mocks": true,
  "no_fallbacks": true
}
```

---

# 6. Repair `run_oof_validation.py`

The script may keep:

```text
fast diagnostic mode
```

but it must stop pretending a single neural checkpoint can produce true 5-fold OOF.

Implement two paths.

## `--mode fast`

Allowed:

```text
mock dense
mock reranker
fallback generator
```

Output banner must say:

```text
DIAGNOSTIC ONLY — NOT VALID FOR MODEL QUALITY OR PROMOTION
```

## `--mode full`

Require either:

### Option A — fold checkpoint map

```yaml
fold_checkpoints:
  0:
    reranker: ...
    adapter: ...
  1:
    reranker: ...
    adapter: ...
  ...
```

or:

### Option B — evaluate only one held-out fold

```bash
--held_out_fold 0
```

and call the result:

```text
held-out screen
```

not “5-fold OOF”.

For true full OOF, each fold must load a checkpoint trained without that fold.

---

# 7. Use the same candidate pipeline in evaluation and inference

The validation script should not manually maintain its own candidate policy.

Move canonical candidate assembly into one importable function/method.

Example:

```python
pipeline.generate_candidates(...)
pipeline.select_candidate(...)
```

Both:

```text
evaluate_checkpoint()
predict_single()
predict_batch()
```

must call the same candidate code.

This removes validation/notebook drift.

---

# 8. Fix the QLoRA reload smoke test

Change:

```python
reload_gen = QwenGenerator.load(
    model_path=model_name,
    adapter_path=output_dir,
    device=dev,
    runtime="torch",
)
```

to:

```python
reload_gen = QwenGenerator.load(
    model_path=model_name,
    adapter_path=output_dir,
    device=dev,
    runtime="torch",
    fail_on_fallback=True,
)

assert reload_gen.runtime == "torch"
assert reload_gen.model is not None
assert reload_gen.tokenizer is not None
```

Also verify PEFT is active.

For example, when PEFT is available:

```python
from peft import PeftModel
assert isinstance(reload_gen.model, PeftModel)
```

or use a robust equivalent check compatible with the selected PEFT version.

Then generate.

A non-empty fallback answer is **not** a successful adapter reload.

---

# 9. Make generator failure policy explicit

Add to `QwenGenerator`:

```python
final_mode: bool = False
```

or:

```python
fail_on_generation_error: bool = False
```

Production load:

```python
QwenGenerator.load(
    ...,
    fail_on_fallback=True,
    final_mode=True,
)
```

`generate()`:

```python
try:
    ...
except Exception as e:
    if self.final_mode:
        raise RuntimeError(
            f"FINAL_PIPELINE_ERROR: Qwen generation failed: {e}"
        ) from e
    return fallback(...)
```

`generate_batch()` should follow the same policy.

Do not silently convert an OOM into an extractive answer in final mode.

---

# 10. Fix smoke-only training for real

Add trainer parameters:

```python
max_steps: Optional[int] = None
max_train_examples: Optional[int] = None
```

## Reranker smoke

```python
train_bge_reranker(
    ...,
    max_steps=30,
    max_train_pairs=256,
)
```

## QLoRA smoke

```python
run_qlora_training(
    ...,
    max_steps=30,
    max_train_examples=128,
)
```

Smoke manifest:

```json
{
  "smoke_only": true,
  "is_final_checkpoint": false
}
```

The notebook must never treat a smoke checkpoint as final.

---

# 11. Final training must use all allowed Task 2 data

Once screen results promote a component:

## Final reranker

```python
train_bge_reranker(
    val_fold=None,
    ...
)
```

## Final generator

```python
run_qlora_training(
    val_fold=None,
    ...
)
```

The final notebook should print:

```text
FINAL RERANKER TRAINING UNIQUE QA: N
FINAL QLORA TRAINING QA: N
EXCLUDED FOLD: None
```

Then validate checkpoint integrity/reload.

Do not call a final all-data model OOF after it has seen all labels.

---

# 12. Do not train QLoRA unless it can affect the final answer

This is critical for score and GPU quota.

Current policy:

```text
train QLoRA
generate QLoRA candidate
always select stitched_extract
```

is not a rational final pipeline.

Choose one of two approaches.

## Approach A — extractive-first production

If corrected held-out/full OOF still shows extracts dominate:

```text
train/tune reranker
use best evidence/extractive candidate
skip QLoRA final training
```

This saves hours and can improve reliability.

## Approach B — gated generator

If QLoRA is useful on a subset:

```text
best fixed extract
+
QLoRA candidate
+
cross-fitted selector/gate
```

The selector must prove:

```text
meta-OOF selected METEOR > best fixed candidate METEOR
```

before deployment.

Do not train a model that the final policy never chooses.

---

# 13. Re-evaluate the fixed candidate after EvidencePacker changes

Historical:

```text
stitched_extract ~0.305
```

came from an earlier pipeline.

Current `stitched_extract` is constructed using the new primary evidence pack.

Therefore the old value is not automatically valid.

At minimum, on a correct held-out fold, evaluate:

```text
focused_extract
stitched_extract
pack_focused
pack_full_article
pack_top2_relevance
generated_base
generated_qlora
strategy_f_300
strategy_f_600
strategy_f_1000
strategy_f_1500
```

Then freeze the best fixed candidate.

Do not hard-code `stitched_extract` until this is reproduced.

---

# 14. Improve reranker training objective before full final training

Current training turns each `(question, positive, negative)` pair row into two independent binary examples:

```text
(q, positive) -> 1
(q, negative) -> 0
```

and duplicates the same positive for every negative.

This works, but ranking is the real objective.

Benchmark a pairwise ranking loss on the screen fold.

Example:

```python
pos_score = model(q, positive)
neg_score = model(q, negative)

loss = -torch.nn.functional.logsigmoid(pos_score - neg_score).mean()
```

or:

```python
MarginRankingLoss
```

Advantages:

```text
directly optimizes positive > hard negative
less dependence on arbitrary sigmoid threshold
better aligned with reranker ranking
```

Do not switch automatically. Compare held-out:

```text
base reranker
binary BCE fine-tune
pairwise fine-tune
```

using:

```text
positive Recall@1/5/8
MRR
downstream best-extractive METEOR
```

Promote only the winner.

---

# 15. Reduce reranker training volume intelligently

`reranker_training_pairs.parquet` can become very large because up to 10 negatives are mined per query and positive text is duplicated.

For Kaggle T4 runtime, prefer a small set of high-information negatives.

For each QA, keep roughly:

```text
1–2 same-document wrong-Article negatives
1–2 top fused false positives
optional 1 cross-document lexical hard negative
```

Target:

```text
2–5 hard negatives per positive
```

before trying 10.

Measure whether additional negatives improve fold-0 ranking enough to justify runtime.

---

# 16. Reranker checkpoint selection is still not truly “best checkpoint”

Current trainer tracks:

```text
best_val_loss
best_accuracy
```

but saves the model only after the entire training loop.

With `epochs=1`, this is harmless.

If epochs become >1, the last epoch is still saved even if an earlier one was better.

Fix now:

```python
if metric improved:
    save checkpoint to best_dir
```

At the end, reload `best_dir`.

Validation “accuracy” should preferably be pairwise ranking accuracy:

```text
score(q, positive) > score(q, negative)
```

not simply sigmoid score `>=0.5`.

---

# 17. QLoRA sequence packing still needs a hard answer-preservation guarantee

Current token-aware builder computes:

```python
"answer_truncated": total_tokens > max_seq_len
```

but still returns the over-length example.

The trainer may then truncate the answer tail.

Fix:

```python
if total_tokens > max_seq_len:
    repack evidence more aggressively
```

If framing + answer alone exceeds the sequence limit:

```python
return diagnostic/fail-or-drop decision
```

Never silently rely on trainer truncation.

## Dataset diagnostic gate

Before full training report at:

```text
2048
3072
4096
```

for the selected tokenizer:

```text
p50 total tokens
p90
p95
p99
max
% evidence truncated
% answers that cannot fit
estimated train VRAM
```

Select the smallest sequence length where answer-loss is acceptably near zero and T4 memory is stable.

---

# 18. Use inference-style evidence during QLoRA training

Current SFT evidence is built from concatenated positive chunk text.

Inference uses:

```text
retriever
reranker
EvidencePacker
```

This remains a train/inference distribution mismatch.

Use a deterministic mixture:

```text
~50–70% oracle positive evidence pack
~30–50% retrieval-produced noisy evidence pack
```

Exact percentages must be screened, not assumed.

The same evidence formatting function should be shared between:

```text
SFT dataset construction
held-out evaluation
public inference
```

At minimum, include document/article/clause structure consistently.

---

# 19. Dense index integrity must be truly strict

Update `DenseRetriever.load_index()`:

```python
def load_index(
    ...,
    expected_model_name: Optional[str] = None,
    expected_dtype: Optional[str] = None,
    verify_embeddings_hash: bool = False,
):
```

Final behavior:

```python
manifest_model = ...
if expected_model_name and manifest_model != expected_model_name:
    raise ValueError(...)

if expected_dtype and str(arr.dtype) != expected_dtype:
    raise ValueError(...)

if expected_dim != arr.shape[1]:
    raise ValueError(...)

if expected_rows != arr.shape[0]:
    raise ValueError(...)

if chunk_hash mismatch:
    raise ValueError(...)
```

For full hash:

```python
if verify_embeddings_hash:
    sha256(embeddings.npy) == manifest["embeddings_sha256"]
```

Production preflight can hash once before training.

---

# 20. Avoid duplicate 801k-corpus Python objects

Currently BM25 load and Dense load can each read:

```text
legal_chunks.parquet
-> DataFrame
-> list[dict]
```

This duplicates a large corpus representation in host RAM.

Refactor toward one shared corpus.

Example:

```python
corpus = load_canonical_corpus_once(
    columns=[
        "chunk_id",
        "doc_name",
        "legal_number",
        "article_number",
        "clause_number",
        "parent_article_id",
        "parent_clause_id",
        "text_raw",
        "text_norm",
        "start_char",
    ]
)

bm25.attach_corpus(corpus)
dense.attach_corpus(corpus)
packer = EvidencePacker(corpus)
```

Or make dense only need:

```text
row -> chunk_id/index
```

and reuse BM25/packer metadata.

Measure host RAM before/after.

---

# 21. `predict_batch()` is only partially batched

V3 now correctly batches:

```text
dense retrieval
Qwen generation
```

but still loops per query for:

```text
BM25
BGE reranker
evidence packing
```

BM25 per-query is acceptable if BM25S latency is small.

The BGE reranker should be flattened across queries.

Add:

```python
BGEReranker.rerank_batch(
    queries,
    candidate_lists,
    top_k=8,
    pair_batch_size=32,
)
```

Implementation:

```text
flatten (query, candidate) pairs
remember query/candidate offsets
CrossEncoder.predict on large batches
reconstruct each query ranking
```

Test exact equality versus per-query rerank on a deterministic fixture.

This can materially reduce public inference time.

---

# 22. Preflight must become a real production gate

For `final_train_and_submit`, require unconditionally:

```python
require_cuda=True
expected_gpu_count=2
```

Validate:

```text
runtime code root
scripts root
configs
training artifacts
public count == 1000
BM25 manifest + mmap files
BM25 corpus_size == canonical corpus rows
DEk21 manifest
DEk21 model == Stack A model
DEk21 dtype
DEk21 shape
DEk21 chunk hash
optional embeddings full hash
Qwen mounted model config identity
reranker base identity
dependency compatibility
parameter base budget
```

Important:

```text
public count != 1000
```

must be an **error**, not merely a warning.

---

# 23. Make artifact path resolution deterministic

The notebook still uses broad recursive globs and takes `[0]`.

Examples:

```python
qa_files[0]
bm25_dirs[0]
dek21_dirs[0]
qwen_configs[0]
```

If multiple datasets/models are mounted, the first match can be wrong.

Resolve the runtime dataset by expected Kaggle dataset name/root.

Then derive:

```text
QA_PATH = runtime_root / "qa_unique.parquet"
CHUNKS_PATH = runtime_root / "legal_chunks.parquet"
BM25_DIR = runtime_root / "indexes/bm25"
DEK21_DIR = runtime_root / "indexes/dek21"
CODE_ROOT = runtime_root / "code/LegalQA"
```

For Qwen, enumerate candidates and verify `config.json`:

```text
model_type
hidden size/layers if useful
expected Qwen2.5-3B identity
```

If more than one valid candidate remains, fail with a clear ambiguity error.

---

# 24. Freeze a working Kaggle dependency tuple

Do not leave:

```text
>=
```

forever.

First add a diagnostic cell:

```python
import importlib.metadata as md

for p in [
    "torch",
    "transformers",
    "trl",
    "peft",
    "bitsandbytes",
    "sentence-transformers",
    "bm25s",
    "datasets",
]:
    print(p, md.version(p))
```

After the clean Kaggle smoke succeeds, record those versions in:

```text
artifacts/run_manifest.json
docs/kaggle_environment.md
```

Then constrain `requirements-kaggle.txt` to that tested compatibility tuple.

Do not modify `torch` unless required.

---

# 25. Add a genuine promotion manifest

Architecture decisions should not be hidden inside notebook constants.

Create:

```text
configs/production_selection.yaml
```

Example:

```yaml
stack: stack_a

reranker:
  use_task_tuned: true
  objective: pairwise
  selected_hparams: ...

generator:
  use_qlora: false
  reason: "did not beat fixed extract on held-out screen"
  selected_hparams: ...

candidate_policy:
  type: fixed
  candidate: pack_top2_relevance

evidence:
  generator_pack: multi_seed_2500_chars
  extract_pack: top2_relevance
```

Or, if QLoRA/selector wins:

```yaml
generator:
  use_qlora: true

candidate_policy:
  type: learned_selector
  checkpoint: ...
```

The final notebook reads this frozen selection.

---

# 26. Highest-score architecture after V4

Do not assume Qwen must dominate.

Use:

```text
Question
  |
  +--> Safe Exact QA Memory
  |
  +--> Fuzzy QA Features/Candidate
  |
  +--> BM25S --------------------+
  |                              |
  +--> DEk21 FP16 Exact ----------+--> RRF
                                      |
                                      v
                              Task-tuned BGE
                                      |
                                      v
                             EvidencePacker
                                      |
                 +--------------------+-------------------+
                 |                    |                   |
                 v                    v                   v
          focused/relevance     article extract      Qwen/QLoRA
              extracts                                   |
                 |                                       |
                 +-------------------+-------------------+
                                     |
                           validated fixed/gated policy
                                     |
                                     v
                                final answer
```

The generator is optional for final score.

---

# 27. Do not spend final GPU quota before these score questions are answered

Before `final_train_and_submit`, produce a real held-out table like:

| Component | Held-out Fold METEOR | Retrieval Recall@8 | Runtime | Promote? |
|---|---:|---:|---:|---|
| pretrained reranker + best extract | ... | ... | ... | |
| tuned reranker + best extract | ... | ... | ... | |
| tuned reranker + base Qwen candidate | ... | ... | ... | |
| tuned reranker + QLoRA candidate | ... | ... | ... | |
| best fixed candidate | ... | — | ... | |
| learned selector | ... | — | ... | |

Do not invent values.

If tuned QLoRA does not improve deployable METEOR, skip final QLoRA training.

If tuned reranker does not improve retrieval/downstream METEOR, use base reranker.

---

# 28. Required regression tests for V4

The V3 commit changed many critical modules but only materially extended packaging tests. Add tests for the actual blockers.

## Execution profiles

- [ ] `smoke_only` passes max-step/subset limits.
- [ ] `screen_fold0` excludes fold 0.
- [ ] `final_train_and_submit` uses `val_fold=None`.
- [ ] screen checkpoint cannot be used as final checkpoint.

## Generator

- [ ] failed neural reload cannot pass through fallback.
- [ ] final-mode `generate()` raises on generation exception.
- [ ] final-mode `generate_batch()` raises on generation exception.
- [ ] loaded QLoRA smoke model is PEFT-backed.
- [ ] over-length SFT example repacks evidence before returning.
- [ ] no returned training example has `answer_truncated=True` unless explicitly dropped/handled.
- [ ] final answer tokens are present at the end of tokenized SFT example.

## Reranker

- [ ] best epoch checkpoint is saved/reloaded.
- [ ] pairwise ranking accuracy is computed correctly.
- [ ] final training uses all folds.
- [ ] fold screen excludes held-out fold.
- [ ] `rerank_batch()` matches per-query rerank ordering.

## OOF / evaluation

- [ ] `fast` output is labeled diagnostic.
- [ ] trained-checkpoint evaluation cannot run in `fast`.
- [ ] explicit reranker checkpoint is accepted.
- [ ] `full` cannot use fallback generator.
- [ ] true 5-fold OOF requires a checkpoint per fold.
- [ ] single fold checkpoint is never reported as 5-fold OOF.
- [ ] candidate generation path is shared with inference.

## Dense

- [ ] expected model mismatch raises.
- [ ] expected dtype mismatch raises.
- [ ] embedding hash mismatch raises when verification is enabled.
- [ ] row and chunk-hash mismatch raise in final mode.

## Preflight

- [ ] one GPU fails production preflight.
- [ ] missing BM25 fails.
- [ ] missing DEk21 fails.
- [ ] wrong dense model fails.
- [ ] public count !=1000 fails.
- [ ] missing reranker pairs fails when reranker final training enabled.

## Submission

- [ ] final checkpoint manifests are `is_final_checkpoint=true`.
- [ ] submission is blocked when a promoted required component is only a screen/smoke checkpoint.

---

# 29. Real Kaggle run sequence after V4

Do not jump directly to full final training.

## Run 1 — clean smoke

Set:

```python
EXECUTION_PROFILE = "smoke_only"
```

Expected:

```text
strict dual-T4 preflight
~30-step reranker train
reranker save/reload
~30-step QLoRA train
QLoRA real PEFT save/reload
2–5 real pipeline predictions
peak VRAM logs
NO public submission
```

If this does not pass from:

```text
Restart Session -> Run All
```

stop.

## Run 2 — component screen

Set:

```python
EXECUTION_PROFILE = "screen_fold0"
```

Expected:

```text
fold0 excluded from both learned components
real dense
real tuned reranker
real base/QLoRA generator
real current candidates
real fold0 METEOR table
promotion manifest
NO public submission
```

## Run 3 — final

Set:

```python
EXECUTION_PROFILE = "final_train_and_submit"
```

Expected:

```text
read frozen promotion config
train only promoted learned components
use ALL allowed QA/pairs
reload final checkpoints
adapter-inclusive parameter audit
real batched inference
strict submission
```

This is the run that should consume the expensive final quota.

---

# 30. Final training runtime optimization

If GPU time becomes too large:

Priority:

1. keep full legal corpus indexes precomputed;
2. keep strongest extractive baseline;
3. task-tune reranker with fewer high-value negatives;
4. train QLoRA only if screen proves a gain;
5. do not train a learned selector unless its oracle gap justifies it.

This order is much more likely to maximize score per GPU-hour than blindly training every component.

---

# 31. README corrections

After V4, README must not claim:

```text
"held-out sanity evaluation"
```

if the notebook is still using fast mocks.

It must distinguish:

```text
smoke
screen
final training
true OOF
```

Parameter table must read the **actual adapter parameter count from the final manifest**, not a hard-coded `~20M` estimate.

Also do not claim "best checkpoint" unless the earlier best state is actually saved and reloaded.

---

# 32. Definition of SAFE TO RUN

The coding agent may write:

```text
SAFE TO RUN FINAL KAGGLE TRAINING
```

only when every item is true:

- [ ] smoke profile is genuinely bounded to ~20–50 steps;
- [ ] smoke QLoRA reload proves a real PEFT neural model, not fallback;
- [ ] production generator never silently falls back after runtime errors;
- [ ] screen profile evaluates exact trained checkpoints with no mocks;
- [ ] OOF/evaluation can load the task-tuned reranker checkpoint;
- [ ] final profile trains promoted components on all allowed Task 2 data;
- [ ] no fold-specific checkpoint can masquerade as a final checkpoint;
- [ ] fixed candidate is revalidated under the current EvidencePacker;
- [ ] QLoRA is trained for final only if it can improve/select final answers;
- [ ] preflight verifies dual T4 + BM25 + DEk21 + index identity/alignment;
- [ ] dense expected model/dtype/hash checks pass;
- [ ] Kaggle dependency tuple is smoke-tested and recorded;
- [ ] final adapter-inclusive parameter count is strictly `<4B`;
- [ ] final public inference uses only final checkpoint manifests;
- [ ] submission passes exact 1,000-ID validation;
- [ ] no CI/test failures remain.

If any item is false, report:

```text
NOT SAFE TO RUN
```

and list the blocker.

---

# 33. Required final coding-agent response

Return evidence, not claims.

## A. Git / tests

```text
HEAD SHA
changed files
pytest passed/failed/skipped
CI status if available
```

## B. Profiles

```text
smoke max steps
screen held-out fold
final training scope
```

## C. Reranker

```text
base/tuned
objective
train QA count
negative count
validation MRR/Recall
best checkpoint path
reload PASS
peak VRAM
```

## D. Generator

```text
base model
sequence length
LoRA config
screen train count
final train count
answer truncation %
adapter params
real PEFT reload PASS
peak VRAM
```

## E. Evaluation provenance

```text
no mocks: yes/no
no fallbacks: yes/no
dense model/index
reranker checkpoint
adapter checkpoint
held-out IDs/fold
candidate METEOR table
```

## F. Promotion decision

```text
best fixed candidate
base Qwen score
QLoRA score
selector score
promoted reranker yes/no
promoted QLoRA yes/no
promoted selector yes/no
```

## G. Final parameter audit

```text
DEk21
BGE reranker
Qwen base if loaded
QLoRA adapter if loaded
selector learned params if counted
TOTAL
<4B PASS
```

## H. Kaggle instructions

Give the exact final sequence for the user.

---

# 34. Final instruction

Do not add another model until this production loop is trustworthy.

The latest repository is close structurally. The remaining score risk is caused primarily by:

```text
validation that does not evaluate the trained models
fold-specific checkpoints being used as final checkpoints
training QLoRA that the final selector ignores
fake smoke behavior
dependency/index identity uncertainty
```

Fix those.

Then run:

```text
smoke -> screen -> final
```

and stop modifying architecture unless measured METEOR indicates a concrete bottleneck.
