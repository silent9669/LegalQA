# LegalQA Kaggle V5 — Last-Mile Production Fix & Final Go/No-Go Gate

> Repository: `https://github.com/silent9669/LegalQA`
>
> Audited HEAD: `476a18e402278a82aff3f101d26fce5d56aabd40`
>
> Previous spec: `LEGALQA_KAGGLE_GO_NO_GO_FIX_V4.md`
>
> **Verdict at this audited commit: NOT SAFE TO START THE EXPENSIVE FINAL KAGGLE RUN YET.**
>
> V4 was partially and usefully implemented. Do **not** redesign the architecture again. This V5 is a narrow last-mile production fix. The target is:
>
> ```text
> clean smoke
> -> real held-out screen
> -> frozen promotion decision
> -> final all-data training of ONLY promoted components
> -> strict dual-T4 inference
> -> submission
> ```
>
> After these fixes pass, stop changing architecture unless real METEOR measurements identify a bottleneck.

---

# 1. What is already good in commit `476a18e4`

Keep these changes:

- four execution profiles now exist conceptually;
- `final_train_and_submit` uses `TRAIN_VAL_FOLD=None`, so final component training can use all allowed Task 2 data;
- smoke `max_steps` support exists in both reranker and QLoRA trainers;
- checkpoint manifests now record training scope and `is_final_checkpoint`;
- `src/task2/evaluation.py` was added for real checkpoint evaluation;
- QLoRA reload now requests strict torch mode and final-mode generation;
- generator final mode now raises instead of silently returning extractive fallback on generation errors;
- dense loading supports expected model name, dtype, and optional embedding-hash verification;
- `BGEReranker.rerank_batch()` exists;
- `configs/production_selection.yaml` exists;
- public count mismatch is now a preflight error;
- final QLoRA adapter parameters are recorded and can be included in the parameter audit.

These are meaningful improvements.

---

# 2. Current P0 blockers

Fix every P0 before a real Kaggle final run.

## P0-1 — `evaluate_checkpoint()` currently crashes with `NameError`

File:

```text
src/task2/evaluation.py
```

The function uses:

```python
candidate_family_meteors: Dict[str, List[float]] = defaultdict(list)
```

but `defaultdict` is not imported.

Add:

```python
from collections import defaultdict
```

Add a unit test that executes `evaluate_checkpoint()` far enough to instantiate the candidate accumulator using mocked lightweight components.

This bug alone means the current `screen_fold0` / smoke evaluation path is not runnable as written.

---

## P0-2 — `evaluate_checkpoint()` can silently stop being held-out evaluation

Current behavior:

```python
val_records = df_qa[df_qa["fold_id"] == held_out_fold]

if val_records.empty:
    val_records = df_qa.head(50)
```

This is not leakage-safe semantics.

Replace with:

```python
if "fold_id" not in df_qa.columns:
    raise RuntimeError("Held-out evaluation requires fold_id assignments.")

if val_records.empty:
    raise RuntimeError(
        f"Held-out fold {held_out_fold} contains no rows; refusing to evaluate a different subset."
    )
```

Never silently substitute another validation set.

---

## P0-3 — held-out METEOR may fail because WordNet resources are not initialized

`src/task2/evaluation.py` calls NLTK `meteor_score` but does not ensure the scorer resources are present.

The legacy OOF script downloads/checks WordNet; the new evaluator does not.

Create one scorer helper, for example:

```text
src/task2/metrics.py
```

with:

```python
ensure_meteor_resources()
official_meteor(reference: str, prediction: str) -> float
```

Use the same helper in:

```text
src/task2/evaluation.py
scripts/run_oof_validation.py
tests
```

The scorer must match the official whitespace-tokenized protocol exactly.

Do not duplicate slightly different scorer implementations.

---

## P0-4 — `production_selection.yaml` is currently dead configuration

The file exists:

```text
configs/production_selection.yaml
```

but the notebook does not read it.

The notebook still hard-codes:

```python
RUN_RERANKER_TRAINING = True
RUN_GENERATOR_TRAINING = True

selector = CandidateSelector(
    policy="fixed_baseline",
    best_fixed_candidate="stitched_extract",
)
```

The current config says:

```yaml
reranker:
  use_task_tuned: true

generator:
  use_qlora: true

candidate_policy:
  type: fixed_baseline
  best_fixed_candidate: stitched_extract
```

This combination means:

```text
train QLoRA
generate QLoRA answers
then usually throw them away because stitched_extract is always selected
```

That wastes GPU time and is not a metric-driven production policy.

### Required fix

Make `configs/production_selection.yaml` authoritative.

Add an importable loader:

```text
src/task2/production_config.py
```

Example:

```python
@dataclass
class ProductionSelection:
    stack: str
    use_task_tuned_reranker: bool
    use_qlora: bool
    candidate_policy: str
    best_fixed_candidate: str | None
    selector_checkpoint: str | None
    primary_evidence_pack: str
    max_new_tokens: int
```

Notebook loads exactly one resolved config:

```python
PRODUCTION_CFG = load_production_selection(PRODUCTION_CONFIG_PATH)
```

Then derive:

```python
RUN_RERANKER_TRAINING = PRODUCTION_CFG.use_task_tuned_reranker
RUN_GENERATOR_TRAINING = PRODUCTION_CFG.use_qlora
```

for final training.

Do not maintain separate conflicting booleans and YAML decisions.

---

## P0-5 — the production config is marked “promoted” without any measured promotion artifact

Current config already says:

```yaml
use_task_tuned: true
use_qlora: true
best_fixed_candidate: stitched_extract
```

but no real held-out V4 screen result is committed or packaged to prove these choices.

Change config schema:

```yaml
schema_version: 3

status: "UNVALIDATED"
source_screen_manifest: null
source_screen_sha256: null

stack: stack_a
...
```

Allowed status:

```text
UNVALIDATED
PROMOTED
```

`final_train_and_submit` must refuse to start when:

```yaml
status: UNVALIDATED
```

unless the user explicitly sets a visible emergency override:

```python
ALLOW_UNVALIDATED_FINAL = False
```

default **False**.

After `screen_fold0`, write:

```text
/kaggle/working/promotion_report.json
```

containing:

```json
{
  "held_out_fold": 0,
  "sample_ids_sha256": "...",
  "reranker_base_metrics": {},
  "reranker_tuned_metrics": {},
  "candidate_family_meteors": {},
  "base_generator_meteor": 0.0,
  "qlora_generator_meteor": 0.0,
  "selected_policy_meteor": 0.0,
  "best_fixed_candidate": "...",
  "recommended_use_task_tuned_reranker": true,
  "recommended_use_qlora": false,
  "recommended_candidate_policy": "fixed_baseline"
}
```

The coding agent must **not invent scores**.

---

## P0-6 — the default notebook profile should not be the expensive final run before screening

Current committed notebook default:

```python
EXECUTION_PROFILE = "final_train_and_submit"
```

That allows an accidental multi-hour run before the smoke/screen gates have passed.

Until a validated `PROMOTED` production config exists, committed default should be:

```python
EXECUTION_PROFILE = "smoke_only"
```

Recommended operational sequence:

```text
1. smoke_only
2. screen_fold0
3. update/freeze production_selection.yaml from measured screen
4. final_train_and_submit
```

Once the repo includes a validated production selection artifact, changing the default to final is acceptable.

---

## P0-7 — `smoke_only` is bounded in optimizer steps but still prepares too much data

The notebook passes:

```python
max_steps=30
```

but does not pass:

```text
max_train_pairs
max_train_examples
```

even though the trainers support those arguments.

Therefore QLoRA smoke can still:

```text
load full 801k corpus
build full chunk map
build/tokenize thousands of SFT examples
then train only 30 steps
```

Reranker smoke can also validate the entire fold after 30 steps.

That is not a fast smoke.

### Notebook smoke settings

Add:

```python
SMOKE_MAX_STEPS = 30
SMOKE_MAX_RERANKER_PAIRS = 256
SMOKE_MAX_RERANKER_VAL_PAIRS = 128
SMOKE_MAX_QA_EXAMPLES = 128
SMOKE_EVAL_QUERIES = 5
```

Pass them explicitly.

Add `max_val_pairs` to reranker training.

Use deterministic sampling by `random_state=SEED`, not accidental `.head()` ordering.

Smoke goal is technical integrity, not quality.

---

## P0-8 — smoke evaluation intentionally allows fallback

Notebook currently calls:

```python
fail_on_fallback=False if EXECUTION_PROFILE == "smoke_only" else True
```

A smoke run is precisely where fallback must **not** hide a broken neural path.

On Kaggle T4 smoke:

```python
fail_on_fallback=True
```

for:

```text
dense
reranker
generator
adapter
```

A smoke that only works through fallback is a failed smoke.

---

## P0-9 — strict adapter-required semantics are still missing

Current `QwenGenerator.load()` only loads the adapter when:

```python
adapter_path and os.path.exists(adapter_path) and PeftModel is not None
```

If `adapter_path` was expected but:

```text
path missing
or PEFT unavailable
```

the code can still return a base Qwen model.

That is wrong for a production config that says:

```yaml
use_qlora: true
```

Add:

```python
require_adapter: bool = False
```

Behavior:

```python
if require_adapter:
    if not adapter_path:
        raise RuntimeError(...)
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(...)
    if PeftModel is None:
        raise RuntimeError(...)
```

After loading:

```python
if require_adapter and not is_peft_model(model):
    raise RuntimeError(...)
```

Notebook:

```python
require_adapter=PRODUCTION_CFG.use_qlora
```

for final/reuse profiles.

---

## P0-10 — `reuse_final_checkpoints_and_submit` is not strict

Current notebook behavior can:

```text
fail to find tuned reranker -> use pretrained base
fail to find adapter -> use base generator
load dense with final_mode=False
load generator with final_mode=False
then still submit
```

That defeats the meaning of “reuse final checkpoints”.

### Required behavior

When profile is:

```text
reuse_final_checkpoints_and_submit
```

load the frozen production config first.

If:

```yaml
reranker.use_task_tuned: true
```

then missing reranker final checkpoint is fatal.

If:

```yaml
generator.use_qlora: true
```

then missing final adapter is fatal.

Validate checkpoint manifests:

```json
"is_final_checkpoint": true,
"training_scope": "all_allowed_task2_data"
```

Reject:

```text
smoke checkpoint
screen checkpoint
fold-specific checkpoint
missing manifest
```

Use:

```python
final_mode=True
fail_on_fallback=True
```

for **both**:

```text
final_train_and_submit
reuse_final_checkpoints_and_submit
```

---

## P0-11 — production dual-T4 preflight is still weakened dynamically

Current notebook still does:

```python
require_cuda=torch.cuda.is_available(),
expected_gpu_count=2 if torch.cuda.device_count() >= 2 else 1,
```

So one GPU can pass.

For:

```text
final_train_and_submit
reuse_final_checkpoints_and_submit
screen_fold0
```

use:

```python
require_cuda=True
expected_gpu_count=2
```

For smoke, also prefer two GPUs because the purpose is to test the actual deployment layout.

If the project intentionally allows one-GPU smoke, make it explicit:

```python
ALLOW_SINGLE_GPU_SMOKE = False
```

default False.

---

## P0-12 — preflight still does not validate BM25 or dense indexes

Although `run_preflight_checks()` receives:

```text
bm25_dir
dek21_dir
```

it still only checks data files, public count, configs, parameters, and GPU availability.

Add strict index validation.

### BM25

Check:

```text
bm25_manifest.json exists
bm25s_index/params.index.json exists
manifest corpus_size == legal_chunks row count
configured k1/b recorded if available
```

### DEk21

Check:

```text
dense/dek21 manifest exists
embeddings.npy exists
model ID == production config dense model
dtype == float16
dimension == expected 768
row count == legal_chunks row count
chunk_ids hash matches corpus
embeddings shape is correct
embeddings hash matches when full hash verification enabled
```

For final run, execute at least one actual strict:

```python
DenseRetriever.load_index(
    ...,
    expected_model_name=...,
    expected_dtype="float16",
    verify_embeddings_hash=True,
    final_mode=True,
)
```

before expensive training starts, then release it if necessary.

Do not wait until public inference to discover a corrupt 1.15 GB index.

---

## P0-13 — final notebook does not use the new `expected_dtype` / hash verification

V4 implemented these options in `DenseRetriever.load_index()` but the notebook currently calls only:

```python
expected_model_name=...
final_mode=True
```

Wire:

```python
expected_dtype="float16"
```

and use full hash verification in preflight once.

Avoid recomputing the full hash repeatedly during inference loading.

---

## P0-14 — path resolution is still broad glob + `[0]`

Current notebook still finds:

```python
qa_files[0]
chunks_files[0]
bm25_dirs[0]
dek21_dirs[0]
qwen_configs[0]
```

This is not deterministic when multiple Kaggle inputs/models are mounted.

### Required resolver

Resolve exactly one runtime dataset root by requiring:

```text
dataset_manifest.json
code_manifest.json
qa_unique.parquet
legal_chunks.parquet
indexes/bm25/
indexes/dek21/
code/LegalQA/
```

Example:

```python
runtime_roots = find_runtime_roots("/kaggle/input")
if len(runtime_roots) != 1:
    raise RuntimeError(...)
RUNTIME_ROOT = runtime_roots[0]
```

Then derive all paths from that root.

For Qwen:

1. enumerate candidate `config.json`;
2. read each config;
3. validate expected Qwen2.5-3B architecture/model identity;
4. fail if ambiguous.

Never silently choose the first recursive match.

---

# 3. Validation semantics still need cleanup

## P0-15 — `run_oof_validation(... held_out_fold=...)` accepts the argument but ignores it

The signature now contains:

```python
held_out_fold: Optional[int] = None
```

but the implementation still loops:

```python
for fold_id in range(n_splits):
```

over every fold.

That is dangerous because the API implies held-out behavior that does not occur.

### Fix options

Preferred:

```python
if held_out_fold is not None:
    fold_ids = [held_out_fold]
else:
    fold_ids = list(range(n_splits))
```

But for neural `mode="full"`:

```python
if held_out_fold is None and no fold_checkpoint_map:
    raise RuntimeError(
        "True full neural OOF requires one checkpoint per fold."
    )
```

Do not evaluate one fold-0-trained adapter across all folds and label it OOF.

---

## P0-16 — full OOF still allows silent generator fallback

Current full mode calls:

```python
QwenGenerator.load(
    ...,
    runtime="torch",
)
```

without:

```text
fail_on_fallback=True
final_mode=True
```

Fix that.

For `mode="full"`:

```text
no mock
no fallback
no index rebuild
```

must be hard invariants.

---

## P0-17 — the new evaluator must be the canonical screen path

For component promotion, use:

```text
src.task2.evaluation.evaluate_checkpoint
```

not legacy `run_oof_validation --fast`.

Keep `run_oof_validation --fast` for cheap CPU diagnostics only.

Document clearly:

```text
fast OOF != checkpoint quality
held-out checkpoint evaluation = component screen
true neural OOF = fold-specific checkpoints
```

---

# 4. Screen quality must be statistically useful

The current notebook screen uses:

```python
sample_size=50
val_records.head(50)
```

This is too fragile for deciding whether to spend final GPU quota on QLoRA.

Use deterministic sampling:

```python
val_subset = val_records.sample(
    n=min(screen_sample_size, len(val_records)),
    random_state=SEED,
)
```

Recommended first screen size:

```text
200–300 held-out questions
```

If generator runtime is too costly:

```text
evaluate all held-out questions for extractive/retrieval candidates
evaluate 150–250 for base/QLoRA generator candidates
```

Record exact evaluated QA IDs and their SHA256.

Do not use `.head()` for promotion decisions.

---

# 5. Compare BASE vs TUNED components, not tuned-only

A screen should answer:

```text
Does training help?
```

Current `screen_fold0` trains tuned models and evaluates them, but does not automatically produce a comparable base-system score under the same held-out sample.

Add a screen matrix:

```text
S0: pretrained BGE + base Qwen
S1: tuned BGE + base Qwen
S2: tuned BGE + QLoRA
```

At minimum compare:

```text
best extractive candidate METEOR
selected fixed candidate METEOR
generated candidate METEOR
oracle candidate METEOR
```

Also retrieval:

```text
positive Article/Chunk Recall@1
Recall@5
Recall@8
MRR
```

Use the **same held-out IDs** for every configuration.

Promotion:

```python
promote_tuned_reranker = tuned_downstream_meteor > base_downstream_meteor + tolerance
promote_qlora = qlora_deployable_meteor > base_deployable_meteor + tolerance
```

Do not require a huge threshold, but do require a real positive signal.

---

# 6. QLoRA should only be final-trained if it can influence final answers

The final notebook must derive:

```python
requires_generator = ...
```

from the frozen production candidate policy.

Examples:

```text
fixed candidate = stitched_extract       -> generator not required
fixed candidate = pack_top2_relevance    -> generator not required
fixed candidate = generated              -> generator required
fixed candidate = strategy_f_1000        -> generator required
learned selector uses generator features -> generator required
```

Then:

```python
RUN_GENERATOR_TRAINING = (
    profile == "final_train_and_submit"
    and PRODUCTION_CFG.use_qlora
    and requires_generator
)
```

If generator is not required:

```text
do not train QLoRA
do not load Qwen
do not generate 1000 unused answers
do not count Qwen in actual loaded-inference parameter total
```

This can dramatically reduce Kaggle time if extractive answers remain best.

---

# 7. Support generator-optional inference

Current `LegalQAPipeline` assumes `generator` always exists and always generates.

Refactor:

```python
generator: Optional[QwenGenerator]
```

Candidate generation:

```python
gen_ans = ""
if self.generator is not None and policy_requires_generator:
    ...
```

`predict_batch()` should avoid building/generating Qwen pairs when the final policy cannot select a generator-based candidate.

This is both faster and cleaner.

Do not use fallback generator as a fake “optional generator”.

---

# 8. Actually use `rerank_batch()`

`BGEReranker.rerank_batch()` now exists, but `LegalQAPipeline.predict_batch()` still calls:

```python
self.reranker.rerank(...)
```

inside the per-query loop.

Refactor batch inference:

```text
exact-memory prepass
-> dense batch
-> BM25 + RRF per query
-> collect fused candidate lists
-> reranker.rerank_batch(all_queries, all_candidate_lists)
-> evidence pack
-> optional Qwen generation batch
-> candidate selection
```

Add equivalence test:

```python
per_query_results == batch_rerank_results
```

for deterministic/mock inputs.

---

# 9. Strict checkpoint manifest validation before submission

Add:

```text
src/task2/checkpoint_manifest.py
```

Functions:

```python
load_reranker_manifest(path)
load_generator_manifest(path)
assert_final_checkpoint(path, expected_base_model, component_name)
```

For final/reuse:

```python
assert man["is_final_checkpoint"] is True
assert man["training_scope"] == "all_allowed_task2_data"
assert man["smoke_only"] is False
assert man["base_model"] == expected_model
```

Also record:

```text
git SHA
training data manifest hash
fold assignment hash
hyperparameters
```

in new training manifests.

Block submission on mismatch.

---

# 10. Reranker smoke needs a real reload check

`train_bge_reranker()` saves the checkpoint but does not explicitly reload it through the exact inference wrapper.

After training:

```python
del model
torch.cuda.empty_cache()

smoke_reranker = BGEReranker(
    model_name=output_dir,
    device=dev,
)

scored = smoke_reranker.rerank(
    "Mức phạt là bao nhiêu?",
    [
        {"text_raw": "Phạt tiền từ 5.000.000 đồng đến 10.000.000 đồng."},
        {"text_raw": "Quy định khác không liên quan."},
    ],
    top_k=2,
)
```

Require:

```text
2 outputs
finite rerank_score
checkpoint actually loaded
```

A final screen will also exercise this, but the trainer should return only after checkpoint integrity is proven.

---

# 11. QLoRA token packing still does not guarantee no answer truncation

Current builder sets:

```python
"answer_truncated": False
```

unconditionally after packing.

But it does not enforce:

```python
total_tokens <= max_seq_len
```

after final prompt construction.

### Correct algorithm

```python
def build_sft_example_token_aware(...):
    answer_tokens = ...
    framing_tokens = ...
    minimum_required = len(framing_tokens) + len(answer_tokens)

    if minimum_required > max_seq_len:
        return DROP/ERROR diagnostic

    evidence_budget = max_seq_len - minimum_required - safety_margin
    pack evidence within evidence_budget

    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    while len(full_ids) > max_seq_len and evidence exists:
        remove/truncate last evidence unit
        rebuild

    assert len(full_ids) <= max_seq_len
    assert answer token suffix is present
```

Do not rely on trainer-side truncation.

Dataset report:

```text
P50 / P90 / P95 / P99 / max tokens
% evidence truncated
% examples dropped because answer cannot fit
```

---

# 12. Do a 2048 / 3072 sequence-length diagnostic before final QLoRA

Historical Task 2 answers are long.

Do not hard-code:

```python
max_seq_len=2048
```

without evidence.

Before screen QLoRA:

```text
build examples at 2048
build examples at 3072
optionally 4096 metadata-only diagnostic
```

Report:

```text
% evidence truncation
% unfit answers
estimated/observed T4 peak VRAM
```

Choose the smallest sequence length that preserves target answers sufficiently and fits T4.

Do not run a large hyperparameter sweep.

---

# 13. Fix candidate type coverage if a learned selector is used

`CandidateSelector.CANDIDATE_ORDER` does not currently include the new pack candidates such as:

```text
pack_focused
pack_full_article
pack_top2_relevance
```

They all fall to the same unknown index `99`.

Add all current candidate families explicitly or encode candidate family categorically in a stable way.

If final policy remains fixed, this is non-blocking.

If a learned selector is promoted, this is mandatory.

---

# 14. Revalidate the fixed candidate under the current pipeline

Do **not** assume old:

```text
stitched_extract ≈ 0.3051
```

still applies.

Current candidate construction/evidence packing has changed.

Screen all current fixed families:

```text
focused_extract
stitched_extract
pack_focused
pack_full_article
pack_top2_relevance
strategy_f_300
strategy_f_600
strategy_f_1000
strategy_f_1500
generated
snapped
```

Then freeze the actual best family.

---

# 15. Improve fixed extracts without changing the architecture

Current:

```python
focused_ext = clean_ev[:800]
stitched_ext = clean_ev[:1500]
```

can cut in the middle of a statutory clause/sentence.

Because `EvidencePacker` already produces structured packs, prefer structured-boundary candidates for the new bake-off.

Keep old 800/1500 char variants as candidates because they historically scored well, but add:

```text
focused_complete_clause
top2_relevance_complete_units
primary_article_budgeted_units
```

Do not replace old candidates until held-out METEOR proves the structured version is better.

---

# 16. Preflight should validate the packaged runtime manifests

`package_kaggle_dataset.py` already produces:

```text
dataset_manifest.json
code_manifest.json
```

Use them.

Preflight should verify:

```text
manifest exists
git SHA recorded
critical data file SHA matches
code manifest exists
production config exists
required index dirs exist
reranker pairs exist if tuned reranker final training is promoted
```

For the giant dense file, full SHA can be verified once at preflight.

---

# 17. Make production packaging profile-aware

Currently:

```text
reranker_training_pairs.parquet = optional
indexes = optional
public-official.json = optional-if-found
```

Create:

```bash
python scripts/package_kaggle_dataset.py --profile final_training
```

For `final_training`, require:

```text
legal_chunks.parquet
qa_unique.parquet
known_qa.json
qa_citations.parquet
retrieval_labels.parquet
fold_assignments.parquet
public-official.json
BM25 index
DEk21 index
code runtime
production_selection.yaml
reranker_training_pairs.parquet IF production config promotes tuned reranker
```

If any required artifact is missing:

```text
packaging fails before upload
```

---

# 18. Dependency reproducibility is still unresolved

`requirements-kaggle.txt` still uses wide lower bounds.

Do not randomly choose pins from local development.

The required workflow is:

### Smoke run

Print exact Kaggle versions:

```text
python
torch
transformers
trl
peft
bitsandbytes
sentence-transformers
datasets
bm25s
nltk
pyvi
```

### After successful smoke

Write:

```text
/kaggle/working/kaggle_environment.json
```

Then update:

```text
requirements-kaggle.txt
```

with a compatibility window or exact versions for packages the notebook installs.

Do not pin/replace Kaggle Torch unless required.

Until that smoke occurs, treat dependency compatibility as an unresolved external risk.

---

# 19. Add CI / regression coverage for the V4/V5 code paths

The latest commit has no GitHub combined CI status and no workflow run attached.

Add a CPU GitHub Actions workflow if permitted:

```text
.github/workflows/tests.yml
```

Run at least:

```bash
pytest tests/ -q
```

with CPU-safe dependencies.

GPU tests can be marked:

```python
@pytest.mark.gpu
```

and run on Kaggle smoke, not GitHub CI.

## Required new tests

### Evaluation

- [ ] `defaultdict`/candidate accumulator path executes.
- [ ] missing held-out fold raises.
- [ ] WordNet/METEOR helper matches official whitespace scorer.
- [ ] exact checkpoint evaluator rejects fallback in strict mode.

### Profiles

- [ ] smoke passes max steps + small data limits.
- [ ] screen excludes fold 0.
- [ ] final training uses `val_fold=None`.
- [ ] final profile refuses `production_selection.status=UNVALIDATED`.
- [ ] reuse profile rejects non-final checkpoint manifest.

### Generator

- [ ] `require_adapter=True` rejects missing path.
- [ ] `require_adapter=True` rejects missing PEFT support.
- [ ] strict loaded model is actually PEFT-backed.
- [ ] final generation error raises.
- [ ] token-aware SFT returns `<= max_seq_len`.
- [ ] answer tokens are preserved.

### Reranker

- [ ] trained checkpoint reloads through `BGEReranker`.
- [ ] batch rerank equals single rerank.
- [ ] smoke train subset/step limits are respected.

### Dense/preflight

- [ ] production one-GPU preflight fails.
- [ ] missing BM25 fails.
- [ ] missing dense index fails.
- [ ] wrong dense model fails.
- [ ] wrong dtype fails.
- [ ] wrong chunk hash fails.
- [ ] embedding SHA mismatch fails when enabled.

### Production config

- [ ] QLoRA off + fixed extract => generator not loaded/trained.
- [ ] QLoRA on + generator-based policy => adapter required.
- [ ] tuned reranker off => base reranker allowed.
- [ ] tuned reranker on => final checkpoint required.

---

# 20. Fix `run_oof_validation.py` or clearly deprecate neural full mode

Do not leave an API that can be misused.

Preferred V5 behavior:

```python
if mode == "full":
    if held_out_fold is not None:
        evaluate only held_out_fold
    elif fold_checkpoint_map is None:
        raise RuntimeError(
            "Full neural 5-fold OOF requires one reranker/adapter checkpoint per fold."
        )
```

For each fold in true neural OOF:

```text
load fold-specific reranker
load fold-specific adapter
exclude entire fold from memory
evaluate only that fold
release models
```

If full 5-fold neural OOF is too expensive, that is fine.

Use:

```text
screen_fold0
```

for architecture screening and label it honestly.

---

# 21. Production screen flow

Implement this exact sequence.

## Stage S0 — base

```text
BM25 + DEk21
pretrained BGE
base Qwen
current candidate family
```

## Stage S1 — tuned reranker

```text
same held-out IDs
tuned BGE
base Qwen
```

## Stage S2 — QLoRA

```text
same held-out IDs
tuned/promoted BGE
QLoRA
```

Write one comparable table.

Do not change retrieval parameters between S0/S1/S2.

---

# 22. Promotion logic

Recommended conservative logic:

## Reranker

Promote if:

```text
retrieval Recall/MRR improves
AND downstream best-fixed/selected METEOR does not regress
```

## QLoRA

Promote only if it improves a **deployable** answer policy.

Examples:

```text
QLoRA generated candidate is better than base
but fixed extract still beats it
and selector cannot exploit QLoRA
=> do NOT final-train QLoRA
```

versus:

```text
QLoRA improves generated candidate
and a validated gate raises selected METEOR
=> promote QLoRA
```

## Selector

Promote only if:

```text
meta-held-out selector METEOR > best fixed candidate METEOR
```

Otherwise use the fixed candidate.

---

# 23. Do not force Qwen into the final parameter stack if unused

If extractive production wins:

```text
BM25
DEk21
BGE reranker
EvidencePacker
fixed extract
```

can be the final pipeline.

That is allowed conceptually if competition rules do not require a generator.

Then actual inference parameter count is much smaller.

Do not load Qwen merely because the project originally planned RAG.

---

# 24. If QLoRA is promoted, tune output length once

Current:

```text
max_new_tokens=384
```

may be short for long gold answers.

On the same held-out screen IDs, compare only a small set:

```text
384
512
```

Optionally test:

```text
dynamic 384/512 based on nearest QA answer-length prior
```

Do not run a large sweep.

Promote only if deployable METEOR improves enough to justify runtime.

---

# 25. Fix run manifest provenance

Final `run_manifest.json` should include:

```json
{
  "git_sha": "...",
  "execution_profile": "final_train_and_submit",
  "production_selection_sha256": "...",
  "dataset_manifest_sha256": "...",
  "code_manifest_sha256": "...",
  "bm25_manifest_sha256": "...",
  "dense_manifest_sha256": "...",
  "dense_embeddings_sha256_verified": true,
  "reranker_manifest": "...",
  "generator_manifest": "...",
  "selector_manifest": null,
  "package_versions": {},
  "total_learned_parameters": 0,
  "candidate_policy": "...",
  "num_queries": 1000
}
```

This makes later leaderboard analysis reproducible.

---

# 26. Final submission strictness

In addition to existing checks, validate:

```text
each value is exactly {"answer": <string>} or official allowed equivalent
no <|im_start|>
no <|im_end|>
no "assistant\n" chat leakage
no [DOCUMENT]
no [ARTICLE]
no [CLAUSE]
no NaN-like string
zip contains exactly submission.json at root
```

After writing the ZIP, reopen it and inspect `namelist()`.

---

# 27. Suggested final notebook structure

Keep the notebook thin.

```text
Cell 1   profile + production config path
Cell 2   environment / secret / deterministic seed
Cell 3   deterministic runtime root/model resolution
Cell 4   dependency compatibility + version manifest
Cell 5   strict preflight incl. indexes + manifests + dual T4
Cell 6   load shared corpus/BM25/memory metadata
Cell 7   profile-aware reranker smoke/screen/final training
Cell 8   profile-aware QLoRA smoke/screen/final training IF required
Cell 9   strict checkpoint manifest + parameter audit
Cell 10  real held-out screen evaluation for smoke/screen profiles
Cell 11  load production inference components based on frozen config
Cell 12  true batch dense + batch reranker + optional batch generator
Cell 13  strict submission validation
Cell 14  save submission/checkpoints/provenance
```

For final profile:

```text
do not run held-out quality evaluation on all-data final models
```

Only checkpoint integrity smoke is valid after all-data training.

---

# 28. Concrete profile configuration

## `smoke_only`

```python
PROFILE = {
    "run_public": False,
    "held_out_fold": 0,
    "reranker_max_steps": 30,
    "reranker_max_pairs": 256,
    "reranker_max_val_pairs": 128,
    "qlora_max_steps": 30,
    "qlora_max_examples": 128,
    "screen_eval_size": 5,
    "require_dual_gpu": True,
}
```

## `screen_fold0`

```python
PROFILE = {
    "run_public": False,
    "held_out_fold": 0,
    "reranker_max_steps": None,
    "qlora_max_steps": None,
    "screen_eval_size": 250,
    "require_dual_gpu": True,
}
```

## `final_train_and_submit`

```python
assert production_cfg.status == "PROMOTED"
PROFILE = {
    "run_public": True,
    "held_out_fold": None,
    "reranker_val_fold": None,
    "qlora_val_fold": None,
    "require_dual_gpu": True,
}
```

---

# 29. Optional P1 score experiment: pairwise reranker loss

Do **not** block smoke on this.

After the production path works, compare current BCE reranker training against a direct pairwise objective:

```python
loss = -F.logsigmoid(pos_score - neg_score).mean()
```

Evaluate:

```text
Recall@1
Recall@5
Recall@8
MRR
downstream best fixed METEOR
```

Use the same fold-0 screen IDs.

Only keep pairwise training if measured better.

---

# 30. Optional P1 score experiment: retrieval-capacity Stack B

Do not implement Stack B runtime before Stack A is trustworthy.

After Stack A smoke/screen/final path is stable, Stack B remains a legitimate future experiment:

```text
BGE-M3 dense
+ BGE reranker
+ Qwen1.5B or no generator
```

But it requires a complete BGE-M3 corpus index and same held-out protocol.

Do not delay the first trustworthy Stack-A submission for this.

---

# 31. Required coding-agent verification report

Do not respond “done” without the following.

## A. Current git state

```text
HEAD SHA
changed files
```

## B. Tests

```text
pytest command
passed
failed
skipped
GitHub Actions URL/status if added
```

## C. Smoke proof

```text
dual T4 detected
reranker 30 optimizer steps
reranker subset size
reranker reload PASS
QLoRA 30 optimizer steps
QLoRA subset size
PEFT reload PASS
5 exact real pipeline predictions
no mocks
no fallback
peak GPU0 VRAM
peak GPU1 VRAM
```

## D. Screen proof

```text
held-out fold
number of screen QA
screen QA ID hash
base reranker metrics
tuned reranker metrics
base generator METEOR
QLoRA METEOR
candidate family METEOR table
best fixed candidate
selected policy
```

## E. Promotion

```text
production_selection.status
use_task_tuned_reranker
use_qlora
candidate_policy
source screen manifest
screen manifest hash
```

## F. Final training proof

```text
reranker training_scope = all_allowed_task2_data or not loaded
QLoRA training_scope = all_allowed_task2_data or not loaded
final checkpoint manifests PASS
```

## G. Index integrity

```text
BM25 corpus rows
dense model ID
dense dtype
dense shape
chunk hash
embedding hash
PASS
```

## H. Parameter audit

Count only actual loaded learned components plus adapters.

## I. Final submission proof

```text
1000 IDs
exact ID-set equality
zip namelist == ["submission.json"]
output paths
```

---

# 32. Definition of SAFE TO RUN FINAL

Only print:

```text
SAFE TO RUN FINAL KAGGLE TRAINING
```

when all are true:

- [ ] `evaluate_checkpoint()` executes without runtime errors;
- [ ] held-out evaluation never silently changes folds;
- [ ] official METEOR helper/resource setup works;
- [ ] smoke uses tiny data + ~30 optimizer steps;
- [ ] smoke allows no neural fallback;
- [ ] QLoRA-required load proves a PEFT model;
- [ ] reranker checkpoint reloads through the actual inference wrapper;
- [ ] screen uses real BM25 + DEk21 + exact reranker checkpoint + exact generator/adapter;
- [ ] screen uses deterministic representative IDs, not `.head(50)`;
- [ ] base-vs-tuned component comparison exists;
- [ ] production selection is frozen from measured screen results;
- [ ] final profile refuses `UNVALIDATED` production config;
- [ ] final profile trains only promoted components;
- [ ] final component training uses all allowed Task 2 data;
- [ ] reuse profile accepts only final all-data checkpoints;
- [ ] final/reuse dense and generator run in strict final mode;
- [ ] dual T4 is required rather than dynamically downgraded;
- [ ] preflight verifies BM25 and dense integrity;
- [ ] dense expected model + dtype + hashes pass;
- [ ] runtime/data/model path resolution is deterministic;
- [ ] QLoRA is not trained/loaded when final policy cannot use it;
- [ ] `rerank_batch()` is actually used in batch inference;
- [ ] SFT examples never silently truncate answer tokens;
- [ ] full neural OOF cannot misuse one checkpoint for every fold;
- [ ] tests pass;
- [ ] Kaggle package version tuple is recorded from a successful smoke;
- [ ] final parameter audit is `< 4,000,000,000`;
- [ ] submission archive passes all checks.

Otherwise report:

```text
NOT SAFE TO RUN
```

with exact remaining blockers.

---

# 33. Final engineering instruction

The architecture is now sufficiently strong.

Do not keep adding components.

The latest audited commit is being held back by **last-mile correctness and experiment gating**, not by lack of another model.

The highest-value work now is:

```text
fix evaluator runtime
make smoke truly small and strict
make screen scientifically valid
make production_selection authoritative
skip QLoRA if it cannot improve the deployed answer
make final/reuse checkpoints and indexes impossible to silently substitute
```

Then run:

```text
smoke_only
-> screen_fold0
-> freeze promotion
-> final_train_and_submit
```

and submit.
