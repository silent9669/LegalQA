# LegalQA Kaggle V7 — Promotion Consistency & Runtime Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make LegalQA HEAD safe for a trustworthy Kaggle Dual-T4 smoke run and ensure the subsequent `screen_fold0 -> promotion -> final_train_and_submit` sequence freezes only a component combination that was actually evaluated.

**Architecture:** Preserve Stack A and all V6 model choices. V7 fixes two remaining correctness boundaries: (1) dependency bootstrap must *prove* Kaggle Torch/CUDA is unchanged, and (2) screening/promotion must evaluate the same reranker/generator/candidate tuple that final inference will deploy. Secondary hardening adds canonical model identity, strict artifact/runtime provenance, complete preflight gates, and exact evaluation semantics.

**Tech Stack:** Python 3.10, PyTorch/CUDA, Hugging Face Transformers, TRL SFTTrainer, PEFT/QLoRA, bitsandbytes, SentenceTransformers/BGE reranker, BM25S, DEk21, pandas/pyarrow, pytest, GitHub Actions, Kaggle Dual T4.

**Spec:** `docs/LEGALQA_KAGGLE_RUNTIME_COMPAT_FIX_V6.md`

## Global Constraints

- Do **not** redesign Stack A or add new models.
- Production stack remains DEk21 + BGE reranker + Qwen2.5-3B-Instruct when generator is selected.
- Learned parameter budget remains strictly `< 4,000,000,000`.
- Official primary metric remains whitespace-tokenized METEOR.
- No silent neural fallback in smoke/screen/final/reuse profiles.
- No mock retrieval/reranking/generation in quality evaluation or promotion.
- `smoke_only` remains the committed notebook default.
- `final_train_and_submit` remains blocked until a measured `PROMOTED` configuration exists.
- Never print or persist `HF_TOKEN`.
- Do not claim GPU readiness from GitHub CPU CI alone.
- TDD: write the regression first, observe failure, implement the minimal fix, then run the focused test and full suite.
- After this V7 patch, **repackage and upload a new Kaggle runtime dataset version** before running the notebook.

---

# 0. Audited Baseline

Audited repository:

```text
https://github.com/silent9669/LegalQA
```

Audited HEAD:

```text
21c682fd245e27dc5a15de3c6609bfda2de5903b
```

V6 made material progress:

```text
- modern TRL prompt/completion dataset structure
- obsolete DataCollatorForCompletionOnlyLM removed
- notebook hard CUDA/T4 gate
- strict index-directory preflight
- streaming SHA256 helper
- runtime/Qwen ambiguity rejection
- retrieve_and_rerank trace
- retrieval label loading and Recall/MRR calculation
- generator-dependent candidate family accounting
- valid fixed_baseline candidate-policy representation
- canonical production promotion script
- final checkpoint manifest validation
- CI: 91 passed, 1 skipped at audited HEAD
```

Do not undo these changes.

The remaining V7 blockers are caused by **runtime protection not being enforceable** and **promotion selecting a configuration from measurements produced by a different component combination**.

---

# 1. File Structure for V7

## Create

```text
tests/test_v7_runtime_integrity.py
tests/test_v7_promotion_consistency.py
configs/runtime_api.yaml
```

Optional only if a focused helper keeps files smaller:

```text
src/task2/promotion.py
```

## Modify

```text
scripts/bootstrap_kaggle_env.py
.github/workflows/tests.yml
src/task2/training/train_generator.py
src/task2/checkpoint_manifest.py
src/task2/evaluation.py
src/task2/production_config.py
scripts/promote_production_selection.py
scripts/preflight_kaggle.py
src/task2/path_resolver.py
scripts/package_kaggle_dataset.py
kaggle_kernel/legalqa_gpu_pipeline.ipynb
src/task2/training/train_reranker.py
scripts/run_oof_validation.py
tests/test_v6_runtime_compat.py
```

Do not move unrelated files.

---

# Task 1 — Make Kaggle Dependency Bootstrap Actually Preserve Torch/CUDA

**Files:**
- Modify: `scripts/bootstrap_kaggle_env.py`
- Create: `tests/test_v7_runtime_integrity.py`
- Modify: `.github/workflows/tests.yml`

**Interfaces:**

Produce:

```python
def get_installed_distribution_version(dist_name: str) -> str | None: ...
def satisfies_spec(version: str, specifier: str) -> bool: ...
def snapshot_protected_versions() -> dict[str, str]: ...
def write_protected_constraints(snapshot: dict[str, str], path: str) -> str: ...
def assert_protected_versions_unchanged(before: dict[str, str], after: dict[str, str]) -> None: ...
def run_pip_check() -> None: ...
def bootstrap_dependencies(...) -> dict[str, object]: ...
```

The bootstrap result must include:

```python
{
    "protected_before": {...},
    "protected_after": {...},
    "installed_or_updated": [...],
    "pip_check_passed": True,
}
```

## Step 1.1 — Write failing protected-version test

- [ ] Add:

```python
def test_bootstrap_protected_snapshot_detects_version_drift(monkeypatch):
    before = {"torch": "2.6.0", "triton": "3.2.0"}
    after = {"torch": "2.7.0", "triton": "3.2.0"}

    with pytest.raises(RuntimeError, match="Protected runtime package changed"):
        assert_protected_versions_unchanged(before, after)
```

- [ ] Run:

```bash
pytest tests/test_v7_runtime_integrity.py::test_bootstrap_protected_snapshot_detects_version_drift -v
```

Expected: FAIL because helper does not exist.

## Step 1.2 — Implement protected snapshot

- [ ] Enumerate installed protected distributions dynamically.

At minimum protect any installed distribution whose normalized name is:

```text
torch
torchvision
torchaudio
triton
```

and every installed distribution starting with:

```text
nvidia-
cuda-
```

Do not hardcode only `*-cu12`; Kaggle/runtime versions may use a different CUDA major.

Use:

```python
from importlib import metadata

def snapshot_protected_versions():
    out = {}
    for dist in metadata.distributions():
        name = (dist.metadata.get("Name") or "").strip()
        norm = name.lower().replace("_", "-")
        if (
            norm in {"torch", "torchvision", "torchaudio", "triton"}
            or norm.startswith("nvidia-")
            or norm.startswith("cuda-")
        ):
            out[norm] = dist.version
    return dict(sorted(out.items()))
```

- [ ] Implement strict pre/post comparison.

Any changed, removed, or newly introduced protected distribution during bootstrap must fail unless an explicit development-only override is passed. The canonical notebook must never pass the override.

## Step 1.3 — Enforce version specifications instead of presence-only checks

- [ ] Use:

```python
from packaging.specifiers import SpecifierSet
from packaging.version import Version
```

- [ ] For every target package:

```python
if current_version is None:
    action = "install"
elif Version(current_version) not in SpecifierSet(specifier):
    action = "update"
else:
    action = "keep"
```

Do not log `[OK]` solely because a package imports.

Target compatibility packages must include:

```text
transformers
accelerate
datasets
peft
trl
bitsandbytes
sentence-transformers
bm25s
scikit-learn
nltk
pyvi
pyyaml
pyarrow
fastparquet
tqdm
```

## Step 1.4 — Generate exact protected constraints

- [ ] Write a temp constraints file:

```text
torch==<preinstalled>
torchvision==<preinstalled>       # if installed
torchaudio==<preinstalled>        # if installed
triton==<preinstalled>            # if installed
nvidia-...==<preinstalled>        # each installed
cuda-...==<preinstalled>          # each installed
```

- [ ] Install user-space requirements with:

```bash
python -m pip install \
  --upgrade-strategy only-if-needed \
  -c /tmp/legalqa_protected_constraints.txt \
  <required-specs...>
```

Do not use a plain unconstrained `pip install --upgrade`.

Do not claim `--no-deps` unless actually using it. Prefer constraints over `--no-deps`, because TRL/PEFT require legitimate user-space dependencies.

## Step 1.5 — Validate after install

- [ ] Run:

```bash
python -m pip check
```

- [ ] Snapshot protected distributions again.

- [ ] Raise if any protected version differs.

- [ ] Import:

```python
import torch
import transformers
import accelerate
import datasets
import peft
import trl
import bitsandbytes
import sentence_transformers
```

- [ ] Verify the current TRL API programmatically:

```python
import inspect
from trl import SFTConfig, SFTTrainer

assert "completion_only_loss" in inspect.signature(SFTConfig).parameters
```

If this API is absent, fail loudly rather than dropping completion-only semantics.

## Step 1.6 — Remove permissive legacy SFTConfig fallback

Current generator trainer catches `TypeError` and eventually constructs `SFTConfig` without:

```python
completion_only_loss=True
```

That can silently change the loss objective.

- [ ] Replace with a fail-loud compatibility helper:

```python
def build_sft_config(...):
    sig = inspect.signature(SFTConfig)
    if "completion_only_loss" not in sig.parameters:
        raise RuntimeError(
            "Installed TRL does not support completion_only_loss; "
            "refusing to train with changed loss semantics."
        )
    ...
```

Support `max_length` vs `max_seq_length` only if needed, but never silently remove `completion_only_loss`.

## Step 1.7 — Add CI proof of preservation

Do not use only the current workflow pattern:

```text
pip install requirements.txt
then bootstrap
```

because that already resolves a new Torch stack.

Add a separate job:

```yaml
bootstrap-protection:
```

Steps:

```text
1. setup Python
2. install a fixed CPU torch fixture version
3. record `python -c "import torch; print(torch.__version__)"`
4. install only minimal test requirements that do not mutate torch
5. run bootstrap in test/check mode
6. record torch version again
7. assert exact equality
8. run tests/test_v7_runtime_integrity.py
```

CPU unit-test job may remain separate.

- [ ] Run all V7 dependency tests.
- [ ] Commit:

```bash
git add scripts/bootstrap_kaggle_env.py tests/test_v7_runtime_integrity.py .github/workflows/tests.yml src/task2/training/train_generator.py
git commit -m "fix(kaggle): enforce immutable torch cuda bootstrap"
```

---

# Task 2 — Canonical Qwen Model Identity Across Train and Reuse

**Files:**
- Modify: `src/task2/training/train_generator.py`
- Modify: `src/task2/checkpoint_manifest.py`
- Modify: `kaggle_kernel/legalqa_gpu_pipeline.ipynb`
- Test: `tests/test_v7_runtime_integrity.py`

## Root cause

The notebook may train QLoRA from a mounted path such as:

```text
/kaggle/input/.../3b-instruct/...
```

The trainer currently persists that resolved path as:

```json
"base_model": "<mounted path>"
```

Fresh-final validation compares against the same path, but reuse validation compares against canonical config:

```text
Qwen/Qwen2.5-3B-Instruct
```

The same model can therefore fail identity validation on a later session.

## Step 2.1 — Write failing canonical-ID test

- [ ] Add:

```python
def test_generator_manifest_uses_canonical_base_model_id(tmp_path):
    manifest = {
        "base_model_id": "Qwen/Qwen2.5-3B-Instruct",
        "resolved_model_path": "/kaggle/input/qwen/3b-instruct/1",
        "training_scope": "all_allowed_task2_data",
        "is_final_checkpoint": True,
        "smoke_only": False,
        "val_fold_excluded": None,
    }
    ...
    assert_final_checkpoint(
        ...,
        expected_base_model="Qwen/Qwen2.5-3B-Instruct",
        component_name="generator",
    )
```

## Step 2.2 — Split logical identity from load path

Change trainer signature to:

```python
def run_qlora_training(
    model_name_or_path: str,
    base_model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    ...
)
```

Use:

```python
model_name_or_path
```

for tokenizer/model loading.

Store:

```json
{
  "base_model_id": "Qwen/Qwen2.5-3B-Instruct",
  "resolved_model_path": "/kaggle/input/...",
  ...
}
```

Do not use machine-specific path as canonical identity.

## Step 2.3 — Update validator

For generator identity:

```python
base_m = (
    manifest.get("base_model_id")
    or manifest.get("base_model")
    or manifest.get("base_model_name_or_path")
)
```

For new V7 manifests, require `base_model_id`.

## Step 2.4 — Notebook passes both values

```python
run_qlora_training(
    model_name_or_path=MODEL_PATH,
    base_model_id=PRODUCTION_CFG.generator_base_model,
    ...
)
```

Fresh and reuse asserts both compare against:

```python
PRODUCTION_CFG.generator_base_model
```

- [ ] Run focused tests.
- [ ] Commit.

---

# Task 3 — Replace Inconsistent S0/S1/S2 Promotion with a Component-Consistent Screen

**Files:**
- Modify: `src/task2/evaluation.py`
- Optional create: `src/task2/promotion.py`
- Create: `tests/test_v7_promotion_consistency.py`
- Modify: `kaggle_kernel/legalqa_gpu_pipeline.ipynb`

This is the highest-priority scoring-correctness task.

## Current invalid dependency

Current matrix effectively measures:

```text
S0 = base reranker  + base generator
S1 = tuned reranker + base generator
S2 = tuned reranker + QLoRA
```

Then it can reject the tuned reranker but still use S2 to choose QLoRA/candidate policy.

That freezes a policy from a system that final inference will not actually deploy.

## Required staged screen

Use the same deterministic validation IDs.

### Stage A — Base

```text
R0G0 = base reranker + base generator
```

### Stage B — Reranker candidate

```text
R1G0 = tuned reranker + base generator
```

Decide:

```text
selected_reranker = R1 if reranker passes guard else R0
base_selected_summary = R1G0 if R1 else R0G0
```

### Stage C — QLoRA candidate under selected reranker

```text
R*G1 = selected_reranker + QLoRA
```

Compare `R*G1` against `R*G0`, where `R*G0 == base_selected_summary`.

This remains only three evaluations, so it does not need a more expensive 2x2 matrix.

## Step 3.1 — Write failing reranker-rejected/QLoRA test

Use synthetic summaries.

Example:

```python
def test_qlora_is_evaluated_with_selected_base_reranker_when_tuned_rejected():
    # R0G0 has stronger retrieval/downstream.
    # R1G0 fails reranker promotion.
    # Assert qlora evaluator receives R0 checkpoint, not R1 checkpoint.
```

This should fail against current hardwired S2 tuned reranker behavior.

## Step 3.2 — Write failing “final candidate comes from rejected QLoRA” test

Synthetic case:

```text
best R*G0 = stitched_extract 0.300
best R*G1 = strategy_f_1000 0.3005
meteor tolerance = 0.001
```

QLoRA is not promoted.

Expected final:

```text
use_qlora = false
best_fixed_candidate = best measured candidate from R*G0
```

Never freeze `strategy_f_1000` from the rejected QLoRA run.

## Step 3.3 — Extract pure decision functions

Prefer pure, testable helpers:

```python
def best_deployable_candidate(summary: dict) -> tuple[str, float]: ...

def decide_reranker_promotion(
    base_summary: dict,
    tuned_summary: dict,
    retrieval_tolerance: float,
    meteor_tolerance: float,
) -> dict: ...

def decide_generator_promotion(
    base_summary: dict,
    qlora_summary: dict,
    meteor_tolerance: float,
) -> dict: ...
```

## Step 3.4 — Reranker guard compares system-level deployable baselines

Do not compare S1’s chosen candidate only to the same family in S0.

Compute:

```python
base_best_name, base_best_score = best_deployable_candidate(R0G0)
tuned_best_name, tuned_best_score = best_deployable_candidate(R1G0)
```

Reranker may be promoted only if:

```python
retrieval_improved
and tuned_best_score >= base_best_score - meteor_tolerance
```

The best family is allowed to differ between R0 and R1.

## Step 3.5 — Make retrieval criterion explicit

Do not use ambiguous mixed MRR.

Prefer:

```text
chunk_mrr
chunk_recall_at_8
```

for reranker promotion if chunk supervision is the direct positive target.

Or use article metrics if the project intentionally optimizes article-level evidence; whichever is selected must be documented and named.

Recommended:

```python
retrieval_improved = (
    tuned.chunk_mrr > base.chunk_mrr + retrieval_tolerance
    or tuned.chunk_recall_at_8 > base.chunk_recall_at_8 + retrieval_tolerance
)
```

## Step 3.6 — Require tuned checkpoint when screen says tuned

In `screen_fold0`:

```python
if tuned_reranker is None or not os.path.isdir(tuned_reranker):
    raise FileNotFoundError(...)
```

Do not silently set:

```python
reranker_to_use = base_reranker
```

## Step 3.7 — Require QLoRA checkpoint when screen says QLoRA

Since canonical `screen_fold0` requests generator training:

```python
if adapter_path is None or not os.path.isdir(adapter_path):
    raise FileNotFoundError(...)
```

Do not alias S2 to S1.

## Step 3.8 — Evaluate QLoRA with selected reranker

Pseudocode:

```python
r0g0 = evaluate_checkpoint(base_reranker, adapter=None)
r1g0 = evaluate_checkpoint(tuned_reranker, adapter=None)

rerank_decision = decide_reranker_promotion(r0g0, r1g0, ...)
selected_reranker = tuned_reranker if rerank_decision["promote"] else base_reranker
selected_base = r1g0 if rerank_decision["promote"] else r0g0

selected_qlora = evaluate_checkpoint(
    reranker_checkpoint=selected_reranker,
    adapter_path=adapter_path,
    ...
)

gen_decision = decide_generator_promotion(
    selected_base,
    selected_qlora,
    ...
)

final_summary = selected_qlora if gen_decision["promote"] else selected_base
final_candidate = best_deployable_candidate(final_summary)
```

## Step 3.9 — Persist exact deployment tuple

Promotion report must contain:

```json
{
  "screen_protocol_version": 7,
  "evaluated_systems": {
    "R0G0": {...},
    "R1G0": {...},
    "R_SELECTED_G1": {...}
  },
  "selected_reranker": {
    "use_task_tuned": true,
    "checkpoint": "...",
    "decision_reason": "..."
  },
  "selected_generator": {
    "use_qlora": false,
    "adapter": null,
    "decision_reason": "..."
  },
  "candidate_policy": {
    "type": "fixed_baseline",
    "best_fixed_candidate": "..."
  },
  "final_measured_system_key": "R1G0"
}
```

The final candidate must come from `final_measured_system_key`.

## Step 3.10 — Run promotion tests and commit

- [ ] Focused tests.
- [ ] Existing evaluation tests.
- [ ] Full suite.
- [ ] Commit:

```bash
git add src/task2/evaluation.py tests/test_v7_promotion_consistency.py kaggle_kernel/legalqa_gpu_pipeline.ipynb
git commit -m "fix(task2): make component promotion measurement consistent"
```

---

# Task 4 — Split Retrieval MRR Semantics and Enforce Coverage

**Files:**
- Modify: `src/task2/evaluation.py`
- Test: `tests/test_v7_promotion_consistency.py`

## Step 4.1 — Add distinct MRR metrics

Current generic MRR matches either chunk or article.

Replace with:

```text
chunk_mrr
article_mrr
```

Calculate independently.

Do not use `zip(retrieved_chunks, retrieved_articles)` as the only semantic.

## Step 4.2 — Add coverage

Persist:

```text
num_eval_queries
num_queries_with_chunk_labels
num_queries_with_article_labels
chunk_label_coverage
article_label_coverage
```

## Step 4.3 — Screen coverage gate

Add:

```python
min_retrieval_label_coverage: float = 0.70
```

For canonical screen, require at least:

```python
chunk_label_coverage >= min_retrieval_label_coverage
```

or document an article-level equivalent if chosen.

“At least one labeled query” is insufficient for promotion.

## Step 4.4 — Add tests

Synthetic test:

```text
250 eval queries
10 labeled
```

Expected: screen raises for insufficient coverage.

Synthetic test:

```text
250 eval queries
220 labeled
```

Expected: allowed.

- [ ] Commit with Task 3 if implemented together.

---

# Task 5 — Make Production Promotion Script Validate the Exact Measured Tuple

**Files:**
- Modify: `scripts/promote_production_selection.py`
- Modify: `src/task2/production_config.py`
- Test: `tests/test_v7_promotion_consistency.py`

## Step 5.1 — Reject old/incomplete report schema

Promoter must require:

```text
screen_protocol_version >= 7
evaluated_systems
final_measured_system_key
selected_reranker
selected_generator
candidate_policy
sample_ids_sha256
sample_size
```

## Step 5.2 — Cross-check consistency

Examples:

If:

```json
"selected_generator": {"use_qlora": false}
```

and final measured system is a QLoRA system, reject.

If:

```json
"selected_reranker": {"use_task_tuned": false}
```

and final measured system uses tuned reranker, reject.

If final candidate is generator dependent, the selected system must contain the corresponding base/QLoRA generator that produced its score.

## Step 5.3 — Freeze report hash

Continue using streaming SHA256.

Write:

```yaml
status: PROMOTED
source_screen_manifest: ...
source_screen_sha256: ...
screen_protocol_version: 7
```

## Step 5.4 — Validate before final profile

`validate_production_selection_for_profile()` should require protocol 7 for newly promoted final configs.

Allow legacy only with an explicit development override, not in canonical notebook.

---

# Task 6 — Strict Preflight: BM25 Index Files and Public-Test Requirement

**Files:**
- Modify: `scripts/preflight_kaggle.py`
- Modify: `kaggle_kernel/legalqa_gpu_pipeline.ipynb`
- Test: `tests/test_v7_runtime_integrity.py`

## Step 6.1 — Check actual BM25S files

When:

```python
check_indexes=True
```

require:

```text
bm25_manifest.json
bm25s_index/params.index.json
```

If BM25S stores additional required files for the installed version, validate those based on the saved index manifest.

## Step 6.2 — Add explicit public requirement

Add:

```python
require_public: bool = False
```

Behavior:

```python
if require_public and (
    public_path is None or not os.path.isfile(public_path)
):
    errors.append("Required public-official.json is missing")
```

If present, still enforce exactly 1000 queries.

## Step 6.3 — Notebook profile wiring

For:

```text
final_train_and_submit
reuse_final_checkpoints_and_submit
```

set:

```python
require_public=True
```

For:

```text
smoke_only
screen_fold0
```

public test can be optional.

## Step 6.4 — Do not open public test in smoke/screen

Cell 12 must start:

```python
if RUN_PUBLIC_INFERENCE:
    with open(TEST_PATH, ...) as f:
        public_test = json.load(f)
    ...
else:
    public_test = {}
    submission = {}
```

This allows pure training/evaluation runtime datasets without public test if desired.

---

# Task 7 — Reject Stale Kaggle Runtime Code/Data

**Files:**
- Create: `configs/runtime_api.yaml`
- Modify: `scripts/package_kaggle_dataset.py`
- Modify: `src/task2/path_resolver.py`
- Modify: `kaggle_kernel/legalqa_gpu_pipeline.ipynb`
- Test: `tests/test_v7_runtime_integrity.py`

## Root cause

The Kaggle dataset contains a full staged copy:

```text
code/LegalQA/src
code/LegalQA/scripts
code/LegalQA/configs
```

The notebook executes packaged code from `/kaggle/input`.

A new GitHub commit does nothing to an already-uploaded Kaggle dataset. A stale dataset can therefore execute old code with the new notebook.

## Step 7.1 — Add runtime API manifest

Create:

```yaml
schema_version: 1
runtime_api_version: 7
stack: stack_a
```

## Step 7.2 — Package runtime API version

Include in:

```text
code_manifest.json
dataset_manifest.json
```

with:

```json
"runtime_api_version": 7
```

Also preserve `git_sha`.

## Step 7.3 — Notebook verifies before imports

Cell 3 must expect:

```python
EXPECTED_RUNTIME_API_VERSION = 7
```

Read staged manifests before importing task modules.

Require:

```python
dataset_manifest["runtime_api_version"] == 7
code_manifest["runtime_api_version"] == 7
```

If absent or stale:

```text
raise RuntimeError("Kaggle runtime dataset is stale; repackage/upload V7.")
```

## Step 7.4 — Optional exact SHA pin

For a release run, support:

```python
EXPECTED_CODE_GIT_SHA = None
```

When set, require exact manifest SHA match.

Do not hardcode a pre-fix SHA in the committed notebook. Runtime API version is the stable default guard.

## Step 7.5 — Test stale manifest failure

Create fixture with runtime API 6 and assert failure.

---

# Task 8 — Make Path Resolution Strict in Kaggle Mode

**Files:**
- Modify: `src/task2/path_resolver.py`
- Modify: `kaggle_kernel/legalqa_gpu_pipeline.ipynb`
- Test: `tests/test_v7_runtime_integrity.py`

Add:

```python
def resolve_runtime_paths(
    base_input_dir="/kaggle/input",
    strict: bool = False,
) -> dict[str, str]:
```

When `strict=True`:

```text
0 LegalQA roots -> RuntimeError
>1 LegalQA roots -> RuntimeError
0 intended Qwen roots -> RuntimeError unless network-loading mode is explicitly allowed
>1 intended Qwen roots -> RuntimeError
```

Do not fall back to local `artifacts` in strict Kaggle mode.

Notebook uses:

```python
resolve_runtime_paths("/kaggle/input", strict=True)
```

---

# Task 9 — Improve QLoRA Sequence-Length Diagnostics with Real Tokens

**Files:**
- Modify: `src/task2/training/train_generator.py`
- Test: `tests/test_v7_runtime_integrity.py`

Current diagnostic recalculates length using:

```python
len(text.split())
```

which is not tokenizer length.

Refactor builder to optionally return aggregate diagnostics:

```python
{
    "kept": ...,
    "dropped": ...,
    "drop_rate": ...,
    "evidence_truncated": ...,
    "evidence_truncated_rate": ...,
    "token_lengths": [...],
}
```

`run_seq_len_diagnostic()` reports tokenizer-derived:

```text
P50
P90
P95
P99
max
drop rate
evidence truncation rate
```

for:

```text
2048
3072
```

Do not auto-switch sequence length; print recommendation and keep explicit notebook setting.

---

# Task 10 — Optional PyArrow Pushdown for Smoke Evidence

**Files:**
- Modify: `src/task2/training/train_generator.py`
- Test: `tests/test_v7_runtime_integrity.py`

This is P1, not a blocker.

Prefer:

```python
import pyarrow.dataset as ds

dataset = ds.dataset(chunks_path, format="parquet")
table = dataset.to_table(
    columns=["chunk_id", "text_raw"],
    filter=ds.field("chunk_id").isin(list(needed_chunk_ids)),
)
```

Fallback to pandas projection only if Arrow filtering fails.

No behavior change to full training examples.

---

# Task 11 — Full OOF Checkpoint Provenance

**Files:**
- Modify: `scripts/run_oof_validation.py`
- Test: existing/new OOF tests

For each fold checkpoint in full mode require manifest:

```text
smoke_only == false
val_fold_excluded == current_fold
base_model_id/base_model matches expected
training_scope == folds_excluding_<fold>
```

For full neural mode:

```python
BM25Retriever.load(..., fail_on_missing_index=True)
DenseRetriever.load_index(
    ...,
    expected_model_name=...,
    expected_dtype="float16",
    final_mode=True,
)
```

No rebuild/fallback.

This is not needed to run the initial fold-0 screen, but it prevents future false “OOF” claims.

---

# Task 12 — CI and Regression Gate

**Files:**
- Modify: `.github/workflows/tests.yml`
- Test: full suite

Required CI jobs:

## `unit-tests`

Runs current CPU tests.

## `bootstrap-protection`

Proves protected Torch fixture version remains unchanged after bootstrap.

## `training-api-compat`

Installs the intended user-space compatibility set and asserts:

```text
SFTConfig has completion_only_loss
SFTTrainer import works
PEFT import works
BitsAndBytesConfig import works
train_generator module import works
```

No 3B model download.

Required V7 regression cases:

```text
[ ] protected Torch version drift raises
[ ] constraints file pins every installed protected package
[ ] incompatible package version is not marked OK
[ ] pip check failure raises
[ ] missing completion_only_loss support raises
[ ] canonical Qwen model ID survives mounted-path train/reuse
[ ] reranker rejected -> QLoRA evaluated with base reranker
[ ] reranker accepted -> QLoRA evaluated with tuned reranker
[ ] QLoRA rejected -> final candidate comes from base-generator system
[ ] QLoRA accepted -> final candidate comes from QLoRA system
[ ] reranker guard compares best deployable R0 vs best deployable R1
[ ] missing tuned reranker checkpoint fails screen
[ ] missing QLoRA adapter fails screen
[ ] chunk_mrr and article_mrr are distinct
[ ] insufficient label coverage fails screen
[ ] promoter rejects component-inconsistent report
[ ] missing bm25s index files fail preflight
[ ] missing required public test fails final/reuse preflight
[ ] smoke/screen do not open public test
[ ] stale runtime API manifest fails
[ ] strict Kaggle path mode refuses local fallback
[ ] sequence diagnostics use tokenizer counts
```

Run:

```bash
pytest tests/ -v
```

No new failures.

Commit:

```bash
git add .
git commit -m "fix(kaggle): close v7 promotion and runtime integrity gates"
```

---

# Task 13 — Repackage Kaggle Runtime Dataset

After tests pass, do not reuse the currently uploaded runtime package.

Generate canonical artifacts first if needed:

```bash
python scripts/package_kaggle_dataset.py \
  --source artifacts/task2 \
  --staging kaggle_dataset/staged \
  --profile final_training
```

Verify staged:

```text
dataset_manifest.json
code_manifest.json
runtime_api_version == 7
legal_chunks.parquet
qa_unique.parquet
known_qa.json
qa_citations.parquet
retrieval_labels.parquet
fold_assignments.parquet
reranker_training_pairs.parquet
indexes/bm25/...
indexes/dek21/...
code/LegalQA/src/...
code/LegalQA/scripts/...
code/LegalQA/configs/runtime_api.yaml
public-official.json     # required for final-training package
```

Upload a **new version** of:

```text
phucdangg/legalqa-task2-clean-data
```

Do not assume GitHub HEAD changes the Kaggle dataset.

---

# Task 14 — Real Kaggle Smoke Gate

Notebook:

```python
EXECUTION_PROFILE = "smoke_only"
```

Use Kaggle Dual T4.

Clean:

```text
Restart Session
Run All
```

Required evidence:

```text
runtime_api_version = 7
protected Torch/CUDA versions unchanged pre/post bootstrap
pip check PASS
CUDA available
2 GPUs
BM25 strict load
DEk21 strict load/hash/identity
30 reranker optimizer steps
reranker reload PASS
reranker peak VRAM recorded
30 QLoRA optimizer steps
PEFT adapter reload PASS
generator peak VRAM recorded
5 held-out real predictions
no mock
no neural fallback
environment manifest saved
```

Save:

```text
/kaggle/working/kaggle_environment.json
/kaggle/working/checkpoints/reranker/best/reranker_manifest.json
/kaggle/working/checkpoints/generator/hf_adapter/generator_manifest.json
/kaggle/working/evaluations/screen_evaluation_summary.json
```

Only after this succeeds is the repo safe to proceed to screen.

---

# Task 15 — Real `screen_fold0` Gate

Set:

```python
EXECUTION_PROFILE = "screen_fold0"
```

Run clean.

Required:

```text
same deterministic held-out IDs across every evaluated system
R0G0
R1G0
selected reranker decision
R_SELECTED_G1
chunk Recall@1/5/8
chunk MRR
article Recall@1/5/8
article MRR
label coverage
all candidate-family METEOR
promotion_report.json protocol 7
```

The report must identify:

```text
final_measured_system_key
exact reranker checkpoint type
exact generator mode
exact selected candidate family
```

No final config is promoted until this report exists.

---

# Task 16 — Freeze Production Selection

Run:

```bash
python scripts/promote_production_selection.py \
  --report /kaggle/working/promotion_report.json \
  --config configs/production_selection.yaml \
  --output /kaggle/working/production_selection.yaml
```

Verify:

```text
status == PROMOTED
screen_protocol_version == 7
report SHA matches
candidate policy uses fixed_baseline + exact winning family
use_task_tuned_reranker matches final measured system
use_qlora matches final measured system
```

Use this exact promoted config for final training.

---

# Task 17 — Final Train and Submit Gate

Set:

```python
EXECUTION_PROFILE = "final_train_and_submit"
```

Use the promoted config.

Train only promoted components.

Required:

```text
val_fold=None
fresh final checkpoint manifests validate
canonical model IDs validate
no smoke checkpoint reused
no held-out-fold checkpoint reused
actual loaded component parameter count <4B
maximum approved Stack-A parameter count <4B
strict production reload
1000 exact public IDs
1000 non-empty answers
no forbidden special tokens
submission.json
submission.json.zip containing only submission.json
run_manifest.json
```

Do not claim final-submission readiness from CPU tests or smoke alone.

---

# Definition of SAFE FOR KAGGLE SMOKE

Coding agent may end with:

```text
SAFE FOR KAGGLE SMOKE
```

only if all code-level V7 gates are implemented and all relevant CPU/compatibility CI passes.

It must **not** mean the GPU smoke itself already passed unless real Kaggle evidence is supplied.

---

# Definition of SAFE FOR `screen_fold0`

Only after real `smoke_only` Restart Session → Run All succeeds with Dual T4, reload checks, and safe VRAM.

---

# Definition of SAFE TO RUN FINAL KAGGLE TRAINING

Only after:

```text
real smoke passes
real component-consistent screen passes
promotion_report protocol 7 exists
PROMOTED config is generated from that report
component tuple matches its measured system
```

---

# Required Coding-Agent Completion Report

Return exactly these sections:

```text
HEAD SHA
Changed files
Root causes fixed
Bootstrap protected-package proof
TRL/SFT API proof
Canonical model identity proof
Promotion matrix proof
Retrieval coverage/metric proof
Runtime API/stale-dataset proof
Preflight proof
CPU pytest result
CI job results
Kaggle GPU evidence (or "not run")
Remaining blockers
Verdict
```

Verdict must be one of:

```text
SAFE FOR KAGGLE SMOKE
NOT SAFE FOR KAGGLE SMOKE
```

Do not output `SAFE TO RUN FINAL KAGGLE TRAINING` unless actual Kaggle smoke + screen evidence exists.

---

# Self-Review Checklist

## Spec coverage

- [x] Torch/CUDA bootstrap protection is enforceable, not merely documented.
- [x] Version constraints are actually checked.
- [x] TRL completion-only semantics cannot silently disappear.
- [x] Train/reuse use canonical Qwen identity.
- [x] Reranker decision is made before QLoRA evaluation.
- [x] QLoRA is evaluated using the reranker that final deployment would use.
- [x] Candidate winner comes only from the finally selected component system.
- [x] Missing tuned checkpoints fail.
- [x] Retrieval MRR semantics and coverage are explicit.
- [x] Promotion report encodes exact measured component tuple.
- [x] BM25S actual index files are preflighted.
- [x] Public test requirement is profile-specific.
- [x] Stale Kaggle dataset/code is detected.
- [x] Strict Kaggle path mode has no local fallback.
- [x] Token diagnostics use tokenizer counts.
- [x] Full OOF provenance is hardened.
- [x] Kaggle runtime package must be rebuilt and re-uploaded.
- [x] Smoke, screen, promotion, and final gates are separately defined.

## Placeholder scan

No TODO/TBD/“implement later” placeholders are permitted in implementation steps.

## Type consistency

Canonical names used throughout:

```text
base_model_id
model_name_or_path
runtime_api_version
chunk_mrr
article_mrr
final_measured_system_key
selected_reranker
selected_generator
fixed_baseline
best_fixed_candidate
```

Do not create alternate spellings in implementation.
