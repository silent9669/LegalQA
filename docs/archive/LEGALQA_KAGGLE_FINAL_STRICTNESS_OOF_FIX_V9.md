# LegalQA Kaggle V9 — Final Strictness & Honest OOF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the remaining V8 fail-open paths and false-positive tests so the canonical Kaggle notebook is genuinely safe to attempt on Dual T4, while making `mode="full"` evaluation provably free of mocks/fallbacks.

**Architecture:** Do not redesign Stack A, retraining, retrieval, candidate generation, or Protocol-8 promotion logic. V9 hardens four boundaries only: canonical notebook runtime resolution, runtime manifest identity, promotion-report schema enforcement, and full-OOF execution semantics. Runtime API is bumped to **9** to force a fresh Kaggle runtime package containing these fixes.

**Tech Stack:** Python 3.10, Kaggle Dual NVIDIA T4, PyTorch/CUDA, Transformers, TRL, PEFT/QLoRA, bitsandbytes, SentenceTransformers/BGE, BM25S, DEk21, pandas/pyarrow, pytest, GitHub Actions.

**Spec:** `docs/LEGALQA_KAGGLE_FAIL_LOUD_INTEGRATION_FIX_V8.md`

## Global Constraints

- Do not change model families or Stack A topology.
- Learned parameter budget remains strictly `< 4,000,000,000`.
- Primary metric remains organizer whitespace-tokenized METEOR.
- Keep `EXECUTION_PROFILE = "smoke_only"` as committed notebook default.
- Canonical Kaggle notebook has **no development fallback**.
- Screen/final/full quality evaluation may not use mock dense, mock reranker, fallback generator, or incomplete fold checkpoint coverage.
- Do not print or persist `HF_TOKEN`.
- Do not claim GPU success from CPU CI.
- Use TDD: reproduce failure first, then minimal fix, focused test, full suite.
- Runtime API becomes **9** and requires a newly packaged Kaggle dataset version.
- Keep screen protocol at **8** unless actual measured-report semantics are changed beyond the validation tightening in this plan.

---

# 0. Audited baseline

```text
Repository: https://github.com/silent9669/LegalQA
Audited HEAD: ae7329cc0a3cfaca4f7dc8234c00c5edaa8ac9c4
Commit: fix(kaggle): enforce V8 fail-loud runtime integration and provenance
Workflow run 33389552419: SUCCESS
CPU Unit Test Suite: SUCCESS
Bootstrap Protected-Version Preservation Proof: SUCCESS
SFT Training API & Module Compatibility: SUCCESS
pytest collected 124 items
pytest result: 123 passed, 1 skipped
```

Use the fresh CI log as authoritative; the commit message's “124 passing tests” is not exact because one item is skipped.

## Confirmed V8 improvements — do not revert

```text
runtime_integrity.py exists
runtime API 8 exists
mounted-Qwen failure mode exists
pip check raises
required import verification can raise
requirements-kaggle aligns with bootstrap floors
promotion-report mirror is byte-identical
Protocol-8 staged screen exists
fold checkpoint helper functions exist
CI is green
```

---

# 1. Root-cause findings

## P0-1 — Canonical notebook still has a development code fallback

Current Cell 3 manually resolves `/kaggle/input/**/code/LegalQA`; when none is found it does:

```python
resolved_code_root = os.path.abspath(".")
```

The strict helper is imported but not used as the authoritative resolver.

**Root cause:** helper implementation and notebook integration diverged.

## P0-2 — Runtime strictness is conditional

Current notebook uses environment heuristics:

```python
strict=True if os.path.exists("/kaggle/input") and any(os.scandir("/kaggle/input")) else False
allow_remote_model_download=False if ... else True
```

Canonical Kaggle execution must never infer development mode. Use literal `strict=True` and `allow_remote_model_download=False`.

## P0-3 — Manifest validation is caller-side fail-open

Current notebook:

```python
if os.path.exists(os.path.join(paths["runtime_root"], "dataset_manifest.json")):
    validate_runtime_manifests(...)
```

This bypasses the helper's missing-manifest error. A runtime recognized from parquet + BM25 can continue with no dataset manifest.

## P0-4 — Notebook regression test is too shallow

The current test proves only that strings such as `resolve_packaged_code_root`, `validate_runtime_manifests`, and `allow_remote_model_download=False` appear somewhere. It does not prove they are called unconditionally. That is why CI passes while P0-1 through P0-3 remain.

## P0-5 — `mode="full"` can silently use mock Dense

Current OOF logic effectively does:

```python
if mode == "full" and dense_index_exists:
    dense = real_dense
else:
    dense = DEk21Retriever(model_name="mock")
```

Missing real index changes the experiment instead of failing.

## P0-6 — `mode="full"` can use mock reranker/fallback generator

Current logic mixes `mode` with device fallbacks. Full mode on a CPU-valued device can instantiate mock reranker or fallback generator. Full mode must mean real components only.

## P0-7 — Incomplete `fold_checkpoint_map` is accepted

A truthy map enables multi-fold full OOF, but folds absent from the map silently reuse the globally loaded model. This can produce a claimed 5-fold OOF result with only some folds truly held out.

## P1-1 — Runtime Git SHA is optional

`validate_runtime_manifests()` checks mismatch only when both SHA values are truthy. Missing/empty SHA values therefore pass. `get_git_sha()` can also return `"unknown"`.

## P1-2 — Promoter checks provenance only when fields exist

Examples:

```python
if "sample_ids_sha256" in sys_summary: ...
if "reranker_checkpoint" in final_summary: ...
if "candidate_family_meteors" in final_summary: ...
```

A corrupted report can omit a field and bypass the check. Required report schema must be enforced before consistency checks.

---

# 2. File map

## Create

```text
tests/test_v9_notebook_contract.py
tests/test_v9_full_oof_contract.py
```

## Modify

```text
configs/runtime_api.yaml
kaggle_kernel/legalqa_gpu_pipeline.ipynb
src/task2/runtime_integrity.py
scripts/package_kaggle_dataset.py
scripts/promote_production_selection.py
scripts/run_oof_validation.py
tests/test_v8_fail_loud_integration.py
tests/test_v8_promotion_provenance.py
.github/workflows/tests.yml
```

No unrelated refactor.

---

# Task 1 — Make the canonical notebook unconditionally strict

**Files:** `kaggle_kernel/legalqa_gpu_pipeline.ipynb`, `tests/test_v9_notebook_contract.py`.

- [ ] **Step 1: Write failing notebook contract tests.**

```python
import json
import re
from pathlib import Path


def notebook_source() -> str:
    nb = json.loads(Path("kaggle_kernel/legalqa_gpu_pipeline.ipynb").read_text())
    parts = []
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        parts.append("".join(src) if isinstance(src, list) else str(src))
    return "\n".join(parts)


def test_notebook_calls_strict_packaged_code_resolver():
    src = notebook_source()
    assert re.search(
        r"resolve_packaged_code_root\(\s*[\"']/kaggle/input[\"']\s*,\s*strict=True\s*\)",
        src,
    )


def test_notebook_runtime_paths_are_unconditionally_strict():
    src = notebook_source()
    assert "strict=True if" not in src
    assert "allow_remote_model_download=False if" not in src
    assert re.search(
        r"resolve_runtime_paths\(\s*[\"']/kaggle/input[\"']\s*,\s*strict=True\s*,\s*allow_remote_model_download=False",
        src,
    )


def test_notebook_has_no_code_root_development_fallback():
    src = notebook_source()
    cell3 = src[src.index("# Cell 3"):src.index("# Cell 4")]
    assert 'resolved_code_root = os.path.abspath(".")' not in cell3
```

- [ ] **Step 2: Run and prove failure on audited HEAD.**

```bash
pytest tests/test_v9_notebook_contract.py -v
```

Expected: FAIL.

- [ ] **Step 3: Remove notebook development fallback.**

Before project imports, bootstrap exactly one packaged root only:

```python
_bootstrap_roots = sorted({
    os.path.abspath(p)
    for p in glob.glob("/kaggle/input/**/code/LegalQA", recursive=True)
    if os.path.isdir(os.path.join(p, "src"))
    and os.path.isdir(os.path.join(p, "scripts"))
})

if len(_bootstrap_roots) != 1:
    raise RuntimeError(
        f"Expected exactly one packaged LegalQA code root, found {_bootstrap_roots}"
    )

sys.path.insert(0, _bootstrap_roots[0])

from src.task2.runtime_integrity import (
    EXPECTED_RUNTIME_API_VERSION,
    resolve_packaged_code_root,
    validate_runtime_manifests,
)

resolved_code_root = resolve_packaged_code_root("/kaggle/input", strict=True)
```

- [ ] **Step 4: Make runtime path call literal.**

```python
paths = resolve_runtime_paths(
    "/kaggle/input",
    strict=True,
    allow_remote_model_download=False,
)
```

- [ ] **Step 5: Run focused tests and commit.**

```bash
pytest tests/test_v9_notebook_contract.py -v
git add kaggle_kernel/legalqa_gpu_pipeline.ipynb tests/test_v9_notebook_contract.py
git commit -m "fix(kaggle): make canonical notebook runtime strictly packaged"
```

---

# Task 2 — Unconditional manifests + real Git identity + runtime API 9

**Files:** `src/task2/runtime_integrity.py`, notebook, packager, `configs/runtime_api.yaml`, tests.

- [ ] **Step 1: Add notebook regression.**

```python
def test_notebook_manifest_validation_is_not_guarded_by_exists():
    src = notebook_source()
    cell3 = src[src.index("# Cell 3"):src.index("# Cell 4")]
    assert "validate_runtime_manifests(" in cell3
    assert (
        'if os.path.exists(os.path.join(paths["runtime_root"], "dataset_manifest.json"))'
        not in cell3
    )
```

- [ ] **Step 2: Call validation unconditionally.**

```python
RUNTIME_PROVENANCE = validate_runtime_manifests(
    runtime_root=paths["runtime_root"],
    code_root=resolved_code_root,
    expected_api_version=EXPECTED_RUNTIME_API_VERSION,
)
```

- [ ] **Step 3: Require real SHA values.**

```python
import re
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _require_git_sha(value: object, field: str) -> str:
    text = str(value or "").strip().lower()
    if not GIT_SHA_RE.fullmatch(text):
        raise RuntimeError(
            f"{field} must contain a real 40-character lowercase Git commit SHA; found {value!r}."
        )
    return text
```

Then always require both SHA fields and equality.

- [ ] **Step 4: Make production packaging fail if Git SHA cannot be resolved.**

```python
def get_git_sha(strict: bool = False) -> str:
    try:
        ...
    except Exception as exc:
        if strict:
            raise RuntimeError("Could not resolve Git HEAD for production package") from exc
        return "unknown"
```

For `profile="final_training"`, resolve once with `strict=True` and reuse the same SHA in both manifests.

- [ ] **Step 5: Bump runtime API.**

```yaml
schema_version: 1
runtime_api_version: 9
stack: stack_a
```

Also set `EXPECTED_RUNTIME_API_VERSION = 9` and package manifest version 9.

- [ ] **Step 6: Add SHA tests.**

```python
@pytest.mark.parametrize("bad_sha", [None, "", "unknown", "abc", "g" * 40])
def test_runtime_manifest_requires_real_git_sha(tmp_path, bad_sha):
    ...
    with pytest.raises(RuntimeError, match="40-character"):
        validate_runtime_manifests(...)
```

Add one passing test with `"a" * 40`.

---

# Task 3 — Promoter schema must reject omitted provenance

**Files:** `scripts/promote_production_selection.py`, `tests/test_v8_promotion_provenance.py`.

Define mandatory evaluated-system fields:

```python
REQUIRED_EVALUATED_SYSTEM_KEYS = {
    "sample_ids_sha256",
    "sample_size",
    "candidate_family_meteors",
    "retrieval_metrics",
    "reranker_checkpoint",
    "generator_model",
    "adapter_path",
    "dense_model",
    "no_mocks",
    "no_fallbacks",
}
```

- [ ] **Step 1: Add missing-field regressions.**

```python
def test_promoter_rejects_missing_system_sample_hash(tmp_path): ...
def test_promoter_rejects_missing_final_reranker_checkpoint(tmp_path): ...
def test_promoter_rejects_missing_candidate_scores(tmp_path): ...
```

Each must delete the field from an otherwise valid report and expect `ValueError`.

- [ ] **Step 2: Require all three systems.**

```python
for key in ("R0G0", "R1G0", "R_SELECTED_G1"):
    if key not in eval_systems:
        raise ValueError(f"Missing required evaluated system: {key}")
```

- [ ] **Step 3: Require every mandatory field for every system.**

No `if field in summary` guards.

- [ ] **Step 4: Require real evaluation semantics.**

```python
if sys_summary["no_mocks"] is not True:
    raise ValueError(...)
if sys_summary["no_fallbacks"] is not True:
    raise ValueError(...)
```

- [ ] **Step 5: Make final identity checks unconditional.**

Directly validate `reranker_checkpoint`, `adapter_path`, and `candidate_family_meteors` after schema validation.

- [ ] **Step 6: Require winner-name equality.**

```python
if report["overall_deployable_winner"] != best_fixed:
    raise ValueError(...)
```

Keep `screen_protocol_version: 8`; measurement semantics are unchanged.

---

# Task 4 — Make `mode="full"` real-only

**Files:** `scripts/run_oof_validation.py`, `tests/test_v9_full_oof_contract.py`.

Create:

```python
def validate_full_mode_contract(
    *,
    bm25_dir: str,
    dek21_dir: str,
    held_out_fold: int | None,
    fold_checkpoint_map: dict | None,
    n_splits: int,
    reranker_checkpoint: str,
    adapter_path: str | None,
    retrieval_device: str | None,
    gen_device: str | None,
) -> None:
    ...
```

- [ ] **Step 1: Add missing-dense failure test.**

```python
def test_full_mode_missing_dense_index_fails(tmp_path):
    with pytest.raises(RuntimeError, match="Dense"):
        validate_full_mode_contract(...)
```

- [ ] **Step 2: Full-mode prerequisites.**

Require before model loading:

```text
BM25 directory exists
DEk21 directory exists
DEk21 embeddings.npy exists
DEk21 manifest exists
retrieval_device starts with cuda
gen_device starts with cuda when generator enabled
```

- [ ] **Step 3: Separate full and fast loading.**

```python
if mode == "full":
    dense = load_real_dense_or_raise(...)
    reranker = BGEReranker(model_name=reranker_checkpoint, device=r_dev)
    generator = QwenGenerator.load(..., fail_on_fallback=True, final_mode=True)
else:
    dense = mock_dense(...)
    reranker = mock_reranker(...)
    generator = fallback_generator(...)
```

No `mode="full"` branch may instantiate `model_name="mock"` or `runtime="fallback"`.

---

# Task 5 — Complete fold-checkpoint coverage

**Files:** OOF script + `tests/test_v9_full_oof_contract.py`.

Create:

```python
def validate_fold_checkpoint_map(
    fold_checkpoint_map: dict[int, dict[str, str]],
    *,
    target_folds: list[int],
    require_reranker: bool,
    require_adapter: bool,
) -> None:
    ...
```

- [ ] **Step 1: Add incomplete-map regression.**

```python
def test_full_oof_incomplete_checkpoint_map_fails():
    fold_map = {
        0: {"reranker": "/r0", "adapter": "/g0"},
        1: {"reranker": "/r1", "adapter": "/g1"},
    }
    with pytest.raises(RuntimeError, match="missing folds"):
        validate_fold_checkpoint_map(
            fold_map,
            target_folds=[0, 1, 2, 3, 4],
            require_reranker=True,
            require_adapter=True,
        )
```

- [ ] **Step 2: Implement exact coverage.**

```python
missing = sorted(set(target_folds) - set(fold_checkpoint_map))
if missing:
    raise RuntimeError(f"fold_checkpoint_map missing folds: {missing}")
```

For each target fold, require the tuned component path when that component is part of the evaluation.

Base pretrained reranker/generator may be shared because they were not fitted on competition folds.

- [ ] **Step 3: Validate all folds before evaluation starts.**

Do not discover missing fold 4 after four expensive folds have already run.

---

# Task 6 — Strengthen checkpoint manifest helpers

Current helpers allow missing scope/base identity. Make these mandatory.

```python
if scope != f"folds_excluding_{fold_id}":
    raise ValueError(...)

if not base_m:
    raise ValueError("checkpoint manifest missing base model identity")
```

Apply to reranker and generator helpers. Add focused tests.

---

# Task 7 — CI contract gates

**File:** `.github/workflows/tests.yml`.

Keep all current jobs and add:

```bash
pytest \
  tests/test_v9_notebook_contract.py \
  tests/test_v9_full_oof_contract.py -v
```

Keep final:

```bash
pytest tests/ -v
```

No GPU/model downloads in CI.

---

# Task 8 — Verification before code-ready verdict

Run fresh:

```bash
pytest tests/test_v9_notebook_contract.py -v
pytest tests/test_v9_full_oof_contract.py -v
pytest tests/test_v8_fail_loud_integration.py -v
pytest tests/test_v8_promotion_provenance.py -v
pytest tests/ -v
```

Static notebook contract:

```bash
python - <<'PY'
import json
from pathlib import Path

nb = json.loads(Path("kaggle_kernel/legalqa_gpu_pipeline.ipynb").read_text())
src = "\n".join(
    "".join(c["source"]) if isinstance(c.get("source"), list) else str(c.get("source", ""))
    for c in nb["cells"]
    if c.get("cell_type") == "code"
)

for forbidden in [
    'resolved_code_root = os.path.abspath(".")',
    'strict=True if',
    'allow_remote_model_download=False if',
    'if os.path.exists(os.path.join(paths["runtime_root"], "dataset_manifest.json"))',
]:
    assert forbidden not in src, forbidden

assert 'resolve_packaged_code_root("/kaggle/input", strict=True)' in src
assert "validate_runtime_manifests(" in src
print("Notebook V9 strictness contract PASS")
PY
```

OOF check: manually verify no `mode="full"` path can reach mock/fallback constructors.

---

# Task 9 — Repackage runtime API 9

Mandatory:

```bash
python scripts/package_kaggle_dataset.py \
  --source artifacts/task2 \
  --staging kaggle_dataset/staged \
  --profile final_training
```

Verify:

```bash
python - <<'PY'
import json
import re
from pathlib import Path

root = Path("kaggle_dataset/staged")
ds = json.loads((root / "dataset_manifest.json").read_text())
code = json.loads((root / "code_manifest.json").read_text())
assert ds["runtime_api_version"] == 9
assert code["runtime_api_version"] == 9
assert ds["git_sha"] == code["git_sha"]
assert re.fullmatch(r"[0-9a-f]{40}", ds["git_sha"])
assert (root / "code/LegalQA/src").is_dir()
assert (root / "code/LegalQA/scripts").is_dir()
assert (root / "indexes/bm25").is_dir()
assert (root / "indexes/dek21").is_dir()
print("V9 package contract PASS")
PY
```

Upload a **new Kaggle dataset version**. Do not reuse V8.

---

# Task 10 — Real Dual-T4 smoke gate

After V9 code + CI are green:

```python
EXECUTION_PROFILE = "smoke_only"
```

Kaggle accelerator:

```text
GPU T4 x2
```

Run:

```text
Restart Session
Run All
```

Required evidence:

```text
exactly one packaged LegalQA code root
runtime API 9
dataset/code real Git SHA match
mounted Qwen resolved
2 T4 GPUs detected
protected Torch/CUDA unchanged
pip check PASS
all required imports PASS
BM25 strict load PASS
DEk21 real FP16 GPU load PASS
30 reranker optimizer steps + reload PASS
30 QLoRA optimizer steps + PEFT reload PASS
5 real held-out predictions
no mock/fallback
no public inference
```

Do not change architecture in response to an unseen error. Capture the exact Kaggle error first.

---

# Verdict definitions

## SAFE FOR KAGGLE SMOKE

Use only when:

```text
V9 notebook contract tests pass
V9 full-OOF contract tests pass
full pytest passes
all GitHub Actions jobs pass
runtime API is 9
fresh V9 package is ready
```

This means code-ready to try GPU smoke, not GPU-verified.

## SAFE FOR SCREEN RUN

Use only after real Dual-T4 `smoke_only` Run All succeeds.

## SAFE FOR FINAL TRAINING

Use only after:

```text
real Dual-T4 smoke passed
Protocol-8 screen completed
PROMOTED config produced from real report
final selected tuple provenance validated
```

---

# Required coding-agent completion report

Return exactly:

```text
HEAD SHA
Changed files
Notebook strictness proof
Runtime API/Git SHA proof
Promotion schema proof
Full-OOF no-mock proof
Fold-map completeness proof
Focused pytest results
Full pytest result
GitHub Actions results
V9 package proof
Kaggle GPU evidence (or "not run")
Remaining blockers
Verdict
```

Verdict exactly:

```text
SAFE FOR KAGGLE SMOKE
```

or:

```text
NOT SAFE FOR KAGGLE SMOKE
```

---

# Self-review

- [x] Removes notebook local code fallback.
- [x] Makes runtime strictness unconditional.
- [x] Makes manifest verification unconditional.
- [x] Requires real Git SHAs.
- [x] Bumps runtime API to 9.
- [x] Replaces shallow notebook tests with contract checks.
- [x] Requires promoter fields instead of checking only if present.
- [x] Prevents mock Dense/reranker/fallback generator in full OOF.
- [x] Rejects incomplete fold checkpoint maps.
- [x] Requires complete checkpoint manifest provenance.
- [x] Keeps Stack A unchanged.
- [x] Keeps Protocol-8 measurement semantics unchanged.
- [x] Separates CPU CI proof from real Kaggle GPU proof.
