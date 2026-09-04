# LegalQA Kaggle V8 — Fail-Loud Integration & Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining V7 integration gaps so the canonical Kaggle notebook cannot silently run stale code, local artifacts, a broken dependency tree, or unverified component provenance.

**Architecture:** Keep Stack A and the existing staged R0G0 → R1G0 → R_SELECTED_G1 screening design. V8 hardens only the execution shell: exact packaged-runtime identity, fatal bootstrap/import validation, mounted-model resolution, canonical report provenance, and direct full-OOF checkpoint provenance. Bump runtime API to 8 so older Kaggle packages are rejected.

**Tech Stack:** Python 3.10, Kaggle Dual T4, PyTorch/CUDA, Transformers, TRL, PEFT/QLoRA, bitsandbytes, SentenceTransformers/BGE, BM25S, DEk21, pandas/pyarrow, pytest, GitHub Actions.

**Spec:** `docs/LEGALQA_KAGGLE_PROMOTION_RUNTIME_FIX_V7.md`

## Global Constraints

- Do not redesign Stack A or change model families.
- Keep learned parameter budget strictly `< 4,000,000,000`.
- Official primary metric remains whitespace-tokenized METEOR.
- `smoke_only` remains the committed notebook default.
- No mock neural components in screen/final quality evaluation.
- No silent local-artifact, CPU neural, dependency, or import fallback in canonical Kaggle execution.
- Never print or persist `HF_TOKEN`.
- A green CPU CI run is not evidence that Dual-T4 Kaggle smoke passed.
- TDD for every V8 behavior: failing regression → minimal fix → focused test → full suite.
- Runtime API becomes **8**; repackage and re-upload Kaggle runtime data before smoke.

---

## 0. Audited baseline

```text
Repository: https://github.com/silent9669/LegalQA
HEAD: 1a48a5fd4dc79ec5a2543022b6b420685e89422d
V7 implementation: 2c0adf1bdac3ff3692a3f56dacd8de8fb4e78620
CI at HEAD: all 3 jobs successful
Full pytest log: 107 passed, 1 skipped
```

Do not revert V7 improvements: protected Torch/CUDA snapshots, canonical Qwen `base_model_id`, strict TRL `completion_only_loss`, staged component-consistent screen, chunk/article retrieval metrics, coverage gate, Protocol-7 promotion logic, strict BM25/Dense preflight, adapter-aware parameter audit, and runtime-manifest packaging.

### Verified remaining defects

1. Canonical notebook calls `resolve_runtime_paths("/kaggle/input", strict=False)`.
2. Notebook validates `code_manifest.json` only if it exists; missing manifest is not fatal.
3. Notebook chooses the first `/kaggle/input/**/code/LegalQA` match instead of rejecting ambiguity.
4. `bootstrap_dependencies()` catches `run_pip_check()` failure and only warns.
5. `verify_runtime_imports()` can record required-module failures without raising.
6. `strict=True` path resolution still falls back to remote `Qwen/Qwen2.5-3B-Instruct` when mounted Qwen is absent.
7. `requirements-kaggle.txt` conflicts with bootstrap (`trl>=0.8.0` vs modern completion-only contract).
8. Promotion-report copies are not byte-identical because hashing occurs before adding `report_sha256` to the mirror copy.
9. Promoter should prove final measured checkpoint/adapter/sample/candidate-score consistency, not only reject a few impossible combinations.
10. Direct `mode="full", held_out_fold=N` OOF does not validate direct checkpoint fold provenance.

---

## 1. File map

### Create

```text
src/task2/runtime_integrity.py
tests/test_v8_fail_loud_integration.py
tests/test_v8_promotion_provenance.py
```

### Modify

```text
configs/runtime_api.yaml
kaggle_kernel/legalqa_gpu_pipeline.ipynb
scripts/bootstrap_kaggle_env.py
src/task2/path_resolver.py
requirements-kaggle.txt
scripts/package_kaggle_dataset.py
src/task2/evaluation.py
scripts/promote_production_selection.py
src/task2/production_config.py
scripts/run_oof_validation.py
.github/workflows/tests.yml
```

---

## Task 1 — Exact packaged-runtime identity

**Files:** create `src/task2/runtime_integrity.py`, create `tests/test_v8_fail_loud_integration.py`, modify `configs/runtime_api.yaml`.

**Interfaces:**

```python
EXPECTED_RUNTIME_API_VERSION = 8

def find_packaged_code_roots(base_input_dir: str = "/kaggle/input") -> list[str]: ...
def resolve_packaged_code_root(base_input_dir: str = "/kaggle/input", *, strict: bool = True) -> str: ...
def validate_runtime_manifests(
    runtime_root: str,
    code_root: str,
    *,
    expected_api_version: int = 8,
    expected_git_sha: str | None = None,
) -> dict[str, object]: ...
```

- [ ] **Step 1: Write failing missing-manifest regression.**

```python
def test_missing_code_manifest_is_fatal(tmp_path):
    runtime = tmp_path / "runtime"
    code = runtime / "code" / "LegalQA"
    code.mkdir(parents=True)
    (runtime / "dataset_manifest.json").write_text(
        json.dumps({"runtime_api_version": 8, "git_sha": "abc"})
    )
    with pytest.raises(RuntimeError, match="code_manifest.json"):
        validate_runtime_manifests(str(runtime), str(code))
```

Run:

```bash
pytest tests/test_v8_fail_loud_integration.py::test_missing_code_manifest_is_fatal -v
```

Expected: FAIL before implementation.

- [ ] **Step 2: Add exact-API and Git-SHA regressions.**

```python
@pytest.mark.parametrize("version", [7, 9])
def test_runtime_api_must_equal_8(tmp_path, version):
    ...
    with pytest.raises(RuntimeError, match="runtime_api_version"):
        validate_runtime_manifests(..., expected_api_version=8)

def test_dataset_code_git_sha_must_match(tmp_path):
    ...
    with pytest.raises(RuntimeError, match="git_sha"):
        validate_runtime_manifests(...)
```

- [ ] **Step 3: Add ambiguous-code-root regression.**

```python
def test_ambiguous_packaged_code_roots_fail(tmp_path):
    for name in ("a", "b"):
        root = tmp_path / name / "code" / "LegalQA"
        (root / "src").mkdir(parents=True)
        (root / "scripts").mkdir()
    with pytest.raises(RuntimeError, match="Ambiguous packaged LegalQA code roots"):
        resolve_packaged_code_root(str(tmp_path), strict=True)
```

- [ ] **Step 4: Implement strict helpers.**

Rules:

```text
0 packaged roots  -> RuntimeError in strict mode
1 packaged root   -> use it
>1 packaged roots -> RuntimeError
missing dataset_manifest.json -> RuntimeError
missing code_manifest.json    -> RuntimeError
dataset API != expected       -> RuntimeError
code API != expected          -> RuntimeError
dataset git_sha != code git_sha -> RuntimeError
expected_git_sha set and mismatch -> RuntimeError
```

Do not include `.`, `/kaggle/working`, or arbitrary local source roots when `strict=True`.

- [ ] **Step 5: Bump `configs/runtime_api.yaml`.**

```yaml
schema_version: 1
runtime_api_version: 8
stack: stack_a
```

- [ ] **Step 6: Run focused tests and commit.**

```bash
pytest tests/test_v8_fail_loud_integration.py -v
git add src/task2/runtime_integrity.py tests/test_v8_fail_loud_integration.py configs/runtime_api.yaml
git commit -m "fix(kaggle): enforce exact runtime package identity"
```

---

## Task 2 — Canonical notebook must actually use strict mode

**Files:** modify `kaggle_kernel/legalqa_gpu_pipeline.ipynb`, test `tests/test_v8_fail_loud_integration.py`.

- [ ] **Step 1: Write notebook-source regression.**

```python
def notebook_source():
    nb = json.loads(Path("kaggle_kernel/legalqa_gpu_pipeline.ipynb").read_text())
    return "\n".join(
        str(c.get("source", ""))
        for c in nb["cells"]
        if c.get("cell_type") == "code"
    )

def test_notebook_uses_strict_runtime_resolution():
    src = notebook_source()
    assert 'strict=True' in src
    assert 'resolve_runtime_paths("/kaggle/input", strict=False' not in src
    assert "validate_runtime_manifests(" in src
```

Run and confirm it fails against audited HEAD.

- [ ] **Step 2: Remove first-match code-root behavior.**

Before importing project modules, collect only `/kaggle/input/**/code/LegalQA` roots containing both `src/` and `scripts/`. Require exactly one; otherwise raise.

- [ ] **Step 3: Make manifest validation unconditional.**

Missing `dataset_manifest.json` or `code_manifest.json` must raise before training imports.

- [ ] **Step 4: Wire strict runtime path resolution.**

Canonical call:

```python
paths = resolve_runtime_paths(
    "/kaggle/input",
    strict=True,
    allow_remote_model_download=False,
)
```

Never call canonical Kaggle runtime with `strict=False`.

- [ ] **Step 5: Expected runtime API is exactly 8.**

Notebook:

```python
EXPECTED_RUNTIME_API_VERSION = 8
```

Validate both code and dataset manifests.

- [ ] **Step 6: Run test and commit.**

---

## Task 3 — Mounted Qwen is mandatory in strict Kaggle mode

**Files:** modify `src/task2/path_resolver.py`, test `tests/test_v8_fail_loud_integration.py`.

Change interface:

```python
def resolve_runtime_paths(
    base_input_dir: str = "/kaggle/input",
    strict: bool = False,
    allow_remote_model_download: bool = True,
) -> dict[str, str]:
```

- [ ] **Step 1: Write failing regression.**

```python
def test_strict_runtime_requires_mounted_qwen(tmp_path):
    create_valid_runtime_without_qwen(tmp_path)
    with pytest.raises(RuntimeError, match="Qwen"):
        resolve_runtime_paths(
            str(tmp_path),
            strict=True,
            allow_remote_model_download=False,
        )
```

- [ ] **Step 2: Implement explicit behavior.**

```python
qwen = find_qwen_model_dir(base_input_dir)
if qwen:
    qwen_dir = qwen
elif allow_remote_model_download:
    qwen_dir = "Qwen/Qwen2.5-3B-Instruct"
else:
    raise RuntimeError("Expected mounted Qwen2.5-3B model was not found.")
```

Canonical notebook passes `False`; local development may use the default `True`.

---

## Task 4 — Dependency bootstrap must fail loud

**Files:** modify `scripts/bootstrap_kaggle_env.py`, test `tests/test_v8_fail_loud_integration.py`.

- [ ] **Step 1: Regression for fatal `pip check`.**

```python
def test_bootstrap_does_not_swallow_pip_check_failure(monkeypatch):
    def boom():
        raise RuntimeError("pip check reported broken dependencies")
    monkeypatch.setattr(bootstrap, "run_pip_check", boom)

    # patch package discovery so no install is required
    ...
    with pytest.raises(RuntimeError, match="pip check"):
        bootstrap.bootstrap_dependencies()
```

- [ ] **Step 2: Remove current warning-only wrapper.**

Replace the `try/except` around `run_pip_check()` with:

```python
run_pip_check()
pip_check_passed = True
```

Return:

```python
{
    "protected_before": protected_before,
    "protected_after": protected_after,
    "installed_or_updated": to_install_or_update,
    "pip_check_passed": True,
}
```

- [ ] **Step 3: Regression for required import failure.**

```python
def test_required_import_failure_is_fatal(monkeypatch):
    real_import = bootstrap.importlib.import_module
    def fake_import(name):
        if name == "bitsandbytes":
            raise ImportError("simulated")
        return real_import(name)
    monkeypatch.setattr(bootstrap.importlib, "import_module", fake_import)

    with pytest.raises(RuntimeError, match="bitsandbytes"):
        bootstrap.verify_runtime_imports(strict=True)
```

- [ ] **Step 4: Implement strict verifier.**

```python
def verify_runtime_imports(strict: bool = True) -> dict[str, str]:
    failures = []
    ...
    if failures and strict:
        raise RuntimeError(
            "Required runtime import verification failed:\n"
            + "\n".join(failures)
        )
```

TRL `completion_only_loss` absence is one of the fatal failures.

- [ ] **Step 5: Persist proof, not secrets.**

Extend environment manifest with:

```json
{
  "pip_check_passed": true,
  "protected_before": {},
  "protected_after": {}
}
```

Do not include token values or arbitrary environment variables.

- [ ] **Step 6: Notebook calls `verify_runtime_imports(strict=True)`.**

---

## Task 5 — Align `requirements-kaggle.txt` with bootstrap

**Files:** modify `requirements-kaggle.txt`, test `tests/test_v8_fail_loud_integration.py`.

Use:

```text
# LegalQA Kaggle user-space compatibility floors.
# Torch/CUDA are intentionally omitted; bootstrap protects Kaggle's preinstalled stack.
transformers>=4.45.0
accelerate>=0.34.0
datasets>=2.20.0
peft>=0.10.0
trl>=0.11.0
bitsandbytes>=0.43.0
sentence-transformers>=3.0.0
bm25s>=0.2.5
scikit-learn>=1.4.0
nltk>=3.8.1
pyvi>=0.1.1
pyyaml>=6.0
pyarrow>=14.0.0
fastparquet>=2024.2.0
tqdm>=4.66.0
```

- [ ] **Step 1: Test `trl>=0.11.0`, no Torch line, and package-name parity with bootstrap target list.**
- [ ] **Step 2: Update file and run test.**
- [ ] **Step 3: Commit.**

---

## Task 6 — Package runtime API 8 and self-validate staging

**Files:** modify `scripts/package_kaggle_dataset.py`, test `tests/test_v8_fail_loud_integration.py`.

- [ ] **Step 1: Import one authoritative constant.**

```python
from src.task2.runtime_integrity import EXPECTED_RUNTIME_API_VERSION
RUNTIME_API_VERSION = EXPECTED_RUNTIME_API_VERSION
```

- [ ] **Step 2: Write API 8 into both manifests.**
- [ ] **Step 3: After staging, call `validate_runtime_manifests(...)` against staged paths.**
- [ ] **Step 4: Fail packaging if API or Git SHA diverges.**
- [ ] **Step 5: Add regression and commit.**

---

## Task 7 — Canonical promotion-report bytes and hash

**Files:** modify `src/task2/evaluation.py`, create `tests/test_v8_promotion_provenance.py`.

Expose:

```python
def write_promotion_report(
    report: dict,
    primary_path: str,
    mirror_path: str | None = None,
) -> dict[str, str]:
    ...
```

- [ ] **Step 1: Failing byte-identity regression.**

```python
def test_promotion_report_mirror_is_byte_identical(tmp_path):
    ...
    hashes = write_promotion_report(report, str(a), str(b))
    assert a.read_bytes() == b.read_bytes()
    assert hashes["sha256"] == sha256_file(a) == sha256_file(b)
```

- [ ] **Step 2: Use canonical JSON serialization.**

```python
payload = json.dumps(
    report,
    ensure_ascii=False,
    indent=2,
    sort_keys=True,
).encode("utf-8")
```

Write exact same bytes to both destinations.

- [ ] **Step 3: Remove mutable self-hash field from report JSON.**

Do not add `report_sha256` after writing. Let promoter compute SHA of the source file and persist it in promoted YAML.

- [ ] **Step 4: Bump report `screen_protocol_version` to 8.**

---

## Task 8 — Promoter proves the exact measured tuple

**Files:** modify `scripts/promote_production_selection.py`, `src/task2/production_config.py`, test `tests/test_v8_promotion_provenance.py`.

- [ ] **Step 1: Require Protocol 8.**

Reject reports `< 8`. Promoted config writes:

```yaml
screen_protocol_version: 8
```

Final/reuse profiles reject promoted configs `< 8`.

- [ ] **Step 2: Require identical evaluation set across all systems.**

For `R0G0`, `R1G0`, `R_SELECTED_G1`:

```python
assert summary["sample_ids_sha256"] == report["sample_ids_sha256"]
assert summary["sample_size"] == report["sample_size"]
```

Raise `ValueError` instead of `assert`.

- [ ] **Step 3: Verify selected reranker checkpoint.**

For `final_measured_system_key`, require:

```python
final_summary["reranker_checkpoint"] == selected_reranker["checkpoint"]
```

- [ ] **Step 4: Verify generator/adapter identity.**

If `use_qlora=True`:

```python
final_summary["adapter_path"] == selected_generator["adapter"]
```

If `use_qlora=False`, final measured key cannot be the QLoRA deployment.

- [ ] **Step 5: Verify winning candidate and score.**

```python
winner = report["candidate_policy"]["best_fixed_candidate"]
scores = final_summary["candidate_family_meteors"]
if winner not in scores:
    raise ValueError(...)
if abs(float(scores[winner]) - float(report["overall_deployable_meteor"])) > 1e-8:
    raise ValueError(...)
```

- [ ] **Step 6: Add regressions for sample hash mismatch, reranker mismatch, adapter mismatch, and score mismatch.**
- [ ] **Step 7: Run focused tests and commit.**

---

## Task 9 — Direct full-OOF checkpoint provenance

**Files:** modify `scripts/run_oof_validation.py`, test `tests/test_v8_promotion_provenance.py`.

Create:

```python
def assert_fold_reranker_checkpoint(
    checkpoint_path: str,
    fold_id: int,
    expected_base_model: str,
) -> dict: ...

def assert_fold_generator_checkpoint(
    adapter_path: str,
    fold_id: int,
    expected_base_model: str,
) -> dict: ...
```

Both require:

```text
smoke_only == false
val_fold_excluded == fold_id
training_scope == folds_excluding_<fold_id>
base model identity matches expected
```

- [ ] **Step 1: Test direct held-out reranker with wrong excluded fold fails.**
- [ ] **Step 2: Test direct held-out adapter with wrong excluded fold fails.**
- [ ] **Step 3: Reuse helpers for `fold_checkpoint_map`.**
- [ ] **Step 4: For `mode="full", held_out_fold=N`, validate direct paths before loading neural models.**
- [ ] **Step 5: Run tests and commit.**

---

## Task 10 — Notebook-level regression coverage

**Files:** modify `tests/test_v8_fail_loud_integration.py`.

Tests must inspect the actual notebook, not only helpers.

Require:

```text
committed EXECUTION_PROFILE == "smoke_only"
runtime API == 8
canonical path call uses strict=True
canonical path call does not use strict=False
manifest validation is unconditional
exactly-one packaged code root is enforced
allow_remote_model_download=False
verify_runtime_imports(strict=True)
final/reuse requires public test
public test is not opened when RUN_PUBLIC_INFERENCE=False
```

This closes the gap where helper tests passed while notebook wiring remained permissive.

---

## Task 11 — CI gates

**Files:** modify `.github/workflows/tests.yml`.

Keep current jobs.

### CPU Unit Test Suite

```bash
pytest tests/ -v
```

### Bootstrap Protected-Version Preservation Proof

Keep Torch before/after equality and add:

```bash
pytest tests/test_v8_fail_loud_integration.py -v
```

### SFT Training API & Module Compatibility

Retain:

```python
assert "completion_only_loss" in inspect.signature(SFTConfig).parameters
```

Run:

```bash
pytest   tests/test_v7_promotion_consistency.py   tests/test_v8_promotion_provenance.py   tests/test_generator_training_data.py -v
```

No 3B model download in CI.

---

## Task 12 — Verification before code-ready verdict

Run fresh:

```bash
pytest tests/test_v8_fail_loud_integration.py -v
pytest tests/test_v8_promotion_provenance.py -v
pytest tests/test_v7_runtime_integrity.py -v
pytest tests/test_v7_promotion_consistency.py -v
pytest tests/ -v
```

Then explicitly search canonical notebook/source:

```bash
grep -R 'resolve_runtime_paths("/kaggle/input", strict=False' -n kaggle_kernel src scripts || true
grep -R 'runtime_api_version: 7' -n configs || true
grep -R 'screen_protocol_version: 7' -n configs src scripts || true
```

Expected:

```text
no canonical strict=False runtime call
runtime API = 8
screen protocol = 8
0 failed tests
```

Commit:

```bash
git add .
git commit -m "fix(kaggle): close v8 fail-loud integration gates"
```

---

## Task 13 — Repackage a new Kaggle runtime dataset version

Mandatory because API changed 7 → 8.

```bash
python scripts/package_kaggle_dataset.py   --source artifacts/task2   --staging kaggle_dataset/staged   --profile final_training
```

Verify:

```python
import json
from pathlib import Path

root = Path("kaggle_dataset/staged")
dataset = json.loads((root / "dataset_manifest.json").read_text())
code = json.loads((root / "code_manifest.json").read_text())

assert dataset["runtime_api_version"] == 8
assert code["runtime_api_version"] == 8
assert dataset["git_sha"] == code["git_sha"]
assert (root / "code/LegalQA/src").is_dir()
assert (root / "code/LegalQA/scripts").is_dir()
assert (root / "indexes/bm25").is_dir()
assert (root / "indexes/dek21").is_dir()
```

Upload a **new version** of:

```text
phucdangg/legalqa-task2-clean-data
```

Do not reuse a V7 package.

---

## Task 14 — Real Kaggle smoke gate

Use:

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
dataset API = 8
code API = 8
dataset/code git SHA match
mounted Qwen resolved
protected Torch/CUDA unchanged
pip check PASS
all required imports PASS
2 CUDA GPUs detected
BM25 strict load PASS
DEk21 strict FP16/GPU load PASS
30 reranker optimizer steps + reload PASS
30 QLoRA optimizer steps + PEFT reload PASS
5 real held-out predictions
no neural fallback
no public submission attempted
```

Only then advance to `screen_fold0`.

---

## Task 15 — Protocol-8 screen and final gate

After smoke succeeds:

```python
EXECUTION_PROFILE = "screen_fold0"
```

Protocol-8 report must prove:

```text
R0G0, R1G0, R_SELECTED_G1
identical sample_ids_sha256/sample_size
selected reranker equals final measured summary
selected adapter equals final measured summary when QLoRA selected
winning candidate exists in final measured summary
winning METEOR equals final measured candidate score
```

Promote only that report. Final/reuse profiles must reject Protocol-7 promoted configs.

---

## Definition — SAFE FOR KAGGLE SMOKE

Coding agent may return exactly:

```text
SAFE FOR KAGGLE SMOKE
```

only after all V8 code gates and all CI tests pass. This means **code-ready to attempt the real GPU smoke**, not that Kaggle GPU execution has already passed.

## Definition — SAFE TO RUN FINAL KAGGLE TRAINING

Do not use this verdict until:

```text
real V8 Dual-T4 smoke passed
real Protocol-8 screen passed
Protocol-8 PROMOTED config exists
exact measured tuple provenance validated
```

---

## Required coding-agent completion report

Return exactly:

```text
HEAD SHA
Changed files
Runtime strictness proof
Manifest/API proof
Bootstrap fail-loud proof
Mounted Qwen proof
Requirements consistency proof
Promotion provenance proof
OOF provenance proof
Focused pytest results
Full pytest result
CI job results
Kaggle GPU evidence (or "not run")
Remaining blockers
Verdict
```

Verdict must be exactly one of:

```text
SAFE FOR KAGGLE SMOKE
NOT SAFE FOR KAGGLE SMOKE
```

---

## Self-review

### Spec coverage

- [x] Canonical notebook switches from `strict=False` to `strict=True`.
- [x] Missing/ambiguous runtime code fails.
- [x] Missing manifests fail.
- [x] Runtime API exact match is enforced.
- [x] Dataset/code Git SHA consistency is enforced.
- [x] Strict Kaggle mode requires mounted Qwen.
- [x] `pip check` failure is fatal.
- [x] Any required runtime import failure is fatal.
- [x] Kaggle requirements match bootstrap floors.
- [x] Runtime API bump forces repackaging.
- [x] Promotion-report copies are byte-identical.
- [x] Promoter proves exact measured tuple/sample/candidate score.
- [x] Direct full-OOF checkpoint provenance is checked.
- [x] Tests inspect actual notebook wiring.
- [x] Real Kaggle GPU smoke remains a separate evidence gate.

### Placeholder scan

No `TODO`, `TBD`, “implement later”, or unspecified validation step remains in this plan.

### Canonical names

Use exactly:

```text
EXPECTED_RUNTIME_API_VERSION
resolve_packaged_code_root
validate_runtime_manifests
allow_remote_model_download
verify_runtime_imports
pip_check_passed
screen_protocol_version
final_measured_system_key
sample_ids_sha256
```
