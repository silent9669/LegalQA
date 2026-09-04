# LegalQA Kaggle V12 — Baseline-Aware `pip check` Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the Kaggle smoke from failing on dependency conflicts that already exist in Kaggle's base image, while still failing loudly if LegalQA's bootstrap introduces any new dependency conflict.

**Architecture:** Replace the current global post-install `pip check == 0` assumption with a baseline/post comparison. Capture Kaggle's pre-existing `pip check` conflicts before installing LegalQA dependencies, run `pip check` again afterward, fail only on newly introduced conflicts, and continue to preserve protected Torch/CUDA versions plus strict LegalQA package/API verification. Bind this packaged runtime change to API 12 and publish a new Kaggle dataset version.

**Tech Stack:** Python 3.12 Kaggle runtime, pip, packaging, pytest, GitHub Actions, Kaggle CLI.

**Spec:** Real manual T4 x2 smoke log from V11 / Git SHA `114ceda55d6c3888273a540f48477a0709c05a05`.

## Global Constraints

- Do not redesign Stack A.
- Do not change BM25, DEk21, reranker, Qwen2.5-3B, QLoRA hyperparameters, candidate selection, scoring, or parameter budget.
- Keep `EXECUTION_PROFILE = "smoke_only"` as the committed notebook default.
- Keep `strict=True` and `allow_remote_model_download=False`.
- Preserve exact protected Torch/CUDA/Triton immutability checks.
- Keep `trl>=0.17.0` and strict TRL API checks.
- Never print or persist `HF_TOKEN`.
- Coding agent must not run Kaggle GPU training; the user runs T4 x2 manually.
- Runtime API for this packaged-code change becomes **12**.

---

## 0. Audited failure and root cause

Latest audited GitHub HEAD:

```text
114ceda55d6c3888273a540f48477a0709c05a05
```

Fresh V11 CI is green:

```text
GitHub Actions run: 33464471713
CPU suite: 157 passed, 2 skipped
Python 3.10 SFT compatibility: PASS
Python 3.12 SFT compatibility: PASS
```

The real Kaggle smoke proves all earlier deployment/runtime gates are fixed:

```text
COMMITTED EXECUTION PROFILE: smoke_only
CUDA GPUs Detected: 2
GPU 0: Tesla T4
GPU 1: Tesla T4
Packaged Code Root resolved
Runtime Root from Code resolved
Verified Runtime Integrity: API v11 | Git SHA: 114ceda55d
Qwen model path resolved
```

Kaggle runtime:

```text
Python 3.12.13
PyTorch 2.10.0+cu128
CUDA 12.8
2 x Tesla T4
```

Bootstrap installs only five missing LegalQA packages successfully:

```text
trl>=0.17.0
bitsandbytes>=0.43.0
bm25s>=0.2.5
pyvi>=0.1.1
fastparquet>=2024.2.0
```

The first real failure is:

```text
RuntimeError: pip check reported broken dependencies:

bigframes requires google-cloud-bigquery-storage
google-adk requires google-cloud-bigquery-storage
google-colab requires jupyter-server==2.14.0
google-colab requires pandas==2.2.2
dopamine-rl requires gym<=0.25.2
moviepy requires decorator<5.0
```

These conflicts are from global Kaggle packages outside the LegalQA target package set. The current bootstrap runs `pip check` only after installation and assumes the whole Kaggle image was clean before LegalQA started. It therefore cannot distinguish a pre-existing base-image conflict from a new conflict introduced by LegalQA. That assumption is the bug.

Do **not** solve this by deleting `pip check`, force-installing Google/Colab packages, downgrading pandas/gym/decorator, or allowing protected Torch/CUDA drift.

---

### Task 1: Add baseline-aware pip conflict collection

**Files:**
- Modify: `scripts/bootstrap_kaggle_env.py`
- Create: `tests/test_v12_pip_check_baseline.py`

**Interfaces:**
- Produces: `collect_pip_check_conflicts() -> list[str]`
- Produces: `assert_no_new_pip_conflicts(before: list[str], after: list[str]) -> list[str]`

- [ ] **Step 1: Write failing tests**

```python
import pytest
import scripts.bootstrap_kaggle_env as bootstrap


def test_same_preexisting_pip_conflicts_are_allowed():
    before = [
        "google-colab 1.0.0 has requirement pandas==2.2.2, but you have pandas 2.3.3.",
        "moviepy 1.0.3 has requirement decorator<5.0,>=4.0.2, but you have decorator 5.3.1.",
    ]
    after = list(reversed(before))

    new = bootstrap.assert_no_new_pip_conflicts(before, after)
    assert new == []


def test_new_pip_conflict_fails_loud():
    before = ["moviepy baseline conflict"]
    after = before + ["trl new conflict"]

    with pytest.raises(RuntimeError, match="new dependency conflict"):
        bootstrap.assert_no_new_pip_conflicts(before, after)


def test_resolved_baseline_conflict_is_allowed():
    before = ["old preexisting conflict"]
    after = []
    assert bootstrap.assert_no_new_pip_conflicts(before, after) == []
```

- [ ] **Step 2: Run and confirm RED**

```bash
pytest tests/test_v12_pip_check_baseline.py -v
```

Expected: FAIL because the functions do not exist.

- [ ] **Step 3: Implement normalized conflict collection**

```python
def collect_pip_check_conflicts() -> List[str]:
    """Return normalized pip-check conflicts without deciding policy."""
    res = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
    )
    text = "\n".join(x for x in [res.stdout, res.stderr] if x)
    return sorted({
        line.strip()
        for line in text.splitlines()
        if line.strip()
    })
```

A `pip check` return code of 1 is expected when conflicts exist. Do not raise here.

- [ ] **Step 4: Implement regression-only policy**

```python
def assert_no_new_pip_conflicts(
    before: List[str],
    after: List[str],
) -> List[str]:
    baseline = set(before)
    post = set(after)
    new_conflicts = sorted(post - baseline)

    if new_conflicts:
        raise RuntimeError(
            "LegalQA bootstrap introduced new dependency conflict(s):\n"
            + "\n".join(f" - {line}" for line in new_conflicts)
        )

    return new_conflicts
```

Exact line comparison is intentional: if a package/version changes and generates a different conflict line, it is treated as a new regression.

- [ ] **Step 5: Re-run focused tests**

```bash
pytest tests/test_v12_pip_check_baseline.py -v
```

Expected: PASS.

---

### Task 2: Integrate baseline/post pip checks into bootstrap

**Files:**
- Modify: `scripts/bootstrap_kaggle_env.py`
- Modify: `tests/test_v8_fail_loud_integration.py`
- Test: `tests/test_v12_pip_check_baseline.py`

- [ ] **Step 1: Add an integration test**

```python
def test_bootstrap_allows_unchanged_base_image_conflicts(monkeypatch):
    snapshots = iter([
        ["preexisting kaggle conflict"],
        ["preexisting kaggle conflict"],
    ])

    monkeypatch.setattr(
        bootstrap,
        "collect_pip_check_conflicts",
        lambda: next(snapshots),
    )
    monkeypatch.setattr(bootstrap, "TARGET_USER_PACKAGES", [])
    monkeypatch.setattr(bootstrap, "snapshot_protected_versions", lambda: {})

    result = bootstrap.bootstrap_dependencies(
        allow_unprotected_drift=True,
    )

    assert result["pip_check_baseline_conflicts"] == [
        "preexisting kaggle conflict"
    ]
    assert result["pip_check_post_conflicts"] == [
        "preexisting kaggle conflict"
    ]
    assert result["pip_check_new_conflicts"] == []
    assert result["pip_check_regression_passed"] is True
```

Add a second integration test with one extra post-install conflict and require `RuntimeError`.

- [ ] **Step 2: Capture baseline before installing anything**

At the start of `bootstrap_dependencies()`, before user-space package installation:

```python
pip_conflicts_before = collect_pip_check_conflicts()

if pip_conflicts_before:
    print(
        f"Pre-existing Kaggle pip conflicts detected: "
        f"{len(pip_conflicts_before)}"
    )
    for line in pip_conflicts_before:
        print(f"  [BASELINE] {line}")
else:
    print("Pre-bootstrap pip check: clean")
```

- [ ] **Step 3: Re-check after installation**

Replace the current unconditional failing `run_pip_check()` with:

```python
pip_conflicts_after = collect_pip_check_conflicts()
new_pip_conflicts = assert_no_new_pip_conflicts(
    pip_conflicts_before,
    pip_conflicts_after,
)

if pip_conflicts_after:
    print(
        "pip check regression guard: PASS "
        f"({len(pip_conflicts_after)} total conflict(s), "
        "0 newly introduced by LegalQA)"
    )
else:
    print("pip check regression guard: PASS (environment clean)")
```

Then preserve protected-package comparison exactly as before.

- [ ] **Step 4: Return provenance**

```python
{
    ...
    "pip_check_baseline_conflicts": pip_conflicts_before,
    "pip_check_post_conflicts": pip_conflicts_after,
    "pip_check_new_conflicts": new_pip_conflicts,
    "pip_check_regression_passed": True,
}
```

- [ ] **Step 5: Replace obsolete V8 test policy**

The old test says any nonzero global `pip check` must abort. Replace it with:
- unchanged baseline conflicts -> PASS;
- resolved baseline conflicts -> PASS;
- newly introduced post-install conflicts -> FAIL.

Do not simply disable pip validation.

---

### Task 3: Re-verify LegalQA dependency floors after installation

**Files:**
- Modify: `scripts/bootstrap_kaggle_env.py`
- Test: `tests/test_v12_pip_check_baseline.py`

Allowing Kaggle baseline conflicts must not weaken LegalQA's own dependency contract.

- [ ] **Step 1: Add helper**

```python
def verify_target_package_versions() -> Dict[str, str]:
    verified: Dict[str, str] = {}
    failures: List[str] = []

    for import_name, specifier, pip_name in TARGET_USER_PACKAGES:
        version = (
            get_installed_distribution_version(pip_name)
            or get_installed_distribution_version(import_name)
        )
        if not satisfies_spec(version, specifier):
            failures.append(
                f"{pip_name}: installed={version!r}, required={specifier}"
            )
        else:
            verified[pip_name] = str(version)

    if failures:
        raise RuntimeError(
            "LegalQA dependency floor verification failed:\n"
            + "\n".join(f" - {x}" for x in failures)
        )

    return verified
```

- [ ] **Step 2: Call it after pip installation**

```python
verified_target_versions = verify_target_package_versions()
```

Include it in the bootstrap result.

- [ ] **Step 3: Keep strict import/API verification**

`verify_runtime_imports(strict=True)` must still fail for missing required packages or missing:
- `SFTConfig.completion_only_loss`
- max-length API
- `SFTTrainer.processing_class`

---

### Task 4: Make bootstrap provenance truthful

**Files:**
- Modify: `scripts/bootstrap_kaggle_env.py`
- Modify: `kaggle_kernel/legalqa_gpu_pipeline.ipynb`
- Test: `tests/test_v12_pip_check_baseline.py`

The old manifest field `"pip_check_passed": true` is misleading when known Kaggle baseline conflicts remain.

- [ ] **Step 1: Update notebook Cell 4**

Change:

```python
bstrap.bootstrap_dependencies()
bstrap.verify_runtime_imports(strict=True)
bstrap.save_bootstrap_manifest()
```

to:

```python
BOOTSTRAP_RESULT = bstrap.bootstrap_dependencies()
bstrap.verify_runtime_imports(strict=True)
bstrap.save_bootstrap_manifest(
    bootstrap_result=BOOTSTRAP_RESULT,
)
```

- [ ] **Step 2: Update manifest writer**

```python
def save_bootstrap_manifest(
    output_path: str = "/kaggle/working/kaggle_environment.json",
    bootstrap_result: Optional[Dict[str, Any]] = None,
) -> None:
```

Persist:

```text
pip_check_regression_passed
pip_check_baseline_conflicts
pip_check_post_conflicts
pip_check_new_conflicts
protected_runtime_unchanged
```

Do not claim global `pip_check_passed=true` if post-install baseline conflicts still exist.

Never include `HF_TOKEN`.

---

### Task 5: Bump runtime release binding to API 12

**Files:**
- Modify: `configs/runtime_api.yaml`
- Modify: `src/task2/runtime_integrity.py`
- Modify: `kaggle_kernel/legalqa_gpu_pipeline.ipynb`
- Modify API-version tests

- [ ] **Step 1: Set config**

```yaml
schema_version: 1
runtime_api_version: 12
stack: "stack_a"
```

- [ ] **Step 2: Set validator**

```python
EXPECTED_RUNTIME_API_VERSION: int = 12
```

- [ ] **Step 3: Set notebook-owned literal**

```python
REQUIRED_RUNTIME_API_VERSION = 12
```

- [ ] **Step 4: Update release-binding tests**

Require:
- stale API 11 package rejected;
- API 12 matching-SHA package accepted;
- 7/8/9/10/11/13 mismatches rejected.

Preserve exact-one packaged code root, manifest-before-resolver order, Git-SHA parity, recursive Kaggle path discovery, strict Qwen mount, and `smoke_only`.

---

### Task 6: Verification and CI

Run:

```bash
pytest tests/test_v12_pip_check_baseline.py -v
pytest tests/test_v11_dependency_contract.py -v
pytest tests/test_v8_fail_loud_integration.py -v
pytest tests/test_v10_runtime_release_binding.py -v
pytest tests/ -v
```

All GitHub Actions jobs must be green on final V12 HEAD.

Report:
- final HEAD;
- clean working tree;
- focused tests;
- full exact pass/skip count;
- Actions run ID;
- Python 3.12 SFT compatibility.

---

### Task 7: Repackage unchanged data/indexes with V12 runtime

Do not rebuild BM25/DEk21 merely because bootstrap policy changed.

```bash
rm -rf kaggle_dataset/staged

python scripts/package_kaggle_dataset.py \
  --source artifacts/task2 \
  --staging kaggle_dataset/staged \
  --profile final_training
```

Verify all three manifests are API 12 and share the final V12 HEAD SHA, and verify the packaged bootstrap contains:

```text
collect_pip_check_conflicts
assert_no_new_pip_conflicts
pip_check_regression_passed
trl>=0.17.0
```

Also require BM25, DEk21 embeddings, public test, reranker pairs, and packaged code.

---

### Task 8: Upload a NEW Kaggle dataset version

Use Kaggle CLI for dataset deployment only.

Target:

```text
phucdangg/legalqa-task2-clean-data
```

Upload a new version and verify the **remote** artifact:
- dataset/root/nested manifests API 12;
- all Git SHAs equal final V12 HEAD;
- baseline-aware pip-check code present;
- TRL floor `>=0.17.0`;
- BM25/DEk21/public test present.

Do not run GPU.

---

### Task 9: Manual smoke handoff

Agent returns exact versions. The user manually runs:

```text
Notebook: newest phucdangg/legalqa-training containing final V12 HEAD
Dataset: new API-12 legalqa-task2-clean-data version
Model: qwen-lm/qwen2.5/transformers/3b-instruct/1
Profile: smoke_only
Accelerator: T4 x2
Internet: On
HF_TOKEN: enabled
```

Then:

```text
Restart Session
Save Version -> Save & Run All
```

Expected next-run bootstrap evidence:

```text
Pre-existing Kaggle pip conflicts detected: <N>
[BASELINE] ...
Installation completed successfully.
pip check regression guard: PASS (... 0 newly introduced by LegalQA)
Protected package integrity check: PASS
TRL SFT API ... PASS
```

Then continue into BM25, DEk21, ~30 reranker optimizer steps + reload, ~30 QLoRA steps + PEFT reload, and 5 real held-out predictions.

---

## Required Agent Completion Report

```text
HEAD SHA
changed files
focused V12 pytest
full pytest
GitHub Actions run ID
runtime API
new Kaggle dataset version
remote API/SHA parity proof
baseline-aware pip-check proof
TRL floor/API proof
exact manual notebook/dataset/model versions
GPU run: NOT RUN BY AGENT
verdict
```

Verdict exactly:

```text
READY FOR MANUAL KAGGLE SMOKE
```

or:

```text
BLOCKED
```

## Self-review

- [x] Root cause is demonstrated by the real T4 x2 log.
- [x] New LegalQA dependency regressions still fail loudly.
- [x] Unrelated Kaggle packages are not mutated to make `pip check` green.
- [x] Torch/CUDA immutability is preserved.
- [x] TRL >=0.17/API verification is preserved.
- [x] Release binding is bumped so stale V11 packages fail loudly.
- [x] GPU execution remains manual.
