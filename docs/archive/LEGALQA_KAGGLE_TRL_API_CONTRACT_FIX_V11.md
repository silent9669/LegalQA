# LegalQA Kaggle V11 — TRL API Contract & Kaggle Python 3.12 Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the packaged Kaggle runtime guarantee the exact TRL APIs used by LegalQA QLoRA training, prove the minimum supported dependency contract on Python 3.12, and release a new runtime package that the user can manually smoke-test on T4 x2.

**Architecture:** Keep Stack A, data, retrieval, reranker, QLoRA hyperparameters, evaluation, and promotion logic unchanged. Fix only the runtime dependency/release contract: raise the TRL floor to the first verified API-compatible family, test that floor instead of only testing latest TRL, bump the packaged runtime API to 11, and republish the Kaggle dataset from one clean final HEAD.

**Tech Stack:** Python 3.12 Kaggle runtime, Python 3.10/3.12 GitHub Actions, TRL SFTTrainer/SFTConfig, Transformers, PEFT, bitsandbytes, pytest, Kaggle CLI.

**Spec:** Real V10 Kaggle runtime contract at HEAD `62d0daf6434e773bcd0deaf1eebae8daf475d012`, uploaded Dataset V6 extracts, and official TRL versioned documentation.

## Global Constraints

- Do not redesign Stack A.
- Do not change BM25, DEk21, RRF, reranker model, Qwen model, candidate policy, training hyperparameters, or scoring.
- Learned parameter budget remains strictly `< 4,000,000,000`.
- Keep `EXECUTION_PROFILE = "smoke_only"` as the committed notebook default.
- Keep `strict=True` and `allow_remote_model_download=False` for mounted Qwen resolution.
- Do not print or persist `HF_TOKEN`.
- Coding agent MUST NOT run Kaggle GPU training. The user will manually run T4 x2.
- Data tables and retrieval indexes are unchanged; only repackage them with the final V11 runtime.
- Runtime API for this release becomes **11**.

---

## 0. Audited baseline and root cause

Latest audited GitHub HEAD:

```text
62d0daf6434e773bcd0deaf1eebae8daf475d012
```

Current V10 release identity from uploaded code manifests:

```text
runtime_api_version = 10
git_sha = 62d0daf6434e773bcd0deaf1eebae8daf475d012
```

The uploaded extracts are structurally sane:

```text
qa_unique.parquet              7,500 rows, 0 nulls in required flat columns
fold_assignments.parquet       7,500 rows, folds 0..4, 0 nulls
qa_citations.parquet          12,499 rows
retrieval_labels.parquet       6,399 rows, 8..15 negatives
reranker_training_pairs       23,844 rows, 0 nulls in required flat columns
public-official.json           1,000 unique IDs/questions, all answers null
```

Both uploaded code manifests are byte-identical and list 53 packaged code/config files. Uploaded `runtime_api.yaml`, `task2.yaml`, `experiments.yaml`, and `requirements-kaggle.txt` match the SHA256 entries recorded in the manifest.

### Concrete remaining blocker

Current runtime declares:

```text
trl>=0.11.0
```

in both:

```text
requirements-kaggle.txt
scripts/bootstrap_kaggle_env.py
```

But LegalQA training hard-requires:

```python
"completion_only_loss" in inspect.signature(SFTConfig).parameters
"processing_class" in inspect.signature(SFTTrainer).parameters
```

and `run_qlora_training()` passes:

```python
SFTConfig(completion_only_loss=True, ...)
SFTTrainer(..., processing_class=tokenizer, ...)
```

Official TRL v0.11.0 and v0.15.2 documentation still documents the legacy `DataCollatorForCompletionOnlyLM` completion-only flow and does not provide the required `SFTConfig.completion_only_loss` contract. Official TRL v0.17.0 documents both `completion_only_loss` and `SFTTrainer.processing_class`.

Therefore `trl>=0.11.0` is not a valid minimum dependency for this code.

The current bootstrap can silently accept an installed TRL 0.11–0.16 because the version satisfies the declared floor, skip upgrading it, and only fail later during `verify_runtime_imports(strict=True)`.

The current CI masks this defect because the compatibility job runs:

```bash
pip install transformers peft trl bitsandbytes accelerate
```

without a version pin, so it proves only that the **latest** TRL works, not that the declared minimum works.

---

## Task 1 — Raise the canonical TRL runtime floor

**Files:**
- Modify: `requirements-kaggle.txt`
- Modify: `scripts/bootstrap_kaggle_env.py`
- Modify: `requirements.txt` comment for documentation consistency
- Test: create `tests/test_v11_dependency_contract.py`

**Interfaces:**
- Consumes: existing `TARGET_USER_PACKAGES`.
- Produces: one canonical TRL floor, `>=0.17.0`.

- [ ] **Step 1: Write a failing contract test.**

```python
from pathlib import Path
from scripts.bootstrap_kaggle_env import TARGET_USER_PACKAGES


def test_trl_floor_guarantees_modern_sft_api():
    req = Path("requirements-kaggle.txt").read_text(encoding="utf-8")
    assert "trl>=0.17.0" in req

    target = {
        pip_name: spec
        for _, spec, pip_name in TARGET_USER_PACKAGES
    }
    assert target["trl"] == ">=0.17.0"
```

- [ ] **Step 2: Run it before the fix.**

```bash
pytest tests/test_v11_dependency_contract.py::test_trl_floor_guarantees_modern_sft_api -v
```

Expected on V10: FAIL because the current floor is `>=0.11.0`.

- [ ] **Step 3: Apply the minimal floor change.**

`requirements-kaggle.txt`:

```text
trl>=0.17.0
```

`scripts/bootstrap_kaggle_env.py`:

```python
("trl", ">=0.17.0", "trl"),
```

`requirements.txt` commented GPU dependency documentation:

```text
# trl>=0.17.0
```

Do not change unrelated package floors in this task.

- [ ] **Step 4: Re-run the focused test.**

```bash
pytest tests/test_v11_dependency_contract.py::test_trl_floor_guarantees_modern_sft_api -v
```

Expected: PASS.

---

## Task 2 — Verify all required TRL APIs, not one parameter only

**Files:**
- Modify: `scripts/bootstrap_kaggle_env.py`
- Test: `tests/test_v11_dependency_contract.py`

**Interfaces:**
- Produces: `verify_trl_sft_api()` or equivalent explicit checks used by `verify_runtime_imports()`.

- [ ] **Step 1: Add a failing source/API contract test.**

The runtime verifier must assert all three requirements:

```text
SFTConfig.completion_only_loss
SFTConfig.max_length OR SFTConfig.max_seq_length
SFTTrainer.processing_class
```

Example test:

```python
import inspect
from trl import SFTConfig, SFTTrainer


def test_installed_trl_exposes_required_legalqa_api():
    config_sig = inspect.signature(SFTConfig)
    trainer_sig = inspect.signature(SFTTrainer)

    assert "completion_only_loss" in config_sig.parameters
    assert (
        "max_length" in config_sig.parameters
        or "max_seq_length" in config_sig.parameters
    )
    assert "processing_class" in trainer_sig.parameters
```

- [ ] **Step 2: Update `verify_runtime_imports(strict=True)`.**

Replace the single TRL parameter check with one explicit compatibility block:

```python
from trl import SFTConfig, SFTTrainer

config_sig = inspect.signature(SFTConfig)
trainer_sig = inspect.signature(SFTTrainer)

missing = []
if "completion_only_loss" not in config_sig.parameters:
    missing.append("SFTConfig.completion_only_loss")
if not (
    "max_length" in config_sig.parameters
    or "max_seq_length" in config_sig.parameters
):
    missing.append("SFTConfig.max_length/max_seq_length")
if "processing_class" not in trainer_sig.parameters:
    missing.append("SFTTrainer.processing_class")

if missing:
    failures.append(
        "Installed TRL lacks LegalQA-required SFT APIs: "
        + ", ".join(missing)
        + ". Require trl>=0.17.0."
    )
```

Do not silently fall back to the obsolete data collator.

- [ ] **Step 3: Keep fail-loud semantics.**

If the required API is missing after bootstrap:

```text
RuntimeError
```

must still be raised before training starts.

---

## Task 3 — Make CI prove the declared minimum API contract

**Files:**
- Modify: `.github/workflows/tests.yml`
- Test: GitHub Actions `training-api-compat`

The existing job installs unpinned latest TRL, which cannot validate a minimum version.

- [ ] **Step 1: Add a minimum-contract Python 3.12 compatibility lane.**

Use Python 3.12 because the real Kaggle runtime observed in smoke logs is Python 3.12.

A minimal approach:

```yaml
training-api-compat:
  name: SFT Training API & Module Compatibility
  runs-on: ubuntu-latest
  strategy:
    matrix:
      python-version: ["3.10", "3.12"]
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install minimum supported TRL contract
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pyyaml pytest packaging
        pip install "trl==0.17.0" transformers peft bitsandbytes accelerate
```

- [ ] **Step 2: Verify both required signatures.**

```bash
python - <<'PY'
import inspect
from trl import SFTConfig, SFTTrainer

sc = inspect.signature(SFTConfig)
st = inspect.signature(SFTTrainer)
assert "completion_only_loss" in sc.parameters
assert "max_length" in sc.parameters or "max_seq_length" in sc.parameters
assert "processing_class" in st.parameters
print("MINIMUM TRL API CONTRACT PASS")
PY
```

- [ ] **Step 3: Keep the training-data tests.**

Run:

```bash
pytest \
  tests/test_generator_training_data.py \
  tests/test_v11_dependency_contract.py \
  -v
```

If `trl==0.17.0` proves incompatible with Python 3.12 or the current Transformers/PEFT stack, do not weaken the API checks. Determine the earliest version that passes both required signatures on Python 3.12, update the canonical floor to that exact verified minimum, and record the evidence.

---

## Task 4 — Bump the runtime release to API 11

**Files:**
- Modify: `configs/runtime_api.yaml`
- Modify: `src/task2/runtime_integrity.py`
- Modify: `kaggle_kernel/legalqa_gpu_pipeline.ipynb`
- Modify explicit API-version tests
- Test: `tests/test_v10_runtime_release_binding.py`
- Test: `tests/test_v11_dependency_contract.py`

This packaged-code change must be distinguishable from Dataset V6/V10.

- [ ] **Step 1: Update runtime config.**

```yaml
schema_version: 1
runtime_api_version: 11
stack: "stack_a"
```

- [ ] **Step 2: Update runtime validator.**

```python
EXPECTED_RUNTIME_API_VERSION: int = 11
```

- [ ] **Step 3: Update notebook-owned release binding.**

```python
REQUIRED_RUNTIME_API_VERSION = 11
```

Keep:

```python
strict=True
allow_remote_model_download=False
```

and preserve manifest validation before `resolve_runtime_paths()`.

- [ ] **Step 4: Add stale-V10 rejection regression.**

A package with:

```json
{
  "runtime_api_version": 10,
  "git_sha": "<valid matching SHA>"
}
```

must fail when the notebook requires API 11.

A matching API-11 package must pass.

---

## Task 5 — Preserve all V10 runtime/data guarantees

**Files:**
- Test only unless regression exposes a defect.

Re-run and keep green:

```bash
pytest tests/test_path_resolver_kaggle_nested.py -v
pytest tests/test_v10_runtime_release_binding.py -v
pytest tests/test_v9_notebook_contract.py -v
pytest tests/test_v8_fail_loud_integration.py -v
```

The V11 patch must not alter:

```text
recursive /kaggle/input/datasets/<owner>/<slug> discovery
followlinks=False
exact-one code root
manifest-before-resolver order
dataset/code Git SHA equality
real 40-char lowercase SHA requirement
mounted Qwen requirement
no local code fallback
```

---

## Task 6 — Run complete CPU verification

Run:

```bash
pytest tests/test_v11_dependency_contract.py -v
pytest tests/ -v
```

Then wait for GitHub Actions on the final V11 HEAD.

Completion evidence must include:

```text
final HEAD SHA
working tree clean
full pytest exact pass/skip count
workflow run ID
Python 3.10 training-api lane PASS
Python 3.12 training-api lane PASS
minimum TRL contract PASS
```

Do not claim V11 is ready based only on local tests.

---

## Task 7 — Repackage the unchanged data with V11 runtime code

The data/index contents do not need to be rebuilt unless package self-validation reports missing artifacts.

From a clean final HEAD:

```bash
git status --short
git rev-parse HEAD

python scripts/package_kaggle_dataset.py \
  --source artifacts/task2 \
  --staging kaggle_dataset/staged \
  --profile final_training
```

Verify locally:

```python
import json
import re
from pathlib import Path

root = Path("kaggle_dataset/staged")
ds = json.loads((root / "dataset_manifest.json").read_text())
code = json.loads((root / "code_manifest.json").read_text())
nested = json.loads((root / "code/LegalQA/code_manifest.json").read_text())

assert ds["runtime_api_version"] == 11
assert code["runtime_api_version"] == 11
assert nested["runtime_api_version"] == 11

assert ds["git_sha"] == code["git_sha"] == nested["git_sha"]
assert re.fullmatch(r"[0-9a-f]{40}", ds["git_sha"])

req = (root / "code/LegalQA/requirements-kaggle.txt").read_text()
assert "trl>=0.17.0" in req

assert (root / "indexes/bm25").is_dir()
assert (root / "indexes/dek21/embeddings.npy").is_file()
assert (root / "public-official.json").is_file()

print("V11 PACKAGE CONTRACT PASS")
```

---

## Task 8 — Upload a NEW Kaggle dataset version using CLI

The coding agent may use Kaggle CLI for dataset deployment only.

Upload a new version of:

```text
phucdangg/legalqa-task2-clean-data
```

Do **not** run the Kaggle GPU notebook.

After upload, download or inspect the uploaded Kaggle artifact itself and prove:

```text
runtime API = 11
dataset/code/nested-code Git SHA = final V11 HEAD
packaged requirements-kaggle.txt contains trl>=0.17.0
packaged bootstrap uses the same TRL floor
recursive path_resolver is present
BM25 present
DEk21 present
public-official.json present
```

Record the new Kaggle dataset version number.

---

## Task 9 — Prepare the notebook for the user's manual smoke

The coding agent must leave:

```python
EXECUTION_PROFILE = "smoke_only"
```

and must not start GPU execution.

Return exact manual attachment instructions:

```text
Notebook: newest phucdangg/legalqa-training version containing final V11 HEAD
Dataset: new API-11 phucdangg/legalqa-task2-clean-data version
Model: qwen-lm/qwen2.5/transformers/3b-instruct/1
Accelerator: T4 x2
Internet: On
HF_TOKEN: enabled
```

The user will manually choose T4 x2 and run:

```text
Restart Session
Save Version -> Save & Run All
```

---

## Manual smoke acceptance

Do not call the smoke passed until the user's real Kaggle log proves, in one run:

```text
COMMITTED EXECUTION PROFILE: smoke_only
CUDA GPUs Detected: 2
Runtime Release API: 11
dataset/code Git SHA = final V11 HEAD
dependency bootstrap PASS
TRL required API checks PASS
pip check PASS
protected Torch/CUDA unchanged
BM25 load PASS
DEk21 load PASS
30 reranker optimizer steps
reranker reload PASS
30 QLoRA optimizer steps
PEFT reload PASS
5 real held-out predictions
no mock/fallback
```

If it fails, diagnose only the first real failure from that log.

---

## Required agent completion report

Return:

```text
HEAD SHA
changed files
TRL minimum chosen and evidence
focused pytest result
full pytest result
GitHub Actions run ID
Python 3.12 compatibility result
new Kaggle dataset version
uploaded API-11 proof
uploaded SHA parity proof
uploaded TRL floor proof
exact manual Kaggle attachment versions
GPU run: NOT RUN BY AGENT
verdict
```

Verdict must be exactly one of:

```text
READY FOR MANUAL KAGGLE SMOKE
BLOCKED
```

## Self-review

- [x] Fixes a demonstrated dependency-contract defect.
- [x] Does not alter model/data architecture.
- [x] Makes CI test the minimum supported API instead of latest only.
- [x] Adds Python 3.12 compatibility coverage matching Kaggle.
- [x] Uses runtime API 11 so stale V10 packages fail loudly.
- [x] Requires a new Kaggle dataset version.
- [x] Keeps GPU execution manual, per user workflow.
