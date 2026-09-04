# LegalQA V12 — QLoRA Transformers 5 Async-Load OOM Hotfix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing API-12 Kaggle smoke load Qwen2.5-3B in 4-bit on a 14.6-GB T4 without the Transformers 5.0.0 async-loader memory spike.

**Architecture:** Keep the current packaged API-12 dataset and code unchanged. The real failure is in the Kaggle notebook execution environment: preinstalled Transformers 5.0.0 uses threaded/async tensor materialization while the Qwen model is being converted to bitsandbytes 4-bit, producing a transient GPU-0 OOM. Set the documented `HF_DEACTIVATE_ASYNC_LOAD=1` escape hatch in Cell 1 before any Transformers/model loading. Do not pin/downgrade Transformers, change QLoRA hyperparameters, or repackage the 2.4-GB dataset for this hotfix.

**Tech Stack:** Kaggle Python 3.12, Transformers 5.0.0, bitsandbytes 0.50.2, Qwen2.5-3B-Instruct, T4 x2.

**Spec:** Real API-12 T4 x2 smoke log `legalqa-training-2.log`, GitHub HEAD `a6692324a47052c835386192dfa78b21e8bf614c`, Transformers v5.0.0 loader behavior.

## Global Constraints

- Do not create API 13.
- Keep Runtime API **12**.
- Do not rebuild or re-upload the V12 dataset.
- Keep the same BM25, DEk21, reranker, Qwen model, QLoRA rank, sequence length, batch size, gradient accumulation, optimizer, scoring, and parameter budget.
- Keep `EXECUTION_PROFILE = "smoke_only"`.
- Keep `strict=True`.
- Keep `trl>=0.17.0`.
- Do not set `PYTORCH_CUDA_ALLOC_CONF` as the primary fix; the observed failure is not allocator fragmentation.
- Do not run Kaggle GPU training; the user runs T4 x2 manually.
- Never print or persist `HF_TOKEN`.

---

## 0. Proven failure

The latest real smoke successfully passes:

```text
Runtime API 12 / SHA a6692324...
2 x Tesla T4
baseline-aware pip regression guard
protected CUDA/Torch integrity
TRL SFT API verification
parameter-budget preflight
BM25 validation and mmap load
DEk21 validation and CUDA probe
reranker 30-step smoke training
reranker checkpoint reload
```

The first failure occurs only when loading Qwen for QLoRA:

```text
AutoModelForCausalLM.from_pretrained(...)
...
core_model_loading.py
Future.result()
_materialize_copy(...)
tensor.to(device=device, dtype=dtype)
...
OutOfMemoryError
```

Observed failure state:

```text
GPU 0 total: 14.56 GiB
GPU 0 free:  28.81 MiB
process GPU memory: 14.53 GiB
PyTorch allocated: 3.44 GiB
PyTorch reserved but unallocated: 41.19 MiB
```

The tiny unallocated reserve shows fragmentation is not the main problem.

Current Kaggle package already uses correct 4-bit QLoRA:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)

AutoModelForCausalLM.from_pretrained(
    ...,
    quantization_config=bnb_config,
    device_map={"": "cuda:0"},
)
```

Transformers **v5.0.0** creates a `ThreadPoolExecutor` unless either:

```text
HF_DEACTIVATE_ASYNC_LOAD=true
```

or disk offload is in the device map. It does not yet disable async loading for on-the-fly quantization.

Newer Transformers source explicitly disables threaded loading when on-the-fly quantization is active because otherwise tensors can be copied to GPU faster than the quantizer can consume them, causing a large memory spike.

Therefore the smallest deterministic fix for the actual Kaggle 5.0.0 environment is:

```text
HF_DEACTIVATE_ASYNC_LOAD=1
```

before model loading.

---

### Task 1 — Add a failing notebook contract test

**Files:**
- Modify: `tests/test_v9_notebook_contract.py`
- Modify: `kaggle_kernel/legalqa_gpu_pipeline.ipynb`

- [ ] Add a test that loads the canonical notebook and asserts:

```python
def test_notebook_disables_transformers_v5_async_model_loading():
    # Read notebook cells as source text.
    # Require this exact assignment in Cell 1:
    expected = 'os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"'
    assert expected in cell1_source

    # Cell 1 must execute before Cell 2/import-heavy bootstrap/model code.
    assert cell1_index < cell2_index
```

Also assert the committed profile and API remain:

```text
EXECUTION_PROFILE = "smoke_only"
REQUIRED_RUNTIME_API_VERSION = 12
```

- [ ] Run the focused test before editing the notebook.

```bash
pytest tests/test_v9_notebook_contract.py -v
```

Expected: new test FAILS.

---

### Task 2 — Apply the minimal notebook hotfix

**File:**
- Modify: `kaggle_kernel/legalqa_gpu_pipeline.ipynb`

In **Cell 1**, immediately after:

```python
import os, sys
```

add:

```python
# Transformers 5.0.0 uses async tensor materialization by default.
# On T4 + on-the-fly bitsandbytes 4-bit loading this can transiently
# materialize weights onto GPU faster than quantization consumes them,
# causing an OOM before QLoRA training starts.
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
print("Transformers async model loading: DISABLED for T4-safe QLoRA load")
```

Do not add a second model copy, CPU offload, a lower `max_seq_len`, or a smaller LoRA rank in this task.

Do not modify:

```text
src/task2/training/train_generator.py
configs/runtime_api.yaml
src/task2/runtime_integrity.py
requirements-kaggle.txt
dataset manifests
```

This is intentionally notebook-only so the existing V12 dataset remains valid.

---

### Task 3 — Add a source-level compatibility explanation test

**Files:**
- Modify: `tests/test_v9_notebook_contract.py`

Require that the notebook does **not** enable the opposite setting:

```python
assert 'HF_DEACTIVATE_ASYNC_LOAD"] = "0"' not in notebook_source
```

Require no accidental API bump:

```python
assert "REQUIRED_RUNTIME_API_VERSION = 12" in notebook_source
assert "REQUIRED_RUNTIME_API_VERSION = 13" not in notebook_source
```

Run:

```bash
pytest tests/test_v9_notebook_contract.py -v
```

Expected: PASS.

---

### Task 4 — Run full CPU/CI verification

Run:

```bash
pytest tests/ -v
```

Then push the notebook-only hotfix and wait for GitHub Actions.

Report:

```text
new Git HEAD
changed files
full pytest exact pass/skip count
GitHub Actions run ID
all jobs PASS
```

Do not claim GPU success from CI.

---

### Task 5 — Push a new Kaggle NOTEBOOK version only

The agent may use Kaggle CLI to upload the updated notebook but MUST NOT execute GPU training.

Keep attached inputs unchanged:

```text
Dataset:
phucdangg/legalqa-task2-clean-data
existing complete API-12 V12 version

Model:
qwen-lm/qwen2.5/transformers/3b-instruct/1
```

Do **not** rebuild or upload another dataset version.

Verify the pushed notebook contains:

```text
EXECUTION_PROFILE = "smoke_only"
REQUIRED_RUNTIME_API_VERSION = 12
HF_DEACTIVATE_ASYNC_LOAD = 1
```

Return the exact new Kaggle notebook version number.

---

### Task 6 — Human T4 x2 smoke

The user manually runs:

```text
Notebook: new notebook-only hotfix version
Dataset: same verified complete API-12 dataset
Model: qwen-lm/qwen2.5/transformers/3b-instruct/1
Accelerator: T4 x2
Internet: On
HF_TOKEN: enabled
Profile: smoke_only
```

Then:

```text
Restart Session
Save Version -> Save & Run All
```

Expected new log evidence before QLoRA:

```text
Transformers async model loading: DISABLED for T4-safe QLoRA load
```

All prior gates should still pass.

At QLoRA, the model should load sequentially rather than via the failing threaded `Future.result()` path.

The smoke is not complete until it proves:

```text
QLoRA model load PASS
30 generator optimizer steps
generator adapter saved
strict PEFT adapter reload PASS
5 held-out predictions/evaluation
```

If a new OOM occurs **after** successful model loading during forward/backward training, that is a different memory problem and must be diagnosed from that new log. Do not preemptively reduce sequence length or rank now.

---

## Secondary non-blocking observations from this run

Do not bundle these into the async-load hotfix:

1. Reranker emitted:

```text
lr_scheduler.step() before optimizer.step()
```

Training still completed and checkpoint reload passed. Audit this separately before final full training because it can skip the first LR schedule value.

2. Dense mmap emitted a non-writable NumPy warning. The probe succeeded; this is not the current failure.

3. `torch_dtype` is deprecated in Transformers 5 in favor of `dtype`. This is not the OOM root cause.

---

## Required Agent Completion Report

```text
HEAD SHA
changed files
notebook contract test result
full pytest result
GitHub Actions run ID
new Kaggle notebook version
dataset version: UNCHANGED
runtime API: 12
HF_DEACTIVATE_ASYNC_LOAD=1 proof
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

- [x] Root cause is evidenced by the real threaded loading stack.
- [x] Matches Transformers v5.0.0 loader behavior.
- [x] Uses the documented synchronous-loading escape hatch.
- [x] Does not change QLoRA hyperparameters to hide the issue.
- [x] Does not touch the already-verified 2.4-GB V12 dataset.
- [x] Does not create unnecessary API 13.
- [x] Keeps GPU execution manual.
