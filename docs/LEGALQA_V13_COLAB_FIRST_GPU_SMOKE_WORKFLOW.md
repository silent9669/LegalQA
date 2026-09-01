# LegalQA V13 — Colab-First GPU Smoke Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace repeated Kaggle smoke runs with a reproducible single-T4 Google Colab GPU gate, while keeping Kaggle notebook execution exclusively manual and preserving one final dual-T4 Kaggle integration gate.

**Architecture:** GitHub CI remains the first gate. After CI is green, use a dedicated Colab smoke harness to exercise the GPU-sensitive reranker/QLoRA code on one T4 with user-space library versions pinned to the last observed Kaggle runtime. The canonical Kaggle notebook remains strict dual-T4 and is never pushed or executed by the coding agent. Only after Colab passes does the user manually run the Kaggle T4x2 notebook.

**Tech Stack:** GitHub Actions, Google Colab single Tesla T4, Python 3.12-compatible code, Transformers 5.0.0, TRL 1.12.0, PEFT 0.19.1, bitsandbytes 0.50.2, Qwen2.5-3B-Instruct.

**Spec:** Current V13 release `f6e4583211d321f40ec6074a40fc05ba07b58d0f` plus the user's permanent operating rule that only the user configures/runs Kaggle.

## Global Constraints

- **Never push, update, trigger, execute, or save a Kaggle notebook via CLI/API.**
- The user alone configures Kaggle accelerator, attaches inputs, and runs the notebook.
- Coding agent may push source/config/test/docs to GitHub and monitor CI.
- Coding agent may use the already-configured browser/Colab MCP to run Google Colab.
- Kaggle canonical notebook remains `EXECUTION_PROFILE = "smoke_only"` and `ALLOW_SINGLE_GPU_SMOKE = False`.
- Colab single-GPU permission must be isolated to the Colab harness; do not change the committed Kaggle default.
- Keep Runtime API **13**. This workflow-only addition does not require API 14.
- Do not repackage/re-upload the V13 dataset solely for this workflow change.
- Keep `HF_DEACTIVATE_ASYNC_LOAD=1`.
- Never print, persist, commit, or echo `HF_TOKEN`.
- Do not force-install/replace Colab's Torch/CUDA stack.
- QLoRA semantics remain: Qwen2.5-3B, 4-bit NF4, LoRA r16/alpha32, max_seq_len 2048, batch 1, grad_accum 8.
- Colab is a component GPU gate, not proof of Kaggle's dual-T4 topology. A final manual Kaggle run remains required.

---

## 0. Audited starting state

Current GitHub HEAD:

```text
f6e4583211d321f40ec6074a40fc05ba07b58d0f
fix(kaggle): enforce single-GPU QLoRA Trainer policy and bump release binding to API 13
```

Current Actions run:

```text
33489612822
completed / success
176 passed, 2 skipped
```

V13 generator code already enforces:

```text
generator target = cuda:0
args._n_gpu = 1
trainer.args.n_gpu == 1
```

and logs:

```text
QLoRA Trainer GPU Policy:
target=cuda:0 | visible_cuda=<N> | trainer_n_gpu=1
```

The canonical notebook remains:

```text
REQUIRED_RUNTIME_API_VERSION = 13
EXECUTION_PROFILE = "smoke_only"
ALLOW_SINGLE_GPU_SMOKE = False
HF_DEACTIVATE_ASYNC_LOAD = 1
```

The supplied complete Drive package is V13:

```text
runtime_api_version = 13
git_sha = f6e4583211d321f40ec6074a40fc05ba07b58d0f
BM25 = 7/7 files
DEk21 = 3/3 files
TRL floor = >=0.17.0
```

The packaged `train_generator.py` SHA256 matches its own code manifest:

```text
3430577c3833baad50aa2efa14ed34e5e97a3e912eb14d7cd96dd4c06eb414ea
```

No code/data blocker is known before the next GPU test.

---

### Task 1 — Add an exact Kaggle-user-space Colab runtime lock

**Files:**
- Create: `requirements-colab-smoke.txt`
- Create: `tests/test_colab_smoke_contract.py`

**Interfaces:**
- Consumes the last real Kaggle T4 log.
- Produces a user-space compatibility lock for Colab.
- Must not include Torch/CUDA packages.

- [ ] **Step 1: Create the lock**

```text
transformers==5.0.0
accelerate==1.13.0
datasets==5.0.0
peft==0.19.1
trl==1.12.0
bitsandbytes==0.50.2
sentence-transformers==5.4.1
bm25s==0.3.11
scikit-learn==1.6.1
nltk==3.9.1
pyvi==0.1.1
pyyaml==6.0.3
pyarrow==24.0.0
fastparquet>=2024.2.0
tqdm>=4.67.0
```

Do not include:

```text
torch
torchvision
torchaudio
triton
cuda-*
nvidia-*
```

- [ ] **Step 2: Add a test**

```python
from pathlib import Path


def test_colab_lock_never_replaces_torch_cuda():
    txt = Path("requirements-colab-smoke.txt").read_text()
    forbidden = [
        "\ntorch==", "\ntorch>=", "\ntorchvision",
        "\ntorchaudio", "\ntriton", "\ncuda-", "\nnvidia-",
    ]
    assert all(x not in "\n" + txt for x in forbidden)
    assert "transformers==5.0.0" in txt
    assert "trl==1.12.0" in txt
    assert "bitsandbytes==0.50.2" in txt
```

---

### Task 2 — Add a reusable single-T4 Colab smoke runner

**Files:**
- Create: `scripts/run_colab_smoke.py`
- Test: `tests/test_colab_smoke_contract.py`

**Interfaces:**
- CLI:
  - `--data-root PATH`
  - `--component generator|reranker|all`
  - `--mode quick|full`
  - `--model-name Qwen/Qwen2.5-3B-Instruct`
  - `--output-dir PATH`
- Produces `colab_smoke_report.json`.

The runner must call existing training functions directly. It must **not** invoke the Kaggle notebook or Kaggle path resolver.

- [ ] **Step 1: Add CLI skeleton**

```python
import argparse
import json
import os
import subprocess
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--component", choices=["generator", "reranker", "all"], default="generator")
    p.add_argument("--mode", choices=["quick", "full"], default="quick")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--output-dir", default="/content/legalqa_colab_smoke")
    return p.parse_args()
```

- [ ] **Step 2: Set the async-load guard before Transformers imports**

At file top, immediately after importing `os`:

```python
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
```

- [ ] **Step 3: Validate hardware**

```python
import torch

if not torch.cuda.is_available():
    raise RuntimeError("COLAB_SMOKE_ERROR: CUDA is unavailable.")

if torch.cuda.device_count() != 1:
    raise RuntimeError(
        f"COLAB_SMOKE_ERROR: expected exactly one Colab GPU, "
        f"found {torch.cuda.device_count()}."
    )

gpu_name = torch.cuda.get_device_name(0)
if "T4" not in gpu_name:
    raise RuntimeError(
        f"COLAB_SMOKE_ERROR: this gate requires Tesla T4 for Kaggle-like "
        f"memory validation, got {gpu_name!r}."
    )

print(f"COLAB GPU: {gpu_name}")
```

- [ ] **Step 4: Validate minimum data files**

Generator requires:

```text
qa_unique.parquet
retrieval_labels.parquet
legal_chunks.parquet
```

Reranker requires:

```text
reranker_training_pairs.parquet
```

Do not require BM25 or DEk21 for this component smoke.

- [ ] **Step 5: Use bounded modes**

```python
if args.mode == "quick":
    generator_steps = 3
    generator_examples = 32
    reranker_steps = 3
    reranker_pairs = 64
else:
    generator_steps = 30
    generator_examples = 128
    reranker_steps = 30
    reranker_pairs = 256
```

For the current V13 verification, run `generator` first. Do not spend reranker time unless generator passes.

- [ ] **Step 6: Call existing QLoRA training**

```python
from src.task2.training.train_generator import run_qlora_training

res = run_qlora_training(
    model_name_or_path=args.model_name,
    base_model_id="Qwen/Qwen2.5-3B-Instruct",
    qa_path=str(data_root / "qa_unique.parquet"),
    labels_path=str(data_root / "retrieval_labels.parquet"),
    chunks_path=str(data_root / "legal_chunks.parquet"),
    output_dir=str(out / "generator"),
    epochs=1,
    batch_size=1,
    grad_accum=8,
    lr=1e-4,
    max_seq_len=2048,
    val_fold=0,
    device="cuda:0",
    max_steps=generator_steps,
    max_train_examples=generator_examples,
    is_final_checkpoint=False,
    fail_on_error=True,
)
```

The existing function already performs adapter save + strict PEFT reload + nonempty generation.

- [ ] **Step 7: Optional reranker smoke**

```python
from src.task2.training.train_reranker import train_bge_reranker

res_reranker = train_bge_reranker(
    pairs_path=str(data_root / "reranker_training_pairs.parquet"),
    output_dir=str(out / "reranker"),
    model_name="BAAI/bge-reranker-v2-m3",
    epochs=1,
    batch_size=2,
    grad_accum=4,
    lr=2e-5,
    val_fold=0,
    device="cuda:0",
    max_length=384,
    max_steps=reranker_steps,
    max_train_pairs=reranker_pairs,
    max_val_pairs=128,
    is_final_checkpoint=False,
    fail_on_error=True,
)
```

- [ ] **Step 8: Write a report**

Report only non-secret metadata:

```json
{
  "git_sha": "...",
  "gpu": "Tesla T4",
  "cuda_device_count": 1,
  "component": "generator",
  "mode": "full",
  "generator_steps": 30,
  "generator_status": "completed",
  "peak_vram_mb": 0,
  "adapter_reload": "pass",
  "status": "PASS"
}
```

Use `git rev-parse HEAD` for SHA. Never serialize environment variables.

---

### Task 3 — Add source-level Colab workflow tests

**Files:**
- Modify: `tests/test_colab_smoke_contract.py`

- [ ] Require the runner to contain:

```text
HF_DEACTIVATE_ASYNC_LOAD
device="cuda:0"
max_seq_len=2048
grad_accum=8
fail_on_error=True
```

- [ ] Require it not to contain:

```text
kaggle kernels push
kaggle kernels output
kaggle kernels status
CUDA_VISIBLE_DEVICES
ALLOW_SINGLE_GPU_SMOKE = True
```

The Colab runner must not mutate the canonical notebook setting.

---

### Task 4 — Document the exact browser/MCP Colab procedure

**Files:**
- Create: `docs/COLAB_FIRST_SMOKE_WORKFLOW.md`

Document this sequence:

```text
1. Push source to GitHub.
2. Wait until all GitHub Actions jobs are green.
3. Open Google Colab with the configured browser MCP.
4. Runtime -> Change runtime type -> T4 GPU.
5. Clone exact GitHub HEAD.
6. Install requirements-colab-smoke.txt WITHOUT Torch/CUDA replacement.
7. Provide a data root containing the 3 generator files
   (plus reranker_training_pairs.parquet only when reranker smoke is needed).
8. Run generator quick smoke (3 steps).
9. If PASS, run generator full smoke (30 steps) in the same Colab runtime.
10. If changed code touches reranker, run reranker quick/full as applicable.
11. Save/capture colab_smoke_report.json and relevant log lines.
12. Report COLAB GPU SMOKE PASS or BLOCKED.
13. Do NOT interact with Kaggle.
14. User manually runs Kaggle T4x2 only after Colab is green.
```

For Drive data, support either:
- Google Drive mounted path; or
- copying the four required smoke files into `/content/legalqa-data`.

Do not require the full 2.4-GB dataset merely to test generator/reranker training.

---

### Task 5 — Make the no-Kaggle-agent rule testable

**Files:**
- Create: `docs/OPERATING_RULES.md`
- Test: `tests/test_colab_smoke_contract.py`

The document must state:

```text
KAGGLE EXECUTION OWNERSHIP

Only the user may:
- configure Kaggle GPU/accelerator;
- attach Kaggle dataset/model inputs;
- trigger Save & Run All;
- push or publish notebook versions;
- start any Kaggle GPU execution.

Coding agents may:
- modify/push GitHub code;
- monitor GitHub CI;
- package/verify files locally when requested;
- run Google Colab smoke through the configured browser MCP.

Coding agents must never push or execute Kaggle notebooks.
```

Add a test that asserts the exact sentence:

```python
def test_operating_rules_reserve_kaggle_for_user():
    txt = Path("docs/OPERATING_RULES.md").read_text()
    assert "Coding agents must never push or execute Kaggle notebooks." in txt
```

---

### Task 6 — Run CI; no runtime API/data change

Run:

```bash
pytest tests/test_colab_smoke_contract.py -v
pytest tests/ -v
```

Push to GitHub and wait for all Actions jobs.

Because this change adds only workflow tooling/docs/tests:

```text
Runtime API stays 13
V13 dataset stays unchanged
No Kaggle dataset repackaging
No Kaggle notebook push
```

---

### Task 7 — Run the current V13 Colab generator gate

After green CI, coding agent uses the configured Colab/browser MCP.

Use:

```bash
git clone https://github.com/silent9669/LegalQA.git
cd LegalQA
git checkout <FINAL_HEAD>

pip install --upgrade-strategy only-if-needed -r requirements-colab-smoke.txt
```

Do not install Torch/CUDA.

Provide:

```text
qa_unique.parquet
retrieval_labels.parquet
legal_chunks.parquet
```

Then:

```bash
python scripts/run_colab_smoke.py \
  --data-root /content/legalqa-data \
  --component generator \
  --mode quick
```

If quick passes:

```bash
python scripts/run_colab_smoke.py \
  --data-root /content/legalqa-data \
  --component generator \
  --mode full
```

Current V13 acceptance requires:

```text
Tesla T4 detected
Transformers async loading disabled
Qwen 4-bit load PASS
QLoRA Trainer n_gpu = 1
3-step quick PASS
30-step full PASS
adapter save PASS
strict PEFT reload PASS
nonempty generation PASS
no CUDA OOM
```

---

### Task 8 — Final manual Kaggle gate remains mandatory

A one-T4 Colab run cannot reproduce:

```text
visible_cuda=2
-> force Trainer from n_gpu=2 to n_gpu=1
```

Therefore V13 still needs one manual Kaggle T4x2 integration smoke after Colab passes.

The user manually configures:

```text
Notebook: canonical API13 notebook
Dataset: complete API13 dataset
Model: qwen-lm/qwen2.5/transformers/3b-instruct/1
Accelerator: T4 x2
Internet: On
HF_TOKEN: enabled
```

Required V13 log:

```text
QLoRA Trainer GPU Policy:
target=cuda:0 | visible_cuda=2 | trainer_n_gpu=1
```

and no DataParallel traceback.

Only the user triggers this run.

---

## Revised Permanent Workflow

```text
CODE CHANGE
   ↓
LOCAL TESTS
   ↓
PUSH GITHUB
   ↓
GITHUB ACTIONS GREEN
   ↓
COLAB T4 QUICK COMPONENT SMOKE
   ↓
COLAB T4 FULL 30-STEP COMPONENT SMOKE
   ↓
STATIC RELEASE/DATA AUDIT
   ↓
USER-ONLY MANUAL KAGGLE T4x2 INTEGRATION SMOKE
   ↓
USER-ONLY FULL KAGGLE TRAINING
```

Optimization rule:

```text
generator-only change -> Colab generator smoke only
reranker-only change  -> Colab reranker smoke only
CPU/retrieval change  -> CI/static tests first; GPU component smoke only if affected
Kaggle/runtime topology change -> Colab where applicable + mandatory final manual Kaggle smoke
```

---

## Required Agent Report

```text
HEAD SHA
changed files
CI run ID
full pytest result
Colab GPU type
Colab package versions
Colab component/mode
quick smoke result
full smoke result
peak VRAM
adapter reload result
colab_smoke_report.json summary
Kaggle notebook pushed: NO
Kaggle GPU run by agent: NO
Runtime API: 13
Dataset: UNCHANGED
verdict
```

Verdict:

```text
READY FOR USER MANUAL KAGGLE T4x2
```

or:

```text
BLOCKED
```

## Self-review

- [x] Kaggle control remains exclusively with the user.
- [x] Colab uses one T4 as a cost-saving GPU component gate.
- [x] Colab does not pretend to validate dual-T4 topology.
- [x] Current V13 data/runtime is not unnecessarily repackaged.
- [x] Exact Kaggle-observed user-space ML versions are mirrored in Colab.
- [x] Torch/CUDA are not replaced.
- [x] Quick then full component smoke minimizes wasted Colab time.
- [x] Generator/reranker can be tested independently based on changed files.
