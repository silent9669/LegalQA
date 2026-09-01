# LegalQA V13 — Single-GPU QLoRA / DataParallel OOM Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent Hugging Face Trainer from wrapping the 4-bit Qwen QLoRA model in `torch.nn.DataParallel` when both Kaggle T4s are visible, so generator training stays exclusively on `cuda:0` as designed.

**Architecture:** Preserve the dual-T4 hardware split: generator only on GPU 0; DEk21 + reranker on GPU 1. The latest smoke proves async model loading is fixed and Qwen now loads successfully. The new failure is the first training forward, where Transformers 5.0.0 sees two GPUs, sets `TrainingArguments.n_gpu=2`, wraps the 4-bit PEFT model in `nn.DataParallel`, and enters `parallel_apply`. Force the SFT Trainer's internal GPU count to one after device setup, fail loudly if the policy is not honored, bump the packaged runtime to API 13, and republish the runtime dataset.

**Tech Stack:** Kaggle Python 3.12, PyTorch 2.10, Transformers 5.0.0, TRL 1.12.0, PEFT 0.19.1, bitsandbytes 0.50.2, Qwen2.5-3B-Instruct, T4 x2.

**Spec:** Real smoke log `legalqa-training-3.log`; current GitHub HEAD `e8762d0c169c03416d2ae35410c2566ad3585db3`.

## Global Constraints

- Do not redesign Stack A.
- Generator must remain on `cuda:0`.
- Retrieval/reranker must remain on `cuda:1`.
- Do not hide GPU 1 globally with `CUDA_VISIBLE_DEVICES`; the pipeline needs it.
- Do not change Qwen model, LoRA rank, sequence length 2048, batch size 1, gradient accumulation 8, optimizer, learning rate, scoring, or parameter budget in this fix.
- Keep `HF_DEACTIVATE_ASYNC_LOAD=1`.
- Keep `trl>=0.17.0` and strict TRL API checks.
- Keep `EXECUTION_PROFILE = "smoke_only"` as committed notebook default.
- Keep strict runtime/model path resolution.
- Do not modify `HF_TOKEN`, print it, or persist it.
- Coding agent must not run Kaggle GPU training; the user runs T4 x2 manually.
- Runtime API becomes **13** because packaged training behavior changes.
- Do not patch TRL's private CE chunk size yet. If a new OOM remains after the log proves `Trainer n_gpu=1` and no DataParallel frame exists, diagnose that separately.

---

## 0. Audited evidence and root cause

Current GitHub HEAD:

```text
e8762d0c169c03416d2ae35410c2566ad3585db3
fix(kaggle): deactivate Transformers v5 async loading in notebook Cell 1
```

Current CI is green:

```text
Run ID: 33485117523
169 passed, 2 skipped
all 4 jobs PASS
```

The latest real T4 x2 smoke proves the previous async-load fix worked:

```text
Transformers async model loading: DISABLED for T4-safe QLoRA load
Runtime API 12 verified
2 x Tesla T4 verified
pip regression guard PASS
TRL SFT API PASS
preflight PASS
BM25 PASS
DEk21 PASS
reranker 30 steps PASS
reranker strict reload PASS
QLoRA dataset built
Qwen model load + SFTTrainer initialization reached
```

The old failure inside `from_pretrained()` is gone.

The first new failure is during `trainer.train()`:

```text
transformers.trainer.training_step
-> model(**inputs)
-> torch.nn.parallel.data_parallel.DataParallel.forward
-> parallel_apply(...)
-> replica 0 on device 0
-> TRL _chunked_cross_entropy_loss
-> h.float() @ w.float().t()
-> CUDA OOM
```

The OOM attempts only 150 MiB:

```text
GPU 0 total: 14.56 GiB
GPU 0 free: 126.81 MiB
process memory: 14.44 GiB
```

This traceback proves the intended single-GPU QLoRA model was wrapped in **DataParallel**.

Transformers 5.0.0 Trainer contains:

```python
if self.args.n_gpu > 1 and not getattr(model, "is_loaded_in_8bit", False):
    model = nn.DataParallel(model)
```

The Qwen model is 4-bit, not 8-bit, so this guard does not protect it.

Transformers 5.0.0 `TrainingArguments` initializes:

```python
self._n_gpu = torch.cuda.device_count()
```

which is `2` on Kaggle. Its `n_gpu` property returns `_n_gpu`.

Therefore this is deterministic:

```text
2 visible T4s
-> SFTConfig/TrainingArguments n_gpu = 2
-> Qwen is loaded_in_4bit, not is_loaded_in_8bit
-> Trainer wraps it in nn.DataParallel
-> model replicas / DP training path violate hardware split
-> GPU-memory pressure reaches OOM on first QLoRA forward
```

The correct fix is to make Trainer believe the generator training job owns exactly one GPU while still leaving both physical GPUs visible to the rest of the notebook.

---

### Task 1 — Add a tested single-device Trainer policy helper

**Files:**
- Modify: `src/task2/training/train_generator.py`
- Create: `tests/test_v13_single_gpu_qlora.py`

**Interfaces:**
- Produces: `enforce_single_gpu_trainer_args(args: Any, device: str) -> None`

- [ ] **Step 1: Write failing unit tests**

Create `tests/test_v13_single_gpu_qlora.py`:

```python
import pytest

from src.task2.training.train_generator import enforce_single_gpu_trainer_args


class DummyArgs:
    def __init__(self, n_gpu=2, device="cuda:0"):
        self._n_gpu = n_gpu
        self._device = device

    @property
    def device(self):
        return self._device

    @property
    def n_gpu(self):
        return self._n_gpu


def test_generator_trainer_forces_one_gpu_when_two_are_visible():
    args = DummyArgs(n_gpu=2, device="cuda:0")

    enforce_single_gpu_trainer_args(args, "cuda:0")

    assert args.n_gpu == 1


def test_generator_trainer_rejects_wrong_cuda_target():
    args = DummyArgs(n_gpu=2, device="cuda:0")

    with pytest.raises(RuntimeError, match="cuda:0"):
        enforce_single_gpu_trainer_args(args, "cuda:1")


def test_generator_trainer_cpu_policy_does_not_fake_gpu_count():
    args = DummyArgs(n_gpu=0, device="cpu")

    enforce_single_gpu_trainer_args(args, "cpu")

    assert args.n_gpu == 0
```

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_v13_single_gpu_qlora.py -v
```

Expected: FAIL because helper does not exist.

- [ ] **Step 3: Implement the helper**

In `src/task2/training/train_generator.py`:

```python
def enforce_single_gpu_trainer_args(args: Any, device: str) -> None:
    """Force HF Trainer to keep LegalQA QLoRA on its dedicated generator GPU.

    Kaggle exposes two T4s in one process. Transformers Trainer otherwise uses
    args.n_gpu > 1 to activate nn.DataParallel, which violates LegalQA's
    generator=cuda:0 / retrieval-reranker=cuda:1 hardware split.
    """
    dev = str(device)

    if not dev.startswith("cuda"):
        return

    if dev != "cuda:0":
        raise RuntimeError(
            f"LegalQA QLoRA generator must target cuda:0, got {dev!r}."
        )

    # Force TrainingArguments._setup_devices to run first. On Kaggle this
    # normally records _n_gpu=2 because both T4s are intentionally visible.
    _ = args.device

    if not hasattr(args, "_n_gpu"):
        raise RuntimeError(
            "Transformers TrainingArguments no longer exposes internal "
            "_n_gpu after device setup; refusing to risk DataParallel."
        )

    args._n_gpu = 1

    if int(args.n_gpu) != 1:
        raise RuntimeError(
            f"Failed to force single-GPU QLoRA Trainer policy; "
            f"Trainer reports n_gpu={args.n_gpu}."
        )
```

- [ ] **Step 4: Verify GREEN**

```bash
pytest tests/test_v13_single_gpu_qlora.py -v
```

Expected: PASS.

---

### Task 2 — Apply the policy before SFTTrainer construction

**Files:**
- Modify: `src/task2/training/train_generator.py`
- Test: `tests/test_v13_single_gpu_qlora.py`

Current flow:

```python
sft_args = build_sft_config(...)
trainer = SFTTrainer(...)
trainer.train(...)
```

Change it to:

```python
sft_args = build_sft_config(max_seq_len=max_seq_len, **sft_kwargs)

enforce_single_gpu_trainer_args(sft_args, dev)

print(
    "QLoRA Trainer GPU Policy: "
    f"target={dev} | visible_cuda={torch.cuda.device_count()} | "
    f"trainer_n_gpu={sft_args.n_gpu}"
)

trainer = SFTTrainer(
    model=model,
    args=sft_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)

if int(trainer.args.n_gpu) != 1:
    raise RuntimeError(
        "FINAL_PIPELINE_ERROR: SFTTrainer changed the generator GPU policy; "
        f"expected n_gpu=1, got {trainer.args.n_gpu}."
    )
```

- [ ] Add a source/runtime contract test:

```python
def test_run_qlora_enforces_policy_before_trainer():
    import inspect
    import src.task2.training.train_generator as mod

    src = inspect.getsource(mod.run_qlora_training)

    force_pos = src.index("enforce_single_gpu_trainer_args(sft_args, dev)")
    trainer_pos = src.index("trainer = SFTTrainer(")

    assert force_pos < trainer_pos
    assert "trainer.args.n_gpu" in src
```

Run:

```bash
pytest tests/test_v13_single_gpu_qlora.py -v
```

Expected: PASS.

---

### Task 3 — Add an explicit anti-DataParallel runtime contract

**Files:**
- Modify: `src/task2/training/train_generator.py`
- Test: `tests/test_v13_single_gpu_qlora.py`

The real smoke must fail loudly if a future Transformers version changes the semantics and enables DataParallel anyway.

After constructing `trainer`, add:

```python
if int(trainer.args.n_gpu) != 1:
    raise RuntimeError(
        "FINAL_PIPELINE_ERROR: QLoRA Trainer must be single-GPU; "
        f"n_gpu={trainer.args.n_gpu} would enable DataParallel."
    )
```

Do not monkeypatch `torch.nn.DataParallel` globally.

Do not set fake model attributes such as:

```text
is_loaded_in_8bit=True
model_parallel=True
```

Those would misrepresent model state.

---

### Task 4 — Preserve the async-load hotfix and all training semantics

**Files:**
- Modify/test only as necessary.

The notebook must still contain in Cell 1:

```python
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
```

The generator must remain:

```text
Qwen/Qwen2.5-3B-Instruct
4-bit NF4
double quantization = true
LoRA r = 16
LoRA alpha = 32
max_seq_len = 2048
per-device batch = 1
grad accumulation = 8
paged_adamw_8bit
gradient checkpointing = true
completion_only_loss = true
```

Add assertions to an existing contract test or V13 test so these critical values are not accidentally reduced merely to make smoke pass.

---

### Task 5 — Bump packaged runtime binding to API 13

**Files:**
- Modify: `configs/runtime_api.yaml`
- Modify: `src/task2/runtime_integrity.py`
- Modify: `kaggle_kernel/legalqa_gpu_pipeline.ipynb`
- Modify release-binding tests

Because `src/task2/training/train_generator.py` is part of the packaged runtime, the existing API-12 dataset cannot contain this fix.

Set:

```yaml
runtime_api_version: 13
```

Set:

```python
EXPECTED_RUNTIME_API_VERSION = 13
```

Notebook:

```python
REQUIRED_RUNTIME_API_VERSION = 13
```

Preserve:

```python
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
EXECUTION_PROFILE = "smoke_only"
strict=True
allow_remote_model_download=False
```

Update stale-package tests so API 12 is rejected by the API-13 notebook/runtime contract.

---

### Task 6 — Full verification

Run focused tests:

```bash
pytest tests/test_v13_single_gpu_qlora.py -v
pytest tests/test_v9_notebook_contract.py -v
pytest tests/test_v10_runtime_release_binding.py -v
pytest tests/test_v12_pip_check_baseline.py -v
```

Then:

```bash
pytest tests/ -v
```

Push only after all pass.

Wait for GitHub Actions and report:

```text
final HEAD SHA
working tree clean
focused V13 tests
full pytest exact pass/skip count
Actions run ID
CPU suite PASS
Python 3.10 SFT compatibility PASS
Python 3.12 SFT compatibility PASS
protected runtime preservation PASS
```

Do not claim GPU success from CI.

---

### Task 7 — Repackage the same data/indexes with API-13 code

The data/index content is already verified; do not rebuild it unless packaging self-validation says something is missing.

From the exact clean final V13 HEAD:

```bash
rm -rf kaggle_dataset/staged

python scripts/package_kaggle_dataset.py \
  --source artifacts/task2 \
  --staging kaggle_dataset/staged \
  --profile final_training
```

Verify:

```python
import json, re
from pathlib import Path

root = Path("kaggle_dataset/staged")

ds = json.loads((root / "dataset_manifest.json").read_text())
cm = json.loads((root / "code_manifest.json").read_text())
nested = json.loads(
    (root / "code/LegalQA/code_manifest.json").read_text()
)

assert ds["runtime_api_version"] == 13
assert cm["runtime_api_version"] == 13
assert nested["runtime_api_version"] == 13

assert ds["git_sha"] == cm["git_sha"] == nested["git_sha"]
assert re.fullmatch(r"[0-9a-f]{40}", ds["git_sha"])

generator_src = (
    root / "code/LegalQA/src/task2/training/train_generator.py"
).read_text()

assert "enforce_single_gpu_trainer_args" in generator_src
assert "trainer.args.n_gpu" in generator_src

assert (root / "indexes/bm25").is_dir()
assert (root / "indexes/dek21/embeddings.npy").is_file()
assert (root / "public-official.json").is_file()

print("LOCAL API13 PACKAGE PASS")
```

---

### Task 8 — Upload new Kaggle dataset + notebook versions

The coding agent may deploy with Kaggle CLI but must not start GPU execution.

Upload a **new dataset version**:

```text
phucdangg/legalqa-task2-clean-data
```

Then inspect/download the remote artifact and prove:

```text
dataset manifest API13 + final HEAD
root code manifest API13 + final HEAD
nested code manifest API13 + final HEAD
packaged train_generator contains single-GPU policy
BM25 complete
DEk21 complete
trl>=0.17.0
```

Push the corresponding notebook version and verify:

```text
REQUIRED_RUNTIME_API_VERSION = 13
EXECUTION_PROFILE = "smoke_only"
HF_DEACTIVATE_ASYNC_LOAD = 1
```

Record exact Kaggle dataset and notebook version numbers.

---

### Task 9 — Human T4 x2 smoke acceptance

The user manually runs:

```text
Notebook: new API-13 notebook version
Dataset: new API-13 dataset version
Model: qwen-lm/qwen2.5/transformers/3b-instruct/1
Accelerator: T4 x2
Internet: On
HF_TOKEN: enabled
Profile: smoke_only
```

Expected generator log:

```text
QLoRA Trainer GPU Policy:
target=cuda:0 | visible_cuda=2 | trainer_n_gpu=1
```

The traceback must contain **no**:

```text
torch.nn.parallel.data_parallel
parallel_apply
replica 0
```

Smoke success requires:

```text
Qwen model load PASS
30 QLoRA optimizer steps
adapter save PASS
strict PEFT reload PASS
5 held-out predictions/evaluation PASS
```

If an OOM remains after the log proves `trainer_n_gpu=1` and no DataParallel frames appear, stop. The next investigation should focus specifically on TRL 1.12's fixed 256-token chunked-CE head (`_CHUNKED_LM_HEAD_CHUNK_SIZE = 256`), not on async loading or DataParallel again.

---

## Non-blocking issues not included in V13

Do not bundle these with the current blocker:

1. Reranker still warns that `lr_scheduler.step()` occurs before `optimizer.step()`.
2. Dense mmap emits a non-writable NumPy warning.
3. Transformers warns `torch_dtype` is deprecated.
4. TRL warns `warmup_ratio` is deprecated.

They should be handled after the smoke reaches the end.

---

## Required Agent Completion Report

```text
HEAD SHA
changed files
focused V13 pytest
full pytest
GitHub Actions run ID
runtime API
new Kaggle dataset version
new Kaggle notebook version
remote API/SHA parity
packaged single-GPU policy proof
HF_DEACTIVATE_ASYNC_LOAD proof
exact manual attachment versions
GPU run: NOT RUN BY AGENT
verdict
```

Verdict exactly one of:

```text
READY FOR MANUAL KAGGLE SMOKE
BLOCKED
```

## Self-review

- [x] Async model loading is already fixed and is not changed again.
- [x] Root cause is directly evidenced by `torch.nn.DataParallel` in the real trace.
- [x] Preserves the intended cuda:0 / cuda:1 hardware split.
- [x] Does not reduce QLoRA sequence length or rank speculatively.
- [x] Does not hide GPU 1 from retrieval/reranking.
- [x] Requires an API bump because packaged source code changes.
- [x] Keeps GPU execution manual.
