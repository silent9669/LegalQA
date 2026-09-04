# LegalQA V15 — T4 QLoRA CE-Chunk + Generator-Only Kaggle Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove and fix the remaining first-step QLoRA OOM on Kaggle T4 without repeatedly spending ~1 hour retraining the already-working reranker.

**Architecture:** Keep Stack A, Qwen2.5-3B, 2048 tokens, LoRA r16/a32, completion-only loss, activation offloading, and single-GPU Trainer policy. The V14 run proves the failure is now isolated to TRL 1.12.0's chunked LM-head cross-entropy at chunk size 256. Make the CE chunk size an explicit LegalQA parameter, use a conservative T4 value of 32, and add a strict `generator_probe` Kaggle profile that trains the full fold-filtered 5,956-example source pool for only 3 optimizer steps while skipping reranker/evaluation. Only after that passes should the user run the expensive `screen_fold0`.

**Tech Stack:** Kaggle 2×Tesla T4, PyTorch 2.10.0+cu128, Transformers 5.0.0, TRL 1.12.0, PEFT 0.19.1, bitsandbytes 0.50.2, Qwen2.5-3B-Instruct.

**Spec:** Real V14 Kaggle log `legalqa-training-3(1).log`, current release HEAD `33c962d2d653f27eff4cc076dd49a936799439b7`.

## Global Constraints

- User alone runs/configures Kaggle notebooks. Agent must never push/trigger/run Kaggle.
- Do not use Colab for this iteration; the user wants the authoritative Kaggle T4 environment.
- Keep Stack A unchanged.
- Keep Qwen/Qwen2.5-3B-Instruct.
- Keep `max_seq_len=2048`.
- Keep 4-bit NF4 + double quantization.
- Keep LoRA `r=16`, `alpha=32`, dropout `0.05`, same target modules.
- Keep batch size `1`, gradient accumulation `8`, learning rate `1e-4`.
- Keep `completion_only_loss=True`.
- Keep `loss_type="chunked_nll"`.
- Keep `activation_offloading=True`.
- Keep `HF_DEACTIVATE_ASYNC_LOAD=1`.
- Keep generator on `cuda:0`; retrieval/reranker on `cuda:1`.
- Keep the `trainer_n_gpu=1` anti-DataParallel guard.
- Keep `trl==1.12.0`.
- Do not reduce sequence length, model size, LoRA rank, or training data in this fix.
- Do not enable fallback generation.
- Runtime API becomes **15** because packaged generator behavior changes.

---

## 0. Root-cause evidence from the real V14 run

The V14 manual Kaggle run proves:

```text
Runtime API 14                              PASS
Git SHA 33c962d2...                         PASS
2×Tesla T4                                 PASS
TRL 1.12.0                                 PASS
activation_offloading API                  PASS
BM25 / DEk21 / data preflight              PASS
full reranker training + reload            PASS
Qwen 4-bit load                            PASS
trainer_n_gpu=1                            PASS
```

Reranker cost before generator:

```text
~3,436 seconds (~57 minutes)
Val loss 0.3343
Val acc 0.8815
Peak GPU1 VRAM 10,866 MiB
```

Generator data:

```text
5,956 kept
13 dropped
P50 = 726
P90 = 1,223
P95 = 1,441
Max = 2,047
Evidence truncated = 1.0%
```

V14 memory diagnostics:

```text
After 4-bit model load:
Allocated 1,969 MiB
Reserved  2,028 MiB
Free      1,566.8 MiB

After SFTTrainer:
Allocated 2,026 MiB
Reserved  2,156 MiB
Free      1,418.8 MiB
```

The first training step then fails at:

```python
trl/trainer/sft_trainer.py
_chunked_cross_entropy_loss(...)
-> _chunk(...)
-> logits = h.float() @ w.float().t()
```

with:

```text
TRL chunk size = 256
requested allocation = 594 MiB
GPU0 free at failure = 226.81 MiB
```

The traceback explicitly enters:

```text
with self.maybe_activation_offload_context
```

so V14 activation offloading is active; it simply does not provide enough headroom.

TRL 1.12.0 documents that chunked-NLL peak LM-head/logit memory scales linearly with `chunk_size`. Its built-in constant is 256. V14's guard retained 256, so the previous "chunk cap" did not actually reduce the default.

### Hypothesis

`chunk_size=256` is too large for Qwen2.5-3B's 151,936-token vocabulary under Kaggle T4 headroom after the decoder forward.

Use `chunk_size=32` for the next isolated experiment.

This changes chunk scheduling, not the NLL objective, labels, training examples, sequence limit, model, LoRA configuration, or parameter budget.

---

### Task 1 — Make LegalQA's TRL CE chunk size explicit and testable

**Files:**
- Modify: `src/task2/training/train_generator.py`
- Modify: `tests/test_v14_qlora_oom_fix.py`
- Create: `tests/test_v15_t4_ce_chunk.py`

**Interfaces:**
- `run_qlora_training(..., ce_chunk_size: int = 32) -> Dict[str, Any]`
- `inspect_and_guard_trl_chunk_size(target_chunk_size: int) -> Dict[str, Any]`

- [ ] **Step 1: Add failing tests**

```python
def test_t4_generator_default_ce_chunk_is_32():
    sig = inspect.signature(run_qlora_training)
    assert sig.parameters["ce_chunk_size"].default == 32


def test_v15_reduces_trl_256_chunk_to_32(monkeypatch):
    # install fake TRL 1.12.0 module with
    # _CHUNKED_LM_HEAD_CHUNK_SIZE = 256
    info = inspect_and_guard_trl_chunk_size(target_chunk_size=32)

    assert info["original_chunk_size"] == 256
    assert info["modified_chunk_size"] == 32
    assert info["action"] == "capped"


def test_chunk_size_must_be_positive_power_of_two():
    with pytest.raises(ValueError):
        validate_ce_chunk_size(0)
    with pytest.raises(ValueError):
        validate_ce_chunk_size(48)
    assert validate_ce_chunk_size(32) == 32
```

- [ ] **Step 2: Verify RED**

```bash
pytest tests/test_v15_t4_ce_chunk.py -v
```

- [ ] **Step 3: Add validation helper**

```python
def validate_ce_chunk_size(value: int) -> int:
    value = int(value)
    if value <= 0 or (value & (value - 1)) != 0:
        raise ValueError(
            "LegalQA CE chunk size must be a positive power of two, "
            f"got {value}."
        )
    if value > 256:
        raise ValueError(
            f"LegalQA T4 CE chunk size must be <=256, got {value}."
        )
    return value
```

- [ ] **Step 4: Add parameter to generator**

```python
def run_qlora_training(
    ...,
    ce_chunk_size: int = 32,
    ...
):
    ...
    ce_chunk_size = validate_ce_chunk_size(ce_chunk_size)
    chunk_info = inspect_and_guard_trl_chunk_size(
        target_chunk_size=ce_chunk_size
    )
    print(
        "LegalQA QLoRA CE Chunk Policy: "
        f"requested={ce_chunk_size} | "
        f"effective={chunk_info['modified_chunk_size']}"
    )
```

Require `effective == requested` under TRL 1.12.0. Fail loudly otherwise.

- [ ] **Step 5: Do not silently retain 256**

For the exact pinned TRL:

```python
if trl_version.startswith("1.12"):
    ...
    if current > target_chunk_size:
        sft_mod._CHUNKED_LM_HEAD_CHUNK_SIZE = target_chunk_size
```

At V15 default:

```text
current 256 -> effective 32
```

Run focused tests again.

---

### Task 2 — Add a fast authoritative Kaggle `generator_probe` profile

**Files:**
- Modify: `kaggle_kernel/legalqa_gpu_pipeline.ipynb`
- Modify: `tests/test_v9_notebook_contract.py`
- Create/modify: `tests/test_v15_generator_probe_profile.py`

The next Kaggle run must not waste ~57 minutes on reranker before testing the generator fix.

Add profile:

```text
generator_probe
```

Profile semantics:

```python
elif EXECUTION_PROFILE == "generator_probe":
    RUN_RERANKER_TRAINING = False
    RUN_GENERATOR_TRAINING = True
    RUN_DEV_EVALUATION = False
    RUN_PUBLIC_INFERENCE = False
    REUSE_EXISTING_CHECKPOINTS = False

    TRAIN_VAL_FOLD = 0

    MAX_RERANKER_STEPS = None
    MAX_RERANKER_PAIRS = None
    MAX_RERANKER_VAL_PAIRS = None

    MAX_GENERATOR_STEPS = 3
    MAX_GENERATOR_EXAMPLES = None

    DEV_EVAL_SIZE = None
```

Critical requirement:

```text
MAX_GENERATOR_EXAMPLES = None
```

The probe must use the full fold-filtered 5,956-example source pool. It is bounded only by 3 optimizer steps.

Committed notebook default for the **next run**:

```python
EXECUTION_PROFILE = "generator_probe"
```

Keep:

```python
ALLOW_SINGLE_GPU_SMOKE = False
ALLOW_UNVALIDATED_FINAL = False
REQUIRED_RUNTIME_API_VERSION = 15
```

The profile still requires Dual T4 so it reproduces the Kaggle production environment exactly.

---

### Task 3 — Pass the CE chunk policy explicitly from notebook to generator

**Files:**
- Modify: `kaggle_kernel/legalqa_gpu_pipeline.ipynb`
- Test: `tests/test_v15_generator_probe_profile.py`

Cell 1:

```python
QLORA_CE_CHUNK_SIZE = 32
```

Cell 8:

```python
res_qlora = run_qlora_training(
    ...,
    ce_chunk_size=QLORA_CE_CHUNK_SIZE,
    ...
)
```

Log before training:

```text
QLORA_CE_CHUNK_SIZE: 32
```

Contract tests must assert:

```text
32
max_seq_len=2048
MAX_GENERATOR_EXAMPLES=None
MAX_GENERATOR_STEPS=3
RUN_RERANKER_TRAINING=False
```

---

### Task 4 — Add probe timing and memory acceptance telemetry

**Files:**
- Modify: `src/task2/training/train_generator.py`
- Test: `tests/test_v15_t4_ce_chunk.py`

Capture around `trainer.train()`:

```python
train_started = time.perf_counter()
trainer.train(...)
train_elapsed = time.perf_counter() - train_started
```

Store in generator manifest:

```json
{
  "ce_chunk_size": 32,
  "train_elapsed_seconds": 0,
  "optimizer_steps": 3,
  "seconds_per_optimizer_step": 0,
  "peak_vram_mb": 0,
  "peak_reserved_mb": 0,
  "free_vram_mb": 0
}
```

For a 3-step probe, print a rough all-data generator estimate:

```python
estimated_total_optimizer_steps = math.ceil(
    len(dataset) / (batch_size * grad_accum)
)

estimated_generator_hours = (
    seconds_per_optimizer_step
    * estimated_total_optimizer_steps
    / 3600
)
```

This estimate is informational only; do not fail based on time.

---

### Task 5 — Preserve all quality-critical training semantics

**Files:**
- Test: `tests/test_v15_t4_ce_chunk.py`

Regression tests must prove V15 did **not** change:

```text
model = Qwen/Qwen2.5-3B-Instruct
max_seq_len = 2048
load_in_4bit = True
NF4
double quantization = True
LoRA r = 16
LoRA alpha = 32
LoRA dropout = 0.05
same target modules
completion_only_loss = True
loss_type = chunked_nll
activation_offloading = True
batch = 1
grad_accum = 8
lr = 1e-4
generator device = cuda:0
trainer n_gpu = 1
TRL = 1.12.0
```

Do not combine sequence-length reductions with the CE-chunk experiment.

---

### Task 6 — Runtime API15 release binding

**Files:**
- Modify: `configs/runtime_api.yaml`
- Modify: `src/task2/runtime_integrity.py`
- Modify: `kaggle_kernel/legalqa_gpu_pipeline.ipynb`
- Modify release-binding tests

Set:

```text
runtime_api_version = 15
EXPECTED_RUNTIME_API_VERSION = 15
REQUIRED_RUNTIME_API_VERSION = 15
```

Stale API14 dataset must be rejected.

---

### Task 7 — CI

Run:

```bash
pytest tests/test_v15_t4_ce_chunk.py -v
pytest tests/test_v15_generator_probe_profile.py -v
pytest tests/test_v14_qlora_oom_fix.py -v
pytest tests/test_v13_single_gpu_qlora.py -v
pytest tests/test_v10_runtime_release_binding.py -v
pytest tests/ -v
```

Push GitHub and require all CI jobs green.

Agent must report exact:
- final HEAD;
- full pass/skip count;
- workflow run ID;
- Python 3.10/3.12 TRL checks.

---

### Task 8 — Repackage API15 data without rebuilding immutable artifacts

Use the same verified Task2 data/indexes.

Repackage from exact clean V15 HEAD:

```bash
python scripts/package_kaggle_dataset.py \
  --source artifacts/task2 \
  --staging kaggle_dataset/staged \
  --profile final_training
```

Require all three manifests:

```text
API = 15
Git SHA = exact final HEAD
```

Require packaged `train_generator.py` contains:
- `ce_chunk_size`;
- default `32`;
- effective chunk verification;
- existing activation offloading;
- existing single-GPU Trainer guard.

Preserve BM25 7/7, DEk21 3/3, public test, QA files and reranker pairs.

Upload a new dataset version with Kaggle CLI if that is part of the existing deployment workflow, but **never run/push the Kaggle notebook**.

Verify the remote dataset artifact after upload.

---

### Task 9 — User-only Kaggle generator probe

Agent ends with:

```text
READY FOR USER MANUAL KAGGLE GENERATOR PROBE
```

User manually attaches:

```text
Notebook: V15 generator_probe notebook
Dataset: new API15 LegalQA dataset
Model: qwen-lm/qwen2.5/transformers/3b-instruct/1
Accelerator: T4 x2
Internet: On
HF_TOKEN: enabled
```

Expected run should skip reranker entirely and reach QLoRA in a few minutes.

Required log:

```text
COMMITTED EXECUTION PROFILE: generator_probe
Verified Runtime Integrity: API v15
Train Reranker: False
Train QLoRA: True

SFT Dataset Stats (5956 kept, ...)
LegalQA QLoRA CE Chunk Policy: requested=32 | effective=32

QLoRA Trainer GPU Policy:
target=cuda:0 | visible_cuda=2 | trainer_n_gpu=1

Immediately Before trainer.train()
...
3 optimizer steps complete
```

Acceptance:

```text
no CUDA OOM
optimizer steps = 3
adapter save PASS
strict PEFT reload PASS
non-empty generation PASS
```

Do not run the expensive `screen_fold0` until this probe passes.

---

### Task 10 — After probe PASS

Do not change API or dataset again if no runtime code changes.

Change only notebook profile back to:

```python
EXECUTION_PROFILE = "screen_fold0"
```

Keep:

```python
QLORA_CE_CHUNK_SIZE = 32
REQUIRED_RUNTIME_API_VERSION = 15
```

Run notebook-contract CI.

Then the user manually runs the real Protocol-8 screen.

If the 32-token probe **still OOMs at the same `w.float()` line**, stop. Do not lower to 16 blindly. The next architectural investigation must isolate the LM-head weight cast / mixed-precision loss path (or a supported fused linear-cross-entropy implementation such as Liger) before another Kaggle run.

---

## Required Agent Completion Report

```text
FINAL HEAD:
RUNTIME API: 15
CE CHUNK POLICY: 32
NOTEBOOK PROFILE: generator_probe
RERANKER IN PROBE: SKIPPED
FULL SOURCE POOL: YES / max_train_examples=None
CI:
REMOTE DATASET VERSION:
REMOTE API/SHA:
KAGGLE NOTEBOOK PUSH/RUN BY AGENT: NO
VERDICT: READY FOR USER MANUAL KAGGLE GENERATOR PROBE / BLOCKED
```

## Self-review

- [x] Root cause is isolated to the TRL chunked-CE path.
- [x] V14 activation offloading is confirmed active rather than assumed.
- [x] The experiment changes one memory variable: CE chunk size.
- [x] Sequence length/model/LoRA/data/scoring are preserved.
- [x] The next Kaggle run skips the ~57-minute reranker.
- [x] The probe still uses the full 5,956-example source pool.
- [x] A failure at chunk=32 triggers architectural investigation rather than endless chunk reductions.
