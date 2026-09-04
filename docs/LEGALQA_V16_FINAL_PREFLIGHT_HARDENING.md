# LegalQA V16 — Final Preflight Hardening Before Manual Kaggle Probe

## Goal

Harden the existing **V16** release before spending another Kaggle Dual‑T4 session.

This is **not** a redesign and must **not** become V17. Keep Runtime API **16**, keep the current model/data/index architecture, and make only the three verification improvements identified in the repository/Drive audit.

Current architecture to preserve:

- Qwen2.5‑3B‑Instruct
- 4‑bit NF4 + double quant
- LoRA `r=16`, `alpha=32`
- `max_seq_len=2048`
- batch size `1`
- gradient accumulation `8`
- completion-only loss
- activation offloading
- gradient checkpointing
- generator fixed to `cuda:0`
- Trainer `n_gpu=1`
- retrieval/reranker on `cuda:1`
- TRL `1.12.0`
- Liger Kernel `0.8.2`
- selective `fused_linear_cross_entropy=True`
- `chunked_nll` disabled while Liger is active
- notebook profile `generator_probe_worstcase`
- strict Dual‑T4 guard
- API16 dataset/index inheritance

Do **not** regenerate legal data, BM25, or DEk21.

---

# 1. Fix Liger environment validation

## Current weakness

`src/task2/generation/liger_backend.py` currently checks the backend approximately via:

```python
has_fused_ce = hasattr(liger_kernel, "transformers")
```

This is not a reliable proof that the exact Qwen2 patch and fused-linear CE symbols are importable.

## Required change

Replace the approximate attribute check with strict symbol imports.

The validator must verify:

```python
from liger_kernel.transformers import apply_liger_kernel_to_qwen2
from liger_kernel.transformers.fused_linear_cross_entropy import (
    LigerFusedLinearCrossEntropyLoss,
)
```

Requirements:

1. `liger-kernel` version must be exactly `0.8.2`.
2. `apply_liger_kernel_to_qwen2` must import successfully.
3. `LigerFusedLinearCrossEntropyLoss` must import successfully.
4. In `strict=True`, any failure must raise `RuntimeError`.
5. The returned status must report fused-linear CE unavailable if either symbol is missing.
6. `train_generator_qlora()` must fail before model loading when strict validation fails.
7. Do not silently fall back to normal CE or TRL `chunked_nll`.

Recommended status fields:

```python
@dataclass(frozen=True)
class LigerBackendStatus:
    version: str
    enabled: bool
    qwen2_patch_available: bool
    fused_linear_ce: bool
    config: Dict[str, bool]
```

## Required tests

Update `tests/unit/test_liger_backend.py` to cover:

- exact version + both symbols present → PASS
- wrong version → FAIL
- missing Qwen2 patch → FAIL
- missing fused-linear CE class → FAIL
- package missing → FAIL in strict mode
- package missing → disabled status in non-strict mode
- selective config remains exactly:

```text
rope                         false
rms_norm                     false
swiglu                       false
cross_entropy                false
fused_linear_cross_entropy   true
```

---

# 2. Complete the Liger loss/gradient parity test

## Current weakness

`tests/gpu/test_liger_qlora_gpu.py` is named as a loss/gradient parity test, but currently validates only the scalar loss.

## Required change

Make it a real forward + backward parity test.

Use **independent cloned tensors** for the reference and Liger branches.

Reference branch:

```python
logits = F.linear(hidden_ref.float(), weight_ref.float())
loss_ref = F.cross_entropy(
    logits.view(-1, vocab_size),
    labels.view(-1),
    ignore_index=-100,
)
loss_ref.backward()
```

Liger branch:

```python
loss_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)
loss_liger = loss_fn(
    hidden_liger.view(-1, hidden_dim),
    weight_liger,
    labels.view(-1),
)
loss_liger.backward()
```

Assertions:

- both losses finite
- absolute loss difference within a documented FP16 tolerance
- hidden-state gradients exist
- LM-head weight gradients exist
- all gradients finite
- gradient shapes match
- gradient cosine similarity is high
- relative/absolute gradient error stays within a documented tolerance suitable for FP16/Triton

The test must remain CUDA-gated and skipped automatically on CPU-only CI.

Do **not** fake GPU success in CPU CI.

## Manual-probe integration

Before QLoRA training begins on Kaggle, add a tiny one-shot Liger backend smoke or equivalent runtime assertion that proves:

```text
Liger 0.8.2
Qwen2 patch importable
fused-linear CE importable
use_liger_kernel=True
fused_linear_cross_entropy=True
loss_type=nll
trainer_n_gpu=1
```

It should run before the expensive model-training step.

---

# 3. Add an exact Kaggle Hugging Face compatibility contract

## Problem

The current CI compatibility job passed on newer packages than the real Kaggle image. The last real Kaggle environment was approximately:

```text
transformers  5.0.0
peft          0.19.1
accelerate    1.13.0
datasets      5.0.0
trl           1.12.0
liger-kernel  0.8.2
```

The V16 release should explicitly prove compatibility with this API surface before another T4 run.

## Required change

Add a lightweight CI compatibility lane or isolated environment that checks the exact user-space package versions above.

Important:

- **Do not** replace or pin GitHub/Kaggle Torch/CUDA/Triton globally.
- This lane is for Hugging Face/TRL/Liger API compatibility.
- Keep the existing normal CI lane as well.
- Avoid dependency commands that unexpectedly replace the runner's protected Torch stack.

The compatibility lane must verify at minimum:

```python
import inspect

from trl import SFTConfig, SFTTrainer

assert "completion_only_loss" in inspect.signature(SFTConfig).parameters
assert "activation_offloading" in inspect.signature(SFTConfig).parameters
assert "loss_type" in inspect.signature(SFTConfig).parameters
assert "processing_class" in inspect.signature(SFTTrainer).parameters
```

Then construct the V16 SFT config and assert:

```text
completion_only_loss == True
activation_offloading == True
use_liger_kernel == True
loss_type == "nll"
max_length/max_seq_length == 2048
```

Also verify the Liger symbols from Section 1 import on the pinned stack.

### Critical regression checks

The compatibility lane must also prove:

- no `DataParallel` policy regression
- `enforce_single_gpu_trainer_args()` leaves `n_gpu == 1`
- no V15 CE chunk override is active in the V16 generator path
- V16 does not request `loss_type="chunked_nll"`
- API16 notebook literal remains `16`
- notebook default profile remains `generator_probe_worstcase`
- async model-loading disable guard remains intact
- strict dual-T4 guard remains intact

---

# 4. Packaging and release binding

Keep Runtime API:

```text
runtime_api_version: 16
```

Do **not** increment to API17 for this hardening patch.

After all source/test changes are final:

1. Commit the final V16 hardening changes.
2. Run the full CPU/contract test suite.
3. Verify all GitHub Actions jobs are green on the **exact final commit**.
4. Repackage the Kaggle runtime from that exact final commit.
5. Reuse the already validated inherited data/index files.
6. Upload a new Kaggle dataset version.
7. Upload/update the canonical notebook version.
8. Verify remote manifests.

Required manifest parity:

```text
dataset_manifest.runtime_api_version == 16
code_manifest.runtime_api_version    == 16
configs/runtime_api.yaml             == 16
notebook REQUIRED_RUNTIME_API_VERSION == 16
dataset_manifest.git_sha             == FINAL_V16_HEAD
code_manifest.git_sha                == FINAL_V16_HEAD
```

The packaged code must contain the new strict Liger validator.

The packaged requirements must retain:

```text
trl==1.12.0
liger-kernel==0.8.2
```

Do not rebuild:

- `legal_chunks.parquet`
- QA parquet/json data
- BM25 index
- DEk21 embeddings/index

Only runtime/code packaging needs refreshing.

---

# 5. Required manual Kaggle run after static verification

The coding agent must **not run the Kaggle GPU notebook**.

The user will manually run it with:

```text
Accelerator: T4 x2
Internet: ON
HF_TOKEN: Kaggle secret
Dataset: newest API16 package
Model: Qwen2.5-3B-Instruct
Notebook: newest canonical V16 notebook
Profile: generator_probe_worstcase
```

Expected high-level log before training:

```text
Runtime API: 16
CUDA GPUs Detected: 2
COMMITTED EXECUTION PROFILE: generator_probe_worstcase
Liger-Kernel: 0.8.2
Qwen2 Liger patch: PASS
Liger fused-linear CE: PASS
use_liger_kernel=True
fused_linear_cross_entropy=True
loss_type=nll
target=cuda:0
trainer_n_gpu=1
```

The probe is successful only if:

1. Qwen loads successfully.
2. No `torch.nn.DataParallel` / `parallel_apply` frames appear.
3. The worst-case selector is used.
4. All 3 generator optimizer steps complete.
5. No CUDA OOM occurs.
6. Adapter saves.
7. Strict PEFT reload succeeds.
8. Reloaded model produces a non-empty generation.
9. Generator manifest is written.
10. Liger runtime path is visibly confirmed in logs.

---

# 6. Explicit non-goals

Do not change any of the following during this patch:

- model family
- Qwen parameter count
- LoRA rank/alpha/targets
- sequence length
- optimizer
- batch size
- grad accumulation
- retrieval architecture
- reranker architecture
- BM25
- DEk21
- Task 2 scorer
- production selection logic
- Protocol 8
- public inference logic
- learned-parameter budget logic

Do not add speculative memory reductions unless the next real T4 probe still OOMs **after** V16 Liger fused-linear CE is proven active.

---

# 7. Acceptance gate

The coding agent may report:

```text
READY FOR MANUAL KAGGLE PROBE
```

only when all of the following are true:

- strict Liger symbol validation implemented
- Liger backend unit tests pass
- CUDA parity test contains real backward/gradient assertions
- exact Kaggle HF-stack compatibility lane passes
- full existing CI remains green
- Runtime API remains 16
- final packaged code SHA equals final GitHub V16 HEAD
- Drive/Kaggle package contains complete BM25 + DEk21 + public data
- no dataset/index regeneration occurred
- canonical notebook still uses `generator_probe_worstcase`
- coding agent did not run the GPU notebook

Otherwise report:

```text
BLOCKED
```

with the exact failing gate and evidence.
