# LegalQA V16 Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a clean API16 LegalQA runtime that inherits the verified V15 dataset and replaces the failing TRL chunked-NLL generator loss path with selective Liger fused-linear cross entropy, then prove it on Kaggle T4×2 before Protocol-8/full training.

**Architecture:** Preserve immutable data/retrieval/evaluation behavior. Refactor generator training into focused modules, use Qwen2.5-3B NF4 QLoRA on `cuda:0`, selective Liger fused-linear CE, and stage-specific use of `cuda:1` for retrieval/reranking. The Kaggle notebook becomes a thin orchestrator with explicit probe/screen/final profiles.

**Tech Stack:** Python 3.12, PyTorch 2.10/CUDA 12.8 Kaggle stack, Transformers 5.0, TRL 1.12.0, PEFT 0.19.1, bitsandbytes 0.50.2, Liger-Kernel 0.8.2, BM25S, DEk21, BGE-reranker-v2-m3, Qwen2.5-3B-Instruct.

**Spec:** `02_FINAL_ARCHITECTURE_V16.md` and `06_LIGER_GENERATOR_DESIGN.md`.

## Global constraints

- Start from V15 HEAD `151313fc3126615ec11c08ca68f154d5b0c5406f`.
- Runtime API target is 16.
- Inherited data/index bytes remain unchanged.
- Only user runs Kaggle GPU.
- Qwen2.5-3B, 2048 tokens, LoRA r16/a32, batch1/gradaccum8 remain unchanged.
- Generator uses `cuda:0`; reranker/retrieval use `cuda:1` outside generator training.
- No DataParallel/DDP/FSDP/ZeRO.
- TRL `chunked_nll` is not used with Liger.
- `liger-kernel==0.8.2` is exact.
- `ALLOW_UNVALIDATED_FINAL=False`.
- Final model policy comes only from Protocol-8 promotion.
- No “ready” verdict without CI + package verification; no full-screen verdict without user GPU evidence.

---

## Task 1 — Create isolated V16 workspace and archive boundary

**Files:**
- Create: `docs/active/v16/README.md`
- Create: `docs/archive/README.md`
- Create branch/worktree: `refresh/v16-liger`

- [ ] Verify clean V15 baseline:
```bash
git status --short
git rev-parse HEAD
```
Expected HEAD: `151313fc3126615ec11c08ca68f154d5b0c5406f`.

- [ ] Create isolated worktree/branch according to repository policy.

- [ ] Add active/archive documentation boundary. Old V7–V15 docs remain historical.

- [ ] Commit:
```bash
git add docs/active/v16 docs/archive
git commit -m "docs: establish V16 active refresh boundary"
```

## Task 2 — Freeze inherited dataset provenance

**Files:**
- Create: `artifacts/inherited/dataset_manifest_v15.json`
- Create: `src/task2/data_contract.py`
- Create: `scripts/verify_inherited_dataset.py`
- Create: `tests/contracts/test_dataset_inheritance.py`

- [ ] Copy the exact verified V15 dataset manifest into `artifacts/inherited/`.

- [ ] Write failing tests for core SHA entries and structural invariants.

- [ ] Implement `verify_inherited_dataset(root, manifest_path)` with streaming SHA256.

- [ ] Validate BM25 7-file and DEk21 3-file layouts.

- [ ] Run:
```bash
pytest tests/contracts/test_dataset_inheritance.py -v
```

- [ ] Commit.

## Task 3 — Split dependency files and pin Liger

**Files:**
- Create: `requirements/base.txt`
- Create: `requirements/kaggle.txt`
- Create: `requirements/gpu-test.txt`
- Keep/update compatibility file: `requirements-kaggle.txt`
- Test: `tests/contracts/test_dependency_lock.py`

- [ ] Add failing tests that forbid Torch/CUDA/Triton replacement and require:
```text
trl==1.12.0
liger-kernel==0.8.2
```

- [ ] Update bootstrap to consume the canonical Kaggle requirement file.

- [ ] Preserve baseline-aware `pip check` and protected distribution checks.

- [ ] Run tests and commit.

## Task 4 — Create generator configuration module

**Files:**
- Create: `src/task2/generation/__init__.py`
- Create: `src/task2/generation/config.py`
- Create: `tests/unit/test_generation_config.py`

- [ ] Write tests for approved production values.

- [ ] Implement `GeneratorTrainConfig`.

- [ ] Reject accidental changes to device, rank, sequence length, or loss backend in production profile.

- [ ] Run tests and commit.

## Task 5 — Move answer-preserving SFT dataset builder

**Files:**
- Create: `src/task2/generation/dataset.py`
- Modify: `src/task2/training/train_generator.py`
- Create: `tests/unit/test_sft_dataset.py`

- [ ] Port current builder behavior without changing semantics.

- [ ] Add `completion_tokens`, `total_tokens`, and `qa_id`.

- [ ] Add deterministic worst-case selector:
```python
select_worst_case_probe(examples, n_total=12, n_completion=12)
```

- [ ] Test:
  - answer never truncated;
  - evidence trimmed first;
  - fold exclusion;
  - deterministic selection;
  - longest examples included.

- [ ] Make old module call/re-export new builder.

- [ ] Commit.

## Task 6 — Add memory lifecycle module

**Files:**
- Create: `src/task2/generation/memory.py`
- Create: `tests/unit/test_memory_policy.py`

- [ ] Implement stage cleanup and snapshots.

- [ ] Implement Trainer callback for 50-step telemetry.

- [ ] Unit-test behavior with mocked CUDA APIs.

- [ ] Commit.

## Task 7 — Add selective Liger backend

**Files:**
- Create: `src/task2/generation/liger_backend.py`
- Create: `tests/unit/test_liger_backend.py`
- Create: `tests/gpu/test_liger_qlora_gpu.py`

**Interfaces:**
```python
validate_liger_environment() -> LigerBackendStatus
build_liger_training_kwargs() -> dict
```

- [ ] Write tests requiring exact Liger version `0.8.2`.

- [ ] Require selective config:
```python
{
 "rope": False,
 "rms_norm": False,
 "swiglu": False,
 "cross_entropy": False,
 "fused_linear_cross_entropy": True,
}
```

- [ ] Ensure `loss_type="chunked_nll"` is absent when Liger is active.

- [ ] Add conditional GPU parity test; skip when CUDA unavailable.

- [ ] Commit.

## Task 8 — Build new QLoRA trainer

**Files:**
- Create: `src/task2/generation/trainer.py`
- Modify: `src/task2/training/train_generator.py`
- Create: `tests/integration/test_generator_training_contract.py`

- [ ] Write source/config contract tests first.

- [ ] Load Qwen in NF4 + double quant.

- [ ] Keep LoRA exact.

- [ ] Use activation offloading and gradient checkpointing.

- [ ] Use Liger fused-linear CE.

- [ ] Force Trainer n_gpu=1 after device setup.

- [ ] Add strict adapter save/reload/generation.

- [ ] Add manifest telemetry.

- [ ] Run tests and commit.

## Task 9 — Extract clean execution profiles

**Files:**
- Create: `src/task2/pipeline/profiles.py`
- Create: `tests/contracts/test_v16_profiles.py`

Profiles:

```text
generator_probe_worstcase
generator_probe_endurance
screen_fold0
final_train_and_submit
```

Exact semantics:

### `generator_probe_worstcase`
```text
reranker train = false
generator train = true
dev eval = false
public inference = false
fold excluded = 0
probe selection = worst_case
max optimizer steps = 3
```

### `generator_probe_endurance`
```text
reranker train = false
generator train = true
dev eval = false
public inference = false
fold excluded = 0
full source pool
max optimizer steps = 30
```

### `screen_fold0`
Full Protocol-8 screen behavior.

### `final_train_and_submit`
Requires `status=PROMOTED` and no unvalidated override.

- [ ] Test each profile and commit.

## Task 10 — Clean pipeline runner and thin notebook

**Files:**
- Create: `src/task2/pipeline/runner.py`
- Modify: `kaggle_kernel/legalqa_gpu_pipeline.ipynb`
- Create: `tests/contracts/test_v16_notebook.py`

- [ ] Move orchestration out of notebook cells.

- [ ] Set committed next-run profile:
```text
generator_probe_worstcase
```

- [ ] Notebook must own literal:
```text
REQUIRED_RUNTIME_API_VERSION = 16
```

- [ ] Keep async-load guard and strict dual-T4 guard.

- [ ] Run notebook-contract tests and commit.

## Task 11 — Runtime API16 binding

**Files:**
- Modify: `configs/runtime_api.yaml`
- Modify: `src/task2/runtime_integrity.py`
- Modify: notebook literal
- Modify release-binding tests

- [ ] Set every active runtime API value to 16.

- [ ] Add stale API15 rejection test.

- [ ] Run release-binding tests and commit.

## Task 12 — Full CI

Run focused tests, then:

```bash
pytest tests/ -v
```

Require all jobs green in GitHub Actions, including Python 3.10/3.12 SFT compatibility.

No ignored failures.

## Task 13 — Package API16 dataset

**Files:**
- Reuse: `scripts/package_kaggle_dataset.py`
- Generated: `kaggle_dataset/staged/`

- [ ] Verify inherited dataset before packaging.

- [ ] Package exact clean final HEAD.

- [ ] Verify:
```text
dataset_manifest.json API16 + final SHA
code_manifest.json API16 + final SHA
code/LegalQA/code_manifest.json API16 + final SHA
```

- [ ] Verify inherited data hashes unchanged.

- [ ] Verify Liger pin present in packaged requirements.

- [ ] Upload new Kaggle dataset version if deployment workflow permits.

- [ ] Re-fetch/inspect remote metadata/manifests.

- [ ] Agent must not start notebook GPU execution.

## Task 14 — User Kaggle worst-case probe

User runs T4×2.

Acceptance:
```text
API16 PASS
Liger 0.8.2 PASS
selective fused-linear CE active
24-ish worst-case microbatches
3 optimizer steps complete
no CUDA OOM
adapter save PASS
strict reload PASS
generation PASS
```

If FAIL: follow `12_FAILURE_RECOVERY_AND_FALLBACKS.md`.

## Task 15 — User Kaggle endurance probe

Change notebook profile only:

```text
generator_probe_endurance
```

No runtime API bump if runtime code is unchanged.

Acceptance:
```text
30 optimizer steps
no monotonic memory leak
finite loss
peak VRAM recorded
seconds/step recorded
strict reload PASS
```

## Task 16 — Protocol-8 screen

Change profile only:

```text
screen_fold0
```

Run full screen, create promotion report/config/handoff.

Stop after screening.

## Task 17 — Final production freeze

Commit the exact promoted config.

Set:

```text
final_train_and_submit
ALLOW_UNVALIDATED_FINAL=False
```

Run CI and final package/provenance checks.

User runs final all-data Kaggle training and public inference.

## Required completion report

```text
FINAL HEAD:
RUNTIME API:
LIGER VERSION:
INHERITED DATA HASH CHECK:
CI:
API16 DATASET VERSION:
REMOTE API/SHA CHECK:
KAGGLE GPU RUN BY AGENT: NO
NEXT PROFILE:
VERDICT:
```
