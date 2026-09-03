# LegalQA V14 — Kaggle Full-Shape QLoRA OOM Fix

## Goal

Fix the **first real `screen_fold0` Kaggle Dual-T4 failure** without changing the competitive architecture or training objective.

The user alone runs Kaggle GPU notebooks. Coding agents may modify/test/package/upload code and datasets, and may run Colab T4 probes, but **must not trigger Kaggle GPU execution**.

---

## 1. Authoritative failure evidence

Real Kaggle run:

- Notebook profile: `screen_fold0`
- Runtime API: `13`
- Dataset/code runtime SHA: `757620ce8ccd41753ff217d4fe6593627a196899`
- Hardware: 2 × Tesla T4, 14.6 GiB each
- Generator: `cuda:0`
- Retrieval/reranker: `cuda:1`
- Preflight: PASS
- BM25: PASS
- DEk21: PASS
- Public test: PASS
- Parameter budget: `3,758,000,000 < 4,000,000,000`
- Reranker full fold-0 training: PASS
- Reranker reload: PASS
- QLoRA model load: PASS
- Single-GPU Trainer guard: PASS (`trainer_n_gpu=1`)
- Failure occurs on the **first real QLoRA training step**.

Observed QLoRA data:

```text
5956 kept
13 dropped
P50 = 726 tokens
P90 = 1223
P95 = 1441
Max = 2047
Evidence truncated = 1.0%
```

Observed failure:

```text
trainer.train(...)
-> TRL SFTTrainer.compute_loss(...)
-> _chunked_cross_entropy_loss(...)
-> _chunk(...)
-> logits = h.float() @ w.float().t()

CUDA OOM
Tried to allocate: 594.00 MiB
GPU0 total:         14.56 GiB
GPU0 free:          182.81 MiB
Process in use:      14.38 GiB
PyTorch allocated:   3.23 GiB
PyTorch reserved:    0.10 GiB
```

This is **not** the old DataParallel issue: `target=cuda:0`, `visible_cuda=2`, `trainer_n_gpu=1` was already verified.

This is also **not** a reranker, data, index, API-binding, or fallback failure.

---

## 2. Why the old Colab gate missed it

The previous Colab “full” generator smoke used:

```text
30 optimizer steps
max_train_examples = 128
```

and passed on a T4 with about 4.3 GiB PyTorch peak allocation.

That gate did not exercise the **full 5956-example length distribution**. The real Kaggle run did, and the first full-shape batch hit the TRL chunked-CE memory path.

Therefore, do **not** treat the old 128-example Colab pass as sufficient evidence for full training.

---

## 3. Constraints — do not regress these

Keep unchanged unless a later measured experiment proves it unavoidable:

```text
Stack A
Qwen/Qwen2.5-3B-Instruct
4-bit NF4 + double quantization
LoRA r=16
LoRA alpha=32
LoRA dropout=0.05
target_modules = q/k/v/o/gate/up/down
max_seq_len = 2048
completion_only_loss = True
batch_size = 1
gradient_accumulation = 8
fold 0 held out during screen training
generator on cuda:0
reranker/retrieval on cuda:1
official METEOR semantics
parameter budget < 4B
ALLOW_UNVALIDATED_FINAL = False
ALLOW_SINGLE_GPU_SMOKE = False
```

Do not solve this by silently dropping long answers, using fallback generation, disabling QLoRA, or reducing the model.

---

## 4. Phase A — reproduce the real failure shape in Colab

Before changing memory behavior, add a **full-shape** generator probe.

### Required probe

Use the exact current Kaggle user-space stack:

```text
transformers 5.0.0
accelerate 1.13.0
datasets 5.0.0
peft 0.19.1
trl 1.12.0
bitsandbytes 0.50.2
sentence-transformers 5.4.1
```

Use a single Tesla T4 and the complete generator data:

```text
qa_unique.parquet
retrieval_labels.parquet
legal_chunks.parquet
```

Run:

```text
val_fold = 0
max_train_examples = None
max_seq_len = 2048
max_steps = 3
batch_size = 1
grad_accum = 8
```

The crucial difference from the old smoke is:

```text
max_train_examples = None
```

Do not random-sample 128 examples.

### Add diagnostics

Before `trainer.train()`, print non-secret diagnostics:

```text
TRL version
Transformers version
PEFT version
bitsandbytes version
actual TRL chunked-LM-head chunk size, if present
torch.cuda.mem_get_info(cuda:0)
torch.cuda.memory_allocated(cuda:0)
torch.cuda.memory_reserved(cuda:0)
```

Log again:

1. after 4-bit base-model load;
2. after `SFTTrainer` construction;
3. immediately before `trainer.train()`;
4. after the first successful optimizer step.

Never print environment variables or `HF_TOKEN`.

### Expected Phase-A result

On unchanged V13 code, this probe should either reproduce the OOM or show dangerously low VRAM headroom. If it does not reproduce, retain the diagnostics and continue only with an evidence-based hypothesis.

---

## 5. Phase B — preferred fix: preserve 2048-token semantics

The real Kaggle runtime uses TRL `1.12.0`. Its `SFTConfig` supports:

```text
completion_only_loss
loss_type
activation_offloading
max_length
processing_class
```

Make the memory policy explicit rather than relying on evolving TRL defaults.

### 5.1 Pin the tested TRL runtime

Change Kaggle/Colab runtime contract from a broad floor to the exact tested version:

```text
trl==1.12.0
```

Do not change Torch/CUDA packages.

Update bootstrap/runtime API checks so the installed TRL is verified against the required SFT features.

### 5.2 Explicitly keep memory-efficient loss

Set:

```python
loss_type="chunked_nll"
completion_only_loss=True
```

Do not switch to standard full-logit NLL as an OOM workaround.

### 5.3 Enable activation offloading for QLoRA on T4

Set:

```python
activation_offloading=True
```

for the QLoRA `SFTConfig`.

This is the first preferred fix because it preserves:

- all 5956 training examples that currently fit;
- the 2048-token limit;
- full-answer-preserving preprocessing;
- LoRA rank/targets;
- loss semantics.

Do not simultaneously reduce `max_seq_len` in the first experiment.

### 5.4 Inspect TRL chunk size before changing it

Read/log:

```python
trl.trainer.sft_trainer._CHUNKED_LM_HEAD_CHUNK_SIZE
```

If the **actual installed wheel** reports a value greater than `256`, test a guarded cap to `256` as the next isolated hypothesis.

Do not patch this private constant blindly. Guard by:

```text
TRL version == 1.12.0
attribute exists
current value > 256
```

and log the before/after value.

If the installed value is already `<=256`, do not modify it.

---

## 6. Phase C — full-shape Colab acceptance gate

After the preferred fix:

### Gate 1

Run the full-shape probe:

```text
5956-example source pool
max_train_examples=None
max_steps=3
max_seq_len=2048
Tesla T4
```

Required:

```text
no CUDA OOM
no fallback
trainer_n_gpu = 1
loss finite
checkpoint save succeeds
strict adapter reload succeeds
non-empty generation succeeds
```

### Gate 2

If Gate 1 passes, run at least:

```text
max_steps=30
max_train_examples=None
```

to cover more of the full dataset ordering.

Capture:

```text
peak PyTorch allocated VRAM
peak reserved VRAM
free GPU memory after step
wall time
```

Do not claim the fix is ready from a 128-example sampled smoke.

---

## 7. Last-resort fallback only if offloading still fails

Only after the full-shape Colab experiment proves `activation_offloading=True` is insufficient:

1. test guarded chunk-size cap if the runtime chunk size is `>256`;
2. rerun the same full-shape probe;
3. only if that also fails, evaluate `max_seq_len=1536`.

Do **not** immediately reduce sequence length.

If 1536 is evaluated, first produce answer-preservation diagnostics:

```text
kept examples
dropped examples
drop rate
P90/P95/max token length
evidence truncation rate
number of answers that cannot fit even with zero evidence
```

Do not accept 1536 unless the quality trade-off is explicitly measured.

---

## 8. Runtime release binding

Because the preferred fix changes packaged runtime code/dependency behavior, create **Runtime API 14**.

Required:

```text
configs/runtime_api.yaml                 -> 14
src/task2/runtime_integrity.py           -> EXPECTED_RUNTIME_API_VERSION = 14
notebook REQUIRED_RUNTIME_API_VERSION    -> 14
dataset_manifest.json                    -> API 14
root code_manifest.json                  -> API 14
code/LegalQA/code_manifest.json          -> API 14
all three manifests use identical final 40-char Git SHA
```

Old API13 packages must fail loud under an API14 notebook.

Do not regenerate immutable legal data or indexes just because runtime code changed.

---

## 9. Tests

Add/adjust tests for:

1. exact TRL runtime contract;
2. `SFTConfig` requires `loss_type` and `activation_offloading`;
3. QLoRA config explicitly uses `chunked_nll`;
4. QLoRA T4 policy enables activation offloading;
5. optional chunk-size guard only activates for TRL 1.12.0 and values `>256`;
6. `max_seq_len` remains 2048;
7. LoRA r/alpha/targets unchanged;
8. completion-only loss remains enabled;
9. single-GPU Trainer policy remains enforced before `SFTTrainer`;
10. full-shape Colab mode uses `max_train_examples=None`;
11. stale API13 package is rejected;
12. API14 nested Kaggle package resolves correctly;
13. no secret scanning regression.

Run full CI, including Python 3.10/3.12 compatibility lanes.

---

## 10. Packaging and remote verification

After code + CI + full-shape Colab gates pass:

1. obtain exact clean final HEAD;
2. rebuild `kaggle_dataset/staged --profile final_training`;
3. verify API14 + final SHA in all manifests;
4. verify packaged `requirements-kaggle.txt` contains the exact TRL contract;
5. preserve existing:
   - `legal_chunks.parquet`
   - `qa_unique.parquet`
   - `known_qa.json`
   - `fold_assignments.parquet`
   - `retrieval_labels.parquet`
   - `reranker_training_pairs.parquet`
   - BM25 index
   - DEk21 index
   - public-official.json
6. upload a **new version** of `phucdangg/legalqa-task2-clean-data`;
7. independently download/inspect the remote metadata/manifests after upload.

Do not run a Kaggle GPU notebook.

---

## 11. First-run checkpoint handling

The first Kaggle run produced a valid **fold-0 screen reranker** checkpoint:

```text
base_model = BAAI/bge-reranker-v2-m3
val_fold_excluded = 0
training_scope = folds_excluding_0
is_final_checkpoint = false
smoke_only = false
num_training_pairs = 19216
best_val_loss = 0.3389
best_val_accuracy = 0.8711
peak_vram_mb = 10874.65
```

Preserve it as evidence/debugging output.

Do **not** label it as a final competition checkpoint because fold 0 was excluded and `is_final_checkpoint=false`.

Do not add a complicated resume path unless the full corrected screen later risks the Kaggle wall-time limit.

---

## 12. Coding-agent completion report

Return exactly:

```text
FINAL HEAD:
CI:
RUNTIME API:
TRL CONTRACT:
FULL-SHAPE COLAB 3-STEP:
FULL-SHAPE COLAB 30-STEP:
QLORA PEAK VRAM:
ADAPTER RELOAD:
DATASET REPACKAGED:
REMOTE KAGGLE DATASET VERSION:
REMOTE API/SHA CHECK:
KAGGLE GPU RUN BY AGENT: NO
VERDICT: READY FOR USER MANUAL KAGGLE SCREEN / BLOCKED
```

No “READY” verdict without fresh full-shape T4 evidence and remote package verification.
