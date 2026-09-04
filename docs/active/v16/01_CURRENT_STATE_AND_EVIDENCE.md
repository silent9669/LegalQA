# Current State and Evidence Register

This document records the evidence that must survive the refresh. Do not reinterpret old failures after implementation begins.

## Baseline repository

```text
GitHub repository: silent9669/LegalQA
V15 HEAD: 151313fc3126615ec11c08ca68f154d5b0c5406f
Commit: feat(generator): add V15 CE chunk size 32 policy and generator_probe profile (API 15)
Runtime API: 15
```

The V15 GitHub Actions run for this exact HEAD completed successfully across:

- CPU unit tests
- bootstrap protected-version preservation
- SFT compatibility on Python 3.10
- SFT compatibility on Python 3.12

## Dataset package verified

The inherited Drive/Kaggle package has:

```text
profile = final_training
runtime_api_version = 15
git_sha = 151313fc3126615ec11c08ca68f154d5b0c5406f
code root = code/LegalQA
code files = 54
BM25 files = 7
DEk21 files = 3
```

The root dataset manifest, root code manifest, nested `code/LegalQA/code_manifest.json`, and `configs/runtime_api.yaml` are all API15 and bound to the same V15 SHA.

## V15 generator probe evidence

Authoritative log: `legalqa-training-5.log`.

### Environment

```text
Python       3.12.13
PyTorch      2.10.0+cu128
CUDA         12.8
Triton       3.6.0
Transformers 5.0.0
Accelerate   1.13.0
Datasets     5.0.0
PEFT         0.19.1
TRL          1.12.0
bitsandbytes 0.50.2
GPU 0        Tesla T4 ~14.6 GiB
GPU 1        Tesla T4 ~14.6 GiB
```

### Runtime controls proven

```text
EXECUTION_PROFILE = generator_probe
runtime API = 15
generator = cuda:0
retrieval/reranker = cuda:1
reranker training = OFF
generator training = ON
max_generator_steps = 3
max_generator_examples = None
trainer_n_gpu = 1
activation offloading active
```

### Training data shape

```text
5,956 examples kept
13 dropped
P50  = 726 tokens
P90  = 1,223
P95  = 1,441
Max  = 2,047
Evidence truncated = 1.0%
```

The current preprocessing is answer-preserving: it trims evidence before the gold answer and drops only examples whose answer plus minimal chat framing cannot fit.

### Memory before failure

```text
After 4-bit base model load:
allocated ≈ 1,969.3 MiB
reserved  ≈ 2,028.0 MiB
free      ≈ 1,566.8 MiB

After SFTTrainer construction:
allocated ≈ 2,026.4 MiB
reserved  ≈ 2,156.0 MiB
free      ≈ 1,418.8 MiB
```

### Experiment that was disproven

V15 successfully changed TRL:

```text
_CHUNKED_LM_HEAD_CHUNK_SIZE:
256 -> 32
```

Yet the first training step still failed with:

```text
OutOfMemoryError
requested allocation: 594.00 MiB
GPU0 free: ~224.81 MiB
failure:
trl/trainer/sft_trainer.py
  _chunk(...)
  logits = h.float() @ w.float().t()
```

Therefore **do not spend another release changing the CE chunk from 32 to 16/8/4**.

## Why the 594 MiB number matters

Qwen2.5-3B-Instruct uses approximately:

```text
vocab_size  = 151,936
hidden_size = 2,048
```

A full half-precision vocabulary weight is:

```text
151936 × 2048 × 2 bytes ≈ 593.5 MiB
```

This closely matches the failed 594 MiB allocation and strongly localizes the problem to the full LM-head weight/cast/projection path.

## Official score semantics that must not change

The repository's official scoring program computes:

```python
meteor_score(
    [str(y_true[k]).split()],
    str(y_pred[k]).split()
)
```

and mean ROUGE-L as the secondary metric.

Consequences:

- whitespace-level lexical overlap is important;
- long unsupported paraphrase can hurt;
- legal wording copied or grounded from evidence can be valuable;
- candidate selection must be validated with the official metric implementation, not an approximate tokenizer.

## Parameter budget

Current Stack A is configured at:

```text
DEk21 v2 dense retriever
+ BGE-reranker-v2-m3
+ Qwen2.5-3B-Instruct
= 3,758,000,000 learned parameters
< 4,000,000,000
margin = 242,000,000
```

The refresh must not add another independently loaded learned model that breaks the budget.

## Decisions already made

Do not reopen these without new evidence:

1. Keep Stack A.
2. Keep Qwen2.5-3B-Instruct.
3. Keep max sequence length 2048 for the first V16 implementation.
4. Keep LoRA r16/alpha32 and existing target modules.
5. Keep batch 1 and grad accumulation 8.
6. Keep full gold answers.
7. Keep fold-0 isolation for Protocol-8 screening.
8. Keep generator on `cuda:0`.
9. Keep reranker/retrieval on `cuda:1` outside generator training.
10. Do not use DataParallel/DDP/FSDP/ZeRO for this 3B QLoRA path.
11. Replace the failing loss backend before reducing information/capacity.
