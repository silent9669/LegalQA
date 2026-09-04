# V16 Final Candidate Architecture

## Objective

Build one production path that is simple enough to debug, memory-safe on Kaggle T4×2, and still maximizes the chance of a strong official METEOR score.

## System architecture

```text
                        LEGALQA TASK 2
                              │
                  immutable inherited data
                              │
       ┌──────────────────────┼──────────────────────┐
       │                      │                      │
   QA memory               BM25S                DEk21 v2
       │                      │                      │
       └──────────── candidates / RRF ───────────────┘
                              │
                              ▼
                    BGE-reranker-v2-m3
                         on cuda:1
                              │
                              ▼
                       evidence packing
                              │
                              ▼
                  candidate generation layer
                  ┌───────────┴───────────┐
                  │                       │
          extractive/stitch        Qwen2.5-3B
              candidates             on cuda:0
                                         │
                                NF4 + LoRA r16/a32
                                         │
                                  Liger fused-linear
                                   cross entropy SFT
                                         │
                                         ▼
                                  QLoRA candidate
                  └───────────┬───────────┘
                              ▼
                    Protocol-8 selection
                              │
                              ▼
                     PROMOTED final policy
```

## GPU lifecycle

The two T4s are not a replicated training pair. They are stage-specific accelerators.

### Retrieval/reranker stage

```text
cuda:1:
- DEk21 query encoder / retrieval work
- BGE reranker training or inference

cuda:0:
- mostly free
```

### Generator training stage

```text
cuda:0:
- Qwen2.5-3B 4-bit base
- LoRA adapters
- decoder activations
- Liger fused-linear CE

cuda:1:
- no reranker training state
- release dense/reranker tensors where possible
- available as safety headroom, but not required by primary design
```

### Evaluation / inference stage

```text
cuda:0: generator
cuda:1: retriever/reranker
```

This avoids `nn.DataParallel` and model replication.

## Generator training contract

```text
base_model            Qwen/Qwen2.5-3B-Instruct
quantization           4-bit NF4
double_quant           true
compute dtype          FP16 on T4
LoRA r                 16
LoRA alpha             32
LoRA dropout           0.05
LoRA targets           q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
max_seq_len            2048
batch_size             1
gradient_accumulation  8
learning_rate          1e-4
optimizer              paged_adamw_8bit
gradient_checkpointing true
activation_offloading  true
completion_only_loss   true
loss backend           Liger fused-linear cross entropy
TRL chunked_nll        DISABLED
DataParallel           DISABLED
generator device       cuda:0
```

## Why Liger is primary

The current TRL 1.12 `chunked_nll` path still executes:

```python
h.float() @ w.float().t()
```

and the V15 probe shows the full vocabulary weight operation still causes a ~594 MiB allocation after token chunking was reduced to 32.

Liger-Kernel v0.8.2 explicitly supports Qwen2/Qwen2.5 fused-linear cross entropy. The fused-linear design avoids the problematic full external FP32 LM-head projection path and is integrated through the Hugging Face `TrainingArguments` interface.

Pin exactly:

```text
liger-kernel==0.8.2
```

Do not float the dependency in the final release.

## Minimal Liger patch policy

Patch only the required final loss operation.

Target configuration:

```python
use_liger_kernel=True
liger_kernel_config={
    "rope": False,
    "rms_norm": False,
    "swiglu": False,
    "cross_entropy": False,
    "fused_linear_cross_entropy": True,
}
```

The implementation must verify the exact supported config at runtime against installed Transformers/Liger versions and fail loudly if the requested selective patch cannot be applied.

Do not turn on unrelated Liger kernels merely for speed.

## Loss semantics

For V16, TRL `chunked_nll` must not be combined with Liger fused-linear CE.

Use the standard NLL path compatible with the Liger patch while preserving completion-only labels.

The acceptance test is not only “no OOM”; a tiny deterministic parity test must compare the selected Liger loss path with the reference non-fused CE on small synthetic inputs within a documented tolerance.

## Memory hygiene

Before QLoRA starts:

```text
1. destroy reranker trainer/model objects if trained;
2. destroy temporary dense probe tensors;
3. gc.collect();
4. torch.cuda.empty_cache() on both GPUs;
5. log allocated/reserved/free VRAM on both devices.
```

During training:

```text
log every 50 optimizer steps:
allocated
reserved
free
max_allocated
```

Optional fragmentation guard:

```text
torch_empty_cache_steps = 50
```

may be enabled only after measuring its throughput cost in the 30-step endurance probe.

## What V16 intentionally does not change

- data corpus
- BM25 index
- DEk21 index
- BGE reranker model
- official scorer
- answer-preserving training example construction
- Protocol-8 promotion logic
- 2048-token maximum
- LoRA capacity
- parameter accounting

The refresh changes the **generator runtime architecture**, not the competition strategy.
