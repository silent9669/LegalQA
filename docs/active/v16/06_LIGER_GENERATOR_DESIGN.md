# Generator Design — Liger Fused-Linear CE

## Problem being solved

Current TRL 1.12 chunked NLL still reaches:

```python
logits = h.float() @ w.float().t()
```

on the first QLoRA training forward. Token chunk reduction from 256 to 32 did not change the 594 MiB failed allocation.

The new backend must remove that external full FP32 LM-head path.

## Module boundary

Create:

```text
src/task2/generation/
├── config.py
├── dataset.py
├── memory.py
├── liger_backend.py
├── trainer.py
└── inference.py
```

### `config.py`

Define immutable validated training configuration:

```python
@dataclass(frozen=True)
class GeneratorTrainConfig:
    model_id: str
    max_seq_len: int = 2048
    batch_size: int = 1
    grad_accum: int = 8
    learning_rate: float = 1e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    activation_offloading: bool = True
    use_liger_fused_ce: bool = True
```

Validation must reject any production config that silently changes the approved values.

### `dataset.py`

Move the existing answer-preserving example builder here with tests proving:

- full gold answer is retained;
- evidence is trimmed before answer;
- examples that cannot fit answer+framing are dropped;
- fold exclusion occurs before training;
- worst-case probe can rank by total/completion token lengths.

Return explicit metadata:

```python
@dataclass
class SFTExample:
    prompt: str
    completion: str
    total_tokens: int
    completion_tokens: int
    qa_id: str
```

### `memory.py`

Provide:

```python
def cleanup_cuda_stage(*objects, devices=(0, 1)) -> None
def snapshot_cuda_memory(label: str) -> dict
class TrainerMemoryCallback(TrainerCallback)
```

The callback records memory every configurable N optimizer steps.

### `liger_backend.py`

Own all Liger-specific behavior.

Interface:

```python
@dataclass(frozen=True)
class LigerBackendStatus:
    version: str
    enabled: bool
    fused_linear_ce: bool
    config: dict

def build_liger_training_kwargs() -> dict
def validate_liger_environment() -> LigerBackendStatus
```

Target kwargs:

```python
{
    "use_liger_kernel": True,
    "liger_kernel_config": {
        "rope": False,
        "rms_norm": False,
        "swiglu": False,
        "cross_entropy": False,
        "fused_linear_cross_entropy": True,
    },
}
```

Do not monkeypatch TRL's `_CHUNKED_LM_HEAD_CHUNK_SIZE` in the V16 active path.

### `trainer.py`

Primary interface:

```python
def train_generator_qlora(
    *,
    model_name_or_path: str,
    qa_path: str,
    labels_path: str,
    chunks_path: str,
    output_dir: str,
    config: GeneratorTrainConfig,
    val_fold: int | None,
    max_steps: int | None,
    probe_mode: str | None,
    device: str = "cuda:0",
) -> dict:
    ...
```

Required steps:

1. validate runtime;
2. build/tokenize data;
3. select probe subset if requested;
4. cleanup CUDA stages;
5. load Qwen 4-bit NF4;
6. configure LoRA;
7. construct SFTConfig with completion-only loss and activation offloading;
8. enable selective Liger fused-linear CE;
9. force Trainer `n_gpu=1`;
10. train;
11. save adapter;
12. strict reload in 4-bit mode;
13. generate non-empty test output;
14. write manifest.

## SFT loss setup

Do not set:

```python
loss_type="chunked_nll"
```

when Liger fused-linear CE is enabled.

The implementation should use the Liger-compatible standard loss path and let the model patch perform fused-linear CE.

## Worst-case probe selection

After constructing all fold-filtered examples:

```text
A = top 12 by total_tokens
B = top 12 by completion_tokens
probe = stable deduplicated union(A, B)
```

If fewer than 24 unique examples remain, include the next-longest examples until at least 24 are present.

Do not alter the examples themselves.

The 3-step probe uses:

```text
batch=1
grad_accum=8
3 optimizer steps
≈ 24 microbatches
```

so it intentionally covers worst-case samples.

## Endurance probe

After worst-case PASS:

```text
probe_mode = endurance
max_train_examples = None
max_steps = 30
sampling = deterministic normal training order/random seed 42
```

This measures:

- peak memory;
- memory growth;
- seconds/optimizer-step;
- strict reload;
- generator stability.

## Manifest

Write:

```json
{
  "runtime_api_version": 16,
  "backend": "liger_fused_linear_ce",
  "liger_version": "0.8.2",
  "model": "Qwen/Qwen2.5-3B-Instruct",
  "max_seq_len": 2048,
  "lora_r": 16,
  "lora_alpha": 32,
  "activation_offloading": true,
  "trainer_n_gpu": 1,
  "probe_mode": null,
  "dataset_size": 0,
  "optimizer_steps": 0,
  "peak_vram_mb": 0,
  "peak_reserved_mb": 0,
  "seconds_per_optimizer_step": 0,
  "strict_reload": "pass"
}
```

## Correctness parity test

Before GPU release, implement a small deterministic test comparing fused-linear CE against a reference CE for a tiny tensor/model configuration.

Acceptance:

```text
finite losses
same label masking semantics
relative/absolute difference within documented tolerance
finite gradients
matching gradient shapes
```

The exact tolerance must be derived from FP16/FP32 test behavior and committed in the test, not left as an arbitrary comment.

## Compatibility wrapper

Keep:

```text
src/task2/training/train_generator.py
```

as a temporary wrapper so current imports and notebook tests do not break during the migration.

Once the V16 notebook uses the new module and all tests pass, the wrapper can be reduced to re-exporting the new entrypoint.
