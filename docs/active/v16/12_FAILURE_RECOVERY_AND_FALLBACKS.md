# Failure Recovery and Fallback Architecture

## No more unbounded tweak loop

Use this exact decision tree.

```text
V16 Liger worst-case probe
        │
        ├── PASS
        │     ↓
        │  30-step endurance
        │     │
        │     ├── PASS -> Protocol-8 screen
        │     └── FAIL -> investigate measured endurance failure
        │
        └── FAIL
              ↓
       classify exact failure
```

## Failure class A — Liger install/import/config

Examples:

```text
package install failure
unsupported selective config
kernel import failure
compile failure on sm_75
```

Action:

1. preserve full version/trace;
2. verify exact `liger-kernel==0.8.2`;
3. test supported standard Liger Qwen2 patch;
4. do not reduce model/sequence length yet.

If v0.8.2 is proven incompatible with T4 + current stack, choose another **exact tested** Liger release only after a minimal compatibility probe.

## Failure class B — Liger still OOMs on first step

Do not change CE chunk size again.

Proceed to **Plan B: remote LM-head CE on GPU1**.

### Plan B concept

```text
cuda:0:
Qwen decoder + LoRA

small hidden chunks
    ↓ peer copy

cuda:1:
detached/frozen LM-head shadow
cross entropy
    ↓ loss/input gradients

cuda:0:
LoRA backward
```

Requirements:

- no FP32 full-head temporary on cuda:0;
- copy half weight to cuda:1 first, cast there;
- preserve loss semantics;
- autograd gradient to hidden state validated;
- free GPU1 shadow before reranker/evaluation stage.

This is a separate API release and requires a new design/test plan.

## Failure class C — remote-head architecture also fails

Only then evaluate:

```text
max_seq_len = 1536
```

Before accepting, compute:

```text
kept/dropped count
answer-fit count
evidence truncation delta
P90/P95/max token lengths
official-metric impact on held-out subset
```

Do not truncate gold answers.

## Failure class D — throughput exceeds Kaggle wall time

Do not automatically reduce quality.

Investigate in order:

1. measured seconds/step;
2. dataloader/tokenization overhead;
3. unnecessary checkpoint saves;
4. safe empty-cache frequency;
5. Liger speed path;
6. one-epoch step count.

Only then consider training-scope/hyperparameter trade-offs.

## Failure class E — NaN/Inf

Capture:

```text
step
loss
learning rate
grad norm if available
batch token lengths
GPU memory
```

First investigate numerical backend/precision, not data deletion.

## Failure class F — strict reload fails after training

Do not rerun full training immediately.

Audit:

```text
adapter_config.json
adapter weight files
base model ID
tokenizer
quantization load path
PEFT version
generator manifest
```

The checkpoint is not accepted until strict neural reload and non-empty generation pass.

## Failure class G — Protocol-8 says QLoRA loses

This is not a runtime failure.

If official held-out METEOR says the non-generator candidate wins, **do not force QLoRA into final production**.

The purpose of screening is to maximize score, not justify training work already spent.
