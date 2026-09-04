# Definition of Done

V16 is not complete when code compiles. It is complete only when each required level is satisfied.

## A. Workspace

- [ ] active V16 docs established;
- [ ] historical docs clearly non-authoritative;
- [ ] clean generator/runtime/profile module boundaries;
- [ ] notebook is a thin launcher;
- [ ] no duplicated live V14/V15 CE-chunk workaround.

## B. Inherited data

- [ ] all core SHA256 values match;
- [ ] BM25 complete;
- [ ] DEk21 complete;
- [ ] public set complete;
- [ ] no regeneration performed.

## C. Generator architecture

- [ ] Qwen2.5-3B-Instruct;
- [ ] NF4 4-bit;
- [ ] double quantization;
- [ ] FP16 T4 compute;
- [ ] LoRA r16/a32/dropout .05;
- [ ] current target modules unchanged;
- [ ] seq 2048;
- [ ] batch1/gradaccum8;
- [ ] completion-only labels;
- [ ] activation offloading;
- [ ] gradient checkpointing;
- [ ] selective Liger fused-linear CE;
- [ ] TRL chunked_nll inactive;
- [ ] Trainer n_gpu=1.

## D. Runtime

- [ ] API16 everywhere;
- [ ] TRL exactly 1.12.0;
- [ ] Liger exactly 0.8.2;
- [ ] Torch/CUDA/Triton protected;
- [ ] async HF load disabled;
- [ ] dual-T4 mapping validated.

## E. Tests

- [ ] unit suite;
- [ ] dataset contract;
- [ ] loss backend contract;
- [ ] profile contract;
- [ ] runtime release binding;
- [ ] package integrity;
- [ ] Python 3.10 SFT lane;
- [ ] Python 3.12 SFT lane;
- [ ] full CI green.

## F. Release

- [ ] exact clean final HEAD;
- [ ] tripartite API16/SHA parity;
- [ ] inherited hashes unchanged;
- [ ] remote Kaggle dataset re-audited;
- [ ] no secrets.

## G. GPU evidence

### Worst-case probe
- [ ] PASS

### 30-step endurance
- [ ] PASS

### Protocol-8
- [ ] complete report;
- [ ] PROMOTED config;
- [ ] handoff archive.

### Final
- [ ] all-data training completes according to promoted policy;
- [ ] strict checkpoint reload;
- [ ] public inference 1,000 answers;
- [ ] submission validation;
- [ ] final hashes/provenance saved.

## Final state

Only after every applicable box above is satisfied:

```text
LEGALQA TASK2 V16 — FINAL TRAINING PIPELINE VERIFIED
```
