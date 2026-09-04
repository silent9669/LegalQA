# LegalQA Task 2 — V16 Fresh-Start Refresh Pack

**Date:** 2026-09-04  
**Repository:** `silent9669/LegalQA`  
**Inherited source baseline:** `151313fc3126615ec11c08ca68f154d5b0c5406f` (V15 / Runtime API 15)  
**Target runtime:** API 16  
**Primary objective:** reach a stable, score-preserving, end-to-end Kaggle Dual-T4 training run without further speculative memory tweaks.

## Why this refresh exists

The current V15 release is structurally healthy: runtime binding, dataset manifests, BM25, DEk21, dual-T4 device assignment, Qwen 4-bit loading, activation offloading, single-GPU Trainer policy, and the generator-only probe all behave as designed. The remaining failure is isolated to the language-model loss backend.

The authoritative V15 Kaggle probe applied `CE chunk 256 -> 32`, loaded the full 5,956-example fold-filtered training source pool, and still failed on the first QLoRA forward with an attempted **594 MiB** allocation at:

```python
logits = h.float() @ w.float().t()
```

This means the active failure is not primarily the token chunk size. The next architecture therefore replaces TRL `chunked_nll` with a supported **Liger fused-linear cross-entropy** backend while preserving the rest of Stack A.

## New source-of-truth hierarchy

From this point forward, coding agents should use the documents in this ZIP in this order:

1. `01_CURRENT_STATE_AND_EVIDENCE.md`
2. `02_FINAL_ARCHITECTURE_V16.md`
3. `03_FRESH_WORKSPACE_STRUCTURE.md`
4. `04_INHERITED_DATASET_CONTRACT.md`
5. `05_RUNTIME_DEPENDENCY_LOCK.md`
6. `06_LIGER_GENERATOR_DESIGN.md`
7. `07_IMPLEMENTATION_PLAN.md`
8. `08_TESTING_AND_ACCEPTANCE_GATES.md`
9. `09_KAGGLE_DUAL_T4_RUNBOOK.md`
10. `10_PROTOCOL8_AND_FINAL_TRAINING.md`
11. `11_RELEASE_AND_PACKAGING_API16.md`
12. `12_FAILURE_RECOVERY_AND_FALLBACKS.md`
13. `13_AGENT_OPERATING_RULES.md`
14. `14_AGENT_PROMPTS.md`
15. `15_DEFINITION_OF_DONE.md`

Older V7–V15 fix documents remain historical evidence only. They must not override this pack.

## What is inherited vs rebuilt

### Inherit without regeneration

- `legal_chunks.parquet`
- `qa_unique.parquet`
- `known_qa.json`
- `qa_citations.parquet`
- `retrieval_labels.parquet`
- `fold_assignments.parquet`
- `reranker_training_pairs.parquet`
- `public-official.json`
- BM25 index
- DEk21 dense index
- official scoring semantics
- Protocol-8 selection logic
- proven retrieval and reranker behavior

### Refresh

- generator training runtime
- memory/loss backend
- execution profiles
- runtime/dependency contracts
- clean orchestration boundaries
- tests around the new generator path
- API16 release package
- Kaggle probe/screen/final run sequence

## Primary architecture decision

```text
Qwen2.5-3B-Instruct
+ 4-bit NF4 / double quantization
+ LoRA r16 / alpha32
+ max_seq_len 2048
+ batch 1 / grad_accum 8
+ completion-only SFT
+ gradient checkpointing
+ activation offloading
+ Liger fused-linear cross entropy
+ generator strictly on cuda:0
+ retrieval/reranker on cuda:1 outside generator training
```

Do **not** use TRL `chunked_nll` for V16.

## Required runtime gates

```text
STATIC + CI
    ↓
worst-case 3-step Kaggle generator probe
    ↓ PASS
30-step Kaggle endurance generator probe
    ↓ PASS
Protocol-8 screen_fold0
    ↓
PROMOTED production config
    ↓
final all-data train + public inference
```

No full screen should run before both generator probes pass.

## First action for the coding agent

Create an isolated V16 branch/worktree from `151313fc3126615ec11c08ca68f154d5b0c5406f`, preserve inherited artifacts by hash, implement the new generator backend and profiles, bump Runtime API to 16, run the full test matrix, package a new API16 Kaggle dataset version, and stop before GPU execution.

Only the user starts Kaggle GPU runs.
