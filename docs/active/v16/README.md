# LegalQA Task 2 — V16 Active Specification

**Status:** ACTIVE SPECIFICATION  
**Target Runtime API:** 16  
**Baseline Commit:** `151313fc3126615ec11c08ca68f154d5b0c5406f`  
**Authoritative Pack:** `LegalQA_V16_Fresh_Start_Refresh_Pack/` (00_START_HERE.md through 15_DEFINITION_OF_DONE.md)

## Core Architectural Principles
1. **Inherit All Verified Data & Indexes:** No recomputation or regeneration of `legal_chunks.parquet`, `qa_unique.parquet`, BM25 index (7 files), DEk21 dense index (3 files), etc.
2. **Selective Liger Fused-Linear CE Backend:** Replace TRL `chunked_nll` with `liger-kernel==0.8.2` fused-linear cross-entropy only (`use_liger_kernel=True`, `fused_linear_cross_entropy=True`).
3. **Qwen2.5-3B-Instruct 4-bit QLoRA on cuda:0:** NF4, double quant, max_seq_len 2048, batch 1, grad_accum 8, completion-only labels, activation offloading, trainer_n_gpu=1.
4. **Dual-T4 Stage Policy:** Generator training strictly on `cuda:0`. Retrieval/reranking on `cuda:1` outside generator training.
5. **Execution Profiles:**
   - `generator_probe_worstcase`: 3 optimizer steps on worst-case token length samples.
   - `generator_probe_endurance`: 30 optimizer steps on full fold-filtered pool.
   - `screen_fold0`: Full Protocol-8 screening.
   - `final_train_and_submit`: Full production training and submission generation (requires PROMOTED status).
6. **Kaggle Execution Guard:** The agent never runs or triggers Kaggle GPU notebooks. All GPU runs are executed manually by the user on Kaggle T4x2.
