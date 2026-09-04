# Coding-Agent Prompts

## Full handoff prompt

```text
Treat the V16 Fresh-Start Refresh Pack as the active source of truth for silent9669/LegalQA Task2. Start from V15 HEAD 151313fc3126615ec11c08ca68f154d5b0c5406f in an isolated V16 worktree. Preserve the verified V15 dataset/index bytes by SHA; do not regenerate data. Build Runtime API16 with clean generation/runtime/pipeline modules. Replace the failing TRL 1.12 chunked_nll path with selective Liger v0.8.2 fused-linear cross entropy only; keep Qwen2.5-3B NF4, seq2048, LoRA r16/a32, batch1, gradaccum8, completion-only loss, activation offloading and single-GPU Trainer cuda:0. Keep retrieval/reranker cuda:1 and Protocol-8 semantics. Add worst-case 3-step and full-pool 30-step generator profiles, strict tests, API15 rejection, CI, API16 packaging and remote manifest verification. Never run Kaggle GPU. Stop at READY FOR USER MANUAL KAGGLE GENERATOR PROBE or BLOCKED.
```

## `/goal` prompt — under 1000 characters

```text
/goal

Use the V16 Fresh-Start Refresh Pack as the only active implementation spec. Start from LegalQA V15 HEAD 151313fc3126615ec11c08ca68f154d5b0c5406f. Inherit all verified dataset/index bytes by SHA; do not regenerate data. Build clean Runtime API16 modules for generation/runtime/profiles. Replace TRL 1.12 chunked_nll with selective liger-kernel==0.8.2 fused-linear CE only. Keep Qwen2.5-3B NF4, seq2048, LoRA r16/a32, batch1, gradaccum8, completion-only labels, activation offloading, trainer_n_gpu=1 on cuda:0; retrieval/reranker stay cuda:1. Add worst-case 3-step and full-pool 30-step generator probe profiles, strict adapter reload, memory telemetry, tests, full CI, stale API15 rejection, API16 repack/upload/remote SHA verification. Never run/push Kaggle GPU notebook; I run T4x2. End READY FOR USER MANUAL KAGGLE GENERATOR PROBE or BLOCKED.
```

## After worst-case probe passes

```text
/goal

The API16 V16 worst-case generator probe passed on Kaggle T4x2. Do not change runtime code, model, data or API. Switch only the committed notebook/profile to generator_probe_endurance: full fold-filtered source pool, max_steps=30. Keep all V16 guards and logging. Run notebook-contract/full CI, verify the attached API16 dataset still matches the exact runtime package. Never execute Kaggle yourself. Return exact HEAD and READY FOR USER MANUAL KAGGLE ENDURANCE PROBE.
```

## After endurance passes

```text
/goal

The API16 V16 30-step generator endurance probe passed on Kaggle T4x2 with stable memory and strict reload. Do not change generator/runtime/data. Switch only to screen_fold0, preserve Protocol-8 R0G0/R1G0/R_SELECTED_G1 evaluation, auto-promotion and screen_handoff.zip. Keep ALLOW_UNVALIDATED_FINAL=False. Run full CI and static release audit. Never run Kaggle. Return READY FOR USER MANUAL KAGGLE SCREEN.
```

## After screen handoff

```text
/goal

Audit the supplied Protocol-8 screen_handoff. Verify protocol=8, hashes/provenance, all required systems, official METEOR, winner and promoted config. Commit exactly the measured PROMOTED production config; do not hand-pick components. Switch notebook to final_train_and_submit with ALLOW_UNVALIDATED_FINAL=False. Run CI/release/package checks. Never run Kaggle. Return READY FOR USER MANUAL FINAL KAGGLE TRAIN or BLOCKED.
```
