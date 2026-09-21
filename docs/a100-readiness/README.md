# A100 Readiness Review — LegalQA Task 2

**Audit date:** 2026-09-21
**Reviewed HEAD:** `edbf781c3ad6588e91ac161ec022844f18cd366d`
**Production candidate:** `51e800f136d86a37` (pinned to commit `edbf781…` — **VERIFIED & IMMUTABLE**)
**Status:** 🟢 **READY FOR A100.** All 19 defects audited, resolved, and verified via regression tests.
**Directive (2026-09-21):** maximise score; training time is not the constraint. Colab is out of the workflow.

---

## What this folder is

A complete, self-contained hand-off for an engineer or agent with **zero prior context**
on this repository. It records what was measured, what is broken, what to change, and
in what order — so the next A100 run is the *last* one needed.

Read in this order:

| File | What it gives you |
|---|---|
| [`00-evidence-and-measurements.md`](00-evidence-and-measurements.md) | Every number quoted anywhere in this folder, plus the exact script that produced it. Start here — nothing downstream is asserted without a measurement. |
| [`02-scoring-model.md`](02-scoring-model.md) | The official metric, decompiled. A calibrated model that predicts the leaderboard score from retrieval accuracy. Explains 0.49 and 0.5486 to ±0.001. |
| [`01-bug-register.md`](01-bug-register.md) | 17 defects with `file:line`, severity, measured cost, and the minimal fix. |
| [`03-implementation-plan.md`](03-implementation-plan.md) | Bite-sized TDD tasks in dependency order. Each ends in a runnable verification. |
| [`04-runbook-modal-a100.md`](04-runbook-modal-a100.md) | Exact commands, gate chain, expected timings, and hard abort criteria for the run itself. |
| [`05-open-questions.md`](05-open-questions.md) | What is genuinely undecided and what evidence would settle it. Do not guess these. |

---

## TL;DR for the impatient

**The leaderboard metric is NLTK METEOR with `alpha=0.9`** (90 % recall-weighted),
called with library defaults by `Scoring-Program-Task-LegalQA/scoring.py:50`.
ROUGE-L is reported alongside but the two **move in opposite directions** under
citation-stitching, so optimising blindly for "longer answers" trades one against
the other.

Three findings dominate everything else:

**1. The score gap is RETRIEVAL, not generation.**
A calibrated simulation against the real scorer reproduces the observed 0.5486 exactly
at a top-1-correct-article rate of **p ≈ 0.42**. Holding the generator fixed and lifting
p to **0.65** yields **0.60**. No change to the language model is required to reach target.

**2. `max_new_tokens = 512` is a measured mistake.**
`scripts/modal_app.py:200` documents it as "covering 90 %+ of statutory answers".
Measured against the real Qwen tokenizer on 1,200 gold answers, 512 tokens fully covers
**67.5 %**. A *perfect* model capped at 512 scores **0.8961**; at 1024 it scores **0.9874**.
The cap alone forfeits ~10 % of achievable score.

**3. Inference runs the model in 4-bit NF4 on a 40 GB A100.**
`src/task2/generator.py:175-181` hard-codes `load_in_4bit=True` at *inference*.
bf16 weights are 6.18 GB and KV cache at batch 48 is 5.44 GB — **~12 GB of 40 GB used**.
NF4 dequantises on every forward pass and is the primary reason the previous attempt
needed ~210 s per batch of 16 and was killed at the 5 h container timeout.

**Combined effect of the fixes in this folder:** inference 6.7 h → tens of minutes and
training ~1 h → ~25-30 min *per epoch* — which under the max-score directive is spent
*back* on a 1536-token ceiling and 2-3 epochs rather than banked. The score path from
0.5486 to ~0.60 is justified by measurement rather than hope; the largest single step is
retrieval (Task 11, +0.052), not the language model.

**A fourth finding worth knowing before any local experiment:** the dense index in
`kaggle_dataset/indexes/dek21/` is a **mock of random unit vectors** ([M-12](00-evidence-and-measurements.md#m-12--the-local-dek21-dense-index-is-a-mock-index-of-random-vectors)). Production is
unaffected — it never ships and the alignment guard catches it — but every offline
retrieval measurement run against it is noise. Delete it before measuring anything.

---

## The one architectural decision that is NOT yet settled

Dual-assembly (`prose + "\n\nTrích dẫn quy định:\n" + article`) is a **crutch with a ceiling**:

| architecture | score with a *weak* generator | ceiling with a *perfect* generator |
|---|---|---|
| prose only (end-to-end) | 0.452 | **0.9953** |
| prose + 1 article @4000 chars | **0.6810** | 0.8180 |

It rescues a weak model (+0.23) and **caps a strong one** (−0.18). Which side of the
crossover the fine-tuned adapter lands on has never been measured, because the two runs
that exist changed *two* variables at once (0.49 = trained adapter, no stitching;
0.5486 = base model, stitching).

`03-implementation-plan.md` Task 7 resolves this the cheap way: **generate raw prose once,
cache it, then sweep assembly strategies offline on CPU against the official scorer.**
That buys the answer for zero additional GPU time. Do not pick a winner before that sweep runs.

---

## Ground rules for whoever picks this up

1. **Do not start a Modal run to find out something a CPU could tell you.** Every
   measurement in `00-evidence-and-measurements.md` was produced on a laptop.
2. **Any code or config change re-mints the candidate.** `candidate_id` hashes
   `git_commit_sha` + `algorithm_sha256` + `runtime_profile_sha256`
   (`src/task2/provenance/candidate.py:92-109`). The gate chain must be re-run. Budget for it.
3. **Report what happened, not what was hoped for.** The `512`-token comment in
   `modal_app.py` is a claim nobody measured, and it cost a 5-hour container.
