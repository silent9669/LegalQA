# 05 — Open Questions

Things this audit could **not** settle. Each states what is unknown, why it matters, and
the cheapest experiment that would resolve it. **Do not guess these into the plan.**

---

## Q-1 🔴 Is dual-assembly still the right architecture *for the fine-tuned adapter*?

**What is unknown.** The two scored runs changed two variables at once:

| run | generator | assembly | score |
|---|---|---|---|
| 0.49 | **fine-tuned adapter** | none (prose only) | 0.49 |
| 0.5486 | **base Qwen**, `USE_TRAINED_ADAPTER=False` | dual-assembly | 0.5486 |

The cell that matters — *fine-tuned adapter + dual-assembly* — has never been run. So has
*base model + no assembly*. The comparison currently drawn between 0.49 and 0.5486 is
confounded.

**Why it matters.** [M-6](00-evidence-and-measurements.md#m-6--prose-length-inside-dual-assembly-n--150-oracle-article) shows the two architectures cross over:

| generator quality | prose only | prose + citation |
|---|---|---|
| weak (120-word summary) | 0.4520 | **0.6813** |
| perfect | **0.9953** | 0.8180 |

Stitching rescues a weak generator by **+0.23** and penalises a strong one by **−0.18**,
because the appended article duplicates text the trained model already quotes: METEOR
aligns each reference token once, so the copy adds `|hyp|` with zero `m` *and* fragments
the alignment (`beta=3`). Picking wrong costs more than every other fix in this folder combined.

**Resolution.** [`03-implementation-plan.md`](03-implementation-plan.md) Task 7. Generate raw prose **once** with the
fine-tuned adapter, cache it (Task 6), then sweep assembly strategies offline on CPU
against the official scorer. Zero extra GPU time.

**Do not** set `best_fixed_candidate` from intuition before that sweep reports.

---

## Q-2 🟠 Which number does the leaderboard rank — METEOR or ROUGE-L?

**What is unknown.** `Scoring-Program-Task-LegalQA/scoring.py:52` returns
`{'rouge': …, 'meteor': …}` and combines nothing. Which one Codabench ranks on is not
visible in the scoring program.

**Why it matters.** The two metrics move in **opposite** directions under stitching
([M-4](00-evidence-and-measurements.md#m-4--assembly-strategy-oracle-retrieval-n--150)): METEOR **+0.229**, ROUGE-L **−0.032**. If the leaderboard ranks ROUGE-L, the
entire dual-assembly strategy is backwards and the correct move is pure end-to-end
generation (ROUGE-L 0.9985 at 1024 tokens vs 0.6671 stitched).

**Current evidence for METEOR.** The calibrated model in [`02-scoring-model.md` §2](02-scoring-model.md) predicts
**0.5483** at p = 0.42 against an observed **0.5486** — within 0.0003. A coincidence at
that precision is unlikely, but it is one data point.

**Resolution (cheap, high value).** Submit one deliberately skewed probe: an answer set
with high METEOR and visibly lower ROUGE-L (e.g. `cite-only @4000`, which [M-4](00-evidence-and-measurements.md#m-4--assembly-strategy-oracle-retrieval-n--150) puts at
METEOR 0.4790 / ROUGE-L 0.5443 — the ordering inverts versus prose-only). Whichever way
the leaderboard moves identifies the ranked metric. Costs one submission slot.

---

## Q-3 🟠 Does bf16 inference on an NF4-trained adapter change output quality?

**What is unknown.** Task 1 trains with a 4-bit NF4 base and serves with a bf16 base. This
is standard QLoRA practice and the delta is normally small — but "normally small" is not a
measurement, and nobody has measured it here.

**Why it matters.** If quality drops materially, the ~3-4× decode speedup is partly paid
for in score.

**Resolution.** During the micro-probe, generate the same 32 held-out prompts twice — once
NF4, once bf16-merged — and compare METEOR against gold. Adds ~2 minutes to a probe that
already loads both paths. If the delta exceeds ~0.01, reconsider: bf16 **without**
`merge_and_unload` (keeping the LoRA layers live) is a middle option that is closer to
training numerics and still far faster than NF4.

---

## Q-4 🔴 How is top-1 article accuracy actually raised from 0.42 to 0.65?

**Status: now Task 11** in [`03-implementation-plan.md`](03-implementation-plan.md), promoted to top priority under the
2026-09-21 max-score directive. It remains an open *question* because the number it turns
on has never been measured.

**What is unknown.** The actual value of `p`. The 0.42 figure is **inferred** from the
observed leaderboard score via the calibrated model, not measured — and it could not be
measured locally because the only local dense index is a mock of random vectors
([M-12](00-evidence-and-measurements.md#m-12--the-local-dek21-dense-index-is-a-mock-index-of-random-vectors), [B-18](01-bug-register.md#b-18)). Until a real index is rebuilt and
`scripts/measure_retrieval.py` runs, every retrieval decision is being made blind.

**What is known.**
- The calibrated model gives `METEOR ≈ 0.4522 + 0.2288 · p`; p ≈ 0.42 today, p ≈ 0.646 needed.
- Candidate levers, none measured: article-level (not chunk-level) reranking; raising
  `candidate_pool` above 50; re-tuning the RRF weights in `configs/task2/algorithm.yaml:52-56`;
  the `encoder_ft_v2` round-2 encoder already on Hugging Face; query rewriting.
- Notebook v13.2 CELL 6d reports its own retrieval numbers and its round-2 fine-tuning
  recipe; it is the best starting point and should be read before anything is re-derived.

**Resolution.** [`03-implementation-plan.md`](03-implementation-plan.md) Task 11, Step 1 first — delete the mock, rebuild a
real index, and measure. The offline harness is cheap:
`retrieval_labels.parquet` gives gold `positive_article_id` for 4,759 questions, so
**top-1 article accuracy is measurable on CPU with no generation at all**. Build that
measurement first; it turns retrieval tuning into a fast offline loop instead of a
GPU-bound guess.

---

## Q-5 🟡 What does dropping 2,354 evidence-free SFT examples actually do to the score?

**What is unknown.** B-07's mechanism is clear — training the model to answer confidently
with no evidence, a condition that never occurs at inference — but the *size* of the
effect is not. It could be worth +0.02, or it could be noise.

**Why it matters.** It also cuts training time by a third, so it is worth doing for speed
regardless. The question is only whether to claim a score benefit.

**Resolution.** Two short training runs (~25 min each on A100) with and without
`require_evidence=True`, evaluated on the same held-out fold. Under the max-score
directive this A/B is **worth running regardless** of Q-1's outcome — it is under an hour
of GPU and it resolves a real train/serve mismatch rather than leaving it assumed.

---

## Q-6 ✅ RESOLVED — `max_seq_len = 2048` is fine; do **not** change it

**Measured 2026-09-21** on 800 evidence-bearing examples with the real Qwen tokenizer
(see [`00-evidence-and-measurements.md` M-10](00-evidence-and-measurements.md#m-10--sft-sequence-budget-at-max_seq_len--2048)):

| `max_seq_len` | examples dropped | evidence budget (mean) | zero-budget |
|---|---|---|---|
| **2048 (current)** | **0.50 %** | 1,396 tok | 0.5 % |
| 3072 | 0.00 % | 2,419 tok | 0.0 % |

Chat framing costs 178 tokens. At 2048 the answer plus framing fits for 99.5 % of
examples and still leaves ~1,400 tokens for evidence — comfortably more than the ~700-token
articles being packed.

**Action: none.** Raising `max_seq_len` would buy 0.5 % of examples at the price of
re-minting the candidate and a longer sequence for every step. Not worth it. The earlier
suspicion that long richly-cited answers were being silently dropped is **wrong** — the
drop counter was right to be quiet.

---

## Q-7 ✅ RESOLVED — the 387 duplicate `qa_id`s are a benign train/warmup overlap

**Measured 2026-09-21:**

```
dup rows: 774 | dup ids: 387
ids whose duplicates DISAGREE on answer:   0
ids whose duplicates DISAGREE on question: 0
source_split of dups: {'train': 387, 'warmup': 387}
```

Every duplicated `qa_id` appears exactly twice — once in `train`, once in `warmup` — with a
**byte-identical question and answer**. There is no contradictory-target problem.

**One real consequence remains.** `build_grounded_training_examples` iterates rows, not
ids, so these 387 examples enter the SFT set **twice** and carry double gradient weight for
no stated reason.

**Action (small, fold into Task 4):** `df_qa = df_qa.drop_duplicates("qa_id")` before
building examples. Removes 387 redundant examples (~8 % of the evidence-bearing set),
shortens training slightly, and eliminates the unintended upweighting. Log the dropped
count in the trainer manifest alongside `dropped_no_evidence`.
