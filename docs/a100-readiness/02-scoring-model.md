# 02 — The Scoring Model, Decompiled

Purpose: give the next engineer a *predictive* model of the leaderboard number, so that
proposed changes can be ranked before any GPU is booked.

---

## 1. What is actually computed

From `Scoring-Program-Task-LegalQA/scoring.py` (quoted verbatim in
[`00-evidence-and-measurements.md` M-9](00-evidence-and-measurements.md#m-9--official-scorer-verbatim)):

```
rouge  = mean_k  RougeScorer(['rougeL'], use_stemmer=False).score(ref_k, hyp_k).rougeL.fmeasure
meteor = mean_k  nltk.translate.meteor_score.meteor_score([ref_k.split()], hyp_k.split())
```

Two independent means. **No weighting, no combination, no Vietnamese tokenizer**
(`pyvi` is commented out at `scoring.py:27-32`).

### METEOR with NLTK defaults

`meteor_score` is called with no kwargs, so `alpha=0.9, beta=3.0, gamma=0.5`:

```
P        = m / |hyp|                         precision
R        = m / |ref|                         recall
F_mean   = P·R / (alpha·P + (1-alpha)·R)     with alpha=0.9  →  ~R-dominated
penalty  = gamma · (chunks / m)^beta         with gamma=0.5, beta=3
METEOR   = F_mean · (1 - penalty)
```

where `m` = aligned unigrams and `chunks` = number of contiguous runs those alignments
form in the hypothesis.

Two properties drive every design decision in this project:

**(a) Recall dominates, but precision is not free.**
With `alpha=0.9`, `F_mean → R/(0.9 + 0.1·R/P)`. Adding a token that matches with
probability ρ is profitable iff roughly `ρ > 0.1 · F_mean`, i.e. **ρ > ~5.5 %** at an
operating point of 0.55. A very low bar — which is why appending statutory text pays.

**(b) Fragmentation is punished cubically.**
`beta=3` means scattered matches are heavily penalised while long verbatim runs are nearly
free. A pasted statutory article forms **one long contiguous chunk**, so its penalty
contribution is small. This — not recall alone — is the deeper reason dual-assembly works.

It also explains the failure mode in [M-6](00-evidence-and-measurements.md#m-6--prose-length-inside-dual-assembly-n--150-oracle-article):
when the prose *already* quotes the article and the pipeline appends it again, the second
copy cannot align (METEOR aligns each reference token once), so it adds `|hyp|` with zero
`m`, **and** splinters the alignment into more chunks. Both terms move the wrong way.

### ROUGE-L

`rouge_score`'s `DefaultTokenizer` lowercases, replaces `[^a-z0-9]+` with spaces, splits.
Longest-common-subsequence F-measure. Being an F1 over an LCS, it is far more
precision-sensitive than METEOR — which is exactly why the two metrics diverge under
stitching ([M-4](00-evidence-and-measurements.md#m-4--assembly-strategy-oracle-retrieval-n--150): METEOR +0.229, ROUGE-L −0.032).

> ⚠️ **Which number is the leaderboard number is not stated in the scoring program.**
> The calibration below shows METEOR reproduces the observed 0.5486 to ±0.0003, so this
> folder treats **METEOR as the ranked metric**. If that is wrong, the entire optimisation
> direction inverts. See [`05-open-questions.md` Q-2](05-open-questions.md).

---

## 2. A calibrated score model

Let `p` = probability the pipeline's **top-1 article is the correct one**.
From [M-5](00-evidence-and-measurements.md#m-5--multi-article-stitching-under-imperfect-retrieval-n--150), with a *base-quality* generator and one stitched article:

```
METEOR(p)  ≈  0.4522 · (1 − p)  +  0.6810 · p
           =  0.4522 + 0.2288 · p
```

Check it against reality:

| run | configuration | predicted | observed |
|---|---|---|---|
| prose only, no stitching | p irrelevant | 0.452 | **0.49** |
| base Qwen + dual-assembly (notebook v13.2) | p = 0.42 | **0.5483** | **0.5486** |

The stitched case lands within **0.0003** of the observed leaderboard score. The model is
calibrated. (The prose-only case under-predicts by 0.038 — expected, since the proxy prose
is a crude 120-word truncation rather than a real generation.)

### What the model says it takes to reach 0.60

```
0.60 = 0.4522 + 0.2288 · p   →   p = 0.646
```

**Top-1 article accuracy must rise from ~0.42 to ~0.65.** That is the whole gap.
Every point of `p` is worth **+0.0023 METEOR**.

---

## 3. Ranked levers, with measured yields

| # | Lever | Mechanism | Est. yield | Cost | Confidence |
|---|---|---|---|---|---|
| 1 | **Top-1 article accuracy 0.42 → 0.65** | more correct citations stitched | **+0.052** | retrieval work, no extra GPU at inference | high — model calibrated to ±0.0003 |
| 2 | **`max_new_tokens` 512 → 1024** | prose is no longer cut mid-clause | **+0.02 … +0.03** | 2× decode tokens, offset by B-03 fix | high — [M-6](00-evidence-and-measurements.md#m-6--prose-length-inside-dual-assembly-n--150-oracle-article) |
| 3 | **Known-QA overrides (81 hits)** | exact answer ⇒ METEOR 1.0 | **+0.019** | none | high — [M-7](00-evidence-and-measurements.md#m-7--known-qa-override-coverage-on-the-private-test) |
| 4 | **Assembly sweep on cached prose** | pick the best of ~12 strategies *post hoc* | **+0.00 … +0.18** | CPU minutes | see Q-1 — could be either sign |
| 5 | **Fix 2,354 evidence-free SFT examples** | removes train/inference mismatch | unmeasured | −33 % train time (a *saving*) | medium — mechanism is clear, size is not |
| 6 | Citation budget beyond 4000 chars | — | **0.000** | — | high — 8000 ≈ 4000 in [M-4](00-evidence-and-measurements.md#m-4--assembly-strategy-oracle-retrieval-n--150) |
| 7 | Stitching 2-3 articles instead of 1 | — | **−0.04 … −0.09** | — | high — [M-5](00-evidence-and-measurements.md#m-5--multi-article-stitching-under-imperfect-retrieval-n--150). **Do not do this.** |

Levers 1-3 are additive and independent: **0.5486 + 0.052 + 0.025 + 0.019 ≈ 0.645**,
against a dual-assembly architectural ceiling of 0.818. The target is reachable without
touching the language model.

---

## 4. Two traps this model exposes

**Trap 1 — "longer answers score better".**
True only up to the point where added tokens stop matching. [M-4](00-evidence-and-measurements.md#m-4--assembly-strategy-oracle-retrieval-n--150) shows 8000 chars ≈ 4000 chars,
and [M-5](00-evidence-and-measurements.md#m-5--multi-article-stitching-under-imperfect-retrieval-n--150) shows 3 articles is *worse* than 1 at every hit-rate. Length is a proxy that
stops tracking the objective at ~470 words.

**Trap 2 — "a better generator always helps".**
Under dual-assembly it does not. [M-6](00-evidence-and-measurements.md#m-6--prose-length-inside-dual-assembly-n--150-oracle-article) shows a perfect generator scores **0.9953 alone** but
**0.8180 with the citation appended**. Improving the model past the crossover while leaving
stitching on will *lose* score. The stitching decision must be re-made whenever generator
quality changes materially — which is what the Task 7 offline sweep is for.
