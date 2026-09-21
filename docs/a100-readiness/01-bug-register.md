# 01 — Bug Register

17 defects found. Severity is by **impact on the next A100 run**: whether it kills the
run, silently costs score, wastes money, or corrupts the provenance record.

Every entry was verified by reading the cited lines at HEAD `5f75cfa`. Where a claim
rests on a measurement rather than on code reading, the measurement is linked.

| id | severity | one line |
|---|---|---|
| [B-01](#b-01) | 🔴 blocker | Candidate manifest is stale; run would stamp the wrong commit onto the release |
| [B-02](#b-02) | 🔴 blocker | `max_new_tokens=512` forfeits ~10 % of achievable score; the code comment is false |
| [B-03](#b-03) | 🔴 blocker | Inference loads 4-bit NF4 on a 40 GB A100 — the cause of the 5 h timeout |
| [B-04](#b-04) | 🔴 blocker | `predicted_inference_seconds` is a hard-coded 3300 s that never matched reality |
| [B-05](#b-05) | 🔴 blocker | `predict_batch` has no checkpointing — any fault loses 100 % of inference |
| [B-06](#b-06) | 🔴 blocker | `group_by_length` unset → ~40-50 % of training compute spent on padding |
| [B-07](#b-07) | 🟠 high | 2,354 SFT examples (33 %) train on an empty evidence block |
| [B-08](#b-08) | 🟠 high | Dev evaluation scores fold 0, which was in the training set — leaked metric on the model card |
| [B-09](#b-09) | 🟠 high | BM25 searched one query at a time, 1,918 times |
| [B-10](#b-10) | 🟠 high | Legal-reference arm called with a 1-element list, 1,918 times |
| [B-11](#b-11) | 🟠 high | Header-only answer can reach `submission.json` and score ~0 |
| [B-12](#b-12) | 🟡 medium | Prompts not length-sorted before batching → pad waste in decode |
| [B-13](#b-13) | 🟡 medium | `completion_only_loss` is passed but never verified to have taken effect |
| [B-14](#b-14) | 🟡 medium | HF auto-upload swallows every exception into a warning |
| [B-15](#b-15) | 🟡 medium | Strict reload loads the full base model a second time |
| [B-16](#b-16) | 🟡 medium | `kaggle_t4x2` gate cannot run on Modal — chain has a manual Kaggle step |
| [B-17](#b-17) | 🟢 low | `submission.json` / `.zip` written with no answer-length telemetry |
| [B-18](#b-18) | 🟠 high | A mock dense index of random vectors sits in `kaggle_dataset/` and poisons every local experiment |
| [B-19](#b-19) | 🟡 medium | `colab_t4` gate stage is dead weight on the critical path |

---

<a name="b-01"></a>
## B-01 🔴 Stale candidate — the release would misattribute its own code

**Where:** `artifacts/candidates/5433e8b4787137c9/candidate_manifest.json:5`,
consumed at `scripts/modal_app.py:334-336` and `scripts/modal_app.py:441-443`.

The manifest pins `git_commit_sha = 51964b1b0aede15d6cbbddaa96f36fbc88928c48`.
HEAD is `5f75cfa5533626e52bce2c9c6dc6419f51b76709` — five commits later, including the
entire `dual_assembled` feature. The container then does:

```python
os.environ["GIT_COMMIT_SHA"] = candidate["git_commit_sha"]
(Path("/root/LegalQA") / ".git_commit_sha").write_text(candidate["git_commit_sha"])
```

**Failure:** the run bundle, the model card, and the Hugging Face release all claim commit
`51964b1b` while executing `5f75cfa`+. `validate_against_config`
(`src/task2/provenance/candidate.py:157-192`) checks the algorithm hash, the seed, model
ids and revisions — **it does not check `git_commit_sha`**, so nothing catches this.

**Compounding:** `candidate_id` is a hash over `git_commit_sha` **and**
`algorithm_sha256` **and** `runtime_profile_sha256`
(`src/task2/provenance/candidate.py:92-109`). Therefore **every fix in this register
re-mints the candidate id**, which invalidates all three gate reports in
`artifacts/gates/5433e8b4787137c9/`.

**Fix:** land all code and config changes first, then re-mint once via
`scripts/freeze_candidate.py`, then re-run the gate chain. Sequencing is in
[`03-implementation-plan.md`](03-implementation-plan.md) Task 9-10.

**Also add:** a guard in `validate_against_config` that compares the manifest's
`git_commit_sha` against the live checkout and refuses on mismatch unless explicitly
overridden. A provenance field nobody validates is not provenance.

---

<a name="b-02"></a>
## B-02 🔴 `max_new_tokens = 512` — documented as 90 %, measured at 67.5 %

**Where:** `scripts/modal_app.py:198-200`

```python
#: Calibrated generation ceiling (tokens) covering 90%+ of statutory answers
#: while preventing repetitive decoder loops and keeping batch inference fast.
MODAL_MAX_NEW_TOKENS = 512
```

**Measured** ([M-2](00-evidence-and-measurements.md#m-2--gold-answer-length-real-qwen-tokenizer), real Qwen tokenizer, 1,200 gold answers): 512 tokens fully covers
**67.5 %**, not 90 %. 768 covers 89.6 %; 1024 covers 96.7 %.

**Cost** ([M-3](00-evidence-and-measurements.md#m-3--perfect-model-ceiling-vs-generation-cap-official-scorer)): a *perfect* generator capped at 512 scores **0.8961** METEOR; at 1024 it
scores **0.9874**. The cap alone discards ~10 % of the achievable score. Inside
dual-assembly the effect is smaller but still real: 0.7864 → 0.8180 ([M-6](00-evidence-and-measurements.md#m-6--prose-length-inside-dual-assembly-n--150-oracle-article)).

**Fix:** raise to **1024**, and delete the unmeasured comment. The 2× decode cost is more
than repaid by B-03. Set it in `configs/task2/runtime/modal_a100.yaml` rather than as a
Python constant, so it is covered by the runtime hash.

---

<a name="b-03"></a>
## B-03 🔴 4-bit NF4 quantisation at *inference* on a 40 GB A100

**Where:** `src/task2/generator.py:174-181`

```python
if BitsAndBytesConfig is not None:
    load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
```

Unconditional for any CUDA device. bitsandbytes NF4 **dequantises weights on every forward
pass**; in autoregressive decode that is once per token, and decode is already
memory-bandwidth-bound. This is the dominant reason the previous attempt measured ~210 s
per batch of 16 and was killed at the container timeout.

**There is no memory justification** ([M-8](00-evidence-and-measurements.md#m-8--vram-budget-bf16-inference-on-a100-40gb)): bf16 weights are **6.18 GB**, and Qwen2.5-3B
uses GQA with only 2 KV heads → **36 KB/token**, so KV at batch 48 × 3072 ctx is
**5.44 GB**. Total ~12-14 GB of 40 GB.

**Note the asymmetry:** 4-bit NF4 is correct for *training* (`generation/trainer.py:284-291`,
that is what QLoRA means) and wrong for *inference on this GPU*. The two paths should not
share a hard-coded policy.

**Fix:** make the inference dtype a runtime setting
(`inference.generator_load_mode: bfloat16 | nf4`), default `nf4` to preserve current
behaviour on small GPUs, and select `bfloat16` in `modal_a100.yaml`. Prefer
`merge_and_unload()` on the LoRA adapter afterwards — it removes 7 target modules × 36
layers of per-token adapter matmuls.

> ⚠️ **This changes numerics** (NF4 → bf16 logits differ), so it is score-affecting, not a
> pure runtime knob. It must go in a hashed config, and the gate chain must be re-run.
> Do not smuggle it in as "runtime only".

---

<a name="b-04"></a>
## B-04 🔴 The deadline guard is a constant that never matched reality

**Where:** `src/task2/pipeline/runner.py:424` and `scripts/modal_app.py:205-214`

```python
int(paths.get("predicted_inference_seconds", 3300)),   # runner.py:424
```

`build_remote_paths` **never sets** `predicted_inference_seconds`, so the default 3300 s
(55 min) always applies. The admission test at `runner.py:421-431` therefore compares
elapsed time against a fixed 55-minute guess, passes, and starts an inference stage that
in the last attempt needed **~6.7 hours** — inside a container whose hard timeout is
18,000 s (`modal_app.py:table at 380`).

**Failure mode observed:** admission returns `OK`, inference starts, container is killed
mid-stage, volume state uncommitted.

**Fix:** derive the prediction instead of guessing it —
`num_queries × max_new_tokens / measured_tokens_per_second × safety_factor` — and take
`measured_tokens_per_second` from the A100 micro-probe rather than a literal. Pass it
explicitly through `build_remote_paths`. A guard whose input is a constant is not a guard.

---

<a name="b-05"></a>
## B-05 🔴 Inference is a single monolithic call with no checkpoint

**Where:** `src/task2/predict.py:487` (`predict_batch`), called once at
`src/task2/pipeline/runner.py:541-548`

All 1,918 questions are retrieved, reranked, and generated inside one call that returns
only at the end. There is no incremental write, no resume, and no deadline check *inside*
the loop — the check at `runner.py:417` happens once, before the stage begins.

**Failure:** any OOM, timeout, or transient fault after 90 % of the work discards 100 % of it.
This is precisely what happened, and why the adapter had to be rescued by hand with
`vol.commit()`.

**Fix:** append each generated answer to a prompt-keyed JSONL as it is produced (notebook
v13.2 does exactly this at its `CELL 11` — `gen_raw_v11.jsonl`, keyed by a hash of the
exact prompt), reload it on start, and skip what is already present. Check the remaining
deadline every N batches and return `INCOMPLETE` with the partial cache intact rather than
being killed. Resume-safety also makes the assembly sweep in Task 7 free.

---

<a name="b-06"></a>
## B-06 🔴 `group_by_length` unset → ~40-50 % of training compute is padding

**Where:** `src/task2/generation/trainer.py:338-355` (the `sft_kwargs` dict)

`grep -rn "group_by_length\|packing" src/ configs/` returns **nothing**. Both default to
`False` in TRL, so every example is padded to `max_seq_len = 2048`
(`configs/task2/algorithm.yaml:16`) regardless of true length.

Example lengths are highly skewed (answers alone: p50 414, p90 779, p99 1,284 tokens —
[M-2](00-evidence-and-measurements.md#m-2--gold-answer-length-real-qwen-tokenizer)), so a large fraction of every batch is `<|pad|>`.

**Fix:** `sft_kwargs["group_by_length"] = True`. One line, no correctness risk, and it
does not change the set of examples or the loss — only their batching order.

Leave `packing = False`: packing concatenates examples across document boundaries, which
would break `completion_only_loss` masking (see B-13).

---

<a name="b-07"></a>
## B-07 🟠 One third of the SFT set trains on an empty evidence block

**Where:** `src/task2/generation/dataset.py:117-118`

```python
pos_pieces = qa_to_pos_evidence.get(qid, [])
raw_evidence = "\n\n".join(pos_pieces) if pos_pieces else ""
```

and `src/task2/generator.py:51`

```python
ev_clean = evidence.strip() if evidence else "Không có căn cứ cụ thể."
```

**Measured** ([M-1](00-evidence-and-measurements.md#m-1--dataset-shape)): of 7,113 unique `qa_id`, only **4,759 (66.9 %)** have a retrieval
label. The remaining **2,354 (33.1 %)** are built with `evidence = ""`, rendering as
`[CĂN CỨ PHÁP LÝ]\nKhông có căn cứ cụ thể.` followed by a full statutory gold answer.

**What that teaches:** "when given no evidence, produce a confident statutory answer from
memory" — i.e. it trains hallucination. At inference every question *has* retrieved
evidence, so the condition never recurs; the capacity spent on it is worse than wasted.

**Fix:** drop examples with no evidence from the SFT set. This removes the mismatch **and**
cuts training time by a third. Record the excluded count in the trainer manifest so the
release states the real training-set size.

---

<a name="b-08"></a>
## B-08 🟠 The dev metric on the model card is computed on training data

**Where:** `src/task2/pipeline/runner.py:312` and `:389-409`

```python
eval_fold = profile.val_fold if profile.val_fold is not None else 0
```

The `modal_a100` profile sets `val_fold = None`
(`src/task2/pipeline/profiles.py:141`) meaning *train on everything*. The expression above
then silently falls back to **fold 0** for evaluation — a fold that was just trained on.

`dev_eval_size` is also `None` for this profile, so `profile.dev_eval_size or 5`
(`runner.py:402`) evaluates **5 samples**.

The resulting `selected_meteor` is passed into `metrics` at `scripts/modal_app.py:585-590`
and published on the Hugging Face model card.

**Failure:** a leaked, 5-sample number presented as a validation metric.

**Fix:** either hold out a fold that training excluded, or set
`run_dev_evaluation = False` for `modal_a100` and mark the field `null` with a reason. A
metric that cannot be honest should be absent, not decorative.

---

<a name="b-09"></a>
## B-09 🟠 BM25 searched once per query, 1,918 times

**Where:** `src/task2/predict.py:548`, inside the `for idx, (qa_id, question)` loop at `:543`

```python
bm25_res = self.bm25.search(ret_q, top_k=pool) if self.bm25 else []
```

Dense retrieval is properly batched just above (`:533`,
`self.dense.search_batch(...)`), but BM25 is not — despite the corpus being 801,863 chunks.

**Fix:** hoist a batched call before the loop if `BM25Retriever` exposes one; otherwise add
`search_batch` that scores the query matrix in one pass. Verify against the per-query path
on a 50-query sample before trusting it.

---

<a name="b-10"></a>
## B-10 🟠 Legal-reference arm invoked with a 1-element list, 1,918 times

**Where:** `src/task2/predict.py:553`

```python
lex_res = search_legal_references([ret_q], self.legal_index, self.legal_rows, k=pool)[0]
```

The function takes a list of queries and returns a list of result lists — it is built to
batch — and is called with `[ret_q]` then immediately indexed `[0]`. `use_legal_reference`
is **on** (`configs/task2/algorithm.yaml:46`), so this executes for every query.

**Fix:** call it once with all `dense_queries`, index into the result inside the loop.

---

<a name="b-11"></a>
## B-11 🟠 A header-only answer can reach the submission

**Where:** `src/task2/candidates.py:96`, `src/task2/selector.py:233`, guard at
`src/task2/predict.py:669-674`

When retrieval returns nothing, `build_citation_header` falls back to the literal
`"Căn cứ quy định của pháp luật:"` (`candidates.py:96`). That string is non-empty, so it
survives the `{k: v for k, v in candidates.items() if v}` filter at `candidates.py:266`,
can be returned by the selector, and passes the emptiness guard:

```python
if not selected or not str(selected).strip():   # predict.py:669 — a header is truthy
```

**Failure:** a 6-word answer scoring ≈ 0 on both metrics, emitted silently.

**Mitigating:** with `best_fixed_candidate = "dual_assembled"` and the fallback order at
`selector.py:233`, this needs *both* an empty generation and empty evidence. Rare — but
unbounded, and invisible when it happens.

**Fix:** treat a candidate that is only a citation header as empty. Add the header-only
check to both guards (`predict.py:472` and `:669`) and count occurrences in the provenance
report so the run states how many fired.

---

<a name="b-12"></a>
## B-12 🟡 Prompts are not length-sorted before batching

**Where:** `src/task2/generator.py:283-285`

```python
while index < total_prompts:
    current = prompts[index:index + batch_size]
```

Consecutive slices of the original order. With left-padding (correctly set at
`generator.py:158`) every sequence in a batch is padded to the batch maximum, so one
2,000-token prompt drags three 300-token prompts up with it.

**Fix:** sort indices by prompt length, generate, then invert the permutation before
returning. The inversion is the part that must be tested — an off-by-one here silently
assigns answers to the wrong questions, which no downstream check would catch.

> Note: notebook v13.2 already does this at `CELL 11`
> (`todo.sort(key=lambda q: -len(prompts[q]))`). The library path never adopted it.

---

<a name="b-13"></a>
## B-13 🟡 `completion_only_loss` is passed but never verified

**Where:** `src/task2/generation/trainer.py:110`, `configs/task2/algorithm.yaml:35`

The flag is forwarded into `SFTConfig`, and `dataset.py:180-181` supplies separate
`prompt` / `completion` fields. But nothing asserts that TRL actually masked the prompt
tokens to `-100`. If the installed TRL ignores the key, training silently optimises the
loss over the prompt — including the entire statutory evidence block — and nothing fails.

**Fix:** after constructing the trainer, pull one batch from the collator and assert that
the count of `label == -100` positions matches the prompt length for that example. It is a
handful of lines and it converts a silent corruption into a loud failure.

---

<a name="b-14"></a>
## B-14 🟡 Hugging Face release failures are downgraded to warnings

**Where:** `scripts/modal_app.py:622-624`

```python
except Exception as e:
    print(f"[!] Warning: Auto-upload to Hugging Face encountered error: {e}")
    hf_res = {"status": "FAILED", "error": str(e)}
```

The `try` spans bundle construction *and* upload (`:536-621`), including the deliberate
`raise ValueError("refusing release: trainer manifest lacks measured optimizer_steps…")`
at `:559-563`. That refusal — designed to block an unevidenced release — is caught and
printed as a warning, and the run still reports `"status": "PASS"` at `:626`.

**Fix:** let integrity refusals propagate; catch only genuine network/transport errors from
the upload call itself, and reflect a failed release in the run's top-level status.

---

<a name="b-15"></a>
## B-15 🟡 Strict reload loads the full base model a second time

**Where:** `src/task2/generation/trainer.py:432-465`

After training, the trainer deletes the model, clears the cache, then calls
`QwenGenerator.load(...)` to verify the adapter round-trips. That is a full Qwen2.5-3B
load (~30-45 s) plus adapter application, to generate a few tokens.

The check is worth keeping for the production run — it is the only thing standing between
a corrupt adapter and a wasted inference stage. But for probes it is pure overhead.

**Fix:** keep the full reload for `modal_a100`; for probe/smoke profiles verify
`adapter_config.json` + `adapter_model.safetensors` integrity instead.

---

<a name="b-16"></a>
## B-16 🟡 The gate chain has a step Modal cannot run

**Where:** `scripts/run_gpu_gate.py:81-85`, `scripts/modal_app.py:313-317`

```python
GATE_PARENTS = {
    "kaggle_t4x2": (),                                  # root — no parent
    "colab_t4": ("kaggle_t4x2",),
    "a100_micro_probe": ("kaggle_t4x2", "colab_t4"),
}
```

`kaggle_t4x2` is the root of the DAG and asserts **two** T4s
(`run_gpu_gate.py:257-262`). `modal_app.py` only declares a single-T4 function
(`gpu="T4"`, `:313`) wired to the `colab_t4` stage. Re-minting the candidate (B-01)
therefore forces a manual Kaggle notebook run before anything else can proceed.

**Fix:** add a `gpu="T4:2"` Modal function for the `kaggle_t4x2` stage. Modal supports
multi-GPU; this makes the whole chain reproducible from one command and removes a manual
dependency from the critical path.

---

<a name="b-17"></a>
## B-17 🟢 The submission is written without answer-length telemetry

**Where:** `src/task2/pipeline/runner.py:557-585`

The stage records counts and SHA-256s, but not the answer-length distribution. Notebook
v13.2 asserts on exactly this (`mean answer N tok — expected ~800`) because a silently
truncated or empty-prose run is otherwise indistinguishable from a good one until the
leaderboard replies.

**Fix:** record mean / median / p90 answer words and the per-source counts from the
provenance report, and fail the stage if the mean falls outside a sanity band.


---

<a name="b-18"></a>
## B-18 🟠 A mock dense index of random vectors is sitting in the dataset folder

**Where:** `kaggle_dataset/indexes/dek21/embeddings.npy` (801,863 × 768 fp16, 1.23 GB),
`kaggle_dataset/indexes/dek21/dek21_manifest.json`

**Measured** ([M-12](00-evidence-and-measurements.md#m-12--the-local-dek21-dense-index-is-a-mock-index-of-random-vectors)): pairwise cosine among 400 stored vectors is
**mean +0.0002, std 0.0361** — identical to the theoretical `1/√768` of random unit vectors,
and re-encoding a chunk's own text scores **cos ≈ 0.01** against its stored vector.
The matrix carries no semantic content, while the manifest beside it declares
`model_id: CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2`.

Almost certainly produced by `DenseRetriever.fit_mock` (see `src/task2/predict.py:229-231`)
during a CPU dev session and never cleaned up.

**What is NOT affected.** Production is safe, on two independent counts:
1. `dataset_manifest.json` lists only `indexes/bm25`; **`indexes/dek21` does not ship**, so
   the Modal container never receives it and cold-rebuilds from the pinned revision
   (`scripts/modal_app.py:482-494`).
2. Even if it did ship, `check_dense_alignment` catches it —
   measured `pass_rate 0.0, aligned False`. The guard works.

**What IS affected.**
- **Any local retrieval experiment against this directory measures noise.** This blocked the
  attempt to measure top-1 article accuracy directly, which is why `p ≈ 0.42` in
  [`02-scoring-model.md`](02-scoring-model.md) is inferred from the observed score rather than measured.
- `scripts/package_kaggle_dataset.py:87-93` picks up `indexes/dek21` **if the directory
  exists**. Re-packaging the dataset today would publish 1.23 GB of random numbers under a
  manifest claiming a real encoder.

**Fix:** delete the directory, and make the packaging script refuse to include a dense index
that fails `check_dense_alignment` rather than trusting its presence. A mock artifact with a
truthful-looking manifest beside it is worse than no artifact.

---

<a name="b-19"></a>
## B-19 🟡 The `colab_t4` gate stage is dead weight

**Where:** `scripts/run_gpu_gate.py:81-85`, `scripts/modal_app.py:313-380`
(`run_modal_t4_remote`), `scripts/modal_app.py:100-107`

```python
GATE_PARENTS = {
    "kaggle_t4x2": (),
    "colab_t4": ("kaggle_t4x2",),
    "a100_micro_probe": ("kaggle_t4x2", "colab_t4"),   # <- either parent is accepted
}
```

`a100_micro_probe` already accepts `kaggle_t4x2` directly, so `colab_t4` is an **optional
middle hop** that adds a GPU stage, a report artifact, and ~10 minutes without gating
anything the A100 probe does not gate itself. The project no longer uses Colab at all;
the stage is named after a platform that is not in the workflow.

**Fix:** drop `colab_t4` from the operational path — the DAG already permits
`kaggle_t4x2 → a100_micro_probe → full`. Removing the stage from the codebase is a larger
change (13 test files reference it); see [`03-implementation-plan.md`](03-implementation-plan.md) Task 12 for the staged
removal. Until then, simply never run it, and stop passing `colab_report` through
`build_modal_request`.
