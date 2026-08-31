# MASTER PROMPT — Fix and Optimize `silent9669/LegalQA` for Maximum DSC 2026 Task 2 Score

> Paste this entire prompt into a capable coding/reasoning LLM that has access to the repository.  
> Repository: `https://github.com/silent9669/LegalQA`  
> Primary target: `kaggle_kernel/legalqa_gpu_pipeline.ipynb`

---

## 0. Your role

You are the **lead ML/NLP competition engineer** responsible for turning this repository into the strongest possible **DSC 2026 Task 2 LegalQA** system under the official constraints.

Do **not** merely review or suggest improvements. **Inspect the entire repository, make the code changes, reconcile inconsistencies, create/update tests, and leave a Kaggle notebook that can run from a clean session end-to-end.**

The only objective that matters is **maximizing the official public/private Task 2 score, especially official METEOR**, while remaining fully compliant and reproducible.

Be empirical. Every architecture decision that can materially affect score must be validated using leakage-safe OOF experiments with the exact official scoring implementation. Do not keep a component merely because it sounds sophisticated or appears in the README.

---

# 1. Hard constraints and source-of-truth priority

Before editing anything, inspect all official files in the repo and the supplied competition data. Resolve constraints in this priority order:

1. Official competition/data overview and official scoring program.
2. Actual dataset schemas/files.
3. Repository code and configs.
4. README/documentation.

If documentation conflicts with executable code, **do not guess**. Determine the intended compliant behavior, fix the code/config, and document the final decision.

Current known constraints to verify from official material:

- Task: Vietnamese Legal Question Answering.
- Primary metric: official **METEOR**.
- Official scorer performs essentially:

```python
meteor_score([reference_answer.split()], predicted_answer.split())
```

- ROUGE-L is also computed, but **optimize METEOR first** unless official rules say otherwise.
- Total learned parameter budget must remain **strictly below 4.0B**.
- Only Task 2 / organizer-provided QA and legal context data may be used as task data. No external QA/legal corpus and no external answer APIs.
- Target runtime: **Kaggle GPU T4 x2**, approximately 16 GB VRAM per GPU.
- User has a Kaggle secret named exactly `HF_TOKEN`. Never print, save, commit, or leak it.
- Mounted Qwen model currently exists as Kaggle model `qwen-lm/qwen2.5/transformers/3b-instruct/1`.
- Final output must be a valid `submission.json` and `submission.json.zip` containing exactly all 1,000 public query IDs with non-empty answers.

The current parameter manifest proposes:

- `CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2` ≈ 100M — dense retriever
- `BAAI/bge-reranker-v2-m3` ≈ 568M — reranker
- `Qwen/Qwen2.5-3B-Instruct` ≈ 3.09B — generator
- base total ≈ 3.758B before any adapter/task head

Keep a real margin below 4B and include LoRA/adapter parameters in the audit. **Do not accidentally use `BAAI/bge-m3` as a 568M dense model together with the 568M reranker and 3.09B Qwen if that causes the full learned stack to exceed 4B.**

---

# 2. Important empirical facts from the supplied data

Recompute these yourself and fail loudly if the current dataset differs, but use them as a warning that the current notebook is underpowered:

- Supplied `train.json`: **7,000 QA records**.
- Supplied `public-official.json`: **1,000 test questions**.
- Training answer whitespace-length distribution observed:
  - mean ≈ **347 words**
  - median ≈ **312 words**
  - p25 ≈ **218 words**
  - p75 ≈ **439 words**
  - p90 ≈ **576 words**
  - only ≈ **16.9%** of gold answers are ≤180 whitespace words
  - only ≈ **25.6%** are ≤220 whitespace words
- Therefore the current `max_new_tokens=180` is very likely too short for the reference-answer style and may severely cap lexical recall.
- In the provided `train.json` vs public test, there were no overlapping IDs and only one exact normalized question duplicate. If `known_qa.json` reports ~40 public hits, most likely come from other organizer-provided data such as warmup; verify this carefully and ensure there is no leakage.

Because official METEOR is based on **whitespace tokens**, and Vietnamese gets little useful English WordNet synonym benefit, this competition should be treated largely as a **lexical/surface-form overlap optimization problem grounded in the correct legal source**, not as a generic “write a concise legally correct answer” task.

A shorter elegant answer can score worse than a longer source-faithful answer if the gold answer reproduces statutory text. Optimize for the actual scorer.

---

# 3. Current repository problems you must audit and fix

Do not assume this list is exhaustive. Inspect all files yourself.

## 3.1 Kaggle notebook is not the architecture claimed by README

Current authoritative notebook:

`kaggle_kernel/legalqa_gpu_pipeline.ipynb`

At present it approximately does:

```text
Exact QA memory
    -> homemade IDF-overlap sparse retrieval
    -> only top 2 chunks
    -> truncate evidence to ~1200 chars
    -> vanilla Qwen2.5-3B generation
    -> unconditional-ish Strategy F source append
    -> submission
```

It currently bypasses most of the stronger code already present under `src/`:

- real BM25
- DEk21 dense retrieval
- RRF
- BGE reranker
- article stitcher
- QLoRA adapter
- full source snapping/candidate selection

The final notebook must use the **best OOF-validated implementation**, not a weaker duplicate implementation.

## 3.2 Current notebook “BM25” is not BM25

The notebook currently uses set-of-token IDF overlap and caps each postings list at the first ~8,000 occurrences:

```python
if doc_freq[t] <= 8000:
    inverted_index[t].append(doc_id)
```

This introduces corpus-order bias and makes many later documents unreachable for common terms.

Replace it with real BM25 (`bm25s` or a correct optimized implementation) with:

- term frequency saturation
- document-length normalization
- full corpus coverage
- no arbitrary first-N postings truncation
- statutory entity boosts only after lexical scoring, with OOF-tuned weights

Use the existing `src/common/bm25.py` only after testing it on the real 801k-chunk corpus for correctness, memory, and speed.

## 3.3 Only one T4 is effectively used

Generic `device="cuda"` / `device_map={"": device}` sends Qwen to GPU 0 while GPU 1 is mostly idle.

Final inference should explicitly use both devices when beneficial, e.g.:

```text
GPU 0: Qwen2.5-3B generator (+ LoRA adapter if validated)
GPU 1: DEk21 query encoder / dense corpus search + BGE reranker
CPU/RAM: BM25 + metadata/index management
```

Use explicit `cuda:0` and `cuda:1`, log VRAM usage, and avoid accidental model duplication.

## 3.4 Configs disagree

Examples that must be reconciled:

- `configs/pipeline.yaml` currently mentions `BAAI/bge-m3` as dense and Qwen2.5-1.5B.
- `configs/models.yaml` / README recommend DEk21 + BGE reranker + Qwen2.5-3B.

Make one canonical configuration and update all references. The notebook and scripts must not silently use a different model stack than the parameter audit.

## 3.5 The full `LegalQAPipeline` can silently fall back to an extractive generator

Audit `src/task2/predict.py` and `src/task2/generator.py`.

Currently, if there is no adapter path, `load_pipeline()` may create `QwenGenerator(runtime="fallback")` instead of loading the base Qwen model. That means “full pipeline” validation can silently evaluate an extractive fallback rather than the intended generator.

Fix this. A normal no-adapter configuration must load the base Qwen generator; fallback should only be used deliberately for CPU tests.

## 3.6 Current OOF validation is not representative of final inference

Audit `scripts/run_oof_validation.py`.

Known issue: it currently constructs `QwenGenerator(runtime="fallback")`, and may use mock dense retrieval. This makes OOF numbers unsuitable for selecting the final Kaggle system.

Build at least two modes:

- `--fast`: cheap retrieval/extractive experiments
- `--full`: exact intended inference path with real dense model, real reranker, real generator/adapter, exact postprocessing

All reported “final” OOF scores must come from `--full` or a demonstrably equivalent pipeline.

## 3.7 `Strategy F` is not validated enough

Current behavior can append up to ~1,500 characters of source text to a generated answer whenever the answer is short.

This may improve METEOR recall but can also destroy precision/alignment if the retrieved source contains irrelevant text.

Do not remove it blindly. Turn it into an **OOF-tuned family of candidates** such as:

- generated only
- snapped generated only
- focused extract only
- stitched extract only
- generated + 300 chars source
- generated + 600 chars source
- generated + 1,000 chars source
- generated + 1,500 chars source

Measure per-example METEOR OOF and learn/tune a safe selector from features available at test time.

## 3.8 Evidence is too small

Current notebook retrieves only top-2 raw chunks and truncates combined evidence to ~1,200 characters.

Gold answers are often several hundred words. This evidence budget is frequently incapable of covering the full expected answer.

Implement retrieval + reranking + structured evidence packing with an OOF-tuned token budget, preserving:

- document title/number
- article number
- clause number
- neighboring clauses when necessary
- exact dates
- exact monetary values
- sanctions/remedies/exceptions

Do not simply concatenate top-8 chunks blindly.

## 3.9 Current output length is probably too short

`max_new_tokens=180` is inconsistent with the observed reference length distribution.

Test at minimum:

- 192
- 256
- 320
- 384
- 512

Potentially use a **dynamic output budget** based on question type, evidence size, nearest-train-answer length, and/or retrieval confidence.

Do not choose the longest value automatically; choose what maximizes full OOF METEOR within runtime.

## 3.10 Dataset packaging does not actually include optional indexes

Audit `scripts/package_kaggle_dataset.py`.

It defines:

```python
OPTIONAL_DIRS = ["indexes/bm25", "indexes/dek21"]
```

but currently does not appear to actually copy those directories into staging.

Fix packaging so precomputed BM25 and dense embeddings/index metadata can be mounted in Kaggle. Final notebook should:

1. prefer precomputed indexes if present;
2. verify checksum/corpus row count/model identity;
3. build missing indexes only as a controlled fallback.

Do not waste every Kaggle run rebuilding 801k dense embeddings if they can legally and reproducibly be packaged as derived organizer-data artifacts.

## 3.11 Source snapping has code quality issues

Audit `src/task2/source_snap.py`, including the unreachable duplicate return in date formatting and the current candidate-selection heuristic.

Extend fact-preserving postprocessing only when OOF supports it. Important lexical entities include:

- legal document numbers
- article/clause/point identifiers
- dates
- money ranges
- percentages
- durations/deadlines
- authority names

Never “snap” to a value merely because it appears somewhere in evidence; require local/contextual compatibility.

## 3.12 Dynamic model path discovery is fragile

Current notebook selects the first matching `config.json` from `/kaggle/input/**`.

Once DEk21 and BGE are mounted/downloaded this can select the wrong checkpoint.

Use explicit deterministic model resolution with validation of `model_type` / expected files. Fail loudly if the intended model is unavailable.

## 3.13 Remove irrelevant startup work

The inference notebook does not need to download NLTK WordNet unless it is actually running local METEOR evaluation. Remove unnecessary network downloads from final submission inference.

Keep NLTK/WordNet only in validation tooling that needs the official scorer.

---

# 4. Optimization philosophy: this is a METEOR competition

The final design must explicitly exploit the behavior of the official metric without breaking task rules.

## 4.1 Favor source-faithful lexical coverage

For each answer, maximize overlap with likely gold wording while minimizing irrelevant tokens.

Gold answers appear to follow a recurring pattern:

```text
legal basis / citation
-> quoted or paraphrased regulation
-> application/conclusion to the asked case
```

The answer generator should learn/reproduce that style rather than produce short ChatGPT-style summaries.

## 4.2 Treat generation as one candidate, not automatically the final answer

For every unseen question, construct multiple answer candidates and OOF-test them.

Candidate families should include:

1. exact QA-memory answer
2. high-confidence near-duplicate training-QA answer
3. focused statutory extract
4. article/clause stitched extract
5. base-Qwen grounded answer
6. QLoRA grounded answer (if trained and superior)
7. snapped generator output
8. generator + carefully selected source continuation
9. hybrid template answer using similar QA style + retrieved current legal evidence

Build a selector/calibrator using only test-time-available features.

---

# 5. Add a leakage-safe **similar-QA memory** branch

This is a high-priority experiment because the training set has ~7k long human/reference answers and METEOR is lexical.

Current memory only uses exact matches. Add a fuzzy/near-duplicate QA retrieval branch over organizer-provided training/warmup QA only.

Use cheap methods first, optionally the already-loaded dense model:

- normalized character n-gram TF-IDF
- BM25 over training questions
- DEk21 embeddings over training questions
- RRF/weighted combination

For the nearest training QA, compute confidence features such as:

- normalized lexical similarity
- dense cosine score
- shared legal document number
- shared article/clause number
- named entity overlap
- question intent overlap
- difference in key amounts/dates/entities

Possible use cases:

### A. Direct answer reuse

Only for extremely high-confidence near duplicates where legal entities/intent agree.

### B. Style/template exemplar

Feed one high-confidence similar organizer-provided QA example into the generator as a style exemplar while forcing factual grounding in the current retrieved legal evidence.

### C. Output-length prior

Use the nearest training answer length as a prior for dynamic generation length.

### Critical leakage rule

OOF validation must build the QA memory **only from the other folds**. Group near-duplicate questions together where possible so the validation question is not trivially represented in training by a paraphrase.

A fuzzy-memory feature that looks amazing only under random leakage must not be shipped.

---

# 6. Retrieval system to implement and validate

Start from this candidate architecture, then keep only components that improve OOF score:

```text
Question
  |
  +--> exact QA memory --------------------------> candidate answer
  |
  +--> similar-QA memory ------------------------> candidate/style prior
  |
  +--> true BM25 over legal chunks -----+
  |                                      |
  +--> DEk21 dense retrieval ------------+--> RRF / calibrated fusion
                                              |
                                              v
                                       candidate pool 30–80
                                              |
                                              v
                                 BGE-reranker-v2-m3
                                              |
                                              v
                                        top 5–10 seeds
                                              |
                                              v
                               structured evidence packing
                                              |
                                              v
                             Qwen / extractive candidates
                                              |
                                              v
                                   source/fact snapping
                                              |
                                              v
                                   OOF-tuned selector
```

## 6.1 BM25

Requirements:

- true BM25
- no truncated posting lists
- corpus-wide
- query normalization consistent with corpus normalization
- legal document/article/clause boosts
- OOF-tune `k1`, `b`, entity boost weights, top-k

Useful search ranges:

- `k1`: 1.2, 1.5, 1.8
- `b`: 0.55, 0.75, 0.9
- top-k before fusion: 30, 50, 80

Do not grid-search everything expensively. Use staged ablations.

## 6.2 DEk21 dense retrieval

Prefer the 100M DEk21 model in the approved parameter stack unless another compliant alternative is empirically superior.

Precompute corpus embeddings from organizer legal contexts and package them.

For 801k x 768 embeddings:

- consider float16 storage for runtime memory if retrieval quality is unchanged
- validate row-to-`chunk_id` alignment with a manifest/checksum
- use batched query encoding
- use an efficient exact/ANN top-k implementation

A strong GPU approach is allowed if practical:

- keep normalized corpus embedding matrix on GPU 1 in FP16
- encode queries in batches
- compute chunked/batched matrix multiplication and `topk`

Or use FAISS if stable in Kaggle. Choose based on measured speed and score, not preference.

## 6.3 Fusion

Test:

- BM25 only
- dense only
- equal RRF
- weighted RRF / calibrated rank fusion

Candidate values around `RRF k=60` are a starting point, not a sacred constant.

## 6.4 Reranker

Use `BAAI/bge-reranker-v2-m3` only if parameter budget and OOF gain justify it.

Test candidate pool sizes such as 20/30/50 and final reranked top-k 5/8/10.

Batch pairs aggressively on GPU 1 and avoid repeated model loading.

## 6.5 Retrieval evaluation

Before generator experiments, measure retrieval independently against `retrieval_labels.parquet` / citation labels:

- Recall@1/5/8/10/20/50
- MRR
- document-level recall
- article-level recall

For every retrieval change, report whether failures are caused by:

- wrong document
- right document/wrong article
- right article/wrong clause
- source missing from labels
- query ambiguity

Do not optimize answer generation while retrieval recall is still obviously deficient.

---

# 7. Evidence construction / article stitching

The current stitcher simply takes siblings of the first seed and concatenates until a character limit. Improve it.

Create an evidence packer that scores and preserves structured legal units.

Possible policy:

1. Start with top reranked seed.
2. Include parent article heading + exact clause.
3. Include adjacent sibling clauses only when:
   - query asks about multiple conditions/remedies/penalties/exceptions; or
   - seed references them; or
   - reranked siblings are also high confidence.
4. Allow evidence from a second legal document when the answer/citation requires an amendment or cross-reference.
5. Deduplicate overlapping text.
6. Preserve document/article headers.
7. Allocate a token budget rather than blind character truncation.

Test evidence budgets approximately 1.5k / 2.5k / 4k / 6k characters or tokenizer-equivalent token limits.

Important: more context is not always better. OOF-test focused evidence vs stitched evidence.

---

# 8. Generator strategy

## 8.1 Base model

Primary generator: `Qwen/Qwen2.5-3B-Instruct`.

Use deterministic decoding first:

- `do_sample=False`
- temperature effectively 0
- modest repetition penalty only if validated

Avoid verbose meta-commentary and generic caveats not present in gold answers.

## 8.2 Prompt design

Train and infer with the same chat template.

The prompt should explicitly request the observed answer style:

- cite the relevant legal basis
- reproduce source wording where useful
- include all conditions, penalties, exceptions, remedies asked for
- preserve exact document/article/clause numbers, dates, money, percentages
- conclude by directly applying the rule to the question
- do not invent unsupported law
- do not over-summarize

Do not tell the model to be “concise” if the references are long.

## 8.3 Output length

OOF-test at least 192/256/320/384/512 `max_new_tokens`.

Also implement a dynamic budget experiment. Example feature-based bins:

- simple definition/form question -> shorter
- penalty/procedure/conditions with multiple clauses -> medium/long
- evidence pack spanning multiple clauses/docs -> long

Remember Hugging Face subword tokens are not equal to whitespace words. Inspect actual tokenizer length distribution of training answers.

## 8.4 QLoRA

Audit and improve `scripts/train_generator_qlora.py`.

Do not assume QLoRA improves METEOR. Prove it OOF.

Current concerns to address:

- examples with no retrieval label may receive empty evidence
- one oracle positive chunk may not resemble noisy inference evidence
- long answers may be truncated by `max_seq_len=2048`
- inference/validation currently may not load the resulting adapter

Build better training examples:

- grounded evidence from positive citation chunks
- stitched relevant legal units where needed
- optionally a mixture of oracle evidence and retrieval-produced evidence to reduce train/inference mismatch
- completion-only loss on assistant answer
- preserve full answer as much as feasible

Inspect tokenizer sequence percentiles before choosing max sequence length. Consider 2048/3072/4096 if memory/runtime permit.

Small hyperparameter sweep, not a huge one:

- LR: 5e-5, 1e-4, 2e-4
- epochs: 1 and possibly 2 if no overfit
- LoRA rank: keep small enough for the <4B audit; test only if meaningful

Select adapter by OOF METEOR, not training loss.

## 8.5 Multi-GPU training

If training inside Kaggle, ensure dual-T4 setup is genuinely correct. Do not claim DDP because `LOCAL_RANK` exists if the notebook is launched as a single Python process.

Use `accelerate launch` / torchrun only if necessary and stable. A simpler single-T4 QLoRA run is acceptable if it fits and lets the other T4 handle preprocessing; score is more important than nominal GPU utilization.

---

# 9. Candidate generation and METEOR-aware selection

This is one of the most important score-maximization areas.

For each OOF question, save all candidate answers, not only the selected one.

At minimum save:

- exact/fuzzy memory candidate
- focused extract
- stitched extract
- base generated
- adapted generated
- snapped generated
- generated + source append variants

For each candidate compute test-time features:

- answer whitespace length
- question whitespace length
- evidence length
- BM25 score/rank
- dense score/rank
- RRF score
- reranker score and margin between rank 1/rank 2
- shared legal entity count
- number of evidence documents/articles
- generated/evidence lexical overlap
- near-QA similarity and entity consistency
- presence of exact requested values/entities

During OOF only, compute oracle candidate METEOR to learn what kinds of questions favor each candidate.

Then choose one of:

- transparent rule-based selector tuned on OOF
- tiny regularized classifier/ranker trained only on OOF-safe features

Do not use a large extra model that threatens the parameter budget.

Report:

- score of every candidate family
- score of selected system
- oracle-best-among-candidates upper bound

The gap between selector score and oracle candidate score tells you whether to improve candidate quality or selector quality.

---

# 10. OOF protocol — mandatory

Do not ship based on intuition.

## 10.1 Exact scorer parity

Create one reusable evaluation function imported from or behavior-identical to the official scorer.

Verify on a small synthetic set that local results match the official scoring program exactly.

## 10.2 Leakage prevention

For validation fold `f`:

- exact/fuzzy QA memory contains only non-`f` records
- any learned selector is trained without the held-out records
- QLoRA validation must not train on fold `f`
- group exact/near-duplicate questions if possible
- avoid citation/document leakage when evaluating generalization claims

Run two validation views if feasible:

### Protocol A — competition-like stratified/grouped QA folds
Useful for model selection.

### Protocol B — stricter document/citation-disjoint diagnostic split
Useful to determine whether gains come only from memorizing seen legal documents.

Do not necessarily optimize final public score on the harsher split, but report both.

## 10.3 Staged experiments

Use this order to save GPU time:

### Phase 1 — retrieval only

Benchmark BM25, dense, fusion, reranker.

### Phase 2 — cheap answer candidates

Evaluate extractive / source append / similar-QA candidate families without Qwen where possible.

### Phase 3 — generator

Evaluate base Qwen on a representative OOF sample.

### Phase 4 — QLoRA

Train/evaluate only after retrieval/evidence is stable.

### Phase 5 — full OOF

Run final 2–3 candidate configurations over the full validation set.

Maintain an ablation table like:

| Run | Retrieval | Reranker | Evidence | Generator | Postprocess | METEOR | Runtime |
|---|---|---|---|---|---|---:|---:|

Never overwrite the prior best configuration without recording its score.

---

# 11. Kaggle runtime engineering

The final notebook must work from **Restart Session + Run All**.

## 11.1 Deterministic paths

Resolve:

- dataset root
- legal chunks
- known QA
- public test
- BM25 index
- dense embeddings
- Qwen model
- dense model
- reranker model
- optional LoRA adapter

with deterministic validation. Print paths and manifests, never secrets.

## 11.2 Dependency handling

Use Kaggle-preinstalled packages when possible. If a package must be installed, pin a compatible version and perform installation in one early cell.

Avoid unnecessary internet downloads during inference. Prefer mounted/precomputed artifacts.

## 11.3 Host RAM

Do not load unused parquet columns.

Use e.g.:

```python
pd.read_parquet(path, columns=[...needed columns...])
```

Avoid duplicating 801k full texts in multiple Python lists/dicts unless needed.

Use categorical/compact metadata or Arrow/Pandas columns where practical.

## 11.4 GPU placement

Use explicit devices:

```python
GEN_DEVICE = "cuda:0"
RETRIEVAL_DEVICE = "cuda:1"
```

Print `torch.cuda.mem_get_info()` at important phases.

If GPU 1 needs both dense embeddings and reranker, measure memory. You may unload the DEk21 encoder after all 1,000 query embeddings are computed while retaining only the corpus embedding matrix, then load the reranker.

## 11.5 Batch everything

Avoid per-query expensive model initialization.

- encode all test questions in batches
- dense search in batches
- rerank candidate pairs in batches if feasible
- generate Qwen answers in length-aware batches

Use length bucketing for Qwen to reduce padding waste.

## 11.6 Checkpoints and recoverability

Kaggle sessions can fail. Save intermediate prediction/cache artifacts under `/kaggle/working` periodically:

- retrieval results
- reranked evidence
- generated answer batches

Make cells idempotent so reruns can reuse compatible caches during development.

The final clean notebook can keep this mechanism without requiring persistence.

---

# 12. Submission safety checks

Before zipping:

- exactly 1,000 IDs
- ID set exactly equals public test ID set
- no empty/None answers
- every value has shape `{"answer": "..."}`
- all answers are strings
- UTF-8 JSON with `ensure_ascii=False`
- no NaN
- no internal tags such as `[DOCUMENT]`, `[ARTICLE]`, `[CLAUSE]` unless intentionally shown and validated
- no chat-template tokens
- no accidental prompts/model commentary
- zip contains exactly `submission.json` at archive root

Print summary statistics:

- total predictions
- exact-memory count
- fuzzy-memory-direct count
- generated count
- extractive-selected count
- mean/median/p90 answer whitespace length
- empty count
- mean retrieval/reranker confidence
- total runtime

---

# 13. Tests you must add/update

At minimum:

1. BM25 correctness vs a tiny hand-computable corpus.
2. No posting-list truncation/corpus-order bias.
3. RRF deterministic ranking.
4. Dense index row alignment with `chunk_id`.
5. Article stitcher ordering/deduplication.
6. Source snapping does not change unrelated numbers/dates.
7. Exact QA memory conflict handling.
8. Fuzzy QA memory leakage-safe fold behavior.
9. Base generator actually loads when no adapter is supplied.
10. Adapter loading path works.
11. Official METEOR parity test.
12. Candidate selector uses only inference-available features.
13. Parameter audit includes adapter/task heads.
14. Submission schema validation.
15. Clean Kaggle path-resolution smoke test.

Run the existing test suite and fix regressions.

---

# 14. Files likely requiring changes

Inspect everything, but expect to modify at least:

```text
kaggle_kernel/legalqa_gpu_pipeline.ipynb
kaggle_kernel/kernel-metadata.json
configs/pipeline.yaml
configs/models.yaml
requirements.txt
src/common/bm25.py
src/common/dense_dek21.py
src/common/reranker.py
src/common/rrf.py
src/common/normalize.py
src/task2/qa_memory.py
src/task2/article_stitcher.py
src/task2/generator.py
src/task2/source_snap.py
src/task2/predict.py
scripts/build_indexes.py
scripts/package_kaggle_dataset.py
scripts/run_oof_validation.py
scripts/train_generator_qlora.py
scripts/audit_parameters.py
README.md
tests/*
```

Do not rewrite stable components unnecessarily. Prefer one reusable implementation imported by both validation and Kaggle notebook so validation/inference cannot drift.

---

# 15. Strong preference: thin Kaggle notebook, real code in modules

The notebook should not contain a second independent implementation of retrieval/generation logic.

Preferred notebook structure:

1. environment + secret setup
2. locate repo/data/models
3. install/import dependencies
4. preflight + parameter audit
5. load canonical pipeline
6. batch retrieval/reranking
7. generation/candidate selection
8. submission validation
9. save ZIP + diagnostics

Core behavior should live in tested Python modules.

If Kaggle does not automatically include the repository package, add a robust `sys.path` setup or package install step, not copy-pasted duplicate algorithms.

---

# 16. What NOT to do

- Do not optimize for subjective answer quality instead of official METEOR.
- Do not claim a component improves score without OOF evidence.
- Do not report mock/fallback OOF as final performance.
- Do not use public test labels; they are `None`.
- Do not leak validation answers through QA memory.
- Do not hardcode or print `HF_TOKEN`.
- Do not use external legal/QA datasets or external answer APIs.
- Do not accidentally exceed 4B learned parameters.
- Do not load multiple full Qwen checkpoints simultaneously.
- Do not use BGE-M3 dense + BGE reranker + Qwen3B without re-auditing the parameter budget.
- Do not use `max_new_tokens=180` merely because it is faster.
- Do not append 1,500 chars of evidence to every short answer without OOF proof.
- Do not rebuild 801k dense embeddings on every clean Kaggle run if compliant precomputed embeddings can be packaged.
- Do not select the first arbitrary `/kaggle/input/**/config.json` as the generator.
- Do not leave README/config/notebook claiming different architectures.

---

# 17. Priority order for expected score gain

Use this as the initial plan, then adapt to measured results:

### P0 — correctness / parity

- fix full OOF so it evaluates the real final path
- exact official METEOR scorer parity
- reconcile configs/model budget
- fix generator fallback bug

### P1 — retrieval quality

- real BM25
- precomputed DEk21
- RRF
- BGE reranker
- structured evidence packer

### P2 — METEOR-specific answer formulation

- longer answer budget
- extractive/generated candidate ensemble
- OOF-tuned Strategy F lengths
- similar-QA memory / answer-style retrieval

### P3 — generator adaptation

- QLoRA with grounded evidence and prompt parity
- output-length calibration

### P4 — engineering

- dual-T4 batching
- precomputed index packaging
- RAM reduction
- resumable cache

Do not spend hours micro-optimizing GPU throughput before P0–P2 are correct.

---

# 18. Required deliverables

When you finish, provide all of the following:

## A. Actual repository edits

Implement the fixes, not just descriptions.

## B. Final architecture summary

A concise diagram showing the exact shipped inference path.

## C. Parameter-budget report

List every learned checkpoint/adapter/task head used at inference with parameter count and strict total <4B.

## D. OOF ablation table

Show the measured METEOR of major variants and identify the chosen final configuration.

## E. Kaggle run instructions

Exact steps for the user:

- which Kaggle datasets/models to attach
- whether Internet must be on/off
- required secret label (`HF_TOKEN`)
- accelerator (`T4 x2`)
- any one-time commands needed to update/precompute Kaggle dataset indexes

## F. Final notebook

`kaggle_kernel/legalqa_gpu_pipeline.ipynb` must be clean, readable, deterministic, and Run-All capable.

## G. Expected logs

Show the key expected startup/output lines so the user can identify a broken run quickly.

## H. Final risk list

Only unresolved real risks; no generic filler.

---

# 19. Definition of done

Do not call the task finished until all of these are true:

- [ ] Final notebook uses the same core code as full OOF validation.
- [ ] Official METEOR implementation is reproduced exactly.
- [ ] Best architecture is selected using measured OOF METEOR.
- [ ] Retrieval uses full-corpus real BM25 without 8k posting cap.
- [ ] Dense retrieval and reranking are either correctly used or explicitly removed because OOF proved they hurt.
- [ ] Both T4s are used sensibly or there is a measured reason not to.
- [ ] Qwen base model actually loads in no-adapter inference.
- [ ] QLoRA is used only if it improves OOF.
- [ ] Strategy F/source append behavior is OOF-tuned, not arbitrary.
- [ ] Answer length is OOF-tuned and no longer hardcoded to an unjustified 180 tokens.
- [ ] Similar-QA memory is tested leakage-safely.
- [ ] Precomputed indexes can be packaged/mounted.
- [ ] Parameter audit includes every learned component and stays strictly <4B.
- [ ] All configs/docs reflect the same model stack.
- [ ] Full unit/integration tests pass.
- [ ] Submission schema check passes for all 1,000 IDs.
- [ ] No secrets are present in repo/output.
- [ ] A clean Kaggle `Restart Session -> Run All` completes and creates `submission.json.zip`.

---

# 20. How to communicate while working

Do not stop after finding the first problem. Work systematically.

Use this response structure while you execute the task:

1. **Repository audit findings** — concrete issues with file/function references.
2. **Plan ranked by expected METEOR gain and implementation risk.**
3. **Changes made** — file-by-file.
4. **Tests/benchmarks run and actual results.**
5. **OOF ablation results.**
6. **Final selected configuration and why.**
7. **Exact Kaggle instructions.**
8. **Remaining risks.**

If you cannot run a required GPU experiment in your environment, still implement the instrumentation/configuration and give exact Kaggle commands, but clearly label the result as **UNMEASURED**. Never invent benchmark numbers.

The final goal is not to make the repository look sophisticated. The goal is to produce the **highest-scoring, compliant, reproducible LegalQA Task 2 submission possible**.
