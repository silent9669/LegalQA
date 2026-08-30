# LegalQA Artifact, Validation, and Model Design

**Date:** 2026-08-25
**Status:** Approved for planning and implementation

## Goal

Establish a reproducible Task 2-only LegalQA workspace, replace the optimistic sampled validator with a benchmark-equivalent full-data protocol, and gate neural retrieval/reranking/generation experiments behind verified evidence labels and the strict learned-parameter budget.

## Competition constraints

- Use only data supplied by the organizers for Task 2 as task training data, retrieval evidence, memory, and evaluation input.
- Do not use Task 1 data, external legal corpora, synthetic/paraphrased QA, or external inference/training APIs.
- Run inference offline.
- Keep the sum of all learned parameters loaded by the system below 4,000,000,000, counting independent checkpoints, adapters, and task-specific learned heads conservatively.
- Compute primary METEOR exactly as `meteor_score([str(reference).split()], str(prediction).split())`.
- Require prediction and reference key sets to match exactly before scoring.

## Artifact authority

`artifacts/` is the sole authoritative hierarchy.

- `artifacts/raw/` contains organizer-provided Task 2 inputs.
- `artifacts/chunks/legal_chunks.parquet` is the active 365,046-row runtime corpus.
- `artifacts/chunks/chunks_output.jsonl` remains a retained reconstruction intermediate until raw-context parsing is reproducible.
- `artifacts/data/` contains canonical QA and exact-memory derivatives.
- `artifacts/labels/` contains derived supervision artifacts.
- `artifacts/submissions/` contains regenerable outputs.
- `artifacts/manifests/` will contain hashes, schemas, counts, provenance, and model manifests.
- `artifacts/archive/` may contain quarantined legacy artifacts and must never be selected by runtime fallback logic.

`trung_artifacts/legal_chunks.parquet` is not runtime-compatible: it has 344,301 rows, 14 columns, and lacks fields required by prediction. It must not replace the active parquet. The differing legacy parquet and memory file will be quarantined with provenance metadata; exact duplicates may be deleted.

## Known data defects to resolve

1. Fourteen normalized-question groups have conflicting answers. Ambiguous groups must not enter normalized-question exact memory.
2. Thirty-eight non-empty selected contexts have no generated chunks.
3. The chunk corpus has 2,059 duplicate `(doc_id, dieu, part)` groups involving 5,156 rows; Article stitching must not blindly concatenate duplicate parts.
4. Existing retrieval labels are citation-only and weakly parsed. They do not contain resolved positive document/article/chunk IDs or hard negatives.
5. Two current submission answers contain literal `None`; 47 contain slug-like citations.

## Collaboration and storage

Git stores code, configs, tests, small manifests, checksums, and download scripts. Git must not store multi-gigabyte model checkpoints, dense indexes, or duplicated raw corpora as ordinary blobs.

Preferred storage:

1. A private Hugging Face organization repository for model checkpoints/adapters and optionally datasets, using pinned revisions and LFS-backed files.
2. An S3-compatible object store such as Cloudflare R2 for immutable artifact bundles, dense indexes, OOF traces, and large generated files.
3. Git LFS only when the team accepts GitHub quota/bandwidth constraints; it is not the primary recommendation for repeated multi-GB model/index updates.

Every external bundle must have a local manifest containing SHA-256, byte size, logical type, source, creation command, schema/version, and remote URI. Secrets or access tokens must never be committed.

## Validation architecture

### Split family A: question-blocked OOF

- Five deterministic folds.
- Exact and near-duplicate question groups are atomic.
- Conflicting-answer groups remain atomic and are reported separately.
- Balance source split, question type, answer length, citation coverage, and document frequency.
- Exact-ID and normalized-question memory are excluded from the primary generalization score and reported separately as deployment diagnostics.

### Split family B: document-held-out stress test

- Reliable resolved document clusters are atomic.
- No fold-trained labels, provision examples, adapters, or selectors may use held-out document supervision.
- The complete organizer context corpus remains searchable, matching deployment.
- Unresolved citation rows remain in an explicit unknown-document group.

### Stage traces

Each prediction trace records memory decision, retrieval IDs/scores, reranked IDs/scores, stitched evidence IDs, all candidate answers, selected candidate, final answer, errors, latency, and per-row metrics.

### Metrics

- Official METEOR and ROUGE-L.
- Document/article/chunk Recall@1/5/10/20/50, MRR, and nDCG.
- Reranker pre/post rank delta.
- Citation and number/date/amount fidelity.
- Article Stitcher win/loss rate.
- Gold-evidence versus retrieved-evidence answer gap.
- Memory-hit and unseen-only scores.
- Grouped paired bootstrap confidence intervals.

### Promotion gates

- Complete 7,113-row evaluation.
- Exact scorer fixture parity and key-set equality.
- No duplicate-group leakage.
- At least +0.005 absolute METEOR versus the promoted baseline.
- Paired bootstrap 95% lower bound above zero.
- No document/article Recall@10 regression.
- No major subgroup regression greater than 0.02 METEOR.
- Deterministic repeated predictions.
- Recorded runtime, peak RAM/VRAM, index size, model revision, license, and conservative parameter count.

## Neural model experiment order

1. Repair artifact integrity and resolved evidence labels.
2. Run BM25/extractive v2 under the new validation harness.
3. Compare dense retrievers with downstream answer construction fixed.
4. Compare neural rerankers on identical cached candidate sets.
5. Compare generators with both gold and retrieved evidence.
6. Fine-tune only on organizer Task 2 data using fold-isolated LoRA/QLoRA.
7. Introduce provision memory and learned candidate selection only after complete OOF candidate traces exist.

Primary candidate stack:

- BM25 plus `Qwen/Qwen3-Embedding-0.6B`.
- `Qwen/Qwen3-Reranker-0.6B`.
- `Qwen/Qwen3-1.7B` in non-thinking mode.
- Existing extractive composer, Article Stitcher, and source alignment as deterministic safeguards.

Conservative total: approximately 3.2233B parameters before adapters/heads.

Required controls:

- Dense retrieval: BGE-M3 and multilingual-E5-large-instruct.
- Reranking: BGE-reranker-v2-m3.
- Generation: Qwen2.5-1.5B-Instruct and extractive-only.

## Documentation policy

Every README, config, and Notion component must be labelled as one of:

- Implemented
- Measured
- Configured but unused
- Planned
- Unverified

Only stored score logs and reproducible validation artifacts may be described as measured. The current `0.3542 ± 0.0223` sampled OOF result is a legacy development signal, not the promotion benchmark.
