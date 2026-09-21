# 00 — Evidence and Measurements

Every number asserted anywhere in this folder appears here with the script that produced it.
All measurements are **CPU-only** and reproduce on a laptop in under 15 minutes.

**Environment used:** `.venv311/bin/python` (Python 3.11.16, transformers 5.17.0),
`HF_HUB_OFFLINE=1` with `Qwen/Qwen2.5-3B-Instruct` tokenizer already in `~/.cache/huggingface/hub`.

---

## M-1 · Dataset shape

```bash
.venv311/bin/python -c "
import pandas as pd
qa  = pd.read_parquet('kaggle_dataset/qa_unique.parquet')
lab = pd.read_parquet('kaggle_dataset/retrieval_labels.parquet')
qa_ids=set(qa['qa_id'].astype(str)); lab_ids=set(lab['qa_id'].astype(str))
print('QA rows                     :', len(qa_ids))
print('QA with >=1 retrieval label :', len(qa_ids & lab_ids))
print('QA with NO evidence         :', len(qa_ids - lab_ids))
print('source_split:', qa['source_split'].value_counts().to_dict())
"
```

| quantity | value |
|---|---|
| unique `qa_id` | **7,113** (parquet has 7,500 rows → 387 duplicate ids) |
| with ≥1 retrieval label | **4,759 (66.9 %)** |
| **with NO evidence** | **2,354 (33.1 %)** |
| mean labels per labelled qa | 1.34 |
| `source_split` | `train` 7,000 · `warmup` 500 |
| `is_conflict=True` | 32 |
| private test questions | **1,918** |

> **Consequence.** `build_grounded_training_examples`
> (`src/task2/generation/dataset.py:117-118`) sets `raw_evidence = ""` when a qa_id has no
> label. One third of the SFT set therefore teaches the model to answer with an **empty
> `[CĂN CỨ PHÁP LÝ]` block** — a condition that never occurs at inference. See bug B-07.

---

## M-2 · Gold answer length, real Qwen tokenizer

```bash
HF_HUB_OFFLINE=1 .venv311/bin/python -c "
import pandas as pd, numpy as np
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
qa = pd.read_parquet('kaggle_dataset/qa_unique.parquet')
s = qa['answer_raw'].astype(str).sample(1200, random_state=0).tolist()
lens = np.array([len(tok.encode(x, add_special_tokens=False)) for x in s])
for q in (50,75,90,95,99): print(f'p{q}: {np.percentile(lens,q):.0f}')
for cap in (384,512,768,1024,1280,1536):
    print(f'covered by {cap}: {(lens<=cap).mean()*100:.1f}%')
"
```

| percentile | tokens |
|---|---|
| p50 | 414 |
| p75 | 564 |
| p90 | 779 |
| p95 | 924 |
| p99 | 1,284 |
| mean / max | 458 / 2,030 |

tokens-per-whitespace-word ratio: **1.33**

| `max_new_tokens` | gold answers fully covered |
|---|---|
| 384 | 44.6 % |
| **512 (current)** | **67.5 %** |
| 768 | 89.6 % |
| **1024** | **96.7 %** |
| 1280 | 98.9 % |

> **`scripts/modal_app.py:198-200` claims 512 covers "90 %+". Measured: 67.5 %.** Bug B-02.

---

## M-3 · Perfect-model ceiling vs generation cap (official scorer)

Hypothesis = the gold answer truncated at N tokens. This isolates the cost of the cap
from every other factor: it is the score an *oracle* generator would receive.

```bash
# full script: see M-3 block in this file's git history, or re-derive:
#   refs = 200 gold answers; hyp = tok.decode(tok.encode(ref)[:cap])
#   METEOR  = nltk.translate.meteor_score.meteor_score([ref.split()], hyp.split())
#   ROUGE-L = rouge_score.RougeScorer(['rougeL'], use_stemmer=False)
```

| cap | METEOR | ROUGE-L |
|---|---|---|
| 384 | 0.8055 | 0.8843 |
| **512** | **0.8961** | 0.9435 |
| 768 | 0.9678 | 0.9854 |
| **1024** | **0.9874** | 0.9952 |
| 1280 | 0.9958 | 0.9986 |
| uncapped | 0.9979 | 0.9993 |

> **512 → 1024 raises the achievable ceiling by +0.091 METEOR absolute.**

---

## M-4 · Assembly strategy, oracle retrieval (n = 150)

Prose proxy = the last 120 words of the gold answer (the `"Theo đó…"` conclusion — what a
summarising model produces). Citation = the full gold article, cleaned via
`clean_statutory_text`, headed by `build_citation_header`.

| variant | METEOR | ROUGE-L | words |
|---|---|---|---|
| A · prose only (~120 w) | 0.4520 | 0.5745 | 119 |
| **B · prose + cite @4000 chars** | **0.6813** | 0.5429 | 473 |
| C · prose + cite @8000 chars | 0.6800 | 0.5406 | 517 |
| D · cite only @4000 chars | 0.4790 | 0.5443 | 350 |
| E · gold (end-to-end ideal) | 1.0000 | 1.0000 | 349 |

> Two things to take from this table.
> **(i)** Stitching is worth **+0.229 METEOR** over prose alone — this is why 0.5486 beat 0.49.
> **(ii)** It *costs* **−0.032 ROUGE-L**. The two official metrics genuinely conflict.
> **(iii)** 8000 chars is no better than 4000 → **4000 is the right citation budget.**

---

## M-5 · Multi-article stitching under imperfect retrieval (n = 150)

Does hedging across the top-k articles beat committing to the top-1?

| scenario | METEOR | ROUGE-L | words |
|---|---|---|---|
| HIT · 1 article (oracle) | **0.6810** | 0.5430 | 475 |
| HIT · 2 articles (gold + 1 wrong) | 0.5881 | 0.4185 | 813 |
| HIT · 3 articles (gold + 2 wrong) | 0.4974 | 0.3283 | 1204 |
| MISS · 1 article | 0.4522 | 0.3601 | 532 |
| MISS · 2 articles | 0.4129 | 0.2908 | 842 |
| MISS · 3 articles | 0.3904 | 0.2551 | 1161 |

Expected METEOR, mixing HIT/MISS by top-1 accuracy `p` (extra articles assumed to add
18 % relative recall each):

| p (top-1 correct) | k=1 | k=2 | k=3 |
|---|---|---|---|
| 0.30 | 0.5208 | 0.4875 | 0.4470 |
| 0.40 | 0.5437 | 0.5019 | 0.4542 |
| **0.42** | **0.5483** | 0.5048 | 0.4557 |
| 0.50 | 0.5666 | 0.5163 | 0.4614 |
| 0.60 | 0.5895 | 0.5306 | 0.4686 |
| **0.65** | **≈0.6010** | — | — |
| 0.70 | 0.6124 | 0.5450 | 0.4758 |

> **k=1 dominates at every hit-rate.** METEOR's 10 % precision term plus the
> fragmentation penalty (`beta=3, gamma=0.5`) punish the added length more than the recall
> gain repays it. **Never stitch more than one article.**
>
> The current `EvidencePacker` pack type `primary_full_article`
> (`src/task2/predict.py:435`) already returns exactly one article. **Keep it that way.**

---

## M-6 · Prose length inside dual-assembly (n = 150, oracle article)

Prose proxy = gold answer truncated to N generated tokens (i.e. a perfectly SFT'd model).

**With the citation block appended:**

| `max_new_tokens` | METEOR | ROUGE-L | words |
|---|---|---|---|
| 0 (cite only) | 0.4790 | 0.5444 | 353 |
| 128 | 0.5986 | 0.5984 | 448 |
| 256 | 0.6920 | 0.6423 | 540 |
| 384 | 0.7542 | 0.6587 | 612 |
| **512** | **0.7864** | 0.6635 | 656 |
| **768** | **0.8121** | 0.6662 | 694 |
| 1024 | 0.8180 | 0.6671 | 704 |

**Without the citation block (pure end-to-end):**

| `max_new_tokens` | METEOR | ROUGE-L | words |
|---|---|---|---|
| 384 | 0.8049 | 0.8877 | 255 |
| 512 | 0.9026 | 0.9506 | 299 |
| 768 | 0.9789 | 0.9908 | 337 |
| **1024** | **0.9953** | 0.9985 | 347 |

> **The crossover.** A *strong* generator is actively harmed by stitching
> (0.9953 → 0.8180, −0.18) because it already quoted the article and the append duplicates
> it: METEOR aligns each reference token once, so the copy earns **+0 numerator and
> +0.1 denominator**, and fragments the alignment.
> A *weak* generator is rescued by it (0.4520 → 0.6813, +0.23).
> **Which regime the fine-tuned adapter is in has never been measured.** See `05-open-questions.md` Q-1.

---

## M-7 · Known-QA override coverage on the private test

```bash
.venv311/bin/python -c "
import json, re, unicodedata
k=json.load(open('kaggle_dataset/known_qa.json')); qm=k['question_map']; im=k['id_map']
priv=json.load(open('private-official.json'))
def norm(s):
    s=unicodedata.normalize('NFC',str(s)).lower()
    return ' '.join(re.sub(r'[^\w\s]',' ',s).split())
kn={norm(q) for q in qm}
print('id overlap :', len(set(map(str,priv)) & set(im)))
print('question hit:', sum(1 for v in priv.values() if norm(v.get('question','')) in kn))
"
```

| quantity | value |
|---|---|
| `known_qa.json` → `question_map` | 7,082 |
| `known_qa.json` → `id_map` | 7,113 |
| private-test **id** overlap | **73** |
| private-test deep-normalised **question** hit | **81 (4.2 %)** |
| METEOR uplift if all 81 are exact | **+0.0190** |

> Free and reliable. Confirms the "82 known-QA overrides" figure from notebook v13.2.

---

## M-8 · VRAM budget, bf16 inference on A100-40GB

```bash
HF_HUB_OFFLINE=1 .venv311/bin/python -c "
from transformers import AutoConfig
c = AutoConfig.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
hd = c.hidden_size // c.num_attention_heads
kv = c.num_hidden_layers * 2 * c.num_key_value_heads * hd * 2
print('layers', c.num_hidden_layers, 'kv_heads', c.num_key_value_heads, 'head_dim', hd)
print('KV KB/token: %.1f' % (kv/1024))
for B,L in ((16,2560),(32,2560),(48,3072),(64,3072)):
    print(f'batch {B} ctx {L}: {B*L*kv/1e9:.2f} GB')
"
```

Qwen2.5-3B: 36 layers · 16 heads · **2 KV heads (GQA)** · head_dim 128 → **36.0 KB/token**.

| component | size |
|---|---|
| bf16 weights | **6.18 GB** |
| KV @ batch 16 × 2560 | 1.51 GB |
| KV @ batch 32 × 2560 | 3.02 GB |
| **KV @ batch 48 × 3072** | **5.44 GB** |
| KV @ batch 64 × 3072 | 7.25 GB |

> **bf16 at batch 48 uses ~12-14 GB of 40 GB.** There is no memory argument for 4-bit
> NF4 at inference on this GPU — only a large speed penalty. Bug B-03.

---

## M-9 · Official scorer, verbatim

`Scoring-Program-Task-LegalQA/scoring.py:24-52`:

```python
rouge_scoring = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

def build_in_tokenizer(string_sent):
    # Dung pyvi cho tokenizer
    # return ViTokenizer.tokenize(str(string_sent))
    # Khong dung Tokenizer
    return string_sent

rouge_result  = np.array([rouge_scoring.score(build_in_tokenizer(str(y_true[k])),
                                              build_in_tokenizer(str(y_pred[k])))['rougeL'].fmeasure
                          for k in ids_preds]).mean()
meteor_result = np.array([meteor_score([build_in_tokenizer(str(y_true[k])).split()],
                                       build_in_tokenizer(str(y_pred[k])).split())
                          for k in ids_preds]).mean()
return {'rouge': rouge_result, 'meteor': meteor_result}
```

`meteor_score` is called with **no keyword arguments**, so NLTK's defaults apply
(verified against the installed NLTK):

```
alpha: float = 0.9,  beta: float = 3.0,  gamma: float = 0.5
preprocess = str.lower,  stemmer = PorterStemmer()
```

- **No Vietnamese tokenizer.** `pyvi` is commented out; both metrics operate on raw
  whitespace splits of the original string.
- **Two independent numbers** are emitted. Nothing in the scoring program combines them.
- Our local `src/task2/metrics.py` mirrors both exactly (whitespace `.split()` for METEOR,
  the official `rouge_scorer` for ROUGE-L) — no divergence found.

See `02-scoring-model.md` for what these constants imply.

---

## M-10 · SFT sequence budget at `max_seq_len = 2048`

Answers whether long gold answers are being silently dropped by the 2048-token window.
Sample: 800 **evidence-bearing** examples (deduplicated by `qa_id`), real Qwen tokenizer,
completion measured as `answer_raw + "<|im_end|>"`, framing measured from
`format_qwen_chat_prompt(q, "")`.

```bash
HF_HUB_OFFLINE=1 .venv311/bin/python -c "
import pandas as pd, numpy as np, sys; sys.path.insert(0,'.')
from transformers import AutoTokenizer
from src.task2.generator import format_qwen_chat_prompt
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
qa  = pd.read_parquet('kaggle_dataset/qa_unique.parquet').drop_duplicates('qa_id')
lab = pd.read_parquet('kaggle_dataset/retrieval_labels.parquet').drop_duplicates('qa_id')
qa  = qa[qa['qa_id'].astype(str).isin(set(lab['qa_id'].astype(str)))].sample(800, random_state=5)
framing = len(tok.encode(format_qwen_chat_prompt('Q','',tokenizer=tok), add_special_tokens=False))
need = np.array([len(tok.encode(str(a)+'<|im_end|>', add_special_tokens=False)) for a in qa['answer_raw']]) + framing
for L in (2048,3072,4096):
    b = np.maximum(L-need-8, 0)
    print(L, f'drop={(need>L).mean()*100:.2f}%', f'ev_budget_mean={b.mean():.0f}')
"
```

| `max_seq_len` | examples dropped | evidence budget mean | p10 | zero-budget |
|---|---|---|---|---|
| **2048 (current)** | **0.50 %** | 1,396 tok | 1,111 | 0.5 % |
| 3072 | 0.00 % | 2,419 tok | 2,135 | 0.0 % |
| 4096 | 0.00 % | 3,443 tok | 3,159 | 0.0 % |

Chat framing (system prompt + template, empty evidence) = **178 tokens**.

> **Conclusion: leave `max_seq_len` at 2048.** It drops half a percent of examples and
> still leaves ~1,400 tokens of evidence budget. Raising it would re-mint the candidate and
> lengthen every training step to recover 0.5 % of the data. Resolves [Q-6](05-open-questions.md).

---

## M-11 · Duplicate `qa_id` rows

```bash
.venv311/bin/python -c "
import pandas as pd
qa = pd.read_parquet('kaggle_dataset/qa_unique.parquet')
d = qa[qa.duplicated('qa_id', keep=False)]
print('dup rows:', len(d), '| dup ids:', d['qa_id'].nunique())
print('answers disagree :', int((d.groupby('qa_id')['answer_raw'].nunique()>1).sum()))
print('questions disagree:', int((d.groupby('qa_id')['question_raw'].nunique()>1).sum()))
print('splits:', d['source_split'].value_counts().to_dict())
"
```

| quantity | value |
|---|---|
| duplicated rows | 774 |
| duplicated ids | **387** |
| ids whose duplicates disagree on the **answer** | **0** |
| ids whose duplicates disagree on the **question** | **0** |
| split distribution | `train` 387 · `warmup` 387 |

> Each duplicated id appears once in `train` and once in `warmup`, byte-identical. Not a
> data conflict — but `build_grounded_training_examples` iterates **rows**, so these 387
> examples receive double gradient weight. Deduplicate by `qa_id`. Resolves [Q-7](05-open-questions.md).

---

## M-12 · The local `dek21` dense index is a MOCK index of random vectors

Found while trying to measure top-1 article accuracy locally. Cosine between a query and
its own gold chunk came out at ≈ 0, so the index itself was tested.

**Test 1 — self-consistency.** Re-encode a chunk's own text with the declared model and
compare to its stored vector. A correct index scores ≈ 0.999.

| chunk | field | cos(stored, re-encoded) |
|---|---|---|
| `doc126324_art2_p0` | `text_raw` | **+0.0119** |
| `doc126324_art2_p0` | `text_norm` | +0.0369 |
| `doc200355_art86_p0` | `text_raw` | −0.0292 |
| `doc290432_art12_p0` | `text_raw` | −0.0053 |
| `doc159081_art1_full` | `text_raw` | +0.0075 |

**Test 2 — geometry.** Pairwise cosine among 400 randomly sampled stored vectors:

| | mean | std |
|---|---|---|
| stored vectors | **+0.0002** | **0.0361** |
| true random unit vectors, 768-d (control) | +0.0001 | 0.0362 |
| theoretical random, `1/√768` | 0.0000 | 0.0361 |
| *a real text encoder on one domain* | *+0.2 … +0.6* | — |

The stored matrix is statistically **indistinguishable from random unit vectors**.
`kaggle_dataset/indexes/dek21/embeddings.npy` (801,863 × 768 fp16, 1.23 GB) carries no
semantic content, despite `dek21_manifest.json` declaring
`model_id: CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2`.

**Test 3 — does the existing guard catch it?** Yes.

```bash
.venv311/bin/python -c "
from scripts.rebuild_dense_index import check_dense_alignment
print(check_dense_alignment('kaggle_dataset/indexes/dek21','kaggle_dataset/legal_chunks.parquet'))"
```

```
{'num_pairs': 400, 'threshold': 0.95, 'mean_cosine': 0.0012,
 'p5_cosine': -0.0612, 'pass_rate': 0.0, 'aligned': False, 'status': 'measured'}
```

**Test 4 — is production affected? No.** `kaggle_dataset/dataset_manifest.json` lists only
`indexes/bm25` under `indexes` — `indexes/dek21` is **not in the manifest and does not ship
to Kaggle**. The Modal container therefore finds no staged dense index, `check_dense_alignment`
returns `missing`, and `modal_app.py:482-494` cold-rebuilds with the pinned revision.

> **Scope, stated precisely.** This is **not** a production scoring bug — the guard and the
> manifest both hold. It is a **local measurement hazard**: any offline retrieval experiment
> run against `kaggle_dataset/indexes/dek21` silently measures noise, and
> `scripts/package_kaggle_dataset.py:87-93` *would* ship the directory if someone re-packaged
> the dataset while it exists on disk. See bug [B-18](01-bug-register.md#b-18).
>
> It also means **top-1 article accuracy could not be measured locally.** The `p ≈ 0.42`
> figure in [`02-scoring-model.md`](02-scoring-model.md) remains an *inference from the observed score*, not a
> direct measurement. Getting a real number requires a rebuilt index — see [Q-4](05-open-questions.md).
