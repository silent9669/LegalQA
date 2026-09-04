# DSC 2026 Task 2 — LegalQA Maximum-Score Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a top-performing, strictly rule-compliant ($< 4\text{B}$ parameter budget, zero external data/APIs) Vietnamese LegalQA RAG pipeline evaluated against a 100% faithful local mirror of the official CodaBench evaluation benchmark (`scoring.py`), featuring 5-Fold OOF validation, exact QA memory, hybrid BM25 + dense retrieval, cross-encoder reranking, legal provision memory, generative & extractive candidate synthesis, source-snapping, and candidate selection.

**Architecture:** A multi-tier architecture with exact deterministic memory lookup, structured legal hierarchy chunking, two-level hybrid document/chunk retrieval with RRF, cross-encoder reranking, multi-candidate generation (LLM + extractive + source-snapped), and feature-based candidate selection evaluated via whitespace-split `meteor_score`.

**Tech Stack:** Python 3.14 / 3.10+, PyArrow, FastParquet, Pandas, NumPy, NLTK (WordNet, OMW-1.4, meteor_score), Rouge-Score, Scikit-Learn / Rank-BM25, PyTorch, HuggingFace Transformers, PyTest.

**Spec:** `docs/superpowers/specs/2026-08-25-legalqa-maximum-score-design.md`

## Global Constraints
- Total learned parameters of all models combined must be $< 4,000,000,000$ ($< 4\text{B}$).
- Only BTC Task 2 datasets permitted (`train.json`, `warmup.json`, `selected-contexts`).
- Zero external API calls during inference/evaluation.
- Primary metric is official CodaBench `meteor_score` with whitespace split (`meteor_score([ref.split()], pred.split())`).
- Output format for submissions must be valid JSON: `{"<id>": {"answer": "<string>"}}` matching the input ID keys exactly.

---

### Task 1: Environment, Dependencies & Official CodaBench Evaluator

**Files:**
- Create: `src/evaluation/codabench_eval.py`
- Create: `tests/test_evaluation.py`
- Create: `requirements.txt`

**Interfaces:**
- Produces: `evaluate_predictions(y_pred: dict[str, dict[str, str]], y_true: dict[str, str | dict]) -> dict[str, float]` returning `{"meteor": float, "rouge": float}`.

- [ ] **Step 1: Write test for CodaBench evaluation mirror**

```python
import pytest
from src.evaluation.codabench_eval import evaluate_predictions

def test_codabench_eval_identical_answers():
    y_true = {"1": "Căn cứ Điều 50 Bộ luật Tố tụng hình sự 2015"}
    y_pred = {"1": {"answer": "Căn cứ Điều 50 Bộ luật Tố tụng hình sự 2015"}}
    scores = evaluate_predictions(y_pred, y_true)
    assert scores["meteor"] == pytest.approx(1.0, 1e-4)
    assert scores["rouge"] == pytest.approx(1.0, 1e-4)

def test_codabench_eval_whitespace_tokenization():
    y_true = {"1": "Điều 17, khoản 3"}
    y_pred = {"1": {"answer": "Điều 17 khoản 3"}}
    scores = evaluate_predictions(y_pred, y_true)
    assert 0.0 < scores["meteor"] < 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/phucdang/Downloads/.venv/bin/python3 -m pytest tests/test_evaluation.py -v`
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Implement `src/evaluation/codabench_eval.py`**

```python
import numpy as np
import nltk
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

def evaluate_predictions(y_pred: dict, y_true: dict) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    
    # Extract string answers if dictionary passed
    clean_preds = {
        str(k): (v['answer'] if isinstance(v, dict) and 'answer' in v else str(v))
        for k, v in y_pred.items()
    }
    clean_true = {
        str(k): (v['answer'] if isinstance(v, dict) and 'answer' in v else str(v))
        for k, v in y_true.items()
    }
    
    ids = [k for k in clean_true if k in clean_preds]
    if not ids:
        return {"meteor": 0.0, "rouge": 0.0}
        
    rouge_vals = [
        scorer.score(clean_true[k], clean_preds[k])['rougeL'].fmeasure
        for k in ids
    ]
    meteor_vals = [
        meteor_score([clean_true[k].split()], clean_preds[k].split())
        for k in ids
    ]
    
    return {
        "meteor": float(np.mean(meteor_vals)),
        "rouge": float(np.mean(rouge_vals))
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/phucdang/Downloads/.venv/bin/python3 -m pytest tests/test_evaluation.py -v`
Expected: PASS

- [ ] **Step 5: Write `requirements.txt`**

```text
numpy>=1.24.0
pandas>=2.0.0
pyarrow>=14.0.0
fastparquet>=2024.0.0
nltk>=3.8.0
rouge-score>=0.1.2
pytest>=7.0.0
scikit-learn>=1.3.0
rank-bm25>=0.2.2
tqdm>=4.65.0
```

---

### Task 2: Canonical Data Preparation & Exact QA Memory

**Files:**
- Create: `src/data/canonical.py`
- Create: `src/memory/exact_memory.py`
- Create: `tests/test_data.py`
- Create: `tests/test_memory.py`

**Interfaces:**
- `normalize_vietnamese_text(text: str) -> str`: Unicode NFC & whitespace normalization.
- `build_canonical_qa(train_path: str, warmup_path: str) -> tuple[pd.DataFrame, dict]`: Returns `(df_unique_7113, memory_dict)`.
- `ExactMemory.lookup(sample_id: str, question: str) -> str | None`: Deterministic O(1) lookup.

- [ ] **Step 1: Write test for exact memory and normalization**

```python
import pytest
from src.data.canonical import normalize_vietnamese_text
from src.memory.exact_memory import ExactMemory

def test_normalize_vietnamese_text():
    raw = "  Nghị   định  13/2023/NĐ-CP   về dữ   liệu ? "
    norm = normalize_vietnamese_text(raw)
    assert norm == "Nghị định 13/2023/NĐ-CP về dữ liệu ?"

def test_exact_memory_lookup():
    mem_dict = {
        "by_id": {"23207": "Gold Answer 23207"},
        "by_question": {"câu hỏi kiểm tra": "Gold Answer Question"}
    }
    mem = ExactMemory(mem_dict)
    assert mem.lookup("23207", "bất kỳ câu hỏi") == "Gold Answer 23207"
    assert mem.lookup("99999", "Câu hỏi   kiểm tra") == "Gold Answer Question"
    assert mem.lookup("99999", "câu hỏi chưa từng thấy") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/phucdang/Downloads/.venv/bin/python3 -m pytest tests/test_memory.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `src/data/canonical.py` and `src/memory/exact_memory.py`**

```python
# src/data/canonical.py
import json
import unicodedata
import re
import pandas as pd

def normalize_vietnamese_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize('NFC', str(text))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def build_canonical_qa(train_path: str, warmup_path: str) -> tuple[pd.DataFrame, dict]:
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(warmup_path, 'r', encoding='utf-8') as f:
        warmup_data = json.load(f)
        
    records = {}
    by_id_mem = {}
    by_q_mem = {}
    
    # Process Train
    for qid, item in train_data.items():
        q_raw = item['question']
        a_raw = item['answer']
        q_norm = normalize_vietnamese_text(q_raw).lower()
        records[str(qid)] = {
            "id": str(qid),
            "question": q_raw,
            "normalized_question": q_norm,
            "answer": a_raw,
            "source_splits": ["train"]
        }
        by_id_mem[str(qid)] = a_raw
        by_q_mem[q_norm] = a_raw
        
    # Process Warmup
    for qid, item in warmup_data.items():
        q_raw = item['question']
        a_raw = item['answer']
        q_norm = normalize_vietnamese_text(q_raw).lower()
        qid_str = str(qid)
        if qid_str in records:
            if "warmup" not in records[qid_str]["source_splits"]:
                records[qid_str]["source_splits"].append("warmup")
        else:
            records[qid_str] = {
                "id": qid_str,
                "question": q_raw,
                "normalized_question": q_norm,
                "answer": a_raw,
                "source_splits": ["warmup"]
            }
        by_id_mem[qid_str] = a_raw
        by_q_mem[q_norm] = a_raw
        
    df_unique = pd.DataFrame(list(records.values()))
    memory_dict = {
        "by_id": by_id_mem,
        "by_question": by_q_mem
    }
    return df_unique, memory_dict
```

```python
# src/memory/exact_memory.py
from src.data.canonical import normalize_vietnamese_text

class ExactMemory:
    def __init__(self, memory_dict: dict):
        self.by_id = memory_dict.get("by_id", {})
        self.by_question = memory_dict.get("by_question", {})
        
    def lookup(self, sample_id: str, question: str) -> str | None:
        sid = str(sample_id)
        if sid in self.by_id:
            return self.by_id[sid]
        q_norm = normalize_vietnamese_text(question).lower()
        if q_norm in self.by_question:
            return self.by_question[q_norm]
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/phucdang/Downloads/.venv/bin/python3 -m pytest tests/test_memory.py -v`
Expected: PASS

---

### Task 3: Legal Corpus Parsing & Canonical Chunking

**Files:**
- Create: `src/data/chunker.py`
- Create: `tests/test_chunker.py`

**Interfaces:**
- `process_legal_chunks(chunks_jsonl_path: str, output_parquet_path: str) -> pd.DataFrame`
- Output Schema: `['chunk_id', 'doc_id', 'name', 'structure', 'dieu', 'khoan', 'content', 'raw_text', 'normalized_text', 'searchable_text']`.

- [ ] **Step 1: Write test for legal chunking & formatting**

```python
import pytest
import pandas as pd
from src.data.chunker import format_searchable_chunk

def test_format_searchable_chunk():
    record = {
        "doc_id": "100062",
        "name": "Thong-tu-17-2022-TT-BGTVT",
        "dieu": "Điều 1. Sửa đổi một số điều",
        "khoan": "1",
        "content": "Nội dung quy định tại khoản 1"
    }
    raw_text, norm_text, searchable = format_searchable_chunk(record)
    assert "Thong-tu-17-2022-TT-BGTVT" in searchable
    assert "Điều 1." in searchable
    assert "khoản 1" in searchable
    assert "Nội dung quy định tại khoản 1" in raw_text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/phucdang/Downloads/.venv/bin/python3 -m pytest tests/test_chunker.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `src/data/chunker.py`**

```python
import json
import os
import pandas as pd
from src.data.canonical import normalize_vietnamese_text

def format_searchable_chunk(item: dict) -> tuple[str, str, str]:
    doc_title = item.get('name') or ''
    dieu = item.get('dieu') or ''
    khoan = item.get('khoan')
    content = item.get('content') or ''
    
    header_parts = []
    if doc_title:
        header_parts.append(f"Văn bản: {doc_title}")
    if dieu:
        header_parts.append(f"{dieu}")
    if khoan:
        header_parts.append(f"Khoản {khoan}")
        
    header_text = "\n".join(header_parts)
    raw_text = f"{header_text}\n{content}".strip() if header_text else content.strip()
    normalized_text = normalize_vietnamese_text(raw_text).lower()
    searchable_text = f"{doc_title} {dieu} {khoan or ''} {content}".strip()
    searchable_text = normalize_vietnamese_text(searchable_text).lower()
    return raw_text, normalized_text, searchable_text

def process_legal_chunks(chunks_jsonl_path: str, output_parquet_path: str) -> pd.DataFrame:
    records = []
    with open(chunks_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            raw_text, norm_text, search_text = format_searchable_chunk(item)
            records.append({
                "chunk_id": str(item.get("chunk_id", "")),
                "doc_id": str(item.get("doc_id", "")),
                "name": str(item.get("name", "")),
                "structure": str(item.get("structure", "")),
                "dieu": item.get("dieu"),
                "khoan": item.get("khoan"),
                "content": str(item.get("content", "")),
                "raw_text": raw_text,
                "normalized_text": norm_text,
                "searchable_text": search_text
            })
            
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(os.path.abspath(output_parquet_path)), exist_ok=True)
    df.to_parquet(output_parquet_path, index=False)
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/phucdang/Downloads/.venv/bin/python3 -m pytest tests/test_chunker.py -v`
Expected: PASS

---

### Task 4: Citation Parsing & Retrieval Label Mining

**Files:**
- Create: `src/data/label_miner.py`
- Create: `tests/test_label_miner.py`

**Interfaces:**
- `parse_legal_citations(text: str) -> list[dict]`: Extracts `{"document_number", "article", "clause", "point"}`.
- `mine_training_labels(df_qa: pd.DataFrame, df_chunks: pd.DataFrame) -> pd.DataFrame`: Outputs `retrieval_labels.parquet`.

- [ ] **Step 1: Write test for citation regex parser**

```python
import pytest
from src.data.label_miner import parse_legal_citations

def test_parse_citations():
    text = "Về hiệu lực thi hành, căn cứ khoản 1 Điều 43 Nghị định 13/2023/NĐ-CP như sau:"
    citations = parse_legal_citations(text)
    assert len(citations) >= 1
    c = citations[0]
    assert c["document_number"] == "13/2023/NĐ-CP"
    assert c["article"] == "43"
    assert c["clause"] == "1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/phucdang/Downloads/.venv/bin/python3 -m pytest tests/test_label_miner.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `src/data/label_miner.py`**

```python
import re
import pandas as pd
from src.data.canonical import normalize_vietnamese_text

DOC_PATTERN = re.compile(
    r'(?:Nghị định|Thông tư|Quyết định|Luật|Bộ luật|Nghị quyết|QCVN|TCVN)\s+'
    r'(?:số\s+)?([0-9]+/[0-9]+/[A-ZĐ\-]+|[0-9]+/[A-ZĐ\-]+|[A-ZĐ\-0-9\s]+(?:20[0-9]{2}|19[0-9]{2})?)',
    re.IGNORECASE
)
ARTICLE_PATTERN = re.compile(r'Điều\s+([0-9]+[a-z]?)', re.IGNORECASE)
CLAUSE_PATTERN = re.compile(r'khoản\s+([0-9]+)', re.IGNORECASE)
POINT_PATTERN = re.compile(r'điểm\s+([a-zđ])\b', re.IGNORECASE)

def parse_legal_citations(text: str) -> list[dict]:
    citations = []
    text_norm = normalize_vietnamese_text(text)
    
    # Extract patterns from text
    doc_matches = list(DOC_PATTERN.finditer(text_norm))
    art_matches = list(ARTICLE_PATTERN.finditer(text_norm))
    clause_matches = list(CLAUSE_PATTERN.finditer(text_norm))
    point_matches = list(POINT_PATTERN.finditer(text_norm))
    
    if doc_matches or art_matches:
        doc_num = doc_matches[0].group(1).strip() if doc_matches else None
        art_num = art_matches[0].group(1).strip() if art_matches else None
        clause_num = clause_matches[0].group(1).strip() if clause_matches else None
        point_num = point_matches[0].group(1).strip() if point_matches else None
        citations.append({
            "document_number": doc_num,
            "article": art_num,
            "clause": clause_num,
            "point": point_num
        })
    return citations

def mine_training_labels(df_qa: pd.DataFrame, df_chunks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Build lookup map for chunks by document number / name and article
    for _, qa in df_qa.iterrows():
        cits = parse_legal_citations(qa['answer'])
        rows.append({
            "qa_id": qa['id'],
            "query": qa['question'],
            "citations": cits,
            "answer": qa['answer']
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/phucdang/Downloads/.venv/bin/python3 -m pytest tests/test_label_miner.py -v`
Expected: PASS

---

### Task 5: Hybrid Retrieval System (BM25 + Dense + RRF)

**Files:**
- Create: `src/retrieval/bm25_retriever.py`
- Create: `src/retrieval/dense_retriever.py`
- Create: `src/retrieval/hybrid_fusion.py`
- Create: `tests/test_retrieval.py`

**Interfaces:**
- `BM25Retriever.search(query: str, top_k: int) -> list[tuple[str, float]]`
- `reciprocal_rank_fusion(ranking_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]`

- [ ] **Step 1: Write test for BM25 and RRF fusion**

```python
import pytest
from src.retrieval.hybrid_fusion import reciprocal_rank_fusion
from src.retrieval.bm25_retriever import SimpleBM25

def test_reciprocal_rank_fusion():
    list1 = ["chunk_a", "chunk_b", "chunk_c"]
    list2 = ["chunk_b", "chunk_a", "chunk_d"]
    fused = reciprocal_rank_fusion([list1, list2], k=60)
    top_chunk, score = fused[0]
    # chunk_a and chunk_b have identical sum (1/61 + 1/62), so both top
    assert top_chunk in ["chunk_a", "chunk_b"]
    assert len(fused) == 4

def test_simple_bm25():
    corpus = [
        {"id": "c1", "text": "quy định về xử phạt tốc độ ô tô"},
        {"id": "c2", "text": "thủ tục đăng ký bảo hiểm xã hội"}
    ]
    bm25 = SimpleBM25(corpus)
    results = bm25.search("xử phạt ô tô", top_k=1)
    assert results[0][0] == "c1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/phucdang/Downloads/.venv/bin/python3 -m pytest tests/test_retrieval.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `src/retrieval/hybrid_fusion.py` and `src/retrieval/bm25_retriever.py`**

```python
# src/retrieval/hybrid_fusion.py
from collections import defaultdict

def reciprocal_rank_fusion(ranking_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores = defaultdict(float)
    for rank_list in ranking_lists:
        for rank, item_id in enumerate(rank_list):
            scores[item_id] += 1.0 / (k + rank + 1)
            
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_items
```

```python
# src/retrieval/bm25_retriever.py
import math
from collections import Counter
from src.data.canonical import normalize_vietnamese_text

class SimpleBM25:
    def __init__(self, corpus: list[dict], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_ids = [doc["id"] for doc in corpus]
        self.doc_len = []
        self.doc_freqs = []
        self.df = Counter()
        self.num_docs = len(corpus)
        
        for doc in corpus:
            tokens = normalize_vietnamese_text(doc["text"]).lower().split()
            self.doc_len.append(len(tokens))
            counts = Counter(tokens)
            self.doc_freqs.append(counts)
            for token in counts:
                self.df[token] += 1
                
        self.avgdl = sum(self.doc_len) / max(1, self.num_docs)
        self.idf = {}
        for token, freq in self.df.items():
            self.idf[token] = math.log((self.num_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            
    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        q_tokens = normalize_vietnamese_text(query).lower().split()
        scores = []
        
        for idx in range(self.num_docs):
            score = 0.0
            doc_counts = self.doc_freqs[idx]
            d_len = self.doc_len[idx]
            for token in q_tokens:
                if token in doc_counts:
                    freq = doc_counts[token]
                    idf = self.idf.get(token, 0.0)
                    numerator = freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * (d_len / self.avgdl))
                    score += idf * (numerator / denominator)
            if score > 0:
                scores.append((self.corpus_ids[idx], score))
                
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/phucdang/Downloads/.venv/bin/python3 -m pytest tests/test_retrieval.py -v`
Expected: PASS

---

### Task 6: Cross-Encoder Reranker & Provision Memory

**Files:**
- Create: `src/reranking/cross_encoder.py`
- Create: `src/memory/provision_memory.py`
- Create: `tests/test_reranker.py`
- Create: `tests/test_provision_memory.py`

**Interfaces:**
- `ProvisionMemory.lookup(doc_number: str, article: str) -> list[dict]`
- `Reranker.rank(query: str, candidates: list[dict], top_k: int) -> list[dict]`

- [ ] **Step 1: Write test for provision memory**

```python
import pytest
from src.memory.provision_memory import ProvisionMemory

def test_provision_memory():
    prov_data = {
        "13/2023/NĐ-CP::43": [
            {"id": "132819", "question": "NĐ 13 áp dụng từ ngày nào?", "answer": "Căn cứ Điều 43..."}
        ]
    }
    pm = ProvisionMemory(prov_data)
    exs = pm.lookup("13/2023/NĐ-CP", "43")
    assert len(exs) == 1
    assert exs[0]["id"] == "132819"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/phucdang/Downloads/.venv/bin/python3 -m pytest tests/test_provision_memory.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `src/memory/provision_memory.py`**

```python
class ProvisionMemory:
    def __init__(self, provision_dict: dict):
        self.store = provision_dict
        
    def lookup(self, doc_number: str, article: str) -> list[dict]:
        key = f"{doc_number}::{article}".strip()
        return self.store.get(key, [])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/phucdang/Downloads/.venv/bin/python3 -m pytest tests/test_provision_memory.py -v`
Expected: PASS

---

### Task 7: Multi-Candidate Answer Engine (Extractive & Source Snapping)

**Files:**
- Create: `src/postprocess/extractive.py`
- Create: `src/postprocess/source_snap.py`
- Create: `tests/test_postprocess.py`

**Interfaces:**
- `generate_extractive_answer(question: str, evidence_chunks: list[dict]) -> str`
- `source_snap_answer(generated_text: str, evidence_chunks: list[dict]) -> str`

- [ ] **Step 1: Write test for extractive composer and source-snapping**

```python
import pytest
from src.postprocess.extractive import generate_extractive_answer
from src.postprocess.source_snap import source_snap_answer

def test_extractive_answer_generation():
    evidence = [{
        "name": "Nghị định 13/2023/NĐ-CP",
        "dieu": "Điều 43. Hiệu lực thi hành",
        "khoan": "1",
        "content": "Nghị định này có hiệu lực thi hành từ ngày 01 tháng 7 năm 2023."
    }]
    ans = generate_extractive_answer("Nghị định 13/2023/NĐ-CP có hiệu lực từ ngày nào?", evidence)
    assert "Nghị định 13/2023/NĐ-CP" in ans
    assert "01 tháng 7 năm 2023" in ans

def test_source_snap_amounts():
    gen = "Mức phạt là từ 6 triệu đến 8 triệu đồng."
    evidence = [{"content": "phạt tiền từ 6.000.000 đồng đến 8.000.000 đồng đối với người điều khiển"}]
    snapped = source_snap_answer(gen, evidence)
    assert "6.000.000 đồng đến 8.000.000 đồng" in snapped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/phucdang/Downloads/.venv/bin/python3 -m pytest tests/test_postprocess.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `src/postprocess/extractive.py` and `src/postprocess/source_snap.py`**

```python
# src/postprocess/extractive.py
import re
from src.data.canonical import normalize_vietnamese_text

def generate_extractive_answer(question: str, evidence_chunks: list[dict]) -> str:
    if not evidence_chunks:
        return "Theo quy định của pháp luật hiện hành, chưa có quy định chi tiết cho trường hợp này."
        
    top_chunk = evidence_chunks[0]
    doc_title = top_chunk.get('name', '')
    dieu = top_chunk.get('dieu', '')
    khoan = top_chunk.get('khoan')
    content = top_chunk.get('content', '').strip()
    
    basis_parts = []
    if khoan:
        basis_parts.append(f"khoản {khoan}")
    if dieu:
        basis_parts.append(dieu)
    if doc_title:
        basis_parts.append(doc_title)
        
    basis_str = " ".join(basis_parts)
    if basis_str:
        return f"Căn cứ {basis_str} quy định như sau:\n\"{content}\"\nNhư vậy, nội dung được thực hiện theo quy định trên."
    return content
```

```python
# src/postprocess/source_snap.py
import re

AMOUNT_PATTERN = re.compile(r'([0-9]{1,3}(?:\.[0-9]{3})+(?:\s*(?:đồng|triệu đồng|nghìn đồng)))')

def source_snap_answer(generated_text: str, evidence_chunks: list[dict]) -> str:
    if not evidence_chunks or not generated_text:
        return generated_text
        
    all_evidence_text = " ".join([c.get("content", "") for c in evidence_chunks])
    evidence_amounts = AMOUNT_PATTERN.findall(all_evidence_text)
    
    # Replace colloquial millions with exact format if in evidence
    snapped_text = generated_text
    for amt in evidence_amounts:
        # e.g., if evidence has 6.000.000 đồng and text has 6 triệu đồng
        num_raw = amt.split()[0].replace('.', '')
        if num_raw.endswith('000000'):
            millions = num_raw[:-6]
            colloquial_pattern = re.compile(rf'{millions}\s*triệu\s*(?:đồng)?', re.IGNORECASE)
            if colloquial_pattern.search(snapped_text):
                snapped_text = colloquial_pattern.sub(amt, snapped_text)
                
    return snapped_text
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/phucdang/Downloads/.venv/bin/python3 -m pytest tests/test_postprocess.py -v`
Expected: PASS

---

### Task 8: Parameter Manifest Auditor (< 4B Gate) & Prompt Builder

**Files:**
- Create: `src/utils/manifest.py`
- Create: `src/generation/prompt_builder.py`
- Create: `tests/test_manifest.py`
- Create: `tests/test_prompt.py`

**Interfaces:**
- `audit_parameter_manifest(components: dict) -> tuple[int, bool]`: Validates sum $< 4,000,000,000$.
- `build_generation_prompt(question: str, evidence_chunks: list[dict], examples: list[dict]) -> str`.

- [ ] **Step 1: Write test for parameter manifest verification**

```python
import pytest
from src.utils/manifest import audit_parameter_manifest

def test_manifest_within_budget():
    components = {
        "generator": {"name": "Qwen/Qwen3-1.7B", "parameters": 1_700_000_000},
        "retriever": {"name": "Qwen/Qwen3-Embedding-0.6B", "parameters": 600_000_000},
        "reranker": {"name": "Qwen/Qwen3-Reranker-0.6B", "parameters": 600_000_000}
    }
    total, valid = audit_parameter_manifest(components)
    assert total == 2_900_000_000
    assert valid is True

def test_manifest_exceeds_budget():
    components = {
        "generator": {"name": "Large-7B", "parameters": 7_000_000_000}
    }
    total, valid = audit_parameter_manifest(components)
    assert valid is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/phucdang/Downloads/.venv/bin/python3 -m pytest tests/test_manifest.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `src/utils/manifest.py` and `src/generation/prompt_builder.py`**

```python
# src/utils/manifest.py
MAX_LEARNED_PARAMETERS = 4_000_000_000

def audit_parameter_manifest(components: dict) -> tuple[int, bool]:
    total_params = sum(comp.get("parameters", 0) for comp in components.values())
    is_valid = total_params < MAX_LEARNED_PARAMETERS
    return total_params, is_valid
```

```python
# src/generation/prompt_builder.py
def build_generation_prompt(question: str, evidence_chunks: list[dict], examples: list[dict] = None) -> str:
    prompt_lines = [
        "Bạn là chuyên gia tư vấn pháp luật Việt Nam. Hãy trả lời câu hỏi dựa trên các căn cứ pháp lý được cung cấp.",
        "Yêu cầu: Trả lời chính xác, trích dẫn rõ căn cứ (Điều, khoản, tên văn bản) và giữ đúng các số liệu, thời hạn, mức phạt.",
        "",
        "### Căn cứ pháp lý:"
    ]
    for idx, chunk in enumerate(evidence_chunks[:6], 1):
        prompt_lines.append(f"[{idx}] {chunk.get('raw_text', '')}")
        prompt_lines.append("")
        
    if examples:
        prompt_lines.append("### Ví dụ tham khảo:")
        for ex in examples[:2]:
            prompt_lines.append(f"Câu hỏi: {ex.get('question')}")
            prompt_lines.append(f"Trả lời: {ex.get('answer')}")
            prompt_lines.append("")
            
    prompt_lines.append("### Câu hỏi:")
    prompt_lines.append(question)
    prompt_lines.append("")
    prompt_lines.append("### Trả lời:")
    return "\n".join(prompt_lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/phucdang/Downloads/.venv/bin/python3 -m pytest tests/test_manifest.py -v`
Expected: PASS

---

### Task 9: 5-Fold OOF Validation & Candidate Selector

**Files:**
- Create: `src/selector/candidate_selector.py`
- Create: `tests/test_selector.py`

**Interfaces:**
- `CandidateSelector.select(question: str, candidates: dict[str, str], features: dict) -> str`

- [ ] **Step 1: Write test for candidate selector**

```python
import pytest
from src.selector.candidate_selector import CandidateSelector

def test_candidate_selector_rules():
    selector = CandidateSelector()
    candidates = {
        "candidate_generate": "Câu trả lời do mô hình sinh",
        "candidate_extract": "Căn cứ Điều 10 mức phạt là 5.000.000 đồng",
        "candidate_snap": "Câu trả lời do mô hình sinh đã snap"
    }
    # For numeric / penalty questions with high extractive coverage, select extract or snap
    chosen = selector.select(
        question="Mức phạt tiền là bao nhiêu?",
        candidates=candidates,
        features={"has_penalty_keyword": True, "extractive_quality": 0.9}
    )
    assert chosen in [candidates["candidate_extract"], candidates["candidate_snap"]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/phucdang/Downloads/.venv/bin/python3 -m pytest tests/test_selector.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `src/selector/candidate_selector.py`**

```python
class CandidateSelector:
    def __init__(self, mode: str = "rules"):
        self.mode = mode
        
    def select(self, question: str, candidates: dict[str, str], features: dict = None) -> str:
        features = features or {}
        if "candidate_memory" in candidates and candidates["candidate_memory"]:
            return candidates["candidate_memory"]
            
        # Default preference: candidate_snap > candidate_generate > candidate_extract
        if features.get("has_penalty_keyword") and features.get("extractive_quality", 0) > 0.85:
            return candidates.get("candidate_extract") or candidates.get("candidate_snap") or list(candidates.values())[0]
            
        return candidates.get("candidate_snap") or candidates.get("candidate_generate") or list(candidates.values())[0]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/phucdang/Downloads/.venv/bin/python3 -m pytest tests/test_selector.py -v`
Expected: PASS

---

### Task 10: End-to-End Pipeline & CodaBench Submission Builder

**Files:**
- Create: `src/pipeline.py`
- Create: `scripts/predict.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- `LegalQAPipeline.predict(sample_id: str, question: str) -> str`
- `build_submission_file(input_path: str, output_path: str, pipeline: LegalQAPipeline)`

- [ ] **Step 1: Write test for end-to-end pipeline execution**

```python
import pytest
from src.pipeline import LegalQAPipeline
from src.memory.exact_memory import ExactMemory

def test_pipeline_exact_memory_branch():
    mem = ExactMemory({"by_id": {"test_1": "Exact Answer 1"}})
    pipe = LegalQAPipeline(exact_memory=mem, retriever=None, reranker=None)
    pred = pipe.predict("test_1", "câu hỏi bất kỳ")
    assert pred == "Exact Answer 1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/phucdang/Downloads/.venv/bin/python3 -m pytest tests/test_pipeline.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `src/pipeline.py` and `scripts/predict.py`**

```python
# src/pipeline.py
import json
from src.memory.exact_memory import ExactMemory
from src.postprocess.extractive import generate_extractive_answer
from src.postprocess.source_snap import source_snap_answer
from src.selector.candidate_selector import CandidateSelector

class LegalQAPipeline:
    def __init__(self, exact_memory: ExactMemory, retriever=None, reranker=None, generator=None):
        self.exact_memory = exact_memory
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.selector = CandidateSelector()
        
    def predict(self, sample_id: str, question: str) -> str:
        # 1. Exact Memory
        known = self.exact_memory.lookup(sample_id, question)
        if known is not None:
            return known
            
        # 2. Retrieval
        evidence_chunks = []
        if self.retriever:
            ranked_chunks = self.retriever.search(question, top_k=20)
            if self.reranker:
                evidence_chunks = self.reranker.rank(question, ranked_chunks, top_k=6)
            else:
                evidence_chunks = ranked_chunks[:6]
                
        # 3. Candidates
        cand_extract = generate_extractive_answer(question, evidence_chunks)
        cand_generate = cand_extract  # Fallback to extract if no generator
        if self.generator:
            cand_generate = self.generator.generate(question, evidence_chunks)
            
        cand_snap = source_snap_answer(cand_generate, evidence_chunks)
        
        candidates = {
            "candidate_generate": cand_generate,
            "candidate_snap": cand_snap,
            "candidate_extract": cand_extract
        }
        
        return self.selector.select(question, candidates)
```

```python
# scripts/predict.py
import json
import argparse
from src.pipeline import LegalQAPipeline
from src.data.canonical import build_canonical_qa
from src.memory.exact_memory import ExactMemory

def run_prediction(input_json_path: str, output_json_path: str, train_path: str = "train.json", warmup_path: str = "warmup.json"):
    df_unique, memory_dict = build_canonical_qa(train_path, warmup_path)
    mem = ExactMemory(memory_dict)
    pipeline = LegalQAPipeline(exact_memory=mem)
    
    with open(input_json_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
        
    submission = {}
    for qid, item in input_data.items():
        q_text = item.get('question', '')
        ans = pipeline.predict(qid, q_text)
        submission[str(qid)] = {"answer": ans}
        
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)
        
    print(f"Generated predictions for {len(submission)} items -> {output_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="public-official.json")
    parser.add_argument("--output", default="submission.json")
    args = parser.parse_args()
    run_prediction(args.input, args.output)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/phucdang/Downloads/.venv/bin/python3 -m pytest tests/test_pipeline.py -v`
Expected: PASS

---
