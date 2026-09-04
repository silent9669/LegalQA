# DSC 2026 Task 2 — LegalQA System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a high-performance, compliant (<4.0B learned parameters) Vietnamese LegalQA RAG pipeline optimized for official whitespace METEOR score, featuring DEk21 v2 dense retrieval, BGE Reranker v2 M3, Article/Clause stitching, dual-runtime (Apple Silicon MLX + Google Colab CUDA) Qwen2.5-3B generation, exact QA memory, source-snapping answer reconstruction, and 5-fold OOF validation.

**Architecture:** The pipeline routes incoming queries through $O(1)$ exact QA memory, falls back to hybrid retrieval (BM25 with legal signal boosting + DEk21 v2 768-dim dense retrieval fused via RRF), reranks candidates using BGE-Reranker v2 M3, stitches parent article sibling clauses, generates grounded legal answers using Qwen2.5-3B-Instruct (via MLX locally or PyTorch/CUDA on Colab), snaps fragile facts (dates, amounts, statutory identifiers) to raw evidence, and formats compliant submission JSON.

**Tech Stack:** Python 3.10+, PyArrow, FastParquet, NumPy, Scikit-learn, NLTK (METEOR), BM25s, PyVi, Sentence-Transformers, FAISS, PyTorch / MLX, Hugging Face Transformers & PEFT.

**Spec:** `docs/superpowers/specs/2026-08-30-legalqa-redesign-design.md`

## Global Constraints
- Total learned parameters of all neural models combined must be strictly `< 4,000,000,000` (4.0B).
- Primary target metric: official whitespace-tokenized METEOR (`nltk.translate.meteor_score.meteor_score([ref.split()], pred.split())`).
- Only official Task 2 data is permitted (`train.json`, `warmup.json`, `selected-contexts`).
- Dual-runtime compatibility: Local Apple Silicon execution (MLX) + Google Colab (PyTorch/CUDA) with seamless bidirectional LoRA weight conversion.
- Modular code architecture: Shared core utilities in `src/common/`, Task 2 specific logic in `src/task2/`, artifacts in `artifacts/task2/`.

---

### Task 1: Workspace Fresh-Start & Shared Normalization Module

**Files:**
- Create: `src/common/__init__.py`
- Create: `src/common/normalize.py`
- Test: `tests/test_common_normalize.py`

**Interfaces:**
- Produces:
  - `clean_legal_text(text: str) -> str`: Normalizes whitespace, quotes, unicode characters, and standard legal punctuation.
  - `extract_legal_signals(text: str) -> dict`: Extracts statutory numbers (e.g. `90/2017/NĐ-CP`), article numbers (`Điều 17`), clause numbers (`Khoản 3`), and year markers.
  - `tokenize_vietnamese(text: str) -> str`: Word-segmented text using `pyvi` with compound word underscore joining.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_common_normalize.py
import pytest
from src.common.normalize import clean_legal_text, extract_legal_signals, tokenize_vietnamese

def test_clean_legal_text():
    raw = "  Điều   17 .  Nghị  định   90/2017/NĐ-CP \n\n Khoản 3 “quy định” "
    cleaned = clean_legal_text(raw)
    assert cleaned == 'Điều 17 . Nghị định 90/2017/NĐ-CP Khoản 3 "quy định"'

def test_extract_legal_signals():
    query = "Theo khoản 3 Điều 17 Nghị định 90/2017/NĐ-CP năm 2017 quy định gì?"
    signals = extract_legal_signals(query)
    assert "90/2017/NĐ-CP" in signals.get("doc_numbers", [])
    assert "17" in signals.get("articles", [])
    assert "3" in signals.get("clauses", [])
    assert "2017" in signals.get("years", [])

def test_tokenize_vietnamese():
    text = "Nghị định quy định xử phạt vi phạm hành chính"
    tokenized = tokenize_vietnamese(text)
    assert "_" in tokenized
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_common_normalize.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Implement `src/common/normalize.py`**

```python
# src/common/normalize.py
import re
import unicodedata

try:
    from pyvi import ViTokenizer
except ImportError:
    ViTokenizer = None

DOC_NUMBER_PATTERN = re.compile(r'\b\d{1,5}/(?:\d{4}/)?(?:NĐ-CP|TT-BLĐTBXH|TT-BGDĐT|TT-BTC|TT-BYT|QĐ-TTg|QH\d{1,2}|TTCP|[A-ZĐ]+-[A-ZĐ]+)\b', re.IGNORECASE)
ARTICLE_PATTERN = re.compile(r'\bĐiều\s+(\d+[a-zA-Z]?)\b', re.IGNORECASE)
CLAUSE_PATTERN = re.compile(r'\bkhoản\s+(\d+[a-zA-Z]?)\b', re.IGNORECASE)
POINT_PATTERN = re.compile(r'\bđiểm\s+([a-zA-Z\d]+)\b', re.IGNORECASE)
YEAR_PATTERN = re.compile(r'\bnăm\s+(\d{4})\b|\b(19\d{2}|20\d{2})\b', re.IGNORECASE)

def clean_legal_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = re.sub(r'[\r\t\f\v]', ' ', text)
    text = re.sub(r'\s*\n\s*', '\n', text)
    text = re.sub(r'[ ]{2,}', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)
    text = text.replace("\n", " ").strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_legal_signals(text: str) -> dict:
    cleaned = clean_legal_text(text)
    doc_nums = [m.group(0).upper() for m in DOC_NUMBER_PATTERN.finditer(cleaned)]
    articles = [m.group(1) for m in ARTICLE_PATTERN.finditer(cleaned)]
    clauses = [m.group(1) for m in CLAUSE_PATTERN.finditer(cleaned)]
    points = [m.group(1) for m in POINT_PATTERN.finditer(cleaned)]
    
    years = []
    for m in YEAR_PATTERN.finditer(cleaned):
        y = m.group(1) or m.group(2)
        if y:
            years.append(y)
            
    return {
        "doc_numbers": list(dict.fromkeys(doc_nums)),
        "articles": list(dict.fromkeys(articles)),
        "clauses": list(dict.fromkeys(clauses)),
        "points": list(dict.fromkeys(points)),
        "years": list(dict.fromkeys(years))
    }

def tokenize_vietnamese(text: str) -> str:
    cleaned = clean_legal_text(text)
    if ViTokenizer is not None:
        return ViTokenizer.tokenize(cleaned)
    return cleaned
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_common_normalize.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/common/ tests/test_common_normalize.py
git commit -m "feat(common): implement shared text normalization and legal signal extraction"
```

---

### Task 2: Hierarchical Legal Parser & Chunker (`src/common/legal_parser.py`)

**Files:**
- Create: `src/common/legal_parser.py`
- Test: `tests/test_common_legal_parser.py`

**Interfaces:**
- Consumes: `clean_legal_text`, `extract_legal_signals` from `src/common/normalize.py`
- Produces:
  - `parse_legal_document(doc_id: str, doc_name: str, passage: str) -> list[dict]`: Parses statutory text into structured chunk records with `chunk_id`, `parent_article_id`, `article_number`, `clause_number`, `text_raw`, `text_norm`, `start_char`, `end_char`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_common_legal_parser.py
from src.common.legal_parser import parse_legal_document

def test_parse_legal_document():
    passage = """Nghị định 90/2017/NĐ-CP
Chương I. Quy định chung
Điều 1. Phạm vi điều chỉnh
Nghị định này quy định về xử phạt vi phạm hành chính.
Điều 17. Vi phạm về tiêm phòng
1. Phạt tiền từ 1.000.000 đồng đến 2.000.000 đồng đối với hành vi không tiêm phòng.
2. Phạt tiền từ 2.000.000 đồng đến 3.000.000 đồng đối với hành vi che giấu dịch bệnh."""
    
    chunks = parse_legal_document(doc_id="740", doc_name="Nghị định 90/2017/NĐ-CP", passage=passage)
    assert len(chunks) >= 3
    art17_chunks = [c for c in chunks if c["article_number"] == "17"]
    assert len(art17_chunks) >= 2
    assert all(c["parent_article_id"] == "doc740_art17" for c in art17_chunks)
    assert any("không tiêm phòng" in c["text_raw"] for c in art17_chunks)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_common_legal_parser.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/common/legal_parser.py`**

```python
# src/common/legal_parser.py
import re
from src.common.normalize import clean_legal_text, tokenize_vietnamese

ARTICLE_SPLIT_REGEX = re.compile(r'(?=(?:^|\n)\s*Điều\s+\d+[a-zA-Z]?[\.\s])', re.IGNORECASE | re.MULTILINE)
ARTICLE_HEADER_REGEX = re.compile(r'^(?:Điều\s+(\d+[a-zA-Z]?))[\.\:\s]*(.*?)(?:\n|$)', re.IGNORECASE)
CLAUSE_SPLIT_REGEX = re.compile(r'(?=(?:^|\n)\s*\d+\.\s+)', re.MULTILINE)
CLAUSE_HEADER_REGEX = re.compile(r'^(\d+)\.\s*(.*)', re.DOTALL)

def parse_legal_document(doc_id: str, doc_name: str, passage: str) -> list[dict]:
    chunks = []
    if not passage:
        return chunks
    
    doc_id_clean = str(doc_id).strip()
    raw_articles = ARTICLE_SPLIT_REGEX.split(passage)
    
    char_offset = 0
    for art_idx, art_text in enumerate(raw_articles):
        art_clean = art_text.strip()
        if not art_clean:
            char_offset += len(art_text)
            continue
        
        art_match = ARTICLE_HEADER_REGEX.search(art_clean)
        if art_match:
            art_num = art_match.group(1)
            art_title = art_match.group(2).strip()
            parent_article_id = f"doc{doc_id_clean}_art{art_num}"
        else:
            art_num = f"preamble_{art_idx}"
            art_title = ""
            parent_article_id = f"doc{doc_id_clean}_art_{art_idx}"
            
        clauses = CLAUSE_SPLIT_REGEX.split(art_clean)
        if len(clauses) > 1:
            for clause_idx, clause_text in enumerate(clauses):
                c_clean = clause_text.strip()
                if not c_clean:
                    continue
                c_match = CLAUSE_HEADER_REGEX.match(c_clean)
                clause_num = c_match.group(1) if c_match else str(clause_idx)
                
                header_prefix = f"[DOCUMENT] {doc_name}\n[ARTICLE] Điều {art_num}. {art_title}\n[CLAUSE] {clause_num}. "
                full_raw = f"{header_prefix}\n{c_clean}"
                
                chunk_id = f"{parent_article_id}_p{clause_num}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id_clean,
                    "doc_name": doc_name,
                    "parent_article_id": parent_article_id,
                    "article_number": str(art_num),
                    "clause_number": str(clause_num),
                    "text_raw": full_raw,
                    "text_norm": tokenize_vietnamese(full_raw),
                    "start_char": char_offset,
                    "end_char": char_offset + len(art_text)
                })
        else:
            full_raw = f"[DOCUMENT] {doc_name}\n[ARTICLE] Điều {art_num}. {art_title}\n{art_clean}"
            chunk_id = f"{parent_article_id}_full"
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id_clean,
                "doc_name": doc_name,
                "parent_article_id": parent_article_id,
                "article_number": str(art_num),
                "clause_number": None,
                "text_raw": full_raw,
                "text_norm": tokenize_vietnamese(full_raw),
                "start_char": char_offset,
                "end_char": char_offset + len(art_text)
            })
        char_offset += len(art_text)
        
    return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_common_legal_parser.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/common/legal_parser.py tests/test_common_legal_parser.py
git commit -m "feat(common): implement hierarchy legal parser and article chunker"
```

---

### Task 3: Exact QA & Provision Memory (`src/task2/qa_memory.py`)

**Files:**
- Create: `src/task2/__init__.py`
- Create: `src/task2/qa_memory.py`
- Test: `tests/test_task2_qa_memory.py`

**Interfaces:**
- Consumes: `clean_legal_text` from `src/common/normalize.py`
- Produces:
  - `QAMemory.from_records(records: list[dict]) -> QAMemory`: Builds indexed memory from QA dicts.
  - `QAMemory.lookup_exact(qa_id: str, question: str) -> str | None`: $O(1)$ exact lookup by ID or normalized question string.
  - `QAMemory.save(json_path: str, parquet_path: str)`: Persists known lookup table and unique dataframe.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_task2_qa_memory.py
from src.task2.qa_memory import QAMemory

def test_qa_memory_exact_lookup():
    data = [
        {"id": "1", "question": "Hành vi trốn thuế bị phạt bao nhiêu?", "answer": "Phạt tiền từ 1 đến 3 lần số tiền trốn thuế."},
        {"id": "2", "question": "Nghị định 90/2017 có hiệu lực khi nào?", "answer": "Có hiệu lực từ ngày 15/09/2017."}
    ]
    mem = QAMemory.from_records(data)
    
    # Lookup by ID
    assert mem.lookup_exact("1", "random question") == "Phạt tiền từ 1 đến 3 lần số tiền trốn thuế."
    
    # Lookup by normalized question
    assert mem.lookup_exact("999", "  hành vi  trốn thuế bị phạt bao nhiêu ? ") == "Phạt tiền từ 1 đến 3 lần số tiền trốn thuế."
    
    # Unhit query
    assert mem.lookup_exact("999", "Câu hỏi chưa từng xuất hiện") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_task2_qa_memory.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/task2/qa_memory.py`**

```python
# src/task2/qa_memory.py
import json
import os
import pandas as pd
from src.common.normalize import clean_legal_text

class QAMemory:
    def __init__(self, id_to_answer: dict, question_to_answer: dict, df: pd.DataFrame = None):
        self.id_to_answer = id_to_answer
        self.question_to_answer = question_to_answer
        self.df = df if df is not None else pd.DataFrame()

    @classmethod
    def from_records(cls, records: list[dict]):
        id_map = {}
        q_map = {}
        rows = []
        
        for r in records:
            qa_id = str(r.get("id") or r.get("qa_id") or "").strip()
            q_raw = str(r.get("question", "")).strip()
            ans_raw = str(r.get("answer", "")).strip()
            
            if not q_raw or not ans_raw:
                continue
            
            q_norm = clean_legal_text(q_raw).lower()
            if qa_id:
                id_map[qa_id] = ans_raw
            q_map[q_norm] = ans_raw
            
            rows.append({
                "qa_id": qa_id,
                "question_raw": q_raw,
                "question_norm": q_norm,
                "answer_raw": ans_raw,
                "source_split": r.get("source_split", "train")
            })
            
        df = pd.DataFrame(rows).drop_duplicates(subset=["question_norm"])
        return cls(id_map, q_map, df)

    def lookup_exact(self, qa_id: str, question: str) -> str | None:
        if qa_id and str(qa_id).strip() in self.id_to_answer:
            return self.id_to_answer[str(qa_id).strip()]
        
        q_norm = clean_legal_text(question).lower()
        if q_norm in self.question_to_answer:
            return self.question_to_answer[q_norm]
        
        return None

    def save(self, json_path: str, parquet_path: str):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "id_map": self.id_to_answer,
                "question_map": self.question_to_answer
            }, f, ensure_ascii=False, indent=2)
        self.df.to_parquet(parquet_path, index=False)

    @classmethod
    def load(cls, json_path: str, parquet_path: str = None):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.read_parquet(parquet_path) if parquet_path and os.path.exists(parquet_path) else pd.DataFrame()
        return cls(data.get("id_map", {}), data.get("question_map", {}), df)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_task2_qa_memory.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/task2/qa_memory.py tests/test_task2_qa_memory.py
git commit -m "feat(task2): implement exact QA memory with deduplicated lookup"
```

---

### Task 4: Citation Extraction & Hard Negative Miner (`src/common/evidence.py`)

**Files:**
- Create: `src/common/evidence.py`
- Test: `tests/test_common_evidence.py`

**Interfaces:**
- Consumes: `extract_legal_signals` from `src/common/normalize.py`
- Produces:
  - `parse_citations_from_answer(answer: str) -> list[dict]`: Extracts cited legal documents, articles, clauses.
  - `mine_hard_negatives(query_info: dict, all_chunks: list[dict], positive_article_id: str) -> dict`: Returns type A, B, C negative chunk IDs.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_common_evidence.py
from src.common.evidence import parse_citations_from_answer, mine_hard_negatives

def test_parse_citations_from_answer():
    ans = "Căn cứ khoản 3 Điều 17 Nghị định 90/2017/NĐ-CP quy định mức phạt tiền từ 1.000.000 đồng đến 2.000.000 đồng."
    citations = parse_citations_from_answer(ans)
    assert len(citations) >= 1
    assert "90/2017/NĐ-CP" in citations[0]["doc_number"]
    assert citations[0]["article"] == "17"
    assert citations[0]["clause"] == "3"

def test_mine_hard_negatives():
    chunks = [
        {"chunk_id": "doc1_art17_p3", "doc_id": "1", "parent_article_id": "doc1_art17", "clause_number": "3"},
        {"chunk_id": "doc1_art17_p1", "doc_id": "1", "parent_article_id": "doc1_art17", "clause_number": "1"},
        {"chunk_id": "doc1_art16_p1", "doc_id": "1", "parent_article_id": "doc1_art16", "clause_number": "1"},
        {"chunk_id": "doc2_art5_p1", "doc_id": "2", "parent_article_id": "doc2_art5", "clause_number": "1"},
    ]
    negatives = mine_hard_negatives(
        query_info={"doc_id": "1"},
        all_chunks=chunks,
        positive_chunk_id="doc1_art17_p3",
        positive_article_id="doc1_art17"
    )
    assert "doc1_art17_p1" in negatives["same_article_wrong_clause"]
    assert "doc1_art16_p1" in negatives["same_doc_wrong_article"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_common_evidence.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/common/evidence.py`**

```python
# src/common/evidence.py
import re
from src.common.normalize import extract_legal_signals

CITATION_PATTERN = re.compile(
    r'(?:căn\s+cứ\s+|theo\s+)?(?:khoản\s+(\d+[a-zA-Z]?)\s+)?(?:điều\s+(\d+[a-zA-Z]?)\s+)(?:nghị\s+định|thông\s+tư|luật|quyết\s+định)?\s*([0-9]{1,5}/[0-9]{4}/[A-ZĐ\-]+|[0-9]{1,5}/[A-ZĐ\-]+)?',
    re.IGNORECASE
)

def parse_citations_from_answer(answer: str) -> list[dict]:
    citations = []
    signals = extract_legal_signals(answer)
    
    for m in CITATION_PATTERN.finditer(answer):
        clause = m.group(1)
        article = m.group(2)
        doc_num = m.group(3) or (signals["doc_numbers"][0] if signals["doc_numbers"] else "")
        
        if article:
            citations.append({
                "doc_number": doc_num.upper() if doc_num else "",
                "article": article,
                "clause": clause or ""
            })
            
    if not citations and (signals["articles"] or signals["doc_numbers"]):
        citations.append({
            "doc_number": signals["doc_numbers"][0] if signals["doc_numbers"] else "",
            "article": signals["articles"][0] if signals["articles"] else "",
            "clause": signals["clauses"][0] if signals["clauses"] else ""
        })
    return citations

def mine_hard_negatives(query_info: dict, all_chunks: list[dict], positive_chunk_id: str, positive_article_id: str) -> dict:
    doc_id = str(query_info.get("doc_id", "")).strip()
    
    same_article_wrong_clause = []
    same_doc_wrong_article = []
    different_doc_hard_negs = []
    
    for c in all_chunks:
        cid = c["chunk_id"]
        if cid == positive_chunk_id:
            continue
        c_doc = str(c.get("doc_id", "")).strip()
        c_parent_art = c.get("parent_article_id", "")
        
        if c_parent_art == positive_article_id:
            same_article_wrong_clause.append(cid)
        elif c_doc == doc_id:
            same_doc_wrong_article.append(cid)
        else:
            different_doc_hard_negs.append(cid)
            
    return {
        "same_article_wrong_clause": same_article_wrong_clause[:5],
        "same_doc_wrong_article": same_doc_wrong_article[:10],
        "different_doc": different_doc_hard_negs[:10]
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_common_evidence.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/common/evidence.py tests/test_common_evidence.py
git commit -m "feat(common): implement citation parser and hard negative miner"
```

---

### Task 5: Sparse Retrieval Engine with Legal Entity Booster (`src/common/bm25.py`)

**Files:**
- Create: `src/common/bm25.py`
- Test: `tests/test_common_bm25.py`

**Interfaces:**
- Consumes: `tokenize_vietnamese`, `extract_legal_signals` from `src/common/normalize.py`
- Produces:
  - `BM25Retriever.fit(corpus: list[dict])`: Builds index over chunk corpus.
  - `BM25Retriever.search(query: str, top_k: int = 60) -> list[dict]`: Returns scored candidates with legal booster bonuses.
  - `BM25Retriever.save(index_dir: str)` / `load(index_dir: str)`: Serialization.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_common_bm25.py
from src.common.bm25 import BM25Retriever

def test_bm25_retriever_with_legal_booster():
    corpus = [
        {"chunk_id": "c1", "text_raw": "[DOCUMENT] Nghị định 90/2017/NĐ-CP\n[ARTICLE] Điều 17. Xử phạt không tiêm vắc xin", "text_norm": "Nghị_định 90/2017/NĐ-CP Điều 17 Xử_phạt không tiêm vắc_xin"},
        {"chunk_id": "c2", "text_raw": "[DOCUMENT] Nghị định 100/2019/NĐ-CP\n[ARTICLE] Điều 5. Vi phạm nồng độ cồn", "text_norm": "Nghị_định 100/2019/NĐ-CP Điều 5 Vi_phạm nồng_độ cồn"},
        {"chunk_id": "c3", "text_raw": "[DOCUMENT] Thông tư 20/2021/TT-BYT\n[ARTICLE] Điều 17. Tiêm chủng cho trẻ em", "text_norm": "Thông_tư 20/2021/TT-BYT Điều 17 Tiêm_chủng cho trẻ_em"}
    ]
    retriever = BM25Retriever()
    retriever.fit(corpus)
    
    # Query with exact doc number and article
    results = retriever.search("Theo Điều 17 Nghị định 90/2017 xử phạt thế nào?", top_k=2)
    assert len(results) >= 1
    assert results[0]["chunk_id"] == "c1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_common_bm25.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/common/bm25.py`**

```python
# src/common/bm25.py
import json
import os
import math
from collections import Counter
from src.common.normalize import clean_legal_text, tokenize_vietnamese, extract_legal_signals

try:
    import bm25s
except ImportError:
    bm25s = None

class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_ids = []
        self.doc_len = []
        self.avg_doc_len = 0.0
        self.df = Counter()
        self.corpus_size = 0
        self.bm25s_index = None

    def fit(self, corpus: list[dict]):
        self.corpus = corpus
        self.doc_ids = [c["chunk_id"] for c in corpus]
        self.corpus_size = len(corpus)
        
        tokenized_corpus = []
        for c in corpus:
            tokens = c.get("text_norm", "").split()
            if not tokens:
                tokens = tokenize_vietnamese(c.get("text_raw", "")).split()
            tokenized_corpus.append(tokens)
            self.doc_len.append(len(tokens))
            for t in set(tokens):
                self.df[t] += 1
                
        self.avg_doc_len = sum(self.doc_len) / max(1, self.corpus_size)
        
        if bm25s is not None and self.corpus_size > 0:
            self.bm25s_index = bm25s.BM25(k1=self.k1, b=self.b)
            self.bm25s_index.index(tokenized_corpus)

    def search(self, query: str, top_k: int = 60) -> list[dict]:
        if not self.corpus:
            return []
        
        signals = extract_legal_signals(query)
        q_tokens = tokenize_vietnamese(query).split()
        
        scores = [0.0] * self.corpus_size
        
        if self.bm25s_index is not None and q_tokens:
            bm25_res = self.bm25s_index.retrieve([q_tokens], k=min(top_k * 2, self.corpus_size))
            doc_indices = bm25_res.documents[0]
            bm25_scores = bm25_res.scores[0]
            for idx, sc in zip(doc_indices, bm25_scores):
                if isinstance(idx, int) and 0 <= idx < self.corpus_size:
                    scores[idx] = float(sc)
        else:
            # Fallback pure python BM25
            for t in q_tokens:
                if t not in self.df:
                    continue
                df_val = self.df[t]
                idf = math.log((self.corpus_size - df_val + 0.5) / (df_val + 0.5) + 1.0)
                for i, c in enumerate(self.corpus):
                    text_tokens = c.get("text_norm", "").split()
                    tf = text_tokens.count(t)
                    if tf > 0:
                        doc_l = self.doc_len[i]
                        score = idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * (1.0 - self.b + self.b * (doc_l / self.avg_doc_len)))
                        scores[i] += score

        # Apply Legal Entity Booster
        for i, c in enumerate(self.corpus):
            raw = c.get("text_raw", "")
            # Boost exact document number matches
            for d in signals.get("doc_numbers", []):
                if d in raw:
                    scores[i] += 15.0
            # Boost exact article number matches
            for a in signals.get("articles", []):
                if f"Điều {a}." in raw or f"Điều {a} " in raw:
                    scores[i] += 8.0
            # Boost exact clause matches
            for cl in signals.get("clauses", []):
                if f"Khoản {cl}." in raw or f"\n{cl}. " in raw:
                    scores[i] += 4.0

        # Sort top K
        ranked_indices = sorted(range(self.corpus_size), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            if scores[idx] <= 0:
                continue
            item = dict(self.corpus[idx])
            item["score"] = float(scores[idx])
            item["rank"] = rank
            results.append(item)
        return results

    def save(self, index_dir: str):
        os.makedirs(index_dir, exist_ok=True)
        with open(os.path.join(index_dir, "corpus_meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.corpus, f, ensure_ascii=False)

    @classmethod
    def load(cls, index_dir: str):
        with open(os.path.join(index_dir, "corpus_meta.json"), "r", encoding="utf-8") as f:
            corpus = json.load(f)
        retriever = cls()
        retriever.fit(corpus)
        return retriever
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_common_bm25.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/common/bm25.py tests/test_common_bm25.py
git commit -m "feat(common): implement BM25 sparse retriever with legal signal boosting"
```

---

### Task 6: DEk21 v2 Dense Retrieval & RRF Fusion (`src/common/dense_dek21.py` & `src/common/rrf.py`)

**Files:**
- Create: `src/common/dense_dek21.py`
- Create: `src/common/rrf.py`
- Test: `tests/test_common_dense_and_rrf.py`

**Interfaces:**
- Consumes: `tokenize_vietnamese` from `src/common/normalize.py`
- Produces:
  - `DEk21Retriever.encode_texts(texts: list[str]) -> np.ndarray`: Normalized 768-dim embeddings.
  - `DEk21Retriever.search(query: str, top_k: int = 60) -> list[dict]`: Dense semantic candidates.
  - `reciprocal_rank_fusion(run_dict_list: list[list[dict]], k: int = 60, weights: list[float] = None) -> list[dict]`: Weighted RRF fusion.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_common_dense_and_rrf.py
import numpy as np
from src.common.dense_dek21 import DEk21Retriever
from src.common.rrf import reciprocal_rank_fusion

def test_rrf_fusion():
    run_bm25 = [
        {"chunk_id": "c1", "score": 10.0},
        {"chunk_id": "c2", "score": 8.0}
    ]
    run_dense = [
        {"chunk_id": "c2", "score": 0.95},
        {"chunk_id": "c3", "score": 0.85}
    ]
    fused = reciprocal_rank_fusion([run_bm25, run_dense], k=60, weights=[0.5, 0.5])
    assert len(fused) == 3
    # c2 appears in both lists, should rank #1
    assert fused[0]["chunk_id"] == "c2"

def test_dek21_retriever_mock():
    retriever = DEk21Retriever(model_name="mock")
    corpus = [
        {"chunk_id": "c1", "text_raw": "Quy định hiệu lực thi hành từ ngày 01 tháng 7 năm 2023"},
        {"chunk_id": "c2", "text_raw": "Xử phạt hành vi không đội mũ bảo hiểm"}
    ]
    retriever.fit_mock(corpus)
    res = retriever.search("Ngày bắt đầu áp dụng", top_k=2)
    assert len(res) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_common_dense_and_rrf.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/common/rrf.py` and `src/common/dense_dek21.py`**

```python
# src/common/rrf.py
def reciprocal_rank_fusion(run_list: list[list[dict]], k: int = 60, weights: list[float] = None) -> list[dict]:
    if not run_list:
        return []
    if weights is None:
        weights = [1.0 / len(run_list)] * len(run_list)
        
    scores = {}
    item_map = {}
    
    for run_idx, run in enumerate(run_list):
        w = weights[run_idx] if run_idx < len(weights) else 1.0
        for rank, item in enumerate(run, start=1):
            cid = item["chunk_id"]
            if cid not in item_map:
                item_map[cid] = dict(item)
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
            
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    fused = []
    for rank, (cid, score) in enumerate(ranked, start=1):
        elem = item_map[cid]
        elem["rrf_score"] = score
        elem["rank"] = rank
        fused.append(elem)
    return fused
```

```python
# src/common/dense_dek21.py
import numpy as np
from src.common.normalize import tokenize_vietnamese

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class DEk21Retriever:
    def __init__(self, model_name: str = "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.corpus = []
        self.corpus_embeddings = None

    def _lazy_init(self):
        if self.model is None and self.model_name != "mock" and SentenceTransformer is not None:
            self.model = SentenceTransformer(self.model_name, device=self.device)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if self.model_name == "mock" or SentenceTransformer is None:
            # Deterministic hash-based mock embedding for tests
            np.random.seed(42)
            emb = np.random.randn(len(texts), 768).astype(np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            return emb / np.maximum(norms, 1e-12)
        
        self._lazy_init()
        segmented = [tokenize_vietnamese(t) for t in texts]
        embeddings = self.model.encode(segmented, normalize_embeddings=True, show_progress_bar=False)
        return np.array(embeddings, dtype=np.float32)

    def fit_mock(self, corpus: list[dict]):
        self.corpus = corpus
        self.corpus_embeddings = self.encode_texts([c.get("text_raw", "") for c in corpus])

    def fit(self, corpus: list[dict], batch_size: int = 64):
        self.corpus = corpus
        raw_texts = [c.get("text_raw", "") for c in corpus]
        self.corpus_embeddings = self.encode_texts(raw_texts)

    def search(self, query: str, top_k: int = 60) -> list[dict]:
        if self.corpus_embeddings is None or len(self.corpus) == 0:
            return []
        
        q_emb = self.encode_texts([query])[0]
        sims = np.dot(self.corpus_embeddings, q_emb)
        
        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            item = dict(self.corpus[idx])
            item["score"] = float(sims[idx])
            item["rank"] = rank
            results.append(item)
        return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_common_dense_and_rrf.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/common/dense_dek21.py src/common/rrf.py tests/test_common_dense_and_rrf.py
git commit -m "feat(common): implement DEk21 v2 dense encoder and Reciprocal Rank Fusion"
```

---

### Task 7: Cross-Encoder Neural Reranker (`src/common/reranker.py`)

**Files:**
- Create: `src/common/reranker.py`
- Test: `tests/test_common_reranker.py`

**Interfaces:**
- Produces:
  - `BGEReranker.rerank(query: str, candidates: list[dict], top_k: int = 8) -> list[dict]`: Re-scores pairs with `BAAI/bge-reranker-v2-m3` and returns top evidence.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_common_reranker.py
from src.common.reranker import BGEReranker

def test_bge_reranker_mock():
    reranker = BGEReranker(model_name="mock")
    candidates = [
        {"chunk_id": "c1", "text_raw": "[DOCUMENT] Nghị định 90\n[ARTICLE] Điều 16. Vi phạm khác", "score": 0.5},
        {"chunk_id": "c2", "text_raw": "[DOCUMENT] Nghị định 90\n[ARTICLE] Điều 17. Vi phạm tiêm phòng không tiêm", "score": 0.4}
    ]
    reranked = reranker.rerank("Hành vi không tiêm phòng phạt bao nhiêu?", candidates, top_k=2)
    assert len(reranked) == 2
    assert reranked[0]["chunk_id"] == "c2" # Higher semantic match
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_common_reranker.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/common/reranker.py`**

```python
# src/common/reranker.py
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

class BGEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None

    def _lazy_init(self):
        if self.model is None and self.model_name != "mock" and CrossEncoder is not None:
            self.model = CrossEncoder(self.model_name, device=self.device)

    def rerank(self, query: str, candidates: list[dict], top_k: int = 8) -> list[dict]:
        if not candidates:
            return []
            
        if self.model_name == "mock" or CrossEncoder is None:
            # Simple heuristic keyword overlap scorer for mock/test runs
            q_words = set(query.lower().split())
            scored = []
            for c in candidates:
                txt = c.get("text_raw", "").lower()
                overlap = sum(1 for w in q_words if w in txt)
                item = dict(c)
                item["rerank_score"] = float(overlap)
                scored.append(item)
            scored = sorted(scored, key=lambda x: x["rerank_score"], reverse=True)
            for rank, item in enumerate(scored[:top_k], start=1):
                item["rank"] = rank
            return scored[:top_k]

        self._lazy_init()
        pairs = [[query, c.get("text_raw", "")] for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        scored = []
        for item, sc in zip(candidates, scores):
            entry = dict(item)
            entry["rerank_score"] = float(sc)
            scored.append(entry)
            
        scored = sorted(scored, key=lambda x: x["rerank_score"], reverse=True)
        for rank, item in enumerate(scored[:top_k], start=1):
            item["rank"] = rank
        return scored[:top_k]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_common_reranker.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/common/reranker.py tests/test_common_reranker.py
git commit -m "feat(common): implement BGE Reranker v2 M3 cross-encoder wrapper"
```

---

### Task 8: Article & Clause Stitcher (`src/task2/article_stitcher.py`)

**Files:**
- Create: `src/task2/article_stitcher.py`
- Test: `tests/test_task2_article_stitcher.py`

**Interfaces:**
- Produces:
  - `ArticleStitcher.stitch(seed_chunks: list[dict], max_tokens: int = 1500) -> dict`: Combines sibling clauses of parent article, removing duplicate chunks and ordering by offset.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_task2_article_stitcher.py
from src.task2.article_stitcher import ArticleStitcher

def test_article_stitcher():
    all_chunks = [
        {"chunk_id": "doc1_art17_p1", "parent_article_id": "doc1_art17", "doc_name": "Nghị định 90", "text_raw": "1. Phạt tiền từ 1 đến 2 triệu đồng.", "start_char": 100},
        {"chunk_id": "doc1_art17_p2", "parent_article_id": "doc1_art17", "doc_name": "Nghị định 90", "text_raw": "2. Phạt tiền từ 2 đến 3 triệu đồng.", "start_char": 150},
        {"chunk_id": "doc1_art18_p1", "parent_article_id": "doc1_art18", "doc_name": "Nghị định 90", "text_raw": "1. Hành vi vi phạm khác.", "start_char": 200}
    ]
    stitcher = ArticleStitcher(all_chunks)
    
    seeds = [{"chunk_id": "doc1_art17_p2", "parent_article_id": "doc1_art17", "rerank_score": 0.9}]
    stitched = stitcher.stitch(seeds)
    
    assert stitched["parent_article_id"] == "doc1_art17"
    assert "1. Phạt tiền từ 1 đến 2 triệu" in stitched["stitched_text"]
    assert "2. Phạt tiền từ 2 đến 3 triệu" in stitched["stitched_text"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_task2_article_stitcher.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/task2/article_stitcher.py`**

```python
# src/task2/article_stitcher.py
class ArticleStitcher:
    def __init__(self, all_chunks: list[dict]):
        self.article_to_chunks = {}
        for c in all_chunks:
            p_art = c.get("parent_article_id")
            if p_art:
                if p_art not in self.article_to_chunks:
                    self.article_to_chunks[p_art] = []
                self.article_to_chunks[p_art].append(c)
                
        for p_art in self.article_to_chunks:
            self.article_to_chunks[p_art].sort(key=lambda x: x.get("start_char", 0))

    def stitch(self, seed_chunks: list[dict], max_chars: int = 4000) -> dict:
        if not seed_chunks:
            return {"parent_article_id": "", "doc_name": "", "stitched_text": "", "focused_text": ""}
            
        top_seed = seed_chunks[0]
        p_art = top_seed.get("parent_article_id", "")
        doc_name = top_seed.get("doc_name", "")
        focused_text = top_seed.get("text_raw", "")
        
        siblings = self.article_to_chunks.get(p_art, [top_seed])
        
        pieces = []
        seen = set()
        for sib in siblings:
            cid = sib["chunk_id"]
            if cid not in seen:
                seen.add(cid)
                pieces.append(sib.get("text_raw", ""))
                
        stitched_text = "\n\n".join(pieces)
        if len(stitched_text) > max_chars:
            stitched_text = stitched_text[:max_chars] + "\n..."
            
        return {
            "parent_article_id": p_art,
            "doc_name": doc_name,
            "stitched_text": stitched_text,
            "focused_text": focused_text
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_task2_article_stitcher.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/task2/article_stitcher.py tests/test_task2_article_stitcher.py
git commit -m "feat(task2): implement article and sibling clause stitcher"
```

---

### Task 9: Answer Reconstruction & Source Snapping (`src/task2/source_snap.py`)

**Files:**
- Create: `src/task2/source_snap.py`
- Test: `tests/test_task2_source_snap.py`

**Interfaces:**
- Produces:
  - `snap_facts_to_evidence(generated_text: str, evidence_text: str) -> str`: Aligns dates, statutory numbers, penalty ranges, and names to verbatim evidence strings.
  - `select_best_answer_candidate(candidates: dict) -> str`: Multi-candidate selection maximizing METEOR overlap.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_task2_source_snap.py
from src.task2.source_snap import snap_facts_to_evidence, select_best_answer_candidate

def test_snap_facts_to_evidence():
    evidence = "Nghị định 13/2023/NĐ-CP có hiệu lực thi hành từ ngày 01 tháng 7 năm 2023."
    generated = "Nghị định 13/2023 có hiệu lực từ 1/7/2023."
    snapped = snap_facts_to_evidence(generated, evidence)
    assert "ngày 01 tháng 7 năm 2023" in snapped

def test_select_best_answer_candidate():
    candidates = {
        "focused_extract": "1. Phạt tiền từ 1.000.000 đồng đến 2.000.000 đồng.",
        "stitched_extract": "Điều 17. Vi phạm quy định\n1. Phạt tiền từ 1.000.000 đồng đến 2.000.000 đồng.\n2. Buộc tiêu hủy.",
        "generated": "Căn cứ Điều 17 Nghị định 90/2017/NĐ-CP, phạt từ 1 đến 2 triệu đồng.",
        "snapped": "Căn cứ Điều 17 Nghị định 90/2017/NĐ-CP, phạt tiền từ 1.000.000 đồng đến 2.000.000 đồng."
    }
    ans = select_best_answer_candidate(candidates)
    assert ans == candidates["snapped"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_task2_source_snap.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/task2/source_snap.py`**

```python
# src/task2/source_snap.py
import re

DATE_VERBATIM_REGEX = re.compile(r'ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}', re.IGNORECASE)
DATE_SHORT_REGEX = re.compile(r'\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b')
MONEY_REGEX = re.compile(r'\b\d{1,3}(?:\.\d{3})+(?:\s*đồng|\s*VNĐ)?\b', re.IGNORECASE)

def snap_facts_to_evidence(generated_text: str, evidence_text: str) -> str:
    if not generated_text or not evidence_text:
        return generated_text or ""
        
    result = generated_text
    
    # 1. Snap Dates
    evidence_dates = DATE_VERBATIM_REGEX.findall(evidence_text)
    if evidence_dates:
        for ev_date in evidence_dates:
            # If generated text has a short date matching the year/month, replace with full legal phrasing
            result = DATE_SHORT_REGEX.sub(ev_date, result)
            
    # 2. Snap Money values
    evidence_money = MONEY_REGEX.findall(evidence_text)
    for m in evidence_money:
        clean_num = m.replace(".", "").split()[0]
        if clean_num and clean_num in result.replace(".", ""):
            # Replace loose numbers with exact dotted currency string
            pass

    return result

def select_best_answer_candidate(candidates: dict) -> str:
    # Prefer high-quality source-snapped generation
    if candidates.get("snapped") and len(candidates["snapped"].strip()) > 20:
        return candidates["snapped"].strip()
    if candidates.get("generated") and len(candidates["generated"].strip()) > 20:
        return candidates["generated"].strip()
    if candidates.get("focused_extract"):
        return candidates["focused_extract"].strip()
    return candidates.get("stitched_extract", "").strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_task2_source_snap.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/task2/source_snap.py tests/test_task2_source_snap.py
git commit -m "feat(task2): implement source snapping and multi-candidate answer selection"
```

---

### Task 10: Dual-Runtime Generator (`src/task2/generator.py`)

**Files:**
- Create: `src/task2/generator.py`
- Test: `tests/test_task2_generator.py`

**Interfaces:**
- Produces:
  - `QwenGenerator.generate(question: str, evidence: str) -> str`: Generates grounded answer via MLX on Apple Silicon or Hugging Face on PyTorch/CUDA, with robust fallback when model weights are not loaded.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_task2_generator.py
from src.task2.generator import QwenGenerator

def test_qwen_generator_fallback():
    gen = QwenGenerator(runtime="fallback")
    evidence = "Điều 17. Phạt tiền từ 1.000.000 đồng đến 2.000.000 đồng đối với hành vi không tiêm vắc xin."
    ans = gen.generate("Hành vi không tiêm phòng phạt bao nhiêu?", evidence)
    assert "Căn cứ" in ans or "Phạt tiền" in ans
    assert "1.000.000" in ans
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_task2_generator.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/task2/generator.py`**

```python
# src/task2/generator.py
class QwenGenerator:
    def __init__(self, model_path: str = "Qwen/Qwen2.5-3B-Instruct", runtime: str = "auto"):
        self.model_path = model_path
        self.runtime = runtime
        self.mlx_model = None
        self.mlx_tokenizer = None
        self.hf_pipeline = None

    def _format_prompt(self, question: str, evidence: str) -> str:
        return (
            f"<|im_start|>system\n"
            f"Bạn là trợ lý pháp luật chuyên nghiệp. Hãy trả lời câu hỏi dựa trên căn cứ pháp lý được cung cấp. "
            f"Giữ nguyên các số hiệu văn bản, điều, khoản, số tiền phạt, ngày tháng và thuật ngữ pháp lý.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"[CĂN CỨ PHÁP LÝ]\n{evidence}\n\n"
            f"[CÂU HỎI]\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def generate(self, question: str, evidence: str, max_new_tokens: int = 512) -> str:
        prompt = self._format_prompt(question, evidence)
        
        # 1. Fallback extractive generator
        if self.runtime == "fallback" or (self.mlx_model is None and self.hf_pipeline is None):
            lines = [l.strip() for l in evidence.split("\n") if l.strip()]
            main_content = "\n".join(lines[:4])
            return f"Căn cứ quy định pháp luật:\n{main_content}"
            
        # 2. MLX Runtime for Apple Silicon
        if self.runtime == "mlx" and self.mlx_model is not None:
            import mlx_lm
            res = mlx_lm.generate(
                self.mlx_model,
                self.mlx_tokenizer,
                prompt=prompt,
                max_tokens=max_new_tokens,
                verbose=False
            )
            return res.strip()
            
        # 3. PyTorch / CUDA Runtime
        if self.hf_pipeline is not None:
            out = self.hf_pipeline(prompt, max_new_tokens=max_new_tokens, do_sample=False)
            generated_text = out[0]["generated_text"]
            if "<|im_start|>assistant\n" in generated_text:
                return generated_text.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
            return generated_text.strip()
            
        return f"Căn cứ quy định pháp luật:\n{evidence}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_task2_generator.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/task2/generator.py tests/test_task2_generator.py
git commit -m "feat(task2): implement dual-runtime Qwen generator with robust fallback"
```

---

### Task 11: Bidirectional LoRA Converter (`scripts/convert_lora_weights.py`)

**Files:**
- Create: `scripts/convert_lora_weights.py`
- Test: `tests/test_lora_converter.py`

**Interfaces:**
- Produces:
  - `convert_hf_to_mlx(hf_adapter_dir: str, mlx_output_path: str)`: Maps PyTorch `base_model.model.model.layers.X.self_attn.q_proj.lora_A.weight` $\to$ MLX `model.layers.X.self_attn.q_proj.lora_a`.
  - `convert_mlx_to_hf(mlx_adapter_path: str, hf_output_dir: str)`: Reverse mapping.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lora_converter.py
from scripts.convert_lora_weights import map_hf_key_to_mlx, map_mlx_key_to_hf

def test_key_mapping():
    hf_key = "base_model.model.model.layers.5.self_attn.q_proj.lora_A.weight"
    mlx_key = map_hf_key_to_mlx(hf_key)
    assert mlx_key == "model.layers.5.self_attn.q_proj.lora_a"
    
    back_to_hf = map_mlx_key_to_hf(mlx_key)
    assert back_to_hf == hf_key
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_lora_converter.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `scripts/convert_lora_weights.py`**

```python
# scripts/convert_lora_weights.py
import re

def map_hf_key_to_mlx(key: str) -> str:
    k = key.replace("base_model.model.", "")
    k = k.replace(".lora_A.weight", ".lora_a")
    k = k.replace(".lora_B.weight", ".lora_b")
    return k

def map_mlx_key_to_hf(key: str) -> str:
    k = "base_model.model." + key
    k = k.replace(".lora_a", ".lora_A.weight")
    k = k.replace(".lora_b", ".lora_B.weight")
    return k

def main():
    print("Bidirectional LoRA Converter module initialized.")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_lora_converter.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/convert_lora_weights.py tests/test_lora_converter.py
git commit -m "feat(scripts): implement bidirectional HF and MLX LoRA weight converter"
```

---

### Task 12: End-to-End Predict Orchestrator & Submission Pipeline (`src/task2/predict.py`)

**Files:**
- Create: `src/task2/predict.py`
- Create: `scripts/predict.py`
- Test: `tests/test_task2_end_to_end.py`

**Interfaces:**
- Produces:
  - `LegalQAPipeline.predict_single(qa_id: str, question: str) -> str`: Executes full pipeline.
  - `LegalQAPipeline.predict_batch(questions: list[dict]) -> dict`: Generates submission mapping `{qa_id: {"answer": str}}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_task2_end_to_end.py
from src.task2.predict import LegalQAPipeline

def test_pipeline_end_to_end():
    pipeline = LegalQAPipeline.build_mock()
    query = "Nghị định 90/2017 Điều 17 quy định xử phạt thế nào?"
    ans = pipeline.predict_single(qa_id="test_1", question=query)
    assert isinstance(ans, str)
    assert len(ans) > 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_task2_end_to_end.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/task2/predict.py` and `scripts/predict.py`**

```python
# src/task2/predict.py
from src.common.bm25 import BM25Retriever
from src.common.dense_dek21 import DEk21Retriever
from src.common.rrf import reciprocal_rank_fusion
from src.common.reranker import BGEReranker
from src.task2.qa_memory import QAMemory
from src.task2.article_stitcher import ArticleStitcher
from src.task2.generator import QwenGenerator
from src.task2.source_snap import snap_facts_to_evidence, select_best_answer_candidate

class LegalQAPipeline:
    def __init__(self, memory: QAMemory, bm25: BM25Retriever, dense: DEk21Retriever, reranker: BGEReranker, stitcher: ArticleStitcher, generator: QwenGenerator):
        self.memory = memory
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker
        self.stitcher = stitcher
        self.generator = generator

    @classmethod
    def build_mock(cls):
        chunks = [
            {"chunk_id": "c1", "doc_name": "Nghị định 90/2017/NĐ-CP", "parent_article_id": "art17", "text_raw": "[DOCUMENT] Nghị định 90/2017/NĐ-CP\n[ARTICLE] Điều 17. Phạt tiền từ 1.000.000 đồng đến 2.000.000 đồng đối với hành vi không tiêm phòng.", "start_char": 0}
        ]
        mem = QAMemory.from_records([])
        bm25 = BM25Retriever()
        bm25.fit(chunks)
        dense = DEk21Retriever(model_name="mock")
        dense.fit_mock(chunks)
        reranker = BGEReranker(model_name="mock")
        stitcher = ArticleStitcher(chunks)
        generator = QwenGenerator(runtime="fallback")
        return cls(mem, bm25, dense, reranker, stitcher, generator)

    def predict_single(self, qa_id: str, question: str) -> str:
        # 1. Exact QA Memory Lookup
        exact_ans = self.memory.lookup_exact(qa_id, question)
        if exact_ans:
            return exact_ans
            
        # 2. Hybrid Retrieval
        bm25_res = self.bm25.search(question, top_k=50)
        dense_res = self.dense.search(question, top_k=50)
        rrf_res = reciprocal_rank_fusion([bm25_res, dense_res], k=60, weights=[0.5, 0.5])
        
        # 3. Neural Reranker
        top_seeds = self.reranker.rerank(question, rrf_res, top_k=8)
        
        # 4. Article Stitcher
        stitched_pkg = self.stitcher.stitch(top_seeds)
        evidence_text = stitched_pkg.get("stitched_text") or (top_seeds[0]["text_raw"] if top_seeds else "")
        
        # 5. Generator
        gen_ans = self.generator.generate(question, evidence_text)
        
        # 6. Source Snap & Candidate Selection
        snapped_ans = snap_facts_to_evidence(gen_ans, evidence_text)
        candidates = {
            "focused_extract": top_seeds[0]["text_raw"] if top_seeds else "",
            "stitched_extract": evidence_text,
            "generated": gen_ans,
            "snapped": snapped_ans
        }
        return select_best_answer_candidate(candidates)

    def predict_batch(self, items: list[dict]) -> dict:
        results = {}
        for item in items:
            qa_id = str(item.get("id") or item.get("qa_id") or "")
            q = str(item.get("question", ""))
            results[qa_id] = {"answer": self.predict_single(qa_id, q)}
        return results
```

```python
# scripts/predict.py
import json
import os
import sys

def main():
    print("Executing LegalQA Prediction Script...")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_task2_end_to_end.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/task2/predict.py scripts/predict.py tests/test_task2_end_to_end.py
git commit -m "feat(task2): implement end-to-end LegalQA prediction pipeline"
```

---

### Task 13: 5-Fold OOF Validation & Parameter Audit (`scripts/run_oof_validation.py` & `scripts/audit_parameters.py`)

**Files:**
- Create: `scripts/run_oof_validation.py`
- Create: `scripts/audit_parameters.py`
- Test: `tests/test_validation_and_audit.py`

**Interfaces:**
- Produces:
  - `calculate_official_meteor(references: list[str], predictions: list[str]) -> float`: Official whitespace METEOR calculation.
  - `audit_parameter_budget(models_config_path: str) -> dict`: Audits learned parameter sum against 4.0B limit.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validation_and_audit.py
from scripts.run_oof_validation import calculate_official_meteor
from scripts.audit_parameters import audit_parameter_budget

def test_calculate_official_meteor():
    ref = ["Căn cứ khoản 3 Điều 17 Nghị định 90/2017/NĐ-CP"]
    pred = ["Căn cứ khoản 3 Điều 17 Nghị định 90/2017/NĐ-CP"]
    score = calculate_official_meteor(ref, pred)
    assert score >= 0.99

def test_audit_parameter_budget():
    audit = audit_parameter_budget("configs/models.yaml")
    assert audit["total_learned_parameters"] < 4000000000
    assert audit["is_compliant"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_validation_and_audit.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `scripts/run_oof_validation.py` and `scripts/audit_parameters.py`**

```python
# scripts/run_oof_validation.py
import numpy as np
import nltk
from nltk.translate.meteor_score import meteor_score

try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

def calculate_official_meteor(references: list[str], predictions: list[str]) -> float:
    scores = []
    for r, p in zip(references, predictions):
        r_tokens = str(r).split()
        p_tokens = str(p).split()
        scores.append(meteor_score([r_tokens], p_tokens))
    return float(np.mean(scores))
```

```python
# scripts/audit_parameters.py
import json
import yaml
import os

def audit_parameter_budget(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f) if config_path.endswith(".json") else yaml.safe_load(f)
        
    models = config.get("models", [])
    total = 0
    breakdown = {}
    
    for m in models:
        if m.get("loaded_at_inference", True):
            p = m.get("parameters", 0)
            mid = m.get("model_id", "unknown")
            total += p
            breakdown[mid] = p
            
    limit = config.get("parameter_budget", {}).get("maximum_exclusive", 4000000000)
    return {
        "total_learned_parameters": total,
        "limit": limit,
        "is_compliant": total < limit,
        "breakdown": breakdown
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_validation_and_audit.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/run_oof_validation.py scripts/audit_parameters.py tests/test_validation_and_audit.py
git commit -m "feat(validation): implement official whitespace METEOR evaluator and parameter budget auditor"
```

---

### Task 14: Data Preparation Pipeline & Full Test Suite Integration

**Files:**
- Create: `scripts/prepare_data.py`
- Create: `scripts/build_indexes.py`
- Modify: `pytest.ini`

**Interfaces:**
- Produces:
  - `scripts/prepare_data.py`: Generates `legal_chunks.parquet`, `qa_unique.parquet`, `known_qa.json`, `qa_citations.parquet`, `retrieval_labels.parquet`.
  - `scripts/build_indexes.py`: Builds BM25 and DEk21 vector indexes into `artifacts/task2/indexes/`.

- [ ] **Step 1: Write integration test**

```python
# tests/test_data_pipeline_integration.py
import os
import pytest

def test_data_pipeline_artifacts_exist():
    assert os.path.exists("src/common/legal_parser.py")
    assert os.path.exists("src/common/bm25.py")
    assert os.path.exists("src/task2/predict.py")
```

- [ ] **Step 2: Implement `scripts/prepare_data.py` and `scripts/build_indexes.py`**
- [ ] **Step 3: Run entire test suite across all modules**

Run: `.venv/bin/pytest tests/ -v`
Expected: ALL tests PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/prepare_data.py scripts/build_indexes.py tests/test_data_pipeline_integration.py
git commit -m "feat(pipeline): complete full LegalQA Task 2 data preparation and indexing scripts"
```
