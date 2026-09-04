"""Answer-preserving SFT dataset builder and deterministic worst-case selector for LegalQA Task 2."""

from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from src.task2.generator import format_qwen_chat_prompt


@dataclass
class SFTExample:
    """Structured SFT training example with token diagnostics."""

    prompt: str
    completion: str
    total_tokens: int
    completion_tokens: int
    qa_id: str
    text: str = ""
    evidence_truncated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "text": self.text or f"{self.prompt}{self.completion}",
            "qa_id": self.qa_id,
            "total_tokens": self.total_tokens,
            "completion_tokens": self.completion_tokens,
            "evidence_truncated": self.evidence_truncated,
        }


def truncate_evidence_preserving_answer(
    question: str,
    evidence: str,
    answer: str,
    max_chars: int = 3000,
) -> str:
    """Answer-preserving character truncation: trims evidence at clause/line boundaries, preserving full gold answer."""
    if not evidence or len(evidence) <= max_chars:
        return evidence.strip()

    paragraphs = evidence.split("\n\n")
    kept_pieces = []
    current_len = 0

    for p in paragraphs:
        p_len = len(p) + 2
        if current_len + p_len <= max_chars:
            kept_pieces.append(p)
            current_len += p_len
        else:
            remaining = max_chars - current_len
            if remaining > 10:
                cut_len = max(0, remaining - 3)
                kept_pieces.append(p[:cut_len].rstrip() + "...")
            break

    return "\n\n".join(kept_pieces)


def build_sft_example_token_aware(
    question: str,
    evidence_text: str,
    answer: str,
    tokenizer: Optional[Any] = None,
    max_seq_len: int = 2048,
    safety_margin: int = 8,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Token-aware example builder guaranteeing gold answer tokens are 100% preserved.

    Returns:
        (full_text, diagnostics_dict) where diagnostics_dict contains 'prompt' and 'completion'
        for modern TRL prompt-completion SFT datasets.
    """
    ans_clean = str(answer).strip()
    q_clean = str(question).strip()
    completion_text = f"{ans_clean}<|im_end|>"

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        ans_tokens = tokenizer.encode(completion_text, add_special_tokens=False)
        ans_token_count = len(ans_tokens)

        empty_prompt = format_qwen_chat_prompt(q_clean, "", tokenizer=tokenizer)
        framing_tokens = tokenizer.encode(empty_prompt, add_special_tokens=False)
        framing_token_count = len(framing_tokens)

        minimum_required = framing_token_count + ans_token_count
        if minimum_required > max_seq_len:
            # Dropped: answer itself plus minimal framing cannot fit within max_seq_len
            return None, {
                "total_tokens": minimum_required,
                "completion_tokens": ans_token_count,
                "evidence_truncated": False,
                "answer_truncated": False,
                "dropped": True,
                "prompt": empty_prompt,
                "completion": completion_text,
            }

        evidence_budget = max_seq_len - minimum_required - safety_margin

        paragraphs = [p.strip() for p in evidence_text.split("\n\n") if p.strip()] if evidence_text else []
        packed_paragraphs = []
        curr_ev_tokens = 0
        ev_truncated = False

        for p in paragraphs:
            p_toks = len(tokenizer.encode(p, add_special_tokens=False))
            if curr_ev_tokens + p_toks <= evidence_budget:
                packed_paragraphs.append(p)
                curr_ev_tokens += p_toks
            else:
                ev_truncated = True
                remaining_toks = evidence_budget - curr_ev_tokens
                if remaining_toks > 15:
                    p_enc = tokenizer.encode(p, add_special_tokens=False)[:remaining_toks]
                    p_dec = tokenizer.decode(p_enc, skip_special_tokens=True).rstrip() + "..."
                    packed_paragraphs.append(p_dec)
                break

        final_evidence = "\n\n".join(packed_paragraphs) if packed_paragraphs else ""
        prompt = format_qwen_chat_prompt(q_clean, final_evidence, tokenizer=tokenizer)
        full_text = f"{prompt}{completion_text}"
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        # While loop safety guard: remove packed paragraphs from the end if tokenizer special tokens exceed max_seq_len
        while len(full_ids) > max_seq_len and packed_paragraphs:
            packed_paragraphs.pop()
            final_evidence = "\n\n".join(packed_paragraphs)
            prompt = format_qwen_chat_prompt(q_clean, final_evidence, tokenizer=tokenizer)
            full_text = f"{prompt}{completion_text}"
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)
            ev_truncated = True

        assert len(full_ids) <= max_seq_len, f"Full text token length {len(full_ids)} > {max_seq_len}"

        diagnostics = {
            "total_tokens": len(full_ids),
            "completion_tokens": ans_token_count,
            "evidence_truncated": ev_truncated,
            "answer_truncated": False,
            "dropped": False,
            "prompt": prompt,
            "completion": completion_text,
        }
        return full_text, diagnostics

    # Fallback when tokenizer is not available
    ev_safe = truncate_evidence_preserving_answer(q_clean, evidence_text, ans_clean, max_chars=3000)
    prompt = format_qwen_chat_prompt(q_clean, ev_safe, tokenizer=None)
    full_text = f"{prompt}{completion_text}"
    ans_words = len(ans_clean.split())
    total_words = len(full_text.split())
    return full_text, {
        "total_tokens": total_words,
        "completion_tokens": ans_words,
        "evidence_truncated": False,
        "answer_truncated": False,
        "dropped": False,
        "prompt": prompt,
        "completion": completion_text,
    }


def build_grounded_training_examples(
    qa_path: Optional[str] = None,
    df_qa: Optional[pd.DataFrame] = None,
    labels_path: Optional[str] = None,
    chunks_path: Optional[str] = None,
    fold_to_exclude: Optional[int] = None,
    tokenizer: Optional[Any] = None,
    max_seq_len: int = 2048,
    max_train_examples: Optional[int] = None,
    return_diagnostics: bool = False,
    return_sft_objects: bool = False,
    seed: int = 42,
) -> Union[
    List[Dict[str, Any]],
    Tuple[List[Dict[str, Any]], Dict[str, Any]],
    List[SFTExample],
    Tuple[List[SFTExample], Dict[str, Any]],
]:
    """Build multi-positive structured SFT training examples with prompt-completion formatting."""
    if df_qa is None:
        if qa_path and os.path.exists(qa_path):
            df_qa = pd.read_parquet(qa_path)
        else:
            return ([], {}) if return_diagnostics else []

    if fold_to_exclude is not None and "fold_id" in df_qa.columns:
        df_qa = df_qa[df_qa["fold_id"] != fold_to_exclude]

    # 1. Deterministic sampling FIRST for bounded smoke subsets
    if max_train_examples is not None and len(df_qa) > max_train_examples:
        df_qa = df_qa.sample(n=max_train_examples, random_state=seed).reset_index(drop=True)
    else:
        df_qa = df_qa.reset_index(drop=True)

    target_qa_ids = set(df_qa["qa_id"].astype(str)) if "qa_id" in df_qa.columns else set()
    if not target_qa_ids and "id" in df_qa.columns:
        target_qa_ids = set(df_qa["id"].astype(str))

    # 2. Filter retrieval labels to only target QA IDs
    needed_chunk_ids: set[str] = set()
    qa_to_pos_chunk_ids: Dict[str, List[str]] = {}

    if labels_path and os.path.exists(labels_path):
        try:
            df_labels = pd.read_parquet(labels_path, columns=["qa_id", "positive_chunk_id"])
        except Exception:
            df_labels = pd.read_parquet(labels_path)

        if "qa_id" in df_labels.columns and "positive_chunk_id" in df_labels.columns:
            if target_qa_ids:
                df_labels = df_labels[df_labels["qa_id"].astype(str).isin(target_qa_ids)]

            for _, row in df_labels.iterrows():
                qid = str(row["qa_id"]).strip()
                cid = str(row.get("positive_chunk_id", "")).strip()
                if cid:
                    if qid not in qa_to_pos_chunk_ids:
                        qa_to_pos_chunk_ids[qid] = []
                    if cid not in qa_to_pos_chunk_ids[qid]:
                        qa_to_pos_chunk_ids[qid].append(cid)
                    needed_chunk_ids.add(cid)

    # 3. Read chunks with selective projection
    chunk_map: Dict[str, str] = {}
    if chunks_path and os.path.exists(chunks_path):
        read_success = False
        if needed_chunk_ids:
            try:
                import pyarrow.dataset as ds
                dataset = ds.dataset(chunks_path, format="parquet")
                table = dataset.to_table(
                    columns=["chunk_id", "text_raw"],
                    filter=ds.field("chunk_id").isin(list(needed_chunk_ids)),
                )
                df_chunks = table.to_pandas()
                chunk_map = dict(zip(df_chunks["chunk_id"].astype(str), df_chunks["text_raw"]))
                read_success = True
            except Exception:
                read_success = False

        if not read_success:
            try:
                df_chunks = pd.read_parquet(chunks_path, columns=["chunk_id", "text_raw"])
            except Exception:
                df_chunks = pd.read_parquet(chunks_path)

            if "chunk_id" in df_chunks.columns and "text_raw" in df_chunks.columns:
                if needed_chunk_ids and len(needed_chunk_ids) < len(df_chunks):
                    df_chunks = df_chunks[df_chunks["chunk_id"].astype(str).isin(needed_chunk_ids)]
                chunk_map = dict(zip(df_chunks["chunk_id"].astype(str), df_chunks["text_raw"]))

    qa_to_pos_evidence: Dict[str, List[str]] = {}
    for qid, cids in qa_to_pos_chunk_ids.items():
        qa_to_pos_evidence[qid] = [chunk_map[c] for c in cids if c in chunk_map]

    examples: List[Dict[str, Any]] = []
    sft_objects: List[SFTExample] = []
    token_lengths = []
    ev_truncated_count = 0
    dropped_count = 0

    for _, row in df_qa.iterrows():
        qid = str(row.get("qa_id") or row.get("id", "")).strip()
        q = str(row.get("question_raw") or row.get("question", "")).strip()
        a = str(row.get("answer_raw") or row.get("answer", "")).strip()

        if not q or not a:
            continue

        pos_pieces = qa_to_pos_evidence.get(qid, [])
        raw_evidence = "\n\n".join(pos_pieces) if pos_pieces else ""

        full_text, diag = build_sft_example_token_aware(
            question=q,
            evidence_text=raw_evidence,
            answer=a,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
        )

        if diag.get("dropped") or full_text is None:
            dropped_count += 1
            continue

        sft_ex = SFTExample(
            prompt=diag.get("prompt", ""),
            completion=diag.get("completion", f"{a}<|im_end|>"),
            total_tokens=diag.get("total_tokens", len(full_text.split())),
            completion_tokens=diag.get("completion_tokens", len(a.split())),
            qa_id=qid,
            text=full_text,
            evidence_truncated=diag.get("evidence_truncated", False),
        )

        examples.append(sft_ex.to_dict())
        sft_objects.append(sft_ex)
        token_lengths.append(diag["total_tokens"])
        if diag.get("evidence_truncated"):
            ev_truncated_count += 1

    diag_summary = {
        "kept_count": len(examples),
        "dropped_count": dropped_count,
        "drop_rate": dropped_count / max(1, len(examples) + dropped_count),
        "evidence_truncated_count": ev_truncated_count,
        "evidence_truncated_rate": ev_truncated_count / max(1, len(examples)),
        "token_lengths": token_lengths,
        "p50_tokens": float(np.percentile(token_lengths, 50)) if token_lengths else 0.0,
        "p90_tokens": float(np.percentile(token_lengths, 90)) if token_lengths else 0.0,
        "p95_tokens": float(np.percentile(token_lengths, 95)) if token_lengths else 0.0,
        "p99_tokens": float(np.percentile(token_lengths, 99)) if token_lengths else 0.0,
        "max_tokens": int(max(token_lengths)) if token_lengths else 0,
    }

    out_data = sft_objects if return_sft_objects else examples
    if return_diagnostics:
        return out_data, diag_summary
    return out_data


def select_worst_case_probe(
    examples: Sequence[Union[SFTExample, Dict[str, Any]]],
    n_total: int = 12,
    n_completion: int = 12,
) -> List[Union[SFTExample, Dict[str, Any]]]:
    """Select a deterministic worst-case probe subset for testing peak memory under high sequence lengths.

    Strategy (from 06_LIGER_GENERATOR_DESIGN.md):
        A = top 12 by total_tokens
        B = top 12 by completion_tokens
        probe = stable deduplicated union(A, B)
        If fewer than 24 unique examples remain, include next-longest examples by total_tokens.
    """
    if not examples:
        return []

    def get_val(ex, key):
        if isinstance(ex, SFTExample):
            return getattr(ex, key)
        return ex.get(key, 0)

    def get_id(ex, idx):
        if isinstance(ex, SFTExample):
            return ex.qa_id or str(idx)
        return ex.get("qa_id", str(idx))

    # Indexed list with stable sorting
    indexed = list(enumerate(examples))

    # Sort descending by total_tokens, then original index for stability
    sorted_total = sorted(
        indexed,
        key=lambda item: (-get_val(item[1], "total_tokens"), item[0]),
    )

    # Sort descending by completion_tokens, then original index for stability
    sorted_completion = sorted(
        indexed,
        key=lambda item: (-get_val(item[1], "completion_tokens"), item[0]),
    )

    chosen_indices: List[int] = []
    seen_ids = set()

    # 1. Top n_total
    for idx, ex in sorted_total[:n_total]:
        ex_id = get_id(ex, idx)
        if ex_id not in seen_ids:
            seen_ids.add(ex_id)
            chosen_indices.append(idx)

    # 2. Top n_completion
    for idx, ex in sorted_completion[:n_completion]:
        ex_id = get_id(ex, idx)
        if ex_id not in seen_ids:
            seen_ids.add(ex_id)
            chosen_indices.append(idx)

    # 3. Fill up to target count if deduplication reduced size below 24
    target_count = min(n_total + n_completion, len(examples))
    if len(chosen_indices) < target_count:
        for idx, ex in sorted_total:
            ex_id = get_id(ex, idx)
            if ex_id not in seen_ids:
                seen_ids.add(ex_id)
                chosen_indices.append(idx)
                if len(chosen_indices) >= target_count:
                    break

    # Preserve original ordering among selected examples for deterministic consistency
    chosen_indices.sort()
    return [examples[i] for i in chosen_indices]
