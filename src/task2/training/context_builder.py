"""Context construction for generator training: retrieved vs gold-assisted.

The record builder consumes NATURALLY RETRIEVED evidence, never mandatory
gold labels: QA examples without retrieval labels remain eligible for SFT
(their evidence is whatever retrieval returned, possibly empty and recorded
as such). Gold-assisted experimental records are constructed separately from
permitted training groups only, through the same token-aware packer, and are
forbidden for dev, lockbox, and public inference.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.task2.generation.dataset import build_chunk_rows

ALLOWED_POLICIES = ("retrieved", "gold_assisted")


def build_context_record(
    qa: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    policy: str,
    max_seq_len: int,
) -> Dict[str, Any]:
    """Build one context record joining a QA example with evidence chunks.

    qa must carry qa_id, qa_group_id, question_raw, answer_raw. evidence is a
    list of chunk dicts with chunk_id/text_raw. policy is "retrieved"
    (deployable) or "gold_assisted" (train-only experiment). max_seq_len is
    recorded so the token-aware packer downstream can enforce the candidate
    sequence budget. Returns primitive fields only.
    """
    if policy not in ALLOWED_POLICIES:
        raise ValueError(f"unknown context policy: {policy}")
    qa_id = str(qa.get("qa_id", "") or "").strip()
    qa_group_id = str(qa.get("qa_group_id", "") or "").strip()
    question = str(qa.get("question_raw", qa.get("question", "")) or "")
    answer = str(qa.get("answer_raw", qa.get("answer", "")) or "")
    if not qa_id or not qa_group_id:
        raise ValueError("build_context_record requires qa_id and qa_group_id")
    if not question.strip() or not answer.strip():
        raise ValueError("build_context_record requires nonempty question and answer")
    if max_seq_len not in (2048, 3072, 4096):
        raise ValueError(f"unapproved sequence budget: {max_seq_len}")
    evidence_texts = [str(e.get("text_raw", "") or "") for e in evidence]
    return {
        "qa_id": qa_id,
        "qa_group_id": qa_group_id,
        "question_raw": question,
        "answer_raw": answer,
        "policy": policy,
        "max_seq_len": int(max_seq_len),
        "evidence_chunk_ids": [str(e.get("chunk_id", "") or "") for e in evidence],
        "evidence_texts": evidence_texts,
        "evidence_count": len(evidence_texts),
        "evidence_chars": sum(len(t) for t in evidence_texts),
    }


def build_training_records(
    qas: List[Dict[str, Any]],
    retrieved: Dict[str, List[str]],
    chunks: List[Dict[str, Any]],
    allowed_group_ids: set,
    max_seq_len: int = 2048,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Join canonical QA examples with naturally retrieved evidence.

    - Only examples whose qa_group_id is in allowed_group_ids are kept;
      held-out groups are excluded and counted, never leaked.
    - Evidence comes from the retrieved mapping (chunk ids per qa_id)
      resolved against ALL physical chunk rows via build_chunk_rows, so
      repeated chunk ids contribute every fragment in order.
    - QA examples with no retrieved evidence (including unlabeled QA) are
      still eligible: they are kept with empty evidence and counted in the
      overflow/missing-evidence ledger, never silently dropped.
    """
    allowed = {str(g).strip() for g in allowed_group_ids}
    needed: set = set()
    for qid, cids in retrieved.items():
        for cid in cids or []:
            if str(cid).strip():
                needed.add(str(cid).strip())
    kept_rows, chunk_report = build_chunk_rows(list(chunks), needed)
    texts_by_id: Dict[str, List[str]] = {}
    for row in kept_rows:
        texts_by_id.setdefault(str(row.get("chunk_id", "")), []).append(str(row.get("text_raw", "") or ""))

    records: List[Dict[str, Any]] = []
    excluded_heldout = 0
    missing_evidence = 0
    for qa in qas:
        gid = str(qa.get("qa_group_id", "") or "").strip()
        qid = str(qa.get("qa_id", "") or "").strip()
        if gid not in allowed:
            excluded_heldout += 1
            continue
        cids = [str(c).strip() for c in (retrieved.get(qid, []) or []) if str(c).strip()]
        evidence = []
        for cid in cids:
            for text in texts_by_id.get(cid, []):
                evidence.append({"chunk_id": cid, "text_raw": text})
        if not evidence:
            missing_evidence += 1
        seq_len = int(qa.get("max_seq_len", max_seq_len))
        records.append(build_context_record(qa, evidence, policy="retrieved", max_seq_len=seq_len))
    report = {
        "input_qa": len(qas),
        "allowed_groups": len(allowed),
        "kept_records": len(records),
        "excluded_heldout": excluded_heldout,
        "missing_evidence_records": missing_evidence,
        "chunk_report": chunk_report,
    }
    return records, report


def build_gold_assisted_records(
    qas: List[Dict[str, Any]],
    gold_by_qa: Dict[str, List[str]],
    chunks: List[Dict[str, Any]],
    allowed_group_ids: set,
    max_seq_len: int = 2048,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build train-only gold-assisted experimental records (80/20 hypothesis).

    Caller must guarantee every input group is a permitted training group;
    this function re-validates against allowed_group_ids and raises on any
    held-out group. Records use the same token-aware packer contract and are
    labeled policy="gold_assisted" so they can never enter dev/lockbox or
    public inference paths.
    """
    allowed = {str(g).strip() for g in allowed_group_ids}
    for qa in qas:
        if str(qa.get("qa_group_id", "") or "").strip() not in allowed:
            raise ValueError("gold-assisted records forbid held-out groups")
    needed: set = set()
    for cids in gold_by_qa.values():
        for cid in cids or []:
            if str(cid).strip():
                needed.add(str(cid).strip())
    kept_rows, chunk_report = build_chunk_rows(list(chunks), needed)
    texts_by_id: Dict[str, List[str]] = {}
    for row in kept_rows:
        texts_by_id.setdefault(str(row.get("chunk_id", "")), []).append(str(row.get("text_raw", "") or ""))
    records = []
    for qa in qas:
        qid = str(qa.get("qa_id", "") or "").strip()
        evidence = [
            {"chunk_id": cid, "text_raw": text}
            for cid in [str(c).strip() for c in (gold_by_qa.get(qid, []) or []) if str(c).strip()]
            for text in texts_by_id.get(cid, [])
        ]
        records.append(build_context_record(qa, evidence, policy="gold_assisted", max_seq_len=max_seq_len))
    return records, {"kept_records": len(records), "chunk_report": chunk_report, "policy": "gold_assisted"}


def recipe_to_train_config(cfg: Any, device: str = "cuda:0") -> Any:
    """Adapt a resolved algorithm+runtime candidate into GeneratorTrainConfig.

    Returns the existing typed GeneratorTrainConfig (never a parallel config
    type). Epochs travel through the trainer's existing ``epochs`` argument,
    so callers must also apply ``cfg.algorithm.generator.num_train_epochs``.
    """
    from src.task2.generation.config import GeneratorTrainConfig

    algo = cfg.algorithm
    runtime = cfg.runtime
    return GeneratorTrainConfig(
        model_id=algo.models.generator.id,
        max_seq_len=int(algo.generator.max_seq_len),
        batch_size=int(runtime.generator_runtime.per_device_train_batch_size),
        grad_accum=int(runtime.generator_runtime.gradient_accumulation_steps),
        learning_rate=float(algo.generator.learning_rate),
        lora_r=int(algo.generator.lora_r),
        lora_alpha=int(algo.generator.lora_alpha),
        lora_dropout=float(algo.generator.lora_dropout),
        target_modules=tuple(algo.generator.target_modules),
        activation_offloading=bool(runtime.generator_runtime.activation_offloading),
        use_liger_fused_ce=bool(algo.generator.use_liger_fused_ce),
        device=device,
        quantization=str(algo.generator.quantization),
        double_quant=bool(algo.generator.double_quant),
        compute_dtype=str(runtime.generator_runtime.compute_dtype),
        optimizer="paged_adamw_8bit",
        gradient_checkpointing=bool(algo.generator.gradient_checkpointing),
        completion_only_loss=bool(algo.generator.completion_only_loss),
        trainer_n_gpu=1,
        warmup_ratio=float(getattr(algo.generator, "warmup_ratio", 0.05)),
        lr_scheduler_type=str(getattr(algo.generator, "lr_scheduler_type", "cosine")),
    )


def recipe_epochs(cfg: Any) -> int:
    """Requested full-scope epochs for the candidate (no silent reduction)."""
    return int(cfg.algorithm.generator.num_train_epochs)
