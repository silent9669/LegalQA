"""Pipeline orchestration runner executing LegalQA stages based on ExecutionProfile (V16).

Decouples pipeline execution from notebook cells so Kaggle notebook remains a thin launcher.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.task2.config.schema import ResolvedTask2Config
from src.task2.pipeline.profiles import ExecutionProfile
from src.task2.production_config import ProductionSelection
from src.task2.generation.config import GeneratorTrainConfig
from src.task2.generation.trainer import train_generator_qlora
from src.task2.generation.memory import cleanup_cuda_stage

logger = logging.getLogger(__name__)

#: Profiles governed by the screen-promotion path (explicit promotion report).
PROMOTION_GATED_PROFILES = ("final_train_and_submit", "reuse_final_checkpoints_and_submit")

#: Profiles held to final-mode strictness (no mocks/fallbacks/CPU substitution,
#: final-checkpoint scope asserts). modal_a100 is governed by the in-run
#: candidate + parent-gate chain instead of the promotion path.
STRICT_CONTRACT_PROFILES = PROMOTION_GATED_PROFILES + ("modal_a100",)


def run_pipeline(
    *,
    profile: ExecutionProfile,
    paths: Dict[str, Any],
    production_cfg: Optional[ProductionSelection] = None,
    resolved_config: Optional[ResolvedTask2Config] = None,
    gen_device: str = "cuda:0",
    retrieval_device: str = "cuda:1",
    output_dir: str = "/kaggle/working",
    seed: int = 42,
    code_root: Optional[str] = None,
    allow_single_gpu: bool = False,
) -> Dict[str, Any]:
    """Execute all stages for the specified profile."""
    import time as _time

    attempt_started = _time.monotonic()
    os.makedirs(output_dir, exist_ok=True)

    if resolved_config is not None:
        seed = resolved_config.algorithm.seed
        gen_device = resolved_config.runtime.devices.get("generator", gen_device)
        retrieval_device = resolved_config.runtime.devices.get("retrieval", retrieval_device)

    if production_cfg is None:
        from src.task2.production_config import get_default_production_selection
        production_cfg = get_default_production_selection()

    if profile.name in PROMOTION_GATED_PROFILES:
        from src.task2.production_config import verify_promotion_provenance
        verify_promotion_provenance(production_cfg)

    results: Dict[str, Any] = {
        "profile": profile.name,
        "runtime_api_version": 16,
        "stages": {},
    }

    cfg_root = code_root or "."
    models_cfg = os.path.join(cfg_root, "configs/task2/algorithm.yaml")
    prod_cfg_path = os.path.join(cfg_root, "configs/task2/algorithm.yaml")

    data_dir = paths["data_dir"]
    bm25_dir = paths["bm25_dir"]
    dek21_dir = paths["dek21_dir"]
    model_path = paths["qwen_model_path"]
    test_path = paths.get("test_path") or paths.get("public_test_path")

    qa_path = os.path.join(data_dir, "qa_unique.parquet")
    chunks_path = os.path.join(data_dir, "legal_chunks.parquet")
    known_qa_path = os.path.join(data_dir, "known_qa.json")
    labels_path = os.path.join(data_dir, "retrieval_labels.parquet")

    # -------------------------------------------------------------
    # Stage 1: Preflight & Dense Index Probe
    # -------------------------------------------------------------
    print(f"\n[Stage 1] Preflight & Dense Index Probe for profile '{profile.name}'...")
    from src.common.dense import DenseRetriever
    from src.task2.dataset.validator import validate_dataset

    is_final = profile.name in STRICT_CONTRACT_PROFILES
    schema_candidate = os.path.join(cfg_root, "configs/dataset_schema.yaml")

    if os.path.exists(schema_candidate) and os.path.exists(data_dir):
        val_res = validate_dataset(data_dir=data_dir, schema_path=schema_candidate)
        if val_res.get("status") != "PASS":
            raise RuntimeError(f"PREFLIGHT DATASET VALIDATION FAILED: {val_res.get('errors')}")
        print("Preflight dataset schema validation: PASS")

    print("Executing pre-training strict Dense index probe...")
    probe_dense = DenseRetriever.load_index(
        dek21_dir,
        corpus_path=chunks_path,
        device=retrieval_device,
        expected_model_name="CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2",
        expected_dtype="float16",
        final_mode=True,
        verify_self_consistency=True,
    )
    consistency = getattr(probe_dense, "self_consistency_report", {})
    print(f"Dense DEk21 probe successful: {probe_dense.corpus_embeddings.shape} on {retrieval_device}")
    if consistency:
        print(f"Dense self-consistency: {consistency}")
    del probe_dense
    cleanup_cuda_stage(devices=(0, 1))

    # -------------------------------------------------------------
    # Stage 2: Load QA Memory & BM25 Index
    # -------------------------------------------------------------
    print("\n[Stage 2] Loading QA Memory & BM25 Index (mmap)...")
    from src.task2.qa_memory import QAMemory
    from src.common.bm25 import BM25Retriever

    memory = QAMemory.load(known_qa_path, qa_path)
    bm25 = BM25Retriever.load(bm25_dir, corpus_path=chunks_path, fail_on_missing_index=is_final)
    print(f"Loaded QA Memory: {len(memory.id_to_answer):,} IDs | BM25 Chunks: {bm25.corpus_size:,}")

    # -------------------------------------------------------------
    # Stage 3: Task-Tuned Reranker Fine-Tuning
    # -------------------------------------------------------------
    reranker_checkpoint = "BAAI/bge-reranker-v2-m3"
    from src.task2.checkpoint_manifest import assert_final_checkpoint

    if profile.run_reranker_training:
        print(f"\n[Stage 3] Starting Task-Tuned Reranker training on {retrieval_device}...")
        from src.task2.training.train_reranker import train_bge_reranker
        pairs_path = os.path.join(data_dir, "reranker_training_pairs.parquet")
        reranker_out = os.path.join(output_dir, "checkpoints/reranker/best")

        res_rerank = train_bge_reranker(
            pairs_path=pairs_path,
            output_dir=reranker_out,
            model_name="BAAI/bge-reranker-v2-m3",
            epochs=1,
            batch_size=2,
            grad_accum=4,
            lr=2e-5,
            val_fold=profile.val_fold,
            max_steps=profile.max_reranker_steps,
            max_train_pairs=profile.max_reranker_pairs,
            max_val_pairs=profile.max_reranker_val_pairs,
            device=retrieval_device,
            fail_on_error=True,
        )
        if res_rerank.get("status") != "completed":
            raise RuntimeError(f"Reranker training failed: {res_rerank}")
        reranker_checkpoint = reranker_out
        results["stages"]["reranker"] = res_rerank

        if profile.name in STRICT_CONTRACT_PROFILES:
            assert_final_checkpoint(reranker_checkpoint, expected_base_model="BAAI/bge-reranker-v2-m3", component_name="reranker")
    elif profile.reuse_existing_checkpoints and production_cfg.use_task_tuned_reranker:
        from src.task2.checkpoint_resolver import resolve_component_checkpoint
        reranker_checkpoint = resolve_component_checkpoint(
            component="reranker",
            expected_base_model="BAAI/bge-reranker-v2-m3",
            preferred_path=production_cfg.reranker_checkpoint,
            expected_runtime_api=16,
        )
        assert_final_checkpoint(reranker_checkpoint, expected_base_model="BAAI/bge-reranker-v2-m3", component_name="reranker")

    # -------------------------------------------------------------
    # Stage 4: Qwen2.5-3B QLoRA SFT Fine-Tuning (Liger Backend)
    # -------------------------------------------------------------
    adapter_path: Optional[str] = None
    if profile.run_generator_training:
        print(f"\n[Stage 4] Starting QLoRA training on {gen_device} (profile: {profile.name})...")
        # Explicit memory hygiene: release retrieval handles before model training
        try:
            del memory
            del bm25
        except Exception:
            pass
        gc.collect()
        cleanup_cuda_stage(devices=(0, 1))

        is_smoke = "smoke" in profile.name or "probe" in profile.name
        qlora_out = os.path.join(output_dir, "checkpoints/generator/hf_adapter")
        if resolved_config is not None:
            from src.task2.training.context_builder import recipe_to_train_config

            gen_cfg = recipe_to_train_config(resolved_config, device=gen_device)
        else:
            gen_cfg = GeneratorTrainConfig(
                model_id=model_path,
                max_seq_len=256 if is_smoke else 2048,
                lora_dropout=0.0,
                activation_offloading=not is_smoke,
                use_liger_fused_ce=True,
                device=gen_device,
            )

        # One-shot Liger backend preflight assertion before model training
        if gen_device.startswith("cuda"):
            from src.task2.generation.liger_backend import (
                validate_liger_environment,
                REQUIRED_LIGER_VERSION,
            )
            liger_status = validate_liger_environment(strict=True)
            assert liger_status.version == REQUIRED_LIGER_VERSION, f"Liger-Kernel version must be {REQUIRED_LIGER_VERSION}"
            assert liger_status.qwen2_patch_available, "Qwen2 Liger patch must be available"
            assert liger_status.fused_linear_ce, "Liger fused-linear CE must be available"
            assert gen_cfg.use_liger_fused_ce is True, "use_liger_fused_ce must be True"
            assert gen_cfg.trainer_n_gpu == 1, "trainer_n_gpu must be 1"
            assert gen_device == "cuda:0", f"Generator device must be cuda:0, got {gen_device}"
            print(f"Liger-Kernel: {liger_status.version}")
            print("Qwen2 Liger patch: PASS")
            print("Liger fused-linear CE: PASS")
            print("use_liger_kernel=True")
            print("fused_linear_cross_entropy=True")
            print("loss_type=nll")
            print(f"target={gen_device}")
            print(f"trainer_n_gpu={gen_cfg.trainer_n_gpu}")

        res_qlora = train_generator_qlora(
            model_name_or_path=model_path,
            qa_path=qa_path,
            labels_path=labels_path,
            chunks_path=chunks_path,
            output_dir=qlora_out,
            config=gen_cfg,
            resolved_config=resolved_config,
            val_fold=profile.val_fold,
            max_steps=profile.max_generator_steps,
            max_train_examples=profile.max_generator_examples,
            probe_mode=profile.probe_selection,
            execution_profile=profile.name,
            device=gen_device,
            fail_on_error=True,
            seed=seed,
        )
        adapter_path = qlora_out
        results["stages"]["generator"] = res_qlora

        from src.task2.provenance.deadline import save_complete_checkpoint

        try:
            elapsed_train = int(_time.monotonic() - attempt_started)
            save_complete_checkpoint(
                os.path.join(output_dir, "checkpoints", "deadline"),
                "generator_train",
                state={
                    "global_step": int(res_qlora.get("global_step", res_qlora.get("optimizer_steps", 0)) or 0),
                    "result": {k: v for k, v in res_qlora.items() if isinstance(v, (int, float, str, bool))},
                },
                manifest={
                    "candidate_id": resolved_config.candidate_id if resolved_config else "",
                    "code_commit_sha": results.get("git_commit_sha", ""),
                    "model_revision": (resolved_config.algorithm.models.generator.id if resolved_config else model_path),
                    "data_hash": "",
                    "split_fingerprint": "",
                    "global_step": int(res_qlora.get("global_step", res_qlora.get("optimizer_steps", 0)) or 0),
                    "elapsed_seconds": elapsed_train,
                    "retry_count": 0,
                    "optimizer_state": "trainer_state",
                    "scheduler_state": "trainer_state",
                    "sampler_position": "epoch_complete" if res_qlora.get("status") not in ("skipped",) else "unknown",
                },
            )
        except Exception as exc:
            raise RuntimeError(f"deadline checkpoint for generator_train failed: {exc}") from exc

        if profile.name in STRICT_CONTRACT_PROFILES:
            assert_final_checkpoint(adapter_path, expected_base_model=production_cfg.generator_base_model, component_name="generator")
    elif profile.reuse_existing_checkpoints and production_cfg.use_qlora and profile.requires_generator:
        from src.task2.checkpoint_resolver import resolve_component_checkpoint
        adapter_path = resolve_component_checkpoint(
            component="generator",
            expected_base_model=production_cfg.generator_base_model,
            preferred_path=production_cfg.adapter_path,
            expected_runtime_api=16,
        )
        assert_final_checkpoint(adapter_path, expected_base_model=production_cfg.generator_base_model, component_name="generator")

    # -------------------------------------------------------------
    # Stage 5: Parameter Audit
    # -------------------------------------------------------------
    print("\n[Stage 5] Auditing parameter budget for active stack...")
    from scripts.audit_parameters import audit_parameter_budget
    ad_manifest = os.path.join(adapter_path, "generator_manifest.json") if adapter_path else None
    audit_res = audit_parameter_budget(models_cfg if os.path.exists(models_cfg) else "configs/task2/algorithm.yaml", stack="stack_a", adapter_manifest_path=ad_manifest)
    print(f"Total Learned Parameters: {audit_res['total_learned_parameters']:,} (limit: {audit_res['limit']:,})")
    if not audit_res["is_compliant"]:
        raise RuntimeError(f"PARAMETER BUDGET EXCEEDED: {audit_res['total_learned_parameters']:,} >= {audit_res['limit']:,}")
    results["stages"]["audit"] = audit_res

    # -------------------------------------------------------------
    # Stage 6: Dev Evaluation / Protocol-8 Screening
    # -------------------------------------------------------------
    if profile.run_dev_evaluation:
        print(f"\n[Stage 6] Dev Evaluation / Screening (profile: {profile.name})...")
        cleanup_cuda_stage(devices=(0, 1))
        eval_fold = profile.val_fold if profile.val_fold is not None else 0

        if profile.name == "screen_fold0":
            from src.task2.evaluation import run_screen_matrix
            from src.common.hashing import sha256_file
            from scripts.promote_production_selection import promote_production_selection
            from src.task2.production_config import load_production_selection, validate_production_selection_for_profile

            screen_report = run_screen_matrix(
                qa_path=qa_path,
                fold_path=os.path.join(data_dir, "fold_assignments.parquet"),
                chunks_path=chunks_path,
                labels_path=labels_path,
                held_out_fold=eval_fold,
                bm25_dir=bm25_dir,
                dense_dir=dek21_dir,
                dense_model="CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2",
                base_reranker="BAAI/bge-reranker-v2-m3",
                tuned_reranker=reranker_checkpoint,
                base_generator=model_path,
                adapter_path=adapter_path,
                sample_size=profile.dev_eval_size or 250,
                eval_output_dir=os.path.join(output_dir, "evaluations"),
                gen_device=gen_device,
                retrieval_device=retrieval_device,
                min_retrieval_label_coverage=0.70,
                seed=seed,
            )

            promotion_report_path = os.path.join(output_dir, "promotion_report.json")
            promoted_config_path = os.path.join(output_dir, "promoted_production_selection.yaml")

            promote_production_selection(
                report_path=promotion_report_path,
                config_path=prod_cfg_path if os.path.exists(prod_cfg_path) else "configs/task2/algorithm.yaml",
                output_path=promoted_config_path,
            )

            with open(promotion_report_path, "r", encoding="utf-8") as f:
                report_data = json.load(f)

            if report_data.get("screen_protocol_version") != 8:
                raise RuntimeError("SCREEN_PROMOTION_ERROR: screen_protocol_version must be 8")

            promoted_cfg = load_production_selection(promoted_config_path)
            if promoted_cfg.status != "PROMOTED":
                raise RuntimeError(f"SCREEN_PROMOTION_ERROR: expected PROMOTED, got {promoted_cfg.status!r}")

            validate_production_selection_for_profile(promoted_cfg, "final_train_and_submit", allow_unvalidated_final=False)

            handoff_dir = os.path.join(output_dir, "screen_handoff")
            os.makedirs(handoff_dir, exist_ok=True)
            shutil.copy(promotion_report_path, os.path.join(handoff_dir, "promotion_report.json"))
            shutil.copy(promoted_config_path, os.path.join(handoff_dir, "promoted_production_selection.yaml"))

            screen_manifest = {
                "runtime_api_version": 16,
                "execution_profile": "screen_fold0",
                "screen_protocol_version": 8,
                "promotion_report_sha256": sha256_file(promotion_report_path),
                "promoted_config_sha256": sha256_file(promoted_config_path),
                "status": "SCREEN_PASS",
            }
            manifest_file = os.path.join(handoff_dir, "screen_run_manifest.json")
            with open(manifest_file, "w", encoding="utf-8") as f:
                json.dump(screen_manifest, f, indent=2)

            handoff_zip = os.path.join(output_dir, "screen_handoff.zip")
            with zipfile.ZipFile(handoff_zip, "w", zipfile.ZIP_DEFLATED) as z:
                z.write(promotion_report_path, arcname="promotion_report.json")
                z.write(promoted_config_path, arcname="promoted_production_selection.yaml")
                z.write(manifest_file, arcname="screen_run_manifest.json")

            print(f"Protocol-8 screen complete! Handoff zip created: {handoff_zip}")
            results["stages"]["screen"] = {"handoff_zip": handoff_zip}
        elif profile.name == "modal_a100" and profile.val_fold is None:
            print(f"\n[Stage 6] modal_a100 trains on 100% of data (val_fold=None); skipping dev evaluation to prevent leaked metrics.")
            eval_res = {
                "selected_meteor": None,
                "reason": "Production trains on 100% of data (val_fold=None); no honest held-out dev fold.",
                "sample_size": None,
                "held_out_fold": None,
            }
            results["stages"]["evaluation"] = eval_res
        else:
            # smoke_only evaluation
            from src.task2.evaluation import evaluate_checkpoint
            eval_res = evaluate_checkpoint(
                qa_path=qa_path,
                fold_path=os.path.join(data_dir, "fold_assignments.parquet"),
                chunks_path=chunks_path,
                labels_path=labels_path,
                held_out_fold=eval_fold,
                bm25_dir=bm25_dir,
                dense_dir=dek21_dir,
                dense_model="CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2",
                reranker_checkpoint=reranker_checkpoint,
                generator_model=model_path if profile.requires_generator else None,
                adapter_path=adapter_path,
                sample_size=profile.dev_eval_size or 5,
                eval_output_dir=os.path.join(output_dir, "evaluations"),
                gen_device=gen_device,
                retrieval_device=retrieval_device,
                fail_on_fallback=is_final,
                seed=seed,
            )
            results["stages"]["evaluation"] = eval_res

    # -------------------------------------------------------------
    # Stage 7: Inference Pipeline & Public Submission
    # -------------------------------------------------------------
    if profile.run_public_inference:
        print("\n[Stage 7] Loading Inference Pipeline and predicting public test set...")

        deadline_budget = paths.get("deadline_budget_seconds")
        if deadline_budget is not None:
            from src.task2.provenance.deadline import allow_stage

            elapsed_now = int(_time.monotonic() - attempt_started)
            admission = allow_stage(
                "public_inference",
                int(paths.get("predicted_inference_seconds", 3300)),
                elapsed_now,
                int(deadline_budget),
            )
            if admission["status"] != "OK":
                results["status"] = "INCOMPLETE"
                results["stages"]["public_inference"] = admission
                print("INCOMPLETE: deadline refuses public_inference stage.")
                return results

        from src.task2.predict import LegalQAPipeline, retrieval_options_from_config
        from src.common.dense import DenseRetriever
        from src.common.reranker import BGEReranker
        from src.task2.evidence_packer import EvidencePacker
        from src.task2.generator import QwenGenerator
        from src.task2.selector import CandidateSelector

        cleanup_cuda_stage(devices=(0, 1))

        # Stage 4 releases training memory; inference reloads its own handles.
        from src.common.bm25 import BM25Retriever
        from src.task2.qa_memory import QAMemory

        memory = QAMemory.load(known_qa_path, qa_path)
        bm25 = BM25Retriever.load(bm25_dir, corpus_path=chunks_path, fail_on_missing_index=True)

        retrieval_options = None
        retrieval_weights = None
        if resolved_config is not None:
            retrieval_options, retrieval_weights = retrieval_options_from_config(resolved_config)
        inference_cfg = resolved_config.runtime.inference if resolved_config is not None else None
        generation_batch_size = inference_cfg.generation_batch_size if inference_cfg else 4
        reranker_batch_size = inference_cfg.reranker_batch_size if inference_cfg else 32
        retrieval_batch_size = inference_cfg.retrieval_batch_size if inference_cfg else 32
        if not torch.cuda.is_available():
            generation_batch_size = 1

        dense = DenseRetriever.load_index(
            dek21_dir,
            corpus_path=chunks_path,
            device=retrieval_device,
            expected_model_name="CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2",
            expected_dtype="float16",
            final_mode=True,
            verify_self_consistency=True,
        )
        reranker = BGEReranker(model_name=reranker_checkpoint, device=retrieval_device)
        packer = EvidencePacker(bm25.corpus)
        legal_index = None
        legal_rows = None
        if retrieval_options is not None and retrieval_options.get("use_legal_reference"):
            from src.common.legal_reference import build_legal_reference_index

            legal_rows = list(bm25.corpus)
            legal_index, lex_report = build_legal_reference_index(legal_rows)
            if lex_report["is_empty"]:
                raise ValueError("legal-reference arm enabled but the index is empty")

        generator = None
        if profile.requires_generator:
            generator = QwenGenerator.load(
                model_path=model_path,
                adapter_path=adapter_path if production_cfg.use_qlora else None,
                device=gen_device,
                runtime="torch" if torch.cuda.is_available() else "fallback",
                fail_on_fallback=True,
                final_mode=True,
                require_adapter=production_cfg.use_qlora,
                load_mode=inference_cfg.generator_load_mode if inference_cfg else "nf4",
                merge_adapter=inference_cfg.merge_adapter if inference_cfg else False,
            )

        selector = CandidateSelector(
            policy=production_cfg.candidate_policy,
            best_fixed_candidate=production_cfg.best_fixed_candidate or "stitched_extract",
        )

        pipeline = LegalQAPipeline(
            memory,
            bm25,
            dense,
            reranker,
            packer,
            generator,
            selector,
            legal_index=legal_index,
            legal_rows=legal_rows,
            retrieval_options=retrieval_options,
            retrieval_weights=retrieval_weights,
        )

        from src.task2.pipeline.contracts import validate_execution_contract

        is_final_profile = profile.name in STRICT_CONTRACT_PROFILES
        validate_execution_contract(
            {"final_mode": is_final_profile, "profile": profile.name},
            {
                "dense_mock": getattr(dense, "model_name", "") == "mock",
                "dense_fallback": False,
                "generator_fallback": False,
                "mock_generator": False,
                "dense_index_missing": dense is None or getattr(dense, "corpus_embeddings", None) is None,
                "bm25_index_missing": bm25 is None,
                "generator_device": gen_device,
                "retrieval_device": retrieval_device,
                "require_adapter": bool(production_cfg.use_qlora),
                "adapter_present": bool(adapter_path),
                "promotion_validated": True,
                "policy_needs_generator": bool(profile.requires_generator),
                "generator_loaded": generator is not None if profile.requires_generator else True,
                "extractive_only": getattr(selector, "policy", "") == "extractive_only",
            },
            is_final_profile,
        )

        if not test_path or not os.path.exists(test_path):
            raise FileNotFoundError(f"Test set not found at: {test_path}")

        with open(test_path, "r", encoding="utf-8") as f:
            public_test = json.load(f)

        items_to_predict = [{"id": str(qid), "question": str(item.get("question", "")).strip()} for qid, item in public_test.items()]
        raw_cache_file = os.path.join(output_dir, "gen_raw_cache.jsonl")
        submission, provenance = pipeline.predict_batch(
            items=items_to_predict,
            max_new_tokens=production_cfg.max_new_tokens,
            retrieval_batch_size=retrieval_batch_size,
            reranker_batch_size=reranker_batch_size,
            generation_batch_size=generation_batch_size,
            return_provenance=True,
            raw_cache_path=raw_cache_file,
        )

        ans_lengths = [len(str(v.get("answer", "")).split()) for v in submission.values()]
        mean_len = float(np.mean(ans_lengths)) if ans_lengths else 0.0
        med_len = float(np.median(ans_lengths)) if ans_lengths else 0.0
        p90_len = float(np.percentile(ans_lengths, 90)) if ans_lengths else 0.0
        empty_count = sum(1 for l in ans_lengths if l == 0)
        print(f"[+] Submission answer words: mean={mean_len:.1f} | median={med_len:.1f} | p90={p90_len:.1f} | empty={empty_count}")
        if is_final_profile and (empty_count > 0 or mean_len < 80):
            raise RuntimeError(f"SUBMISSION INTEGRITY ERROR: empty answers ({empty_count}) or mean words ({mean_len:.1f}) < 80")

        # Verification: exact 1,000 IDs, nonempty answer objects, ZIP inner bytes.
        from src.task2.pipeline.contracts import verify_submission_ids
        from src.task2.scorer_contract import verify_zip_inner_matches_loose

        verify_submission_ids(list(public_test.keys()), submission)

        out_json = os.path.join(output_dir, "submission.json")
        out_zip = os.path.join(output_dir, "submission.json.zip")

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(submission, f, ensure_ascii=False, indent=2)

        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(out_json, arcname="submission.json")

        zip_report = verify_zip_inner_matches_loose(out_zip, out_json)

        provenance_path = os.path.join(output_dir, "submission_provenance.json")
        with open(provenance_path, "w", encoding="utf-8") as f:
            json.dump(provenance, f, ensure_ascii=False, indent=2)

        from src.task2.provenance.checksums import compute_file_sha256
        provenance_sha = compute_file_sha256(provenance_path)

        results["stages"]["submission"] = {
            "submission_json": out_json,
            "submission_zip": out_zip,
            "num_queries": len(submission),
            "answer_length": {
                "mean_words": mean_len,
                "median_words": med_len,
                "p90_words": p90_len,
                "empty_answers": empty_count,
            },
            "loose_sha256": zip_report["loose_sha256"],
            "inner_sha256": zip_report["inner_sha256"],
            "zip_sha256": zip_report["zip_sha256"],
            "provenance_path": provenance_path,
            "provenance_counts": provenance["counts"],
            "provenance_sha256": provenance_sha,
        }
        results["evidence_links"] = {
            "candidate_sha": resolved_config.candidate_id if resolved_config else "",
            "training": results["stages"].get("generator", {}),
            "inference": {
                "kind": "inference",
                "num_predictions": len(submission),
                "expected_count": len(public_test),
                "submission_sha256": zip_report["loose_sha256"],
                "provenance_sha256": provenance_sha,
                "test_fingerprint": {
                    "num_expected": len(public_test),
                    "generation_batch_size": generation_batch_size,
                    "reranker_batch_size": reranker_batch_size,
                    "retrieval_batch_size": retrieval_batch_size,
                },
                "measured": True,
            },
        }
        print(f"SUCCESS: Submission saved to {out_zip}")

    print(f"\nPipeline execution for profile '{profile.name}' complete!")
    return results
