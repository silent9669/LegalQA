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

from src.task2.pipeline.profiles import ExecutionProfile
from src.task2.production_config import ProductionSelection
from src.task2.generation.config import GeneratorTrainConfig
from src.task2.generation.trainer import train_generator_qlora
from src.task2.generation.memory import cleanup_cuda_stage

logger = logging.getLogger(__name__)


def run_pipeline(
    *,
    profile: ExecutionProfile,
    paths: Dict[str, Any],
    production_cfg: ProductionSelection,
    gen_device: str = "cuda:0",
    retrieval_device: str = "cuda:1",
    output_dir: str = "/kaggle/working",
    seed: int = 42,
    code_root: Optional[str] = None,
    allow_single_gpu: bool = False,
) -> Dict[str, Any]:
    """Execute all stages for the specified profile."""
    os.makedirs(output_dir, exist_ok=True)
    results: Dict[str, Any] = {
        "profile": profile.name,
        "runtime_api_version": 16,
        "stages": {},
    }

    data_dir = paths["data_dir"]
    bm25_dir = paths["bm25_dir"]
    dek21_dir = paths["dek21_dir"]
    model_path = paths["qwen_model_path"]
    test_path = paths.get("public_test_path")

    qa_path = os.path.join(data_dir, "qa_unique.parquet")
    chunks_path = os.path.join(data_dir, "legal_chunks.parquet")
    known_qa_path = os.path.join(data_dir, "known_qa.json")
    labels_path = os.path.join(data_dir, "retrieval_labels.parquet")

    # -------------------------------------------------------------
    # Stage 1: Preflight & Dense Index Probe
    # -------------------------------------------------------------
    print(f"\n[Stage 1] Preflight & Dense Index Probe for profile '{profile.name}'...")
    from scripts.preflight_kaggle import run_preflight_checks
    from src.common.dense import DenseRetriever

    is_final = profile.name in ("final_train_and_submit", "reuse_final_checkpoints_and_submit")
    cfg_root = code_root or "."
    pipeline_cfg = os.path.join(cfg_root, "configs/pipeline.yaml")
    models_cfg = os.path.join(cfg_root, "configs/models.yaml")
    prod_cfg_path = os.path.join(cfg_root, "configs/production_selection.yaml")

    preflight_res = run_preflight_checks(
        pipeline_config_path=pipeline_cfg if os.path.exists(pipeline_cfg) else "configs/pipeline.yaml",
        models_config_path=models_cfg if os.path.exists(models_cfg) else "configs/models.yaml",
        production_config_path=prod_cfg_path if os.path.exists(prod_cfg_path) else "configs/production_selection.yaml",
        require_cuda=torch.cuda.is_available(),
        expected_gpu_count=2 if not allow_single_gpu else 1,
        allow_single_gpu=allow_single_gpu,
        check_dataset_files=True,
        check_indexes=True,
        require_public=is_final,
        data_dir=data_dir,
        bm25_dir=bm25_dir,
        dek21_dir=dek21_dir,
        public_path=test_path,
        stack="stack_a",
        require_training_files=profile.run_reranker_training,
        verify_dense_hash=is_final,
    )

    if not preflight_res["passed"]:
        raise RuntimeError(f"PREFLIGHT FAILED: {preflight_res['errors']}")

    print("Executing pre-training strict Dense index probe...")
    probe_dense = DenseRetriever.load_index(
        dek21_dir,
        corpus_path=chunks_path,
        device=retrieval_device,
        expected_model_name="CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2",
        expected_dtype="float16",
        final_mode=True,
    )
    print(f"Dense DEk21 probe successful: {probe_dense.corpus_embeddings.shape} on {retrieval_device}")
    del probe_dense
    cleanup_cuda_stage(devices=(0, 1))

    # -------------------------------------------------------------
    # Stage 2: Load QA Memory & BM25 Index
    # -------------------------------------------------------------
    print("\n[Stage 2] Loading QA Memory & BM25 Index (mmap)...")
    from src.task2.qa_memory import QAMemory
    from src.common.bm25 import BM25Retriever

    memory = QAMemory.load(known_qa_path, qa_path)
    bm25 = BM25Retriever.load(bm25_dir, corpus_path=chunks_path, fail_on_missing_index=True)
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

        if profile.name == "final_train_and_submit":
            assert_final_checkpoint(reranker_checkpoint, expected_base_model="BAAI/bge-reranker-v2-m3", component_name="reranker")
    elif profile.reuse_existing_checkpoints and production_cfg.use_task_tuned_reranker:
        import glob
        cands = glob.glob("/kaggle/input/**/checkpoints/reranker/best", recursive=True) or glob.glob("checkpoints/reranker/best", recursive=True)
        if not cands:
            raise FileNotFoundError("Reusing checkpoints requested but no reranker checkpoint found!")
        reranker_checkpoint = cands[0]
        assert_final_checkpoint(reranker_checkpoint, expected_base_model="BAAI/bge-reranker-v2-m3", component_name="reranker")

    # -------------------------------------------------------------
    # Stage 4: Qwen2.5-3B QLoRA SFT Fine-Tuning (Liger Backend)
    # -------------------------------------------------------------
    adapter_path: Optional[str] = None
    if profile.run_generator_training:
        print(f"\n[Stage 4] Starting QLoRA training on {gen_device} (profile: {profile.name})...")
        qlora_out = os.path.join(output_dir, "checkpoints/generator/hf_adapter")
        gen_cfg = GeneratorTrainConfig(
            model_id=model_path,
            max_seq_len=2048,
            activation_offloading=True,
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
            val_fold=profile.val_fold,
            max_steps=profile.max_generator_steps,
            max_train_examples=profile.max_generator_examples,
            probe_mode=profile.probe_selection,
            device=gen_device,
            fail_on_error=True,
            seed=seed,
        )
        adapter_path = qlora_out
        results["stages"]["generator"] = res_qlora

        if profile.name == "final_train_and_submit":
            assert_final_checkpoint(adapter_path, expected_base_model=production_cfg.generator_base_model, component_name="generator")
    elif profile.reuse_existing_checkpoints and production_cfg.use_qlora and profile.requires_generator:
        import glob
        ad_cands = glob.glob("/kaggle/input/**/checkpoints/generator/hf_adapter", recursive=True) or glob.glob("checkpoints/generator/hf_adapter", recursive=True)
        if not ad_cands:
            raise FileNotFoundError("Reusing checkpoints requested but no adapter found!")
        adapter_path = ad_cands[0]
        assert_final_checkpoint(adapter_path, expected_base_model=production_cfg.generator_base_model, component_name="generator")

    # -------------------------------------------------------------
    # Stage 5: Parameter Audit
    # -------------------------------------------------------------
    print("\n[Stage 5] Auditing parameter budget for active stack...")
    from scripts.audit_parameters import audit_parameter_budget
    ad_manifest = os.path.join(adapter_path, "generator_manifest.json") if adapter_path else None
    audit_res = audit_parameter_budget(models_cfg if os.path.exists(models_cfg) else "configs/models.yaml", stack="stack_a", adapter_manifest_path=ad_manifest)
    print(f"Total Learned Parameters: {audit_res['total_learned_parameters']:,} (limit: {audit_res['limit']:,})")
    if not audit_res["is_compliant"]:
        raise RuntimeError(f"PARAMETER BUDGET EXCEEDED: {audit_res['total_learned_parameters']:,} >= {audit_res['limit']:,}")
    results["stages"]["audit"] = audit_res

    # -------------------------------------------------------------
    # Stage 6: Dev Evaluation / Protocol-8 Screening
    # -------------------------------------------------------------
    if profile.run_dev_evaluation:
        print(f"\n[Stage 6] Dev Evaluation / Screening (profile: {profile.name})...")
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
                config_path=prod_cfg_path if os.path.exists(prod_cfg_path) else "configs/production_selection.yaml",
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
                fail_on_fallback=True,
                seed=seed,
            )
            results["stages"]["evaluation"] = eval_res

    # -------------------------------------------------------------
    # Stage 7: Inference Pipeline & Public Submission
    # -------------------------------------------------------------
    if profile.run_public_inference:
        print("\n[Stage 7] Loading Inference Pipeline and predicting public test set...")
        from src.task2.predict import LegalQAPipeline
        from src.common.dense import DenseRetriever
        from src.common.reranker import BGEReranker
        from src.task2.evidence_packer import EvidencePacker
        from src.task2.generator import QwenGenerator
        from src.task2.selector import CandidateSelector

        cleanup_cuda_stage(devices=(0, 1))

        dense = DenseRetriever.load_index(
            dek21_dir,
            corpus_path=chunks_path,
            device=retrieval_device,
            expected_model_name="CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2",
            expected_dtype="float16",
            final_mode=True,
        )
        reranker = BGEReranker(model_name=reranker_checkpoint, device=retrieval_device)
        packer = EvidencePacker(bm25.corpus)

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
            )

        selector = CandidateSelector(
            policy=production_cfg.candidate_policy,
            best_fixed_candidate=production_cfg.best_fixed_candidate or "stitched_extract",
        )

        pipeline = LegalQAPipeline(memory, bm25, dense, reranker, packer, generator, selector)

        if not test_path or not os.path.exists(test_path):
            raise FileNotFoundError(f"Public test set not found at: {test_path}")

        with open(test_path, "r", encoding="utf-8") as f:
            public_test = json.load(f)

        items_to_predict = [{"id": str(qid), "question": str(item.get("question", "")).strip()} for qid, item in public_test.items()]
        batch_size_gen = 4 if torch.cuda.is_available() else 1
        submission = pipeline.predict_batch(
            items=items_to_predict,
            max_new_tokens=production_cfg.max_new_tokens,
            retrieval_batch_size=32,
            reranker_batch_size=32,
            generation_batch_size=batch_size_gen,
        )

        # Verification
        assert len(submission) == 1000, f"Submission count mismatch! Expected 1000, got {len(submission)}"
        assert set(public_test.keys()) == set(submission.keys()), "Submission ID keys mismatch!"

        out_json = os.path.join(output_dir, "submission.json")
        out_zip = os.path.join(output_dir, "submission.json.zip")

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(submission, f, ensure_ascii=False, indent=2)

        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(out_json, arcname="submission.json")

        results["stages"]["submission"] = {
            "submission_json": out_json,
            "submission_zip": out_zip,
            "num_queries": len(submission),
        }
        print(f"SUCCESS: Submission saved to {out_zip}")

    print(f"\nPipeline execution for profile '{profile.name}' complete!")
    return results
