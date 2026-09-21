"""Loader and resolver for Task 2 authoritative configurations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Set, Union
import yaml

from src.task2.config.schema import (
    AlgorithmConfig,
    EvaluationConfig,
    FinalTrainingConfig,
    GeneratorAlgorithmConfig,
    GeneratorRuntimeConfig,
    InferenceRuntimeConfig,
    MicroProbeConfig,
    ModelEntry,
    ModelsConfig,
    ProductionRuntimeConfig,
    ResolvedTask2Config,
    RetrievalConfig,
    RuntimeConfig,
    SmokeConfig,
)

# Protected algorithm field names that must never be set in runtime profiles
PROTECTED_ALGORITHM_KEYS: Set[str] = {
    "schema_version",
    "seed",
    "models",
    "generator",
    "final_training",
    "evaluation",
    "retrieval",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "target_modules",
    "learning_rate",
    "lr_scheduler_type",
    "warmup_ratio",
    "effective_batch_size",
    "num_train_epochs",
    "completion_only_loss",
    "use_liger_fused_ce",
    "gradient_checkpointing",
    "max_seq_len",
    "quantization",
    "double_quant",
}

ALLOWED_ALGORITHM_TOP_KEYS: Set[str] = {
    "schema_version",
    "seed",
    "models",
    "generator",
    "final_training",
    "evaluation",
    "retrieval",
}

ALLOWED_RUNTIME_TOP_KEYS: Set[str] = {
    "profile_name",
    "required_gpu_count",
    "required_gpu_name_contains",
    "devices",
    "generator_runtime",
    "smoke",
    "a100_micro_probe",
    "production",
    "outputs",
    "inference",
}


def canonical_json_dumps(data: Any) -> str:
    """Produce deterministic canonical JSON string."""
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_sha256(data: Any) -> str:
    """Compute SHA-256 hash over canonical JSON representation."""
    canonical = canonical_json_dumps(data)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse_algorithm_config(raw: Dict[str, Any]) -> AlgorithmConfig:
    """Validate and construct AlgorithmConfig dataclass."""
    for k in raw:
        if k not in ALLOWED_ALGORITHM_TOP_KEYS:
            raise ValueError(f"Unknown algorithm key: {k}")

    req_keys = {"schema_version", "seed", "models", "generator", "final_training", "evaluation"}
    missing = req_keys - set(raw.keys())
    if missing:
        raise ValueError(f"Missing required algorithm keys: {missing}")
    models_raw = raw["models"]
    models = ModelsConfig(
        generator=ModelEntry(**models_raw["generator"]),
        reranker=ModelEntry(**models_raw["reranker"]),
        dense=ModelEntry(**models_raw["dense"]),
    )

    gen_raw = raw["generator"]
    generator = GeneratorAlgorithmConfig(
        max_seq_len=int(gen_raw.get("max_seq_len", 2048)),
        quantization=str(gen_raw.get("quantization", "4bit_nf4")),
        double_quant=bool(gen_raw.get("double_quant", True)),
        lora_r=int(gen_raw.get("lora_r", 16)),
        lora_alpha=int(gen_raw.get("lora_alpha", 32)),
        lora_dropout=float(gen_raw.get("lora_dropout", 0.0)),
        target_modules=list(gen_raw.get("target_modules", [])),
        learning_rate=float(gen_raw.get("learning_rate", 1.0e-4)),
        lr_scheduler_type=str(gen_raw.get("lr_scheduler_type", "cosine")),
        warmup_ratio=float(gen_raw.get("warmup_ratio", 0.05)),
        effective_batch_size=int(gen_raw.get("effective_batch_size", 8)),
        num_train_epochs=int(gen_raw.get("num_train_epochs", 3)),
        completion_only_loss=bool(gen_raw.get("completion_only_loss", True)),
        use_liger_fused_ce=bool(gen_raw.get("use_liger_fused_ce", True)),
        gradient_checkpointing=bool(gen_raw.get("gradient_checkpointing", True)),
    )

    ft_raw = raw["final_training"]
    final_training = FinalTrainingConfig(
        training_scope=str(ft_raw.get("training_scope", "all_allowed_train")),
        val_fold=ft_raw.get("val_fold"),
    )

    eval_raw = raw["evaluation"]
    evaluation = EvaluationConfig(
        primary_metric=str(eval_raw.get("primary_metric", "whitespace_meteor")),
        secondary_metric=str(eval_raw.get("secondary_metric", "rouge_l")),
    )

    ret_raw = raw.get("retrieval", {}) or {}
    if not isinstance(ret_raw, dict):
        raise ValueError("Algorithm 'retrieval' section must be a mapping")
    retrieval = RetrievalConfig(
        rrf_k=int(ret_raw.get("rrf_k", 60)),
        candidate_pool=int(ret_raw.get("candidate_pool", 50)),
        use_legal_reference=bool(ret_raw.get("use_legal_reference", False)),
        use_acronyms=bool(ret_raw.get("use_acronyms", False)),
        use_weighted_rrf=bool(ret_raw.get("use_weighted_rrf", False)),
        diversify_context=bool(ret_raw.get("diversify_context", False)),
        lost_in_middle=bool(ret_raw.get("lost_in_middle", False)),
        max_parts_per_article=int(ret_raw.get("max_parts_per_article", 2)),
        w_bm25_plain=float(ret_raw.get("w_bm25_plain", 0.5)),
        w_dense_plain=float(ret_raw.get("w_dense_plain", 0.5)),
        w_bm25_lex=float(ret_raw.get("w_bm25_lex", 1.0 / 3.0)),
        w_dense_lex=float(ret_raw.get("w_dense_lex", 1.0 / 3.0)),
        w_lexref_lex=float(ret_raw.get("w_lexref_lex", 1.0 / 3.0)),
    )
    if retrieval.rrf_k <= 0 or retrieval.candidate_pool <= 0 or retrieval.max_parts_per_article < 1:
        raise ValueError("retrieval rrf_k/candidate_pool must be positive, max_parts_per_article >= 1")
    for w in (retrieval.w_bm25_plain, retrieval.w_dense_plain, retrieval.w_bm25_lex,
              retrieval.w_dense_lex, retrieval.w_lexref_lex):
        if w < 0:
            raise ValueError("retrieval RRF weights must be non-negative")
    if abs(retrieval.w_bm25_plain + retrieval.w_dense_plain - 1.0) > 1e-6:
        raise ValueError("retrieval plain weights must sum to 1.0")
    if abs(retrieval.w_bm25_lex + retrieval.w_dense_lex + retrieval.w_lexref_lex - 1.0) > 1e-6:
        raise ValueError("retrieval lex weights must sum to 1.0")

    return AlgorithmConfig(
        schema_version=int(raw["schema_version"]),
        seed=int(raw["seed"]),
        models=models,
        generator=generator,
        final_training=final_training,
        evaluation=evaluation,
        retrieval=retrieval,
    )


def _parse_runtime_config(raw: Dict[str, Any]) -> RuntimeConfig:
    """Validate and construct RuntimeConfig dataclass."""
    for k in raw:
        if k in PROTECTED_ALGORITHM_KEYS:
            raise ValueError(f"Protected field or unknown runtime key: {k}")
        if k not in ALLOWED_RUNTIME_TOP_KEYS:
            raise ValueError(f"Protected field or unknown runtime key: {k}")

    req_keys = {"profile_name", "required_gpu_count", "required_gpu_name_contains", "devices", "generator_runtime"}
    missing = req_keys - set(raw.keys())
    if missing:
        raise ValueError(f"Missing required runtime keys: {missing}")

    gen_rt_raw = raw["generator_runtime"]
    generator_runtime = GeneratorRuntimeConfig(
        compute_dtype=str(gen_rt_raw.get("compute_dtype", "float16")),
        per_device_train_batch_size=int(gen_rt_raw.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(gen_rt_raw.get("gradient_accumulation_steps", 8)),
        activation_offloading=bool(gen_rt_raw.get("activation_offloading", True)),
    )

    smoke = SmokeConfig(**raw["smoke"]) if "smoke" in raw and raw["smoke"] is not None else None
    probe = MicroProbeConfig(**raw["a100_micro_probe"]) if "a100_micro_probe" in raw and raw["a100_micro_probe"] is not None else None
    prod = ProductionRuntimeConfig(**raw["production"]) if "production" in raw and raw["production"] is not None else None
    inference_raw = raw.get("inference", None)
    if inference_raw is None:
        inference = InferenceRuntimeConfig()
    elif isinstance(inference_raw, dict):
        gen_mode = str(inference_raw.get("generator_load_mode", "nf4"))
        if gen_mode not in ("nf4", "bfloat16"):
            raise ValueError(f"unknown generator load mode: {gen_mode!r}")
        inference = InferenceRuntimeConfig(
            generation_batch_size=int(inference_raw.get("generation_batch_size", 4)),
            reranker_batch_size=int(inference_raw.get("reranker_batch_size", 32)),
            retrieval_batch_size=int(inference_raw.get("retrieval_batch_size", 32)),
            generator_load_mode=gen_mode,
            merge_adapter=bool(inference_raw.get("merge_adapter", False)),
            max_new_tokens=int(inference_raw.get("max_new_tokens", 1536)),
            best_fixed_candidate=str(inference_raw.get("best_fixed_candidate", "dual_assembled")),
        )
    else:
        raise ValueError("Runtime 'inference' section must be a mapping")
    if inference.generation_batch_size < 1 or inference.reranker_batch_size < 1 or inference.retrieval_batch_size < 1:
        raise ValueError("inference batch sizes must be positive")

    return RuntimeConfig(
        profile_name=str(raw["profile_name"]),
        required_gpu_count=int(raw["required_gpu_count"]),
        required_gpu_name_contains=str(raw["required_gpu_name_contains"]),
        devices=dict(raw["devices"]),
        generator_runtime=generator_runtime,
        smoke=smoke,
        a100_micro_probe=probe,
        production=prod,
        outputs=dict(raw["outputs"]) if "outputs" in raw and raw["outputs"] is not None else None,
        inference=inference,
    )


def load_resolved_config(
    algorithm_path: Union[str, Path],
    runtime_path: Union[str, Path],
    candidate_id: Optional[str] = None,
) -> ResolvedTask2Config:
    """Load, strictly validate, and resolve algorithm and runtime configuration YAMLs.

    Ensures:
    1. Unknown keys in either YAML fail closed.
    2. Runtime profiles cannot override protected algorithm parameters.
    3. per_device_train_batch_size * gradient_accumulation_steps == effective_batch_size.
    4. Deterministic canonical SHA256 hashes are computed.
    """
    algo_p = Path(algorithm_path)
    rt_p = Path(runtime_path)

    if not algo_p.is_file():
        raise FileNotFoundError(f"Algorithm config file not found: {algo_p}")
    if not rt_p.is_file():
        raise FileNotFoundError(f"Runtime config file not found: {rt_p}")

    with open(algo_p, "r", encoding="utf-8") as f:
        algo_raw = yaml.safe_load(f)
    with open(rt_p, "r", encoding="utf-8") as f:
        rt_raw = yaml.safe_load(f)

    if not isinstance(algo_raw, dict):
        raise ValueError(f"Expected dict in algorithm config {algo_p}, got {type(algo_raw)}")
    if not isinstance(rt_raw, dict):
        raise ValueError(f"Expected dict in runtime config {rt_p}, got {type(rt_raw)}")

    algorithm = _parse_algorithm_config(algo_raw)
    runtime = _parse_runtime_config(rt_raw)

    # Invariant: effective batch size must match
    rt_batch = runtime.generator_runtime.per_device_train_batch_size
    rt_accum = runtime.generator_runtime.gradient_accumulation_steps
    calc_effective = rt_batch * rt_accum
    if calc_effective != algorithm.generator.effective_batch_size:
        raise ValueError(
            f"Effective batch size mismatch! Algorithm requires {algorithm.generator.effective_batch_size}, "
            f"but runtime profile {runtime.profile_name} computes {rt_batch} * {rt_accum} = {calc_effective}."
        )

    algo_dict = algorithm.to_dict()
    rt_dict = runtime.to_dict()

    algo_sha = canonical_sha256(algo_dict)
    rt_sha = canonical_sha256(rt_dict)
    bundle_sha = canonical_sha256({"algorithm": algo_dict, "runtime": rt_dict})

    return ResolvedTask2Config(
        algorithm=algorithm,
        runtime=runtime,
        candidate_id=candidate_id,
        algorithm_sha256=algo_sha,
        runtime_sha256=rt_sha,
        bundle_sha256=bundle_sha,
    )
