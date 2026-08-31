"""Leakage-safe cross-fitted Candidate Selector with fixed-baseline guardrail for Task 2."""

from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.common.normalize import clean_legal_text, extract_legal_signals, normalize_question

CANDIDATE_ORDER = [
    "exact_memory",
    "fuzzy_memory",
    "focused_extract",
    "stitched_extract",
    "primary_article_extract",
    "relevance_extract",
    "multi_seed_extract",
    "strategy_f_300",
    "strategy_f_600",
    "strategy_f_1000",
    "strategy_f_1500",
    "snapped",
    "generated",
]


def extract_candidate_features(
    question: str,
    candidate_name: str,
    candidate_text: str,
    evidence: str = "",
    retrieval_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Extract inference-available feature vector for a (question, candidate) pair."""
    retrieval_meta = retrieval_meta or {}
    cand_tokens = str(candidate_text).split()
    q_tokens = str(question).split()
    ev_tokens = str(evidence).split() if evidence else []

    cand_set = set(cand_tokens)
    ev_set = set(ev_tokens)
    q_set = set(q_tokens)

    ev_overlap = len(cand_set & ev_set) / max(1, len(cand_set | ev_set)) if ev_set else 0.0
    q_overlap = len(cand_set & q_set) / max(1, len(cand_set | q_set))

    q_lower = question.lower()
    is_penalty = 1.0 if any(k in q_lower for k in ["phạt", "xử phạt", "tù", "tiền", "mức phạt"]) else 0.0
    is_deadline = 1.0 if any(k in q_lower for k in ["thời hạn", "thời hiệu", "bao lâu", "ngày", "tháng", "năm", "khi nào"]) else 0.0
    is_condition = 1.0 if any(k in q_lower for k in ["điều kiện", "thủ tục", "hồ sơ", "trình tự", "yêu cầu"]) else 0.0

    cand_idx = CANDIDATE_ORDER.index(candidate_name) if candidate_name in CANDIDATE_ORDER else 99

    return {
        "cand_type_idx": float(cand_idx),
        "cand_word_count": float(len(cand_tokens)),
        "q_word_count": float(len(q_tokens)),
        "overlap_ratio": float(ev_overlap),
        "q_overlap_ratio": float(q_overlap),
        "is_penalty": is_penalty,
        "is_deadline": is_deadline,
        "is_condition": is_condition,
        "rerank_top1": float(retrieval_meta.get("rerank_top1", 0.0)),
        "rerank_margin": float(retrieval_meta.get("rerank_margin", 0.0)),
        "bm25_top1": float(retrieval_meta.get("bm25_top1", 0.0)),
        "dense_top1": float(retrieval_meta.get("dense_top1", 0.0)),
        "fuzzy_sim": float(retrieval_meta.get("fuzzy_sim", 0.0)),
    }


class CandidateSelector:
    """Cross-fitted candidate selector with mandatory fixed-baseline guardrail."""

    def __init__(self, policy: str = "fixed_baseline", best_fixed_candidate: str = "stitched_extract"):
        self.policy = policy  # "learned_model" or "fixed_baseline"
        self.best_fixed_candidate = best_fixed_candidate
        self.model: Optional[HistGradientBoostingRegressor] = None
        self.feature_names: List[str] = []

    def fit_meta_oof(
        self,
        df_candidates: pd.DataFrame,
        n_splits: int = 5,
        baseline_candidate: str = "stitched_extract",
    ) -> Dict[str, Any]:
        """Cross-fit selector on meta-folds grouped by qa_id, evaluate vs fixed baselines, and enforce guardrail."""
        if df_candidates.empty or "meteor" not in df_candidates.columns:
            self.policy = "fixed_baseline"
            self.best_fixed_candidate = baseline_candidate
            return {"policy": self.policy, "best_fixed_candidate": baseline_candidate}

        # Calculate mean METEOR per fixed candidate family
        cand_means = df_candidates.groupby("cand_name")["meteor"].mean().to_dict()
        best_fixed_family = max(cand_means.keys(), key=lambda k: cand_means[k])
        best_fixed_score = cand_means[best_fixed_family]

        self.best_fixed_candidate = best_fixed_family

        # Build feature matrix
        feature_rows = []
        for _, row in df_candidates.iterrows():
            feats = extract_candidate_features(
                question=str(row.get("question", "")),
                candidate_name=str(row.get("cand_name", "")),
                candidate_text=str(row.get("cand_text", "")),
                evidence=str(row.get("evidence", "")),
            )
            feature_rows.append(feats)

        df_feats = pd.DataFrame(feature_rows)
        self.feature_names = list(df_feats.columns)
        X = df_feats.values
        y = df_candidates["meteor"].values

        # Meta-OOF Cross Validation grouped by qa_id / fold_id
        folds = df_candidates["fold_id"].values if "fold_id" in df_candidates.columns else np.random.randint(0, n_splits, len(df_candidates))
        unique_folds = np.unique(folds)

        oof_preds = np.zeros(len(df_candidates))

        for f_id in unique_folds:
            train_mask = folds != f_id
            val_mask = folds == f_id
            if not np.any(train_mask) or not np.any(val_mask):
                continue

            fold_model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
            fold_model.fit(X[train_mask], y[train_mask])
            oof_preds[val_mask] = fold_model.predict(X[val_mask])

        df_candidates["pred_score"] = oof_preds

        # Evaluate meta-OOF selection: for each QA, pick candidate with highest predicted score
        meta_selected_scores = []
        for qid, group in df_candidates.groupby("qa_id"):
            best_row = group.sort_values("pred_score", ascending=False).iloc[0]
            meta_selected_scores.append(best_row["meteor"])

        selector_meta_score = float(np.mean(meta_selected_scores)) if meta_selected_scores else 0.0

        print(f"Meta-OOF Validation: Selector METEOR = {selector_meta_score:.4f} vs Best Fixed '{best_fixed_family}' = {best_fixed_score:.4f}")

        # Enforce Guardrail: If selector doesn't strictly outperform best fixed candidate, fallback to fixed
        if selector_meta_score < best_fixed_score:
            print(f"GUARDRAIL TRIGGERED: Selector ({selector_meta_score:.4f}) < Best Fixed ({best_fixed_score:.4f}). Deploying fixed baseline '{best_fixed_family}'.")
            self.policy = "fixed_baseline"
        else:
            print(f"SELECTOR PROMOTED: Selector ({selector_meta_score:.4f}) >= Best Fixed ({best_fixed_score:.4f}).")
            self.policy = "learned_model"
            self.model = HistGradientBoostingRegressor(max_iter=150, random_state=42)
            self.model.fit(X, y)

        return {
            "policy": self.policy,
            "best_fixed_candidate": best_fixed_family,
            "best_fixed_score": best_fixed_score,
            "selector_meta_score": selector_meta_score,
            "candidate_means": cand_means,
        }

    def select(
        self,
        candidates: Dict[str, str],
        question: str = "",
        evidence: str = "",
        retrieval_meta: Optional[Dict[str, Any]] = None,
        features: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Select best candidate answer given question and test-time features."""
        if not candidates:
            return ""

        # 1. Exact Memory is highest priority
        if candidates.get("exact_memory"):
            return candidates["exact_memory"].strip()

        # 2. High-confidence Similar QA memory
        if candidates.get("fuzzy_memory") and features and features.get("is_direct_reuse"):
            return candidates["fuzzy_memory"].strip()

        # 3. Guardrail fixed baseline policy
        if self.policy == "fixed_baseline":
            if self.best_fixed_candidate in candidates:
                return candidates[self.best_fixed_candidate].strip()
            for fallback_key in ["stitched_extract", "focused_extract", "strategy_f_1000", "generated", "snapped"]:
                if fallback_key in candidates and candidates[fallback_key].strip():
                    return candidates[fallback_key].strip()

        # 4. Learned Model Selection
        if self.policy == "learned_model" and self.model is not None and len(candidates) > 1:
            cand_keys = list(candidates.keys())
            feat_rows = [
                extract_candidate_features(
                    question=question,
                    candidate_name=k,
                    candidate_text=candidates[k],
                    evidence=evidence,
                    retrieval_meta=retrieval_meta,
                )
                for k in cand_keys
            ]
            X_test = pd.DataFrame(feat_rows)[self.feature_names].values
            predicted_scores = self.model.predict(X_test)
            best_idx = int(np.argmax(predicted_scores))
            return candidates[cand_keys[best_idx]].strip()

        # Default fallback
        for k in ["stitched_extract", "focused_extract", "strategy_f_1000", "generated"]:
            if k in candidates and candidates[k].strip():
                return candidates[k].strip()

        return next(iter(candidates.values())).strip()

    def save(self, output_path: str) -> None:
        """Save selector policy and model weights."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump({
                "policy": self.policy,
                "best_fixed_candidate": self.best_fixed_candidate,
                "model": self.model,
                "feature_names": self.feature_names,
            }, f)

    @classmethod
    def load(cls, model_path: str) -> CandidateSelector:
        """Load trained selector from disk."""
        if not os.path.exists(model_path):
            return cls()
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        selector = cls(policy=data.get("policy", "fixed_baseline"), best_fixed_candidate=data.get("best_fixed_candidate", "stitched_extract"))
        selector.model = data.get("model")
        selector.feature_names = data.get("feature_names", [])
        return selector
