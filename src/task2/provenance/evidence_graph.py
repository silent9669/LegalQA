"""End-to-end evidence graph: one candidate from freeze to receipt.

Joins gate statuses, platform evidence, final checkpoint scope, offline
labelled metrics, 1,000-inference verification, and the immutable external
receipt. Missing links and fabricated timing/counts are rejected; a valid
experimental release is never automatically a score-qualified best.
"""

from __future__ import annotations

from typing import Any, Dict, List

REQUIRED_GATE_STAGES = ("kaggle_t4x2", "colab_t4", "a100_micro_probe")


def join_evidence_reports(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Join heterogeneous evidence links into one candidate graph.

    Every input must declare kind, candidate_sha, and measured=True.
    Estimated/placeholder timing or counts are rejected as fabricated.
    Offline metrics and the official receipt stay separate links.
    """
    if not reports:
        raise ValueError("join_evidence_reports requires at least one report")
    candidate_shas = {str(r.get("candidate_sha", "") or "") for r in reports}
    candidate_shas.discard("")
    if len(candidate_shas) != 1:
        raise ValueError(f"evidence reports must share one candidate_sha, got {sorted(candidate_shas)}")
    candidate_sha = next(iter(candidate_shas))
    graph: Dict[str, Any] = {
        "candidate_sha": candidate_sha,
        "gates": {},
        "training": None,
        "offline_metrics": None,
        "inference": None,
        "a100": None,
        "receipt": None,
    }
    for report in reports:
        kind = report.get("kind")
        if report.get("measured") is not True or report.get("estimated"):
            raise ValueError(f"fabricated timing/counts rejected in {kind} report")
        if kind == "gate":
            stage = report.get("stage")
            if report.get("status") != "PASS":
                raise ValueError(f"gate {stage} is not PASS")
            graph["gates"][stage] = report
        elif kind == "training":
            steps = report.get("optimizer_steps")
            samples = report.get("training_samples")
            if not isinstance(steps, int) or steps <= 0:
                raise ValueError("training evidence requires measured optimizer_steps > 0")
            if not isinstance(samples, int) or samples <= 0:
                raise ValueError("training evidence requires measured training_samples > 0")
            graph["training"] = report
        elif kind == "offline_metrics":
            if report.get("meteor") is None or report.get("rouge") is None:
                raise ValueError("offline metrics evidence requires meteor and rouge")
            if not report.get("checkpoint") or not report.get("split_fingerprint"):
                raise ValueError("offline metrics require checkpoint and split provenance")
            graph["offline_metrics"] = report
        elif kind == "inference":
            if int(report.get("num_predictions", 0)) != 1000:
                raise ValueError("inference evidence requires exactly 1000 verified predictions")
            if not report.get("submission_sha256"):
                raise ValueError("inference evidence requires submission_sha256")
            graph["inference"] = report
        elif kind == "a100":
            graph["a100"] = report
        elif kind == "receipt":
            if not report.get("commit_sha"):
                raise ValueError("receipt evidence requires commit_sha")
            graph["receipt"] = report
        else:
            raise ValueError(f"unknown evidence kind: {kind}")
    return graph


def validate_evidence_graph(graph: Dict[str, Any]) -> None:
    """Validate a joined evidence graph; raises ValueError naming the gap."""
    if not graph.get("candidate_sha"):
        raise ValueError("evidence graph missing candidate_sha")
    a100 = graph.get("a100") or {}
    if a100.get("status") != "PASS":
        raise ValueError("evidence graph missing A100 PASS measurement")
    if a100.get("wall_seconds") in (None, ""):
        raise ValueError("evidence graph missing A100 measured timing (wall_seconds)")
    gates = graph.get("gates", {})
    for stage in REQUIRED_GATE_STAGES:
        if stage not in gates:
            raise ValueError(f"evidence graph missing gate: {stage}")
    training = graph.get("training")
    if not training or not training.get("optimizer_steps") or not training.get("training_samples"):
        raise ValueError("evidence graph missing full training scope and measured counts")
    offline = graph.get("offline_metrics")
    if not offline:
        raise ValueError("evidence graph missing offline labelled metrics")
    inference = graph.get("inference")
    if not inference or int(inference.get("num_predictions", 0)) != 1000:
        raise ValueError("evidence graph missing exact public-ID inference verification")
    receipt = graph.get("receipt")
    if not receipt or not receipt.get("commit_sha"):
        raise ValueError("evidence graph missing immutable receipt commit_sha")
    if graph.get("final_scope") is not None:
        scope = graph["final_scope"]
        if scope.get("val_fold") is not None or scope.get("training_scope") not in ("all_allowed_train", None):
            raise ValueError("final checkpoint scope must be all allowed data with val_fold null")


def final_run_checklist(graph: Dict[str, Any]) -> Dict[str, Any]:
    """Render the final-run checklist; overall PASS only when all links hold."""
    checks = []

    def _check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    gates = graph.get("gates", {}) if isinstance(graph.get("gates"), dict) else {}
    for stage in REQUIRED_GATE_STAGES:
        _check(f"gate:{stage}", gates.get(stage, {}).get("status") == "PASS")
    training = graph.get("training") or {}
    _check("training:measured_counts",
           isinstance(training.get("optimizer_steps"), int) and training.get("optimizer_steps", 0) > 0
           and isinstance(training.get("training_samples"), int) and training.get("training_samples", 0) > 0)
    a100 = graph.get("a100") or {}
    _check("a100:measured_timing", a100.get("status") == "PASS" and a100.get("wall_seconds") not in (None, ""))
    offline = graph.get("offline_metrics") or {}
    _check("offline_metrics:separate_from_receipt", offline.get("meteor") is not None and offline.get("rouge") is not None)
    inference = graph.get("inference") or {}
    _check("inference:exact_1000_ids", int(inference.get("num_predictions", 0)) == 1000)
    receipt = graph.get("receipt") or {}
    _check("receipt:immutable_commit", bool(receipt.get("commit_sha")))
    scope = graph.get("final_scope") or {}
    _check("final_scope:all_allowed_val_fold_null",
           not scope or (scope.get("val_fold") is None and scope.get("training_scope", "all_allowed_train") == "all_allowed_train"))
    overall = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
    note = "" if overall == "PASS" else "a valid experimental release is not automatically a score-qualified best"
    return {"overall": overall, "checks": checks, "note": note}
