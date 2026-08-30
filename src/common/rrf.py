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
