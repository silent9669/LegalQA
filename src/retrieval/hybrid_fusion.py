from collections import defaultdict

def reciprocal_rank_fusion(ranking_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """
    Combines multiple ranked lists using Reciprocal Rank Fusion (RRF).
    RRF(d) = sum_{r} 1 / (k + rank_r(d))
    """
    scores = defaultdict(float)
    for rank_list in ranking_lists:
        for rank, item_id in enumerate(rank_list):
            scores[item_id] += 1.0 / (k + rank + 1)

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_items
