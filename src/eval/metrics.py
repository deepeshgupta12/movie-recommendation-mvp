from __future__ import annotations

import math
from typing import Dict, List, Sequence, Set  # noqa: UP035


def recall_at_k(recs: Sequence[int], truth: Set[int], k: int) -> float:
    if not truth:
        return 0.0
    topk = recs[:k]
    hit = sum(1 for i in topk if i in truth)
    return hit / len(truth)


def dcg_at_k(recs: Sequence[int], truth: Set[int], k: int) -> float:
    dcg = 0.0
    for idx, item in enumerate(recs[:k], start=1):
        if item in truth:
            dcg += 1.0 / math.log2(idx + 1)
    return dcg


def ndcg_at_k(recs: Sequence[int], truth: Set[int], k: int) -> float:
    if not truth:
        return 0.0

    # Ideal DCG assumes all relevant items appear first
    ideal_recs = list(truth)
    idcg = dcg_at_k(ideal_recs, set(ideal_recs), k)
    if idcg == 0:
        return 0.0

    return dcg_at_k(recs, truth, k) / idcg


def aggregate_metrics(
    user_recs: Dict[int, List[int]],
    user_truth: Dict[int, Set[int]],
    k: int
) -> Dict[str, float]:
    recalls: List[float] = []
    ndcgs: List[float] = []

    for u, truth in user_truth.items():
        recs = user_recs.get(u, [])
        recalls.append(recall_at_k(recs, truth, k))
        ndcgs.append(ndcg_at_k(recs, truth, k))

    n = max(len(recalls), 1)
    return {
        f"recall@{k}": sum(recalls) / n,
        f"ndcg@{k}": sum(ndcgs) / n,
        "users_evaluated": float(len(recalls)),
    }