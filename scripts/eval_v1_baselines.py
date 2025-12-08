from __future__ import annotations

from typing import Dict, Set  # noqa: UP035

import polars as pl

from src.config.settings import settings
from src.eval.metrics import aggregate_metrics
from src.models.als_cf import ALSRecommender
from src.models.popularity import PopularityRecommender


def build_truth(test_path: str) -> Dict[int, Set[int]]:
    df = pl.read_parquet(test_path)

    # Use positives only for truth
    df = df.filter(pl.col("is_positive") == 1)

    truth: Dict[int, Set[int]] = {}

    for u, i in df.select("user_idx", "item_idx").iter_rows():
        u_i = int(u)
        i_i = int(i)
        if u_i not in truth:
            truth[u_i] = set()
        truth[u_i].add(i_i)

    return truth


def main() -> None:
    test_path = str(settings.PROCESSED_DIR / "test.parquet")
    truth = build_truth(test_path)

    users = list(truth.keys())
    if not users:
        raise RuntimeError(
            "No positive interactions found in test set. "
            "Check is_positive threshold or split logic."
        )

    ks = [10, 20, 50]
    max_k = max(ks)

    # Popularity baseline
    pop = PopularityRecommender().fit()
    pop_recs = pop.batch_recommend(users, k=max_k)

    print("\n[Popularity Results]")
    for k in ks:
        print(aggregate_metrics(pop_recs, truth, k))

    # ALS CF baseline
    als = ALSRecommender().fit()
    als_recs = als.batch_recommend(users, k=max_k)

    print("\n[ALS CF Results]")
    for k in ks:
        print(aggregate_metrics(als_recs, truth, k))


if __name__ == "__main__":
    main()