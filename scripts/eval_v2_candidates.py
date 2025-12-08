from __future__ import annotations

from typing import Dict, Set  # noqa: UP035

import polars as pl

from src.config.settings import settings
from src.eval.metrics import aggregate_metrics
from src.models.als_cf_conf import ALSConfidenceRecommender
from src.models.item_item import ItemItemSimilarityRecommender
from src.models.popularity import PopularityRecommender
from src.retrieval.genre_neighbors import GenreNeighborsRecommender
from src.retrieval.hybrid import HybridCandidateBlender, SourceWeight


def build_truth(test_path: str) -> Dict[int, Set[int]]:
    df = pl.read_parquet(test_path)
    df = df.filter(pl.col("confidence") > 0)

    truth: Dict[int, Set[int]] = {}
    for u, i in df.select("user_idx", "item_idx").iter_rows():
        truth.setdefault(int(u), set()).add(int(i))
    return truth


def main() -> None:
    test_path = str(settings.PROCESSED_DIR / "test_conf.parquet")
    truth = build_truth(test_path)
    users = list(truth.keys())

    ks = [10, 20, 50]
    max_k = max(ks)

    # Sources
    pop = PopularityRecommender().fit()
    item_item = ItemItemSimilarityRecommender().fit()
    als = ALSConfidenceRecommender().fit()
    genre_nb = GenreNeighborsRecommender().fit()

    blender = HybridCandidateBlender(
        sources=[
            SourceWeight("pop", 0.4),
            SourceWeight("item_item", 1.0),
            SourceWeight("als", 1.0),
            SourceWeight("genre_nb", 0.8),
        ]
    )

    user_recs = {}

    for u in users:
        cands_by_source = {
            "pop": pop.recommend(u, k=max_k),
            "item_item": item_item.recommend(u, k=max_k),
            "als": als.recommend(u, k=max_k),
            "genre_nb": genre_nb.recommend(u, k=max_k),
        }
        user_recs[u] = blender.blend(cands_by_source, k=max_k)

    print("\n[V2 Hybrid Candidate Results]")
    for k in ks:
        print(aggregate_metrics(user_recs, truth, k))


if __name__ == "__main__":
    main()