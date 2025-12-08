from __future__ import annotations

from typing import Dict, List, Set  # noqa: UP035

import joblib
import polars as pl

from src.config.settings import settings
from src.eval.metrics import aggregate_metrics
from src.models.als_cf_conf import ALSConfidenceRecommender
from src.models.item_item import ItemItemSimilarityRecommender
from src.models.popularity import PopularityRecommender
from src.ranking.features import (
    build_item_genre_lookup,
    build_popularity_scores,
    build_user_genre_affinity,
)
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

    # Candidate sources
    pop = PopularityRecommender().fit()
    item_item = ItemItemSimilarityRecommender().fit()
    als = ALSConfidenceRecommender().fit()

    blender = HybridCandidateBlender(
        sources=[
            SourceWeight("pop", 0.5),
            SourceWeight("item_item", 1.0),
            SourceWeight("als", 1.0),
        ]
    )

    # Load ranker
    ranker_path = settings.PROJECT_ROOT / "reports" / "models" / "ranker_hgb.pkl"
    if not ranker_path.exists():
        raise FileNotFoundError("Ranker model not found. Run train_ranker first.")

    model = joblib.load(ranker_path)

    # Build feature lookups
    train_conf = pl.read_parquet(settings.PROCESSED_DIR / "train_conf.parquet").select(
        "user_idx", "item_idx", "confidence", "timestamp"
    )
    item_genres = build_item_genre_lookup()
    user_genre_aff = build_user_genre_affinity(train_conf, item_genres)
    pop_scores = build_popularity_scores(train_conf)

    def item_features_for_user(u: int, i: int) -> List[float]:
        genres = item_genres.get(i, [])
        u_aff_d = user_genre_aff.get(u, {})
        aff = sum(u_aff_d.get(g, 0.0) for g in genres) / len(genres) if genres else 0.0

        pop_conf = pop_scores.get(i, 0.0)
        genre_count = len(genres)

        # pop_bucket derived from pop_conf rough binning
        # simple deterministic bins for MVP
        if pop_conf >= 5000:
            pop_bucket = 1
        elif pop_conf >= 1500:
            pop_bucket = 2
        elif pop_conf >= 300:
            pop_bucket = 3
        else:
            pop_bucket = 4

        return [pop_bucket, aff, pop_conf, genre_count]

    user_recs_ranked: Dict[int, List[int]] = {}

    for u in users:
        cands_by_source = {
            "pop": pop.recommend(u, k=max_k),
            "item_item": item_item.recommend(u, k=max_k),
            "als": als.recommend(u, k=max_k),
        }
        blended = blender.blend(cands_by_source, k=max_k)

        # Score candidates with ranker
        X = [item_features_for_user(u, i) for i in blended]
        if not X:
            user_recs_ranked[u] = []
            continue

        scores = model.predict_proba(X)[:, 1]
        ranked = [i for i, _ in sorted(zip(blended, scores, strict=False), key=lambda x: x[1], reverse=True)]

        user_recs_ranked[u] = ranked[:max_k]

    print("\n[Ranked Hybrid Results]")
    for k in ks:
        print(aggregate_metrics(user_recs_ranked, truth, k))


if __name__ == "__main__":
    main()