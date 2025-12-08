from __future__ import annotations

import json
from typing import Dict, List, Set  # noqa: UP035

import joblib
import polars as pl
from tqdm import tqdm

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

    # Candidate models
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

    models_dir = settings.PROJECT_ROOT / "reports" / "models"
    ranker_path = models_dir / "ranker_hgb_v2.pkl"
    meta_path = models_dir / "ranker_hgb_v2.features.json"

    if not ranker_path.exists():
        raise FileNotFoundError("ranker_hgb_v2.pkl not found. Run train_ranker_v2 first.")
    if not meta_path.exists():
        raise FileNotFoundError("ranker_hgb_v2.features.json not found. Re-run train_ranker_v2.")

    model = joblib.load(ranker_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols: List[str] = meta["feature_cols"]
    print(f"[OK] Loaded feature order: {feature_cols}")

    # Load feature store tables
    user_feat = pl.read_parquet(settings.PROCESSED_DIR / "user_features.parquet")
    item_feat = pl.read_parquet(settings.PROCESSED_DIR / "item_features.parquet")

    # Build lookup dicts
    user_feat_map = {
        int(r[0]): {
            "user_interactions": float(r[1]),
            "user_conf_sum": float(r[2]),
            "user_conf_decay_sum": float(r[3]),
            "user_days_since_last": float(r[4]),
        }
        for r in user_feat.select(
            "user_idx",
            "user_interactions",
            "user_conf_sum",
            "user_conf_decay_sum",
            "user_days_since_last",
        ).iter_rows()
    }

    item_feat_map = {
        int(r[0]): {
            "item_interactions": float(r[1]),
            "item_conf_sum": float(r[2]),
            "item_conf_decay_sum": float(r[3]),
            "item_days_since_last": float(r[4]),
        }
        for r in item_feat.select(
            "item_idx",
            "item_interactions",
            "item_conf_sum",
            "item_conf_decay_sum",
            "item_days_since_last",
        ).iter_rows()
    }

    def feature_dict(u: int, i: int) -> Dict[str, float]:
        ud = user_feat_map.get(u, {})
        idd = item_feat_map.get(i, {})

        # Merge with defaults
        out = {
            "user_interactions": 0.0,
            "user_conf_sum": 0.0,
            "user_conf_decay_sum": 0.0,
            "user_days_since_last": 0.0,
            "item_interactions": 0.0,
            "item_conf_sum": 0.0,
            "item_conf_decay_sum": 0.0,
            "item_days_since_last": 0.0,
        }
        out.update(ud)
        out.update(idd)
        return out

    user_recs_ranked: Dict[int, List[int]] = {}

    print("[START] Ranking V2 hybrid candidates per user...")
    for u in tqdm(users, desc="Ranking users (V2)"):
        cands_by_source = {
            "pop": pop.recommend(u, k=max_k),
            "item_item": item_item.recommend(u, k=max_k),
            "als": als.recommend(u, k=max_k),
            "genre_nb": genre_nb.recommend(u, k=max_k),
        }
        blended = blender.blend(cands_by_source, k=max_k)

        if not blended:
            user_recs_ranked[u] = []
            continue

        # Build X in the exact saved order
        X = []
        for i in blended:
            fd = feature_dict(u, i)
            X.append([fd[c] for c in feature_cols])

        scores = model.predict_proba(X)[:, 1]
        ranked = [i for i, _ in sorted(zip(blended, scores), key=lambda x: x[1], reverse=True)]

        user_recs_ranked[u] = ranked[:max_k]

    print("\n[Ranked V2 Hybrid Results]")
    for k in ks:
        print(aggregate_metrics(user_recs_ranked, truth, k))


if __name__ == "__main__":
    main()