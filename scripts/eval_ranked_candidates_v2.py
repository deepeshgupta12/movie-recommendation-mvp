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
    meta_path = models_dir / "ranker_hgb_v2.meta.json"

    if not ranker_path.exists():
        raise FileNotFoundError("ranker_hgb_v2.pkl not found. Run train_ranker_v2 first.")
    if not meta_path.exists():
        raise FileNotFoundError("ranker_hgb_v2.meta.json not found. Re-run train_ranker_v2.")

    model = joblib.load(ranker_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols: List[str] = meta["feature_cols"]
    positive_class = meta.get("positive_class", 1)

    classes_list = list(model.classes_)
    if positive_class not in classes_list:
        raise ValueError(f"Positive class {positive_class} not present in model.classes_: {classes_list}")

    pos_idx = classes_list.index(positive_class)

    print(f"[OK] Loaded feature order: {feature_cols}")
    print(f"[OK] Model classes_: {classes_list} | positive_class={positive_class} | pos_idx={pos_idx}")

    # Load feature store artifacts
    user_feat = pl.read_parquet(settings.PROCESSED_DIR / "user_features.parquet")
    item_feat = pl.read_parquet(settings.PROCESSED_DIR / "item_features.parquet")
    item_genre_count = (
        pl.read_parquet(settings.PROCESSED_DIR / "item_genres_expanded.parquet")
        .group_by("item_idx")
        .agg(pl.len().alias("item_genre_count"))
    )
    user_item_aff = None  # noqa: F841

    # Precompute user-item affinity for fast lookup by reading from feature table join source
    # We'll build a lightweight map from user_genre_affinity + item_genres.
    item_genres = pl.read_parquet(settings.PROCESSED_DIR / "item_genres_expanded.parquet")
    user_genre = pl.read_parquet(settings.PROCESSED_DIR / "user_genre_affinity.parquet")

    # Build user->genre maps (top-level dict of dicts)
    user_genre_map: Dict[int, Dict[str, float]] = {}
    for u, g, aff, aff_d in user_genre.select(
        "user_idx", "genre", "user_genre_aff", "user_genre_aff_decay"
    ).iter_rows():
        ud = user_genre_map.setdefault(int(u), {})
        # store both in one packed float is messy; store main aff only here
        ud[str(g)] = float(aff)

    # Item->genres map
    item_genres_map: Dict[int, List[str]] = {}
    for i, g in item_genres.select("item_idx", "genre").iter_rows():
        item_genres_map.setdefault(int(i), []).append(str(g))

    # User feature map
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

    # Item feature map
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

    # Item genre count map
    item_genre_count_map = {
        int(i): float(c)
        for i, c in item_genre_count.select("item_idx", "item_genre_count").iter_rows()
    }

    defaults = {
        "user_interactions": 0.0,
        "user_conf_sum": 0.0,
        "user_conf_decay_sum": 0.0,
        "user_days_since_last": 0.0,
        "item_interactions": 0.0,
        "item_conf_sum": 0.0,
        "item_conf_decay_sum": 0.0,
        "item_days_since_last": 0.0,
        "item_genre_count": 0.0,
        "user_item_genre_aff": 0.0,
        "user_item_genre_aff_decay": 0.0,  # will remain 0 in this lightweight path
    }

    def feature_dict(u: int, i: int) -> Dict[str, float]:
        out = dict(defaults)
        out.update(user_feat_map.get(u, {}))
        out.update(item_feat_map.get(i, {}))
        out["item_genre_count"] = item_genre_count_map.get(i, 0.0)

        genres = item_genres_map.get(i, [])
        if genres:
            ug = user_genre_map.get(u, {})
            if ug:
                out["user_item_genre_aff"] = sum(ug.get(g, 0.0) for g in genres) / len(genres)
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

        X = []
        for i in blended:
            fd = feature_dict(u, i)
            X.append([fd.get(c, 0.0) for c in feature_cols])

        scores = model.predict_proba(X)[:, pos_idx]
        ranked = [i for i, _ in sorted(zip(blended, scores), key=lambda x: x[1], reverse=True)]

        user_recs_ranked[u] = ranked[:max_k]

    print("\n[Ranked V2 Hybrid Results]")
    for k in ks:
        print(aggregate_metrics(user_recs_ranked, truth, k))


if __name__ == "__main__":
    main()