from __future__ import annotations

from typing import Dict, List, Set, Tuple  # noqa: UP035

import polars as pl

from src.config.settings import settings


def _parse_genres(genres: str | None) -> List[str]:
    if not genres:
        return []
    return [g.strip() for g in genres.split("|") if g.strip()]


def build_user_genre_affinity(train_conf: pl.DataFrame, item_genres: Dict[int, List[str]]) -> Dict[int, Dict[str, float]]:
    """
    Simple affinity score:
    sum of confidence per genre per user, normalized by total confidence.
    """
    user_genre_sum: Dict[int, Dict[str, float]] = {}
    user_total: Dict[int, float] = {}

    for u, i, c in train_conf.select("user_idx", "item_idx", "confidence").iter_rows():
        u_i = int(u)
        i_i = int(i)
        c_f = float(c)

        user_total[u_i] = user_total.get(u_i, 0.0) + c_f

        for g in item_genres.get(i_i, []):
            d = user_genre_sum.setdefault(u_i, {})
            d[g] = d.get(g, 0.0) + c_f

    # Normalize
    for u, d in user_genre_sum.items():
        total = user_total.get(u, 1.0)
        for g in list(d.keys()):
            d[g] = d[g] / total

    return user_genre_sum


def build_item_genre_lookup() -> Dict[int, List[str]]:
    items_map = pl.read_parquet(settings.PROCESSED_DIR / "items.parquet")
    movies = pl.read_parquet(settings.PROCESSED_DIR / "movies.parquet")

    meta = (
        items_map.join(movies, on="movieId", how="left")
        .select("item_idx", "genres")
    )

    lookup: Dict[int, List[str]] = {}
    for item_idx, genres in meta.iter_rows():
        lookup[int(item_idx)] = _parse_genres(genres)

    return lookup


def build_popularity_scores(train_conf: pl.DataFrame) -> Dict[int, float]:
    pop = (
        train_conf.group_by("item_idx")
        .agg(pl.col("confidence").sum().alias("pop_conf_sum"))
    )
    return {int(i): float(s) for i, s in pop.select("item_idx", "pop_conf_sum").iter_rows()}


def build_feature_table(pairs_path: str) -> pl.DataFrame:
    train_conf = pl.read_parquet(settings.PROCESSED_DIR / "train_conf.parquet").select(
        "user_idx", "item_idx", "confidence", "timestamp"
    )

    pairs = pl.read_parquet(pairs_path)

    item_genres = build_item_genre_lookup()
    user_genre_aff = build_user_genre_affinity(train_conf, item_genres)
    pop_scores = build_popularity_scores(train_conf)

    # Build features row-wise
    rows = []
    for u, i, label, pop_bucket in pairs.select("user_idx", "item_idx", "label", "pop_bucket").iter_rows():
        u_i = int(u)
        i_i = int(i)

        genres = item_genres.get(i_i, [])
        # user affinity aggregated across item genres
        aff = 0.0
        u_aff_d = user_genre_aff.get(u_i, {})
        if genres:
            aff = sum(u_aff_d.get(g, 0.0) for g in genres) / len(genres)

        pop = pop_scores.get(i_i, 0.0)

        rows.append(
            (u_i, i_i, float(label), int(pop_bucket), float(aff), float(pop), len(genres))
        )

    return pl.DataFrame(
        rows,
        schema=[
            "user_idx",
            "item_idx",
            "label",
            "pop_bucket",
            "user_genre_aff",
            "item_pop_conf",
            "item_genre_count",
        ],
    )