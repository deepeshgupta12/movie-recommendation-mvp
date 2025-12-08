from __future__ import annotations

import polars as pl

from src.config.settings import settings


def build_feature_table_v2(pairs_path: str) -> pl.DataFrame:
    pairs = pl.read_parquet(pairs_path)

    user_feat_path = settings.PROCESSED_DIR / "user_features.parquet"
    item_feat_path = settings.PROCESSED_DIR / "item_features.parquet"
    item_genres_path = settings.PROCESSED_DIR / "item_genres_expanded.parquet"
    user_genre_path = settings.PROCESSED_DIR / "user_genre_affinity.parquet"

    if not user_feat_path.exists() or not item_feat_path.exists():
        raise FileNotFoundError("user_features/item_features not found. Run feature_store first.")
    if not item_genres_path.exists() or not user_genre_path.exists():
        raise FileNotFoundError("item_genres_expanded/user_genre_affinity not found. Run updated feature_store.")

    user_features = pl.read_parquet(user_feat_path)
    item_features = pl.read_parquet(item_feat_path)
    item_genres = pl.read_parquet(item_genres_path)
    user_genre = pl.read_parquet(user_genre_path)

    # Base joins
    df = (
        pairs.join(user_features, on="user_idx", how="left")
             .join(item_features, on="item_idx", how="left")
    ).fill_null(0)

    # Item genre count
    item_genre_count = (
        item_genres.group_by("item_idx")
        .agg(pl.len().alias("item_genre_count"))
    )

    df = df.join(item_genre_count, on="item_idx", how="left").fill_null(0)

    # User-item genre affinity:
    # pairs -> item genres -> user genre aff
    pair_genres = (
        pairs.select("user_idx", "item_idx", "label")
        .join(item_genres, on="item_idx", how="left")
        .join(user_genre, on=["user_idx", "genre"], how="left")
        .with_columns(
            pl.col("user_genre_aff").fill_null(0.0),
            pl.col("user_genre_aff_decay").fill_null(0.0),
        )
    )

    user_item_aff = (
        pair_genres.group_by("user_idx", "item_idx")
        .agg(
            pl.mean("user_genre_aff").alias("user_item_genre_aff"),
            pl.mean("user_genre_aff_decay").alias("user_item_genre_aff_decay"),
        )
    )

    df = df.join(user_item_aff, on=["user_idx", "item_idx"], how="left").fill_null(0)

    keep = [
        "user_idx",
        "item_idx",
        "label",

        # User
        "user_interactions",
        "user_conf_sum",
        "user_conf_decay_sum",
        "user_days_since_last",

        # Item
        "item_interactions",
        "item_conf_sum",
        "item_conf_decay_sum",
        "item_days_since_last",

        # Cross/genre
        "item_genre_count",
        "user_item_genre_aff",
        "user_item_genre_aff_decay",
    ]

    existing = [c for c in keep if c in df.columns]
    return df.select(existing)