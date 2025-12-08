from __future__ import annotations

import time

import polars as pl

from src.config.settings import settings


def _half_life_decay(ts: pl.Expr, max_ts: int, half_life_days: int = 30) -> pl.Expr:
    half_life_sec = half_life_days * 24 * 3600
    delta = (pl.lit(max_ts) - ts).cast(pl.Float64)
    return (pl.lit(0.5) ** (delta / pl.lit(half_life_sec))).alias("decay_w")


def main() -> None:
    t0 = time.time()

    train_path = settings.PROCESSED_DIR / "train_conf.parquet"
    movies_path = settings.PROCESSED_DIR / "movies.parquet"
    items_path = settings.PROCESSED_DIR / "items.parquet"

    if not train_path.exists():
        raise FileNotFoundError("train_conf.parquet not found.")
    if not movies_path.exists() or not items_path.exists():
        raise FileNotFoundError("movies.parquet/items.parquet not found.")

    print("[START] Loading train_conf...")
    train = pl.read_parquet(train_path).select(
        "user_idx", "item_idx", "confidence", "timestamp"
    )
    print(f"[OK] train_conf rows: {train.height}")

    max_ts = int(train["timestamp"].max())
    print(f"[OK] max timestamp: {max_ts}")

    print("[START] Adding decay weights...")
    train = train.with_columns(
        _half_life_decay(pl.col("timestamp"), max_ts, half_life_days=30)
    ).with_columns(
        (pl.col("confidence") * pl.col("decay_w")).alias("conf_decay")
    )

    # ---------------------------
    # Item features
    # ---------------------------
    print("[START] Building item features...")
    item_features = (
        train.group_by("item_idx")
        .agg(
            pl.len().alias("item_interactions"),
            pl.col("confidence").sum().alias("item_conf_sum"),
            pl.col("conf_decay").sum().alias("item_conf_decay_sum"),
            pl.col("timestamp").max().alias("item_last_ts"),
        )
        .with_columns(
            ((pl.lit(max_ts) - pl.col("item_last_ts")) / 86400.0).alias("item_days_since_last")
        )
    )

    # ---------------------------
    # User features
    # ---------------------------
    print("[START] Building user features...")
    user_features = (
        train.group_by("user_idx")
        .agg(
            pl.len().alias("user_interactions"),
            pl.col("confidence").sum().alias("user_conf_sum"),
            pl.col("conf_decay").sum().alias("user_conf_decay_sum"),
            pl.col("timestamp").max().alias("user_last_ts"),
        )
        .with_columns(
            ((pl.lit(max_ts) - pl.col("user_last_ts")) / 86400.0).alias("user_days_since_last")
        )
    )

    # ---------------------------
    # Item genres expanded
    # ---------------------------
    print("[START] Building item_genres_expanded...")
    items_map = pl.read_parquet(items_path).select("item_idx", "movieId")
    movies = pl.read_parquet(movies_path).select("movieId", "genres")

    item_genres = (
        items_map.join(movies, on="movieId", how="left")
        .with_columns(
            pl.col("genres").fill_null("").str.split("|").alias("genre_list")
        )
        .explode("genre_list")
        .rename({"genre_list": "genre"})
        .filter(pl.col("genre") != "")
        .select("item_idx", "genre")
    )

    # ---------------------------
    # Genre priors (time-decayed)
    # ---------------------------
    print("[START] Building genre priors...")
    genre_item = (
        train.join(item_genres, on="item_idx", how="left")
        .filter(pl.col("genre").is_not_null())
    )

    genre_priors = (
        genre_item.group_by("genre", "item_idx")
        .agg(
            pl.col("confidence").sum().alias("genre_item_conf"),
            pl.col("conf_decay").sum().alias("genre_item_conf_decay"),
        )
        .sort(["genre", "genre_item_conf_decay"], descending=[False, True])
    )

    # ---------------------------
    # User genre affinity (normalized)
    # ---------------------------
    print("[START] Building user_genre_affinity...")
    user_genre = (
        train.filter(pl.col("confidence") > 0)
        .join(item_genres, on="item_idx", how="left")
        .filter(pl.col("genre").is_not_null())
        .group_by("user_idx", "genre")
        .agg(
            pl.col("confidence").sum().alias("user_genre_conf"),
            pl.col("conf_decay").sum().alias("user_genre_conf_decay"),
        )
    )

    user_totals = (
        user_genre.group_by("user_idx")
        .agg(
            pl.col("user_genre_conf").sum().alias("user_genre_total"),
            pl.col("user_genre_conf_decay").sum().alias("user_genre_total_decay"),
        )
    )

    user_genre_aff = (
        user_genre.join(user_totals, on="user_idx", how="left")
        .with_columns(
            pl.when(pl.col("user_genre_total") > 0)
              .then(pl.col("user_genre_conf") / pl.col("user_genre_total"))
              .otherwise(0.0)
              .alias("user_genre_aff"),

            pl.when(pl.col("user_genre_total_decay") > 0)
              .then(pl.col("user_genre_conf_decay") / pl.col("user_genre_total_decay"))
              .otherwise(0.0)
              .alias("user_genre_aff_decay"),
        )
        .select("user_idx", "genre", "user_genre_aff", "user_genre_aff_decay")
    )

    # ---------------------------
    # Write outputs
    # ---------------------------
    out_user = settings.PROCESSED_DIR / "user_features.parquet"
    out_item = settings.PROCESSED_DIR / "item_features.parquet"
    out_genre = settings.PROCESSED_DIR / "genre_item_priors.parquet"
    out_item_genres = settings.PROCESSED_DIR / "item_genres_expanded.parquet"
    out_user_genre = settings.PROCESSED_DIR / "user_genre_affinity.parquet"

    user_features.write_parquet(out_user)
    item_features.write_parquet(out_item)
    genre_priors.write_parquet(out_genre)
    item_genres.write_parquet(out_item_genres)
    user_genre_aff.write_parquet(out_user_genre)

    t1 = time.time()

    print("[DONE] Feature store created (V2+genre cross).")
    print(f"[PATH] {out_user}")
    print(f"[PATH] {out_item}")
    print(f"[PATH] {out_genre}")
    print(f"[PATH] {out_item_genres}")
    print(f"[PATH] {out_user_genre}")
    print(f"[OK] Total time: {t1 - t0:.2f}s")


if __name__ == "__main__":
    main()