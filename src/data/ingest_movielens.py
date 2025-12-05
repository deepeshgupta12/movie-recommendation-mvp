from __future__ import annotations

from pathlib import Path

import polars as pl

from src.config.settings import settings


def _resolve_root() -> Path:
    """
    MovieLens extract structure typically:
      data/raw/ml-20m/ml-20m/ratings.csv
    This function finds the folder that contains ratings.csv.
    """
    base = settings.RAW_DIR / settings.MOVIELENS_VARIANT

    # Search for ratings.csv
    matches = list(base.rglob("ratings.csv"))
    if not matches:
        raise FileNotFoundError(f"ratings.csv not found under {base}")

    return matches[0].parent


def ingest() -> None:
    settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    root = _resolve_root()

    ratings_path = root / "ratings.csv"
    movies_path = root / "movies.csv"
    tags_path = root / "tags.csv"
    links_path = root / "links.csv"

    # Lazy reads for speed
    ratings = (
        pl.scan_csv(ratings_path)
        .with_columns(
            pl.col("userId").cast(pl.Int64),
            pl.col("movieId").cast(pl.Int64),
            pl.col("rating").cast(pl.Float32),
            pl.col("timestamp").cast(pl.Int64),
        )
    )

    movies = (
        pl.scan_csv(movies_path)
        .with_columns(
            pl.col("movieId").cast(pl.Int64),
            pl.col("title").cast(pl.Utf8),
            pl.col("genres").cast(pl.Utf8),
        )
    )

    tags = None
    if tags_path.exists():
        tags = (
            pl.scan_csv(tags_path)
            .with_columns(
                pl.col("userId").cast(pl.Int64),
                pl.col("movieId").cast(pl.Int64),
                pl.col("tag").cast(pl.Utf8),
                pl.col("timestamp").cast(pl.Int64),
            )
        )

    links = None
    if links_path.exists():
        links = (
            pl.scan_csv(links_path)
            .with_columns(
                pl.col("movieId").cast(pl.Int64),
                pl.col("imdbId").cast(pl.Int64, strict=False),
                pl.col("tmdbId").cast(pl.Int64, strict=False),
            )
        )

    # Write Parquet
    ratings_out = settings.PROCESSED_DIR / "ratings.parquet"
    movies_out = settings.PROCESSED_DIR / "movies.parquet"
    tags_out = settings.PROCESSED_DIR / "tags.parquet"
    links_out = settings.PROCESSED_DIR / "links.parquet"

    ratings.collect(streaming=True).write_parquet(ratings_out)
    movies.collect(streaming=True).write_parquet(movies_out)

    if tags is not None:
        tags.collect(streaming=True).write_parquet(tags_out)

    if links is not None:
        links.collect(streaming=True).write_parquet(links_out)

    print("[DONE] Parquet files created in data/processed")


def main() -> None:
    ingest()


if __name__ == "__main__":
    main()