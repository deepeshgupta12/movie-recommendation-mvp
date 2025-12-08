from __future__ import annotations

import polars as pl

from src.config.settings import settings


def main() -> None:
    ratings_path = settings.PROCESSED_DIR / "ratings.parquet"
    if not ratings_path.exists():
        raise FileNotFoundError("ratings.parquet not found. Run ingestion first.")

    df = pl.read_parquet(ratings_path)

    # Implicit positive signal threshold for candidate generation
    # This is a pragmatic default for MVP.
    df = df.with_columns(
        (pl.col("rating") >= 3.5).cast(pl.Int8).alias("is_positive")
    )

    # Create stable integer indices
    users = df.select("userId").unique().sort("userId").with_row_index("user_idx")
    items = df.select("movieId").unique().sort("movieId").with_row_index("item_idx")

    interactions = (
        df.join(users, on="userId", how="inner")
          .join(items, on="movieId", how="inner")
          .select(
              "userId", "movieId",
              "user_idx", "item_idx",
              "rating", "is_positive", "timestamp"
          )
    )

    users_out = settings.PROCESSED_DIR / "users.parquet"
    items_out = settings.PROCESSED_DIR / "items.parquet"
    inter_out = settings.PROCESSED_DIR / "interactions.parquet"

    users.write_parquet(users_out)
    items.write_parquet(items_out)
    interactions.write_parquet(inter_out)

    print("[DONE] users/items/interactions parquet created.")


if __name__ == "__main__":
    main()