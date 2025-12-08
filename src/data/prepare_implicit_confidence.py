from __future__ import annotations

import polars as pl

from src.config.settings import settings


def main() -> None:
    ratings_path = settings.PROCESSED_DIR / "ratings.parquet"
    users_path = settings.PROCESSED_DIR / "users.parquet"
    items_path = settings.PROCESSED_DIR / "items.parquet"

    if not ratings_path.exists():
        raise FileNotFoundError("ratings.parquet not found. Run ingestion first.")

    # If mappings don't exist, create them using the same logic as Step 2
    if not users_path.exists() or not items_path.exists():
        raise FileNotFoundError(
            "users.parquet/items.parquet not found. "
            "Run src.data.prepare_interactions first."
        )

    ratings = pl.read_parquet(ratings_path)
    users = pl.read_parquet(users_path)
    items = pl.read_parquet(items_path)

    # Join to stable indices
    df = (
        ratings.join(users, on="userId", how="inner")
               .join(items, on="movieId", how="inner")
               .select("userId", "movieId", "user_idx", "item_idx", "rating", "timestamp")
    )

    # Convert explicit ratings to implicit confidence
    # Simple, robust MVP rule:
    # - confidence = max(rating - 2.5, 0)
    #   so ratings 3,4,5 become 0.5,1.5,2.5
    df = df.with_columns(
        (pl.col("rating") - 2.5).clip(lower_bound=0.0).alias("confidence")
    )

    out = settings.PROCESSED_DIR / "implicit_confidence.parquet"
    df.write_parquet(out)

    print("[DONE] implicit_confidence.parquet created.")


if __name__ == "__main__":
    main()