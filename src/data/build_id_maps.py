from __future__ import annotations

from pathlib import Path

import polars as pl

from src.config.settings import settings


def _processed_dir() -> Path:
    return Path(settings.PROCESSED_DIR)


def main():
    """
    Build stable user/item id maps for local MVP.

    Assumption:
    - user_idx and item_idx were created using sorted unique IDs.
    This is used for UI/demo/title resolution and future V3 blending diagnostics.
    """
    processed = _processed_dir()

    ratings_path = processed / "ratings.parquet"
    movies_path = processed / "movies.parquet"

    if not ratings_path.exists():
        raise FileNotFoundError(f"Missing {ratings_path}. Expected from Step 1 ingestion.")
    if not movies_path.exists():
        raise FileNotFoundError(f"Missing {movies_path}. Expected from Step 1 ingestion.")

    print("\n[START] Loading ratings + movies...")
    ratings = pl.read_parquet(ratings_path).select(["userId", "movieId"]).unique()
    movies = pl.read_parquet(movies_path).select(["movieId", "title"]).unique()

    print(f"[OK] unique userIds: {ratings['userId'].n_unique()}")
    print(f"[OK] unique movieIds: {ratings['movieId'].n_unique()}")

    print("[START] Building dim_users...")
    users = (
        ratings.select("userId")
        .unique()
        .sort("userId")
        .with_row_index(name="user_idx", offset=0)
        .select(["user_idx", "userId"])
    )

    print("[START] Building dim_items...")
    items = (
        ratings.select("movieId")
        .unique()
        .sort("movieId")
        .with_row_index(name="item_idx", offset=0)
        .select(["item_idx", "movieId"])
        .join(movies, on="movieId", how="left")
    )

    users_out = processed / "dim_users.parquet"
    items_out = processed / "dim_items.parquet"

    users.write_parquet(users_out)
    items.write_parquet(items_out)

    print("[DONE] ID maps created.")
    print(f"[PATH] {users_out}")
    print(f"[PATH] {items_out}")


if __name__ == "__main__":
    main()