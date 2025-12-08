from __future__ import annotations

from typing import Dict, List  # noqa: UP035

import polars as pl

from src.config.settings import settings


class GenreNeighborsRecommender:
    """
    Candidate generator based on time-decayed top items per genre.

    Uses genre_item_priors from feature_store.
    """

    def __init__(self, per_genre_k: int = 200) -> None:
        self.per_genre_k = per_genre_k
        self.genre_top: Dict[str, List[int]] = {}

        self.user_genre_aff: Dict[int, List[str]] = {}

    def fit(self) -> GenreNeighborsRecommender:
        priors_path = settings.PROCESSED_DIR / "genre_item_priors.parquet"
        train_path = settings.PROCESSED_DIR / "train_conf.parquet"
        items_path = settings.PROCESSED_DIR / "items.parquet"
        movies_path = settings.PROCESSED_DIR / "movies.parquet"

        if not priors_path.exists():
            raise FileNotFoundError("genre_item_priors.parquet not found. Run feature_store first.")

        priors = pl.read_parquet(priors_path)

        # Build top items per genre
        top = (
            priors.group_by("genre")
            .agg(
                pl.col("item_idx").head(self.per_genre_k).alias("top_items")
            )
        )

        self.genre_top = {
            row[0]: [int(i) for i in row[1]]
            for row in top.iter_rows()
        }

        # Build lightweight user->top genres from train_conf
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

        train = pl.read_parquet(train_path).select("user_idx", "item_idx", "confidence")

        user_genre = (
            train.join(item_genres, on="item_idx", how="left")
            .filter(pl.col("genre").is_not_null())
            .group_by("user_idx", "genre")
            .agg(pl.col("confidence").sum().alias("user_genre_conf"))
            .sort(["user_idx", "user_genre_conf"], descending=[False, True])
        )

        # Take top 3 genres per user
        user_top = (
            user_genre.group_by("user_idx")
            .agg(pl.col("genre").head(3).alias("top_genres"))
        )

        self.user_genre_aff = {
            int(u): list(genres)
            for u, genres in user_top.iter_rows()
        }

        return self

    def recommend(self, user_idx: int, k: int = 50) -> List[int]:
        genres = self.user_genre_aff.get(user_idx, [])
        if not genres:
            return []

        out: List[int] = []
        seen = set()

        for g in genres:
            for item in self.genre_top.get(g, []):
                if item not in seen:
                    seen.add(item)
                    out.append(item)
                if len(out) >= k:
                    return out

        return out[:k]

    def batch_recommend(self, user_ids: List[int], k: int = 50) -> Dict[int, List[int]]:
        return {u: self.recommend(u, k) for u in user_ids}