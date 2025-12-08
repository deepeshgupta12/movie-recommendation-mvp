from __future__ import annotations

from typing import Dict, List  # noqa: UP035

import polars as pl

from src.config.settings import settings


class PopularityRecommender:
    def __init__(self) -> None:
        self.top_items: List[int] = []

    def fit(self, train_path: str | None = None) -> PopularityRecommender:
        path = train_path or str(settings.PROCESSED_DIR / "train.parquet")
        df = pl.read_parquet(path)

        # Count positives for popularity
        pop = (
            df.filter(pl.col("is_positive") == 1)
              .group_by("item_idx")
              .len()
              .sort("len", descending=True)
        )

        self.top_items = pop["item_idx"].to_list()
        return self

    def recommend(self, user_idx: int, k: int = 50) -> List[int]:
        return self.top_items[:k]

    def batch_recommend(self, user_ids: list[int], k: int = 50) -> Dict[int, List[int]]:
        return {u: self.recommend(u, k) for u in user_ids}