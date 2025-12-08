from __future__ import annotations

from typing import Dict, List, Tuple, Union  # noqa: UP035

import numpy as np
import polars as pl
from implicit.nearest_neighbours import ItemItemRecommender
from scipy.sparse import csr_matrix

from src.config.settings import settings


class ItemItemSimilarityRecommender:
    """
    Item-Item KNN recommender for candidate generation.

    Compatibility fixes:
    - Some implicit versions require float64 ('double') buffers for all_pairs_knn.
    - recommend() output shape varies across versions:
        A) list[(item, score)]
        B) (item_ids_array, scores_array)
      We normalize both.
    """

    def __init__(self, K: int = 100) -> None:
        self.K = K

        self.model: ItemItemRecommender | None = None
        self.user_item: csr_matrix | None = None

        self.n_users: int = 0
        self.n_items: int = 0

    def _load_global_dimensions(self) -> Tuple[int, int]:
        users_df = pl.read_parquet(settings.PROCESSED_DIR / "users.parquet")
        items_df = pl.read_parquet(settings.PROCESSED_DIR / "items.parquet")
        return int(users_df.height), int(items_df.height)

    def _build_user_item_matrix(self, path: str, n_users: int, n_items: int) -> csr_matrix:
        df = pl.read_parquet(path).select("user_idx", "item_idx", "confidence")

        if df.is_empty():
            return csr_matrix((n_users, n_items), dtype=np.float64)

        users = df["user_idx"].to_numpy()
        items = df["item_idx"].to_numpy()
        vals = df["confidence"].to_numpy().astype(np.float64)

        return csr_matrix((vals, (users, items)), shape=(n_users, n_items), dtype=np.float64)

    def fit(self, train_path: str | None = None) -> ItemItemSimilarityRecommender:
        path = train_path or str(settings.PROCESSED_DIR / "train_conf.parquet")

        n_users, n_items = self._load_global_dimensions()
        self.n_users, self.n_items = n_users, n_items

        user_item = self._build_user_item_matrix(path, n_users, n_items)

        # Store for recommend-time filtering
        self.user_item = user_item

        # implicit expects item-user for fitting
        item_user = user_item.T.tocsr().astype(np.float64)

        model = ItemItemRecommender(K=self.K)
        model.fit(item_user)

        self.model = model
        return self

    def _normalize_recommend_output(
        self,
        recs: Union[List[Tuple[int, float]], Tuple[np.ndarray, np.ndarray]]
    ) -> List[int]:
        # Option B
        if isinstance(recs, tuple) and len(recs) == 2:
            item_ids = recs[0]
            return [int(i) for i in item_ids.tolist()]

        # Option A
        if isinstance(recs, list):
            out: List[int] = []
            for row in recs:
                try:
                    out.append(int(row[0]))
                except Exception:
                    continue
            return out

        return []

    def recommend(self, user_idx: int, k: int = 50) -> List[int]:
        if self.model is None or self.user_item is None:
            raise RuntimeError("ItemItemSimilarityRecommender is not fitted yet.")

        if user_idx < 0 or user_idx >= self.n_users:
            return []

        user_items_1row = self.user_item[user_idx]

        recs = self.model.recommend(
            userid=user_idx,
            user_items=user_items_1row,
            N=k,
            filter_already_liked_items=True,
        )

        return self._normalize_recommend_output(recs)

    def batch_recommend(self, user_ids: List[int], k: int = 50) -> Dict[int, List[int]]:
        return {u: self.recommend(u, k) for u in user_ids}