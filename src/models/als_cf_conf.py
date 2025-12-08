from __future__ import annotations

from typing import Dict, List, Tuple, Union  # noqa: UP035

import numpy as np
import polars as pl
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

from src.config.settings import settings


class ALSConfidenceRecommender:
    """
    ALS trained on implicit confidence signals.
    This should behave far better than the binary version.
    """

    def __init__(
        self,
        factors: int = 128,
        regularization: float = 0.08,
        iterations: int = 20,
        alpha: float = 20.0,
        random_state: int = 42,
    ) -> None:
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.random_state = random_state

        self.model: AlternatingLeastSquares | None = None
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
            return csr_matrix((n_users, n_items), dtype=np.float32)

        users = df["user_idx"].to_numpy()
        items = df["item_idx"].to_numpy()
        vals = df["confidence"].to_numpy().astype(np.float32)

        return csr_matrix((vals, (users, items)), shape=(n_users, n_items))

    def _normalize_recommend_output(
        self,
        recs: Union[List[Tuple[int, float]], Tuple[np.ndarray, np.ndarray]]
    ) -> List[int]:
        if isinstance(recs, tuple) and len(recs) == 2:
            return [int(i) for i in recs[0].tolist()]
        if isinstance(recs, list):
            return [int(row[0]) for row in recs if row]
        return []

    def fit(self, train_path: str | None = None) -> ALSConfidenceRecommender:
        path = train_path or str(settings.PROCESSED_DIR / "train_conf.parquet")

        n_users, n_items = self._load_global_dimensions()
        self.n_users, self.n_items = n_users, n_items

        user_item = self._build_user_item_matrix(path, n_users, n_items)
        self.user_item = user_item

        item_user = user_item.T.tocsr()
        item_user = item_user * self.alpha

        model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
        )
        model.fit(item_user)

        self.model = model
        return self

    def recommend(self, user_idx: int, k: int = 50) -> List[int]:
        if self.model is None or self.user_item is None:
            raise RuntimeError("ALSConfidenceRecommender is not fitted yet.")

        if user_idx < 0 or user_idx >= self.n_users:
            return []

        if hasattr(self.model, "user_factors"):
            if user_idx >= self.model.user_factors.shape[0]:
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