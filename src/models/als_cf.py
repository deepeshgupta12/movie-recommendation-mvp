from __future__ import annotations

from typing import Dict, List, Tuple, Union  # noqa: UP035

import numpy as np
import polars as pl
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

from src.config.settings import settings


class ALSRecommender:
    """
    Implicit-feedback ALS for candidate generation.

    Correctness + stability fixes:
    - Use GLOBAL n_users/n_items from users/items mapping files.
      This prevents index errors for users absent in train after time-split.
    - Build USER-ITEM matrix for recommend-time filtering.
    - Fit ALS on ITEM-USER matrix (transpose), as expected by implicit.
    - Pass a 1-row user_items slice to recommend().
    - Support multiple recommend() return shapes across implicit versions.
    """

    def __init__(
        self,
        factors: int = 128,
        regularization: float = 0.08,
        iterations: int = 20,
        alpha: float = 40.0,
        random_state: int = 42,
    ) -> None:
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.random_state = random_state

        self.model: AlternatingLeastSquares | None = None
        self.user_item: csr_matrix | None = None  # (n_users, n_items)

        self.n_users: int = 0
        self.n_items: int = 0

    def _load_global_dimensions(self) -> Tuple[int, int]:
        users_path = settings.PROCESSED_DIR / "users.parquet"
        items_path = settings.PROCESSED_DIR / "items.parquet"

        if not users_path.exists() or not items_path.exists():
            raise FileNotFoundError(
                "users.parquet/items.parquet not found. "
                "Run src.data.prepare_interactions first."
            )

        users_df = pl.read_parquet(users_path)
        items_df = pl.read_parquet(items_path)

        # users/items were created with row_index as *_idx
        n_users = users_df.height
        n_items = items_df.height

        return int(n_users), int(n_items)

    def _build_user_item_matrix(self, path: str, n_users: int, n_items: int) -> csr_matrix:
        df = pl.read_parquet(path).select("user_idx", "item_idx", "is_positive")

        if df.is_empty():
            # No train interactions; return an all-zero matrix
            return csr_matrix((n_users, n_items), dtype=np.float32)

        users = df["user_idx"].to_numpy()
        items = df["item_idx"].to_numpy()
        vals = df["is_positive"].to_numpy().astype(np.float32)

        mat = csr_matrix((vals, (users, items)), shape=(n_users, n_items))
        return mat

    def fit(self, train_path: str | None = None) -> "ALSRecommender":
        path = train_path or str(settings.PROCESSED_DIR / "train.parquet")

        n_users, n_items = self._load_global_dimensions()
        self.n_users = n_users
        self.n_items = n_items

        user_item = self._build_user_item_matrix(path, n_users, n_items)
        self.user_item = user_item

        # implicit expects item-user for fitting
        item_user = user_item.T.tocsr()

        # Confidence scaling
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

    def _normalize_recommend_output(
        self,
        recs: Union[List[Tuple[int, float]], Tuple[np.ndarray, np.ndarray]]
    ) -> List[int]:
        # Option B: (item_ids_array, scores_array)
        if isinstance(recs, tuple) and len(recs) == 2:
            item_ids = recs[0]
            return [int(i) for i in item_ids.tolist()]

        # Option A: list of tuples
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
            raise RuntimeError("ALSRecommender is not fitted yet.")

        # Guard: if user outside global bounds
        if user_idx < 0 or user_idx >= self.n_users:
            return []

        # Guard: if implicit internally returns smaller user_factors
        # (rare, but safe for local MVP)
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