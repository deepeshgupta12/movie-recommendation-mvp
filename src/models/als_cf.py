from __future__ import annotations

from typing import Dict, List, Tuple  # noqa: UP035

import numpy as np
import polars as pl
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

from src.config.settings import settings


class ALSRecommender:
    def __init__(
        self,
        factors: int = 128,
        regularization: float = 0.08,
        iterations: int = 20,
        alpha: float = 40.0,
    ) -> None:
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha

        self.model: AlternatingLeastSquares | None = None
        self.user_item: csr_matrix | None = None

    def _build_matrix(self, path: str) -> Tuple[csr_matrix, int, int]:
        df = pl.read_parquet(path).select("user_idx", "item_idx", "is_positive")
        # implicit expects confidence > 0
        users = df["user_idx"].to_numpy()
        items = df["item_idx"].to_numpy()
        vals = df["is_positive"].to_numpy().astype(np.float32)

        n_users = int(df["user_idx"].max()) + 1
        n_items = int(df["item_idx"].max()) + 1

        mat = csr_matrix((vals, (users, items)), shape=(n_users, n_items))
        return mat, n_users, n_items

    def fit(self, train_path: str | None = None) -> ALSRecommender:
        path = train_path or str(settings.PROCESSED_DIR / "train.parquet")
        mat, _, _ = self._build_matrix(path)

        # confidence scaling
        mat = mat * self.alpha

        self.user_item = mat

        model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=42,
        )
        model.fit(mat)

        self.model = model
        return self

    def recommend(self, user_idx: int, k: int = 50) -> List[int]:
        if self.model is None or self.user_item is None:
            raise RuntimeError("Model not fitted.")

        recs = self.model.recommend(
            userid=user_idx,
            user_items=self.user_item,
            N=k,
            filter_already_liked_items=True
        )
        return [int(i) for i, _ in recs]

    def batch_recommend(self, user_ids: list[int], k: int = 50) -> Dict[int, List[int]]:
        return {u: self.recommend(u, k) for u in user_ids}