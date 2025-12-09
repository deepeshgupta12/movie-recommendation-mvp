from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.neural.contracts import SequenceConfig


@dataclass
class GruArtifacts:
    model_path: str
    config: SequenceConfig


class Gru4RecModel:
    """
    Contract placeholder for GRU-based sequential recommender.
    """

    def __init__(self, config: Optional[SequenceConfig] = None):
        self.config = config or SequenceConfig()
        self._model = None

    def fit(self, train_df, val_df=None) -> GruArtifacts:
        raise NotImplementedError("GRU training will be implemented in Step 8.6.")

    def save(self, path: str) -> str:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> Gru4RecModel:
        raise NotImplementedError

    def recommend_next(self, user_idx: int, k: int = 200):
        raise NotImplementedError