from __future__ import annotations

from typing import List  # noqa: UP035

from src.neural.contracts import RetrievalResult
from src.neural.paths import HNSW_INDEX, USER_EMB


def retrieve_two_tower_candidates(
    user_idx: int,
    k: int = 200,
) -> RetrievalResult:
    """
    Contract-first placeholder.

    In Step 8.5:
    - load user embedding
    - query HNSW index
    - return top-k candidates

    Notes:
    - k here is candidate pool size (not final UI k).
    """
    raise NotImplementedError("Two-tower ANN retrieval will be implemented in Step 8.5.")