from __future__ import annotations

from src.neural.contracts import RetrievalResult


def retrieve_sequence_candidates(
    user_idx: int,
    k: int = 200,
) -> RetrievalResult:
    """
    Contract-first placeholder.

    In Step 8.6:
    - load trained GRU model
    - produce next-item candidates
    """
    raise NotImplementedError("Sequence candidate retrieval will be implemented in Step 8.6.")