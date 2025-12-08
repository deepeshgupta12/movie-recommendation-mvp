# app/deps.py
from __future__ import annotations

from typing import Optional

from src.service.recommender_service import V2RecommenderService

_service: Optional[V2RecommenderService] = None  # noqa: UP045


def init_service(max_k: int = 50) -> None:
    """
    Initialize the global V2RecommenderService singleton.
    Called once on FastAPI startup.
    """
    global _service
    if _service is None:
        _service = V2RecommenderService(max_k=max_k).load()


def get_service() -> V2RecommenderService:
    """
    Retrieve the global V2RecommenderService singleton.
    Raises if not initialized (should only happen if startup hook failed).
    """
    if _service is None:
        raise RuntimeError("Recommender service not initialized. Startup may have failed.")
    return _service