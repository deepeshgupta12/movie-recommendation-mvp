# app/api/routes/recommend.py
from __future__ import annotations

from typing import Any, Dict, List  # noqa: UP035

from fastapi import APIRouter, HTTPException, Path, Query

from app.deps import get_service

router = APIRouter(prefix="/recommend", tags=["recommend"])


class RecommendItemSchema(dict):
    """
    Lightweight schema-like mapping for response items.
    Using dict subclass to avoid Pydantic dependency if you want to stay minimal.
    """

    @classmethod
    def from_domain(cls, obj) -> RecommendItemSchema:
        return cls(
            item_idx=obj.item_idx,
            title=obj.title,
            genres=obj.genres,
            score=obj.score,
            reasons=obj.reasons,
        )


@router.get("/user/{user_idx}")
async def recommend_for_user(
    user_idx: int = Path(..., description="Internal user_idx from the Movielens mapping."),
    k: int = Query(10, ge=1, le=50, description="Number of recommendations to return."),
) -> List[dict]:
    """
    Return top-k ranked movie recommendations for a given user_idx.
    """
    service = get_service()

    try:
        recs = service.recommend(user_idx=user_idx, k=k)
    except Exception as e:
        # Surface as 500 with message for now; can add better structured errors later.
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {e}") from e

    return [RecommendItemSchema.from_domain(r) for r in recs]


@router.get("/user/{user_idx}/debug")
async def recommend_for_user_debug(
    user_idx: int = Path(..., description="Internal user_idx from the Movielens mapping."),
    k: int = Query(10, ge=1, le=50, description="Number of recommendations to debug."),
) -> Dict[str, Any]:
    """
    Debug endpoint returning:
    - candidates per source
    - blended list
    - ranked top-k with feature vectors and reasons
    """
    service = get_service()

    try:
        debug_payload = service.recommend_debug(user_idx=user_idx, k=k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to debug recommendations: {e}") from e

    return debug_payload