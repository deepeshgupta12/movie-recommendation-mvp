from __future__ import annotations

from typing import Any, Dict, List, Optional  # noqa: UP035

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from src.service.reco_service_v4 import get_v4_service

app = FastAPI(title="Movie Recommendation MVP - V4", version="v4")


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "v4"


class FeedbackRequest(BaseModel):
    user_idx: int = Field(..., ge=0)
    item_idx: int = Field(..., ge=-1)
    event: str = Field(..., description="like, unlike, watched, watch_later, remove_watch_later, start, remove_start, reset")
    split: str = Field("val")


class FeedbackResponse(BaseModel):
    status: str = "ok"
    user_idx: int
    state: Dict[str, List[int]]


class RecItem(BaseModel):
    item_idx: int
    movieId: Optional[int] = None
    title: Optional[str] = None
    poster_url: Optional[str] = None
    score: float
    reason: str
    has_tt: int = 0
    has_seq: int = 0
    has_v2: int = 0
    short_term_boost: float = 0.0
    sess_hot: int = 0
    sess_warm: int = 0
    sess_cold: int = 0


class RecommendResponse(BaseModel):
    user_idx: int
    k: int
    split: str
    items: List[RecItem]
    # Backward compatibility for older UI hooks:
    recommendations: List[RecItem]
    # New: sectioned UI rows
    sections: Dict[str, List[RecItem]] = Field(default_factory=dict)
    debug: Optional[Dict[str, Any]] = None


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.get("/recommend", response_model=RecommendResponse)
def recommend(
    user_idx: int = Query(..., ge=0),
    k: int = Query(20, ge=1, le=200),
    include_titles: bool = True,
    debug: bool = False,
    split: str = Query("val"),
    apply_feedback: bool = Query(True),
) -> RecommendResponse:
    svc = get_v4_service(split=split)
    out = svc.recommend(
        user_idx=user_idx,
        k=k,
        include_titles=include_titles,
        debug=debug,
        apply_feedback=apply_feedback,
    )
    # Ensure both keys exist for response validation
    if "recommendations" not in out:
        out["recommendations"] = out.get("items", [])
    if "items" not in out:
        out["items"] = out.get("recommendations", [])
    if "sections" not in out:
        out["sections"] = {}
    return RecommendResponse(**out)


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest) -> FeedbackResponse:
    svc = get_v4_service(split=req.split)
    if req.event.lower().strip() == "reset":
        svc.feedback.reset_user(req.user_idx)
    else:
        svc.record_feedback(req.user_idx, req.item_idx, req.event)

    state = svc.get_user_state(req.user_idx)
    return FeedbackResponse(user_idx=req.user_idx, state=state)


@app.get("/user_state", response_model=Dict[str, List[int]])
def user_state(
    user_idx: int = Query(..., ge=0),
    split: str = Query("val"),
) -> Dict[str, List[int]]:
    svc = get_v4_service(split=split)
    return svc.get_user_state(user_idx)