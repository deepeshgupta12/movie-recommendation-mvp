from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional  # noqa: UP035

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from src.service.reco_service_v4 import get_v4_service

app = FastAPI(title="Movie Recommendation MVP - V4", version="v4")


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "v4"


class RecommendationItem(BaseModel):
    item_idx: int
    movieId: Optional[int] = None
    title: Optional[str] = None
    poster_url: Optional[str] = None

    score: float
    reason: str
    category: str

    has_tt: int = 0
    has_seq: int = 0
    has_v2: int = 0

    short_term_boost: float = 0.0
    sess_hot: int = 0
    sess_warm: int = 0
    sess_cold: int = 0

    blend_score: float = 0.0
    ranker_score: float = 0.0
    feedback_boost: float = 0.0


class RecommendResponse(BaseModel):
    user_idx: int
    k: int
    split: str
    recommendations: List[RecommendationItem]
    debug: Optional[Dict[str, Any]] = None


class FeedbackRequest(BaseModel):
    user_idx: int = Field(..., ge=0)
    item_idx: int = Field(..., ge=0)
    event: Literal["like", "watched", "watch_later", "start", "skip"]


class FeedbackResponse(BaseModel):
    status: str = "ok"
    user_idx: int
    item_idx: int
    event: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.get("/recommend", response_model=RecommendResponse)
def recommend(
    user_idx: int = Query(..., ge=0),
    k: int = Query(20, ge=1, le=200),
    include_titles: bool = True,
    debug: bool = False,
    split: str = Query("val", pattern="^(val|test)$"),
) -> RecommendResponse:
    svc = get_v4_service(split=split)
    out = svc.recommend(user_idx=user_idx, k=k, include_titles=include_titles, debug=debug)
    return RecommendResponse(**out)


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest) -> FeedbackResponse:
    svc = get_v4_service(split="val")  # feedback is split-agnostic for live UI loop
    svc.record_feedback(req.user_idx, req.item_idx, req.event)
    return FeedbackResponse(user_idx=req.user_idx, item_idx=req.item_idx, event=req.event)