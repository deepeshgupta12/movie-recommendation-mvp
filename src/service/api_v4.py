# src/service/api_v4.py

from __future__ import annotations

from typing import Any, Dict, List, Optional  # noqa: UP035

from fastapi import FastAPI, Query
from pydantic import BaseModel, ConfigDict, Field

from src.service.reco_service_v4 import get_v4_service

app = FastAPI(title="Movie Recommendation MVP - V4 API", version="v4")


# ---------------------------
# Models
# ---------------------------

class ItemRec(BaseModel):
    model_config = ConfigDict(extra="ignore")

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
    sess_cold: int = 1

    blend_source_raw: Optional[str] = None


class RecommendResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_idx: int
    k: int
    split: str
    recommendations: List[ItemRec] = Field(default_factory=list)
    debug: Optional[Dict[str, Any]] = None


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_idx: int
    item_idx: int
    event_type: str


class FeedbackResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ok: bool
    event: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    event_type: Optional[str] = None


# ---------------------------
# Routes
# ---------------------------

@app.get("/health")
def health():
    return {"status": "ok", "version": "v4"}


@app.get("/recommend", response_model=RecommendResponse)
def recommend(
    user_idx: int = Query(..., ge=0),
    k: int = Query(20, ge=1, le=200),
    include_titles: bool = True,
    debug: bool = False,
    split: str = Query("val", pattern="^(val|test)$"),
    apply_diversity: bool = True,
):
    svc = get_v4_service(split=split)
    out = svc.recommend(
        user_idx=user_idx,
        k=k,
        include_titles=include_titles,
        debug=debug,
        apply_diversity=apply_diversity,
    )
    return out


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    svc = get_v4_service(split="val")  # feedback always writes global v4 file
    res = svc.record_feedback(req.user_idx, req.item_idx, req.event_type)
    return res