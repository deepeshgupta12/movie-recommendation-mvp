# src/service/api_v4.py

from __future__ import annotations

from typing import List, Literal, Optional  # noqa: UP035

from fastapi import FastAPI, Query
from pydantic import BaseModel, ConfigDict, Field

from src.service.reco_service_v4 import (
    V4RecommenderService,
    V4ServiceConfig,
    get_v4_service,
)

Split = Literal["val", "test"]
FeedbackEvent = Literal[
    "like",
    "remove_like",
    "watched",
    "watch_later",
    "remove_watch_later",
    "skip",
]


app = FastAPI(title="Movie Recommendation MVP - V4", version="v4")


class RecommendationItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    item_idx: int
    movieId: Optional[int] = None
    title: Optional[str] = None
    poster_url: Optional[str] = None
    score: float

    reason: str
    bucket: str

    has_tt: int = 0
    has_seq: int = 0
    has_v2: int = 0

    short_term_boost: float = 0.0
    sess_hot: int = 0
    sess_warm: int = 0
    sess_cold: int = 0


class RecommendResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_idx: int
    k: int
    split: Split

    recommendations: List[RecommendationItem] = Field(default_factory=list)
    sections: Optional[dict] = None
    debug: Optional[dict] = None


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_idx: int
    item_idx: int
    event: FeedbackEvent


class FeedbackResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: str
    detail: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "version": "v4"}


@app.get("/recommend", response_model=RecommendResponse)
def recommend(
    user_idx: int = Query(..., ge=0),
    k: int = Query(20, ge=1, le=200),
    include_titles: bool = Query(True),
    debug: bool = Query(False),
    split: Split = Query("val"),
    apply_diversity: bool = Query(True),
):
    svc = get_v4_service(V4ServiceConfig(split=split, apply_diversity=apply_diversity))

    out = svc.recommend(
        user_idx=user_idx,
        k=k,
        include_titles=include_titles,
        debug=debug,
        apply_diversity=apply_diversity,
    )

    # Map to API schema
    items = out.get("items", [])
    sections = out.get("sections")

    return {
        "user_idx": user_idx,
        "k": k,
        "split": split,
        "recommendations": items,
        "sections": sections,
        "debug": out.get("debug") if debug else None,
    }


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    svc = get_v4_service(V4ServiceConfig(split="val"))
    result = svc.record_feedback(req.user_idx, req.item_idx, req.event)
    return result