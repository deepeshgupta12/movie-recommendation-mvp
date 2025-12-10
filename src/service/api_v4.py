# src/service/api_v4.py
from __future__ import annotations

from typing import Any, Dict, List, Optional  # noqa: UP035

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict

from src.service.reco_service_v4 import get_v4_service

app = FastAPI(title="Movie Reco MVP - V4", version="v4")


# -----------------------------
# Models
# -----------------------------

class RecommendItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    item_idx: int
    movieId: Optional[int] = None
    title: str = ""
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
    model_config = ConfigDict(extra="ignore")

    user_idx: int
    k: int
    items: List[RecommendItem] = Field(default_factory=list)
    split: str = "val"
    debug: Optional[Dict[str, Any]] = None


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_idx: int
    item_idx: int
    event: str


class FeedbackResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ok: bool = True
    user_idx: int
    item_idx: int
    event: str


class UserStateResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_idx: int
    like: List[int] = Field(default_factory=list)
    watched: List[int] = Field(default_factory=list)
    watch_later: List[int] = Field(default_factory=list)
    dislike: List[int] = Field(default_factory=list)


# -----------------------------
# Routes
# -----------------------------

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
):
    try:
        svc = get_v4_service(split=split)
        out = svc.recommend(
            user_idx=user_idx,
            k=k,
            include_titles=include_titles,
            debug=debug,
        )
        return out
    except Exception as e:
        # Ensure we always send a response (avoid RemoteDisconnected)
        return JSONResponse(
            status_code=500,
            content={
                "error": "recommend_failed",
                "detail": str(e),
                "user_idx": int(user_idx),
                "k": int(k),
                "split": split,
            },
        )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest, split: str = Query("val", pattern="^(val|test)$")):
    svc = get_v4_service(split=split)
    out = svc.record_feedback(req.user_idx, req.item_idx, req.event)
    return out


@app.get("/user_state", response_model=UserStateResponse)
def user_state(
    user_idx: int = Query(..., ge=0),
    split: str = Query("val", pattern="^(val|test)$"),
):
    svc = get_v4_service(split=split)
    st = svc.get_user_state(user_idx)
    return {"user_idx": int(user_idx), **st}