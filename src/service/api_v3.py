# src/service/api_v3.py
from __future__ import annotations

from typing import Any, Dict, Optional  # noqa: UP035

from fastapi import FastAPI, Query
from pydantic import BaseModel

from src.service.reco_service_v3 import (
    V3RecommenderService,  # noqa: F401
    V3ServiceConfig,  # noqa: F401
    get_v3_service,
)

app = FastAPI(title="Movie Recommendation V3 API")


# -------------------------
# Schemas
# -------------------------

class FeedbackIn(BaseModel):
    user_idx: int
    item_idx: int
    action: str  # like/unlike/watched/unwatch/start/unstart


# -------------------------
# Routes
# -------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "v3"}


@app.get("/recommend")
def recommend(
    user_idx: int = Query(..., ge=0),
    k: int = Query(20, ge=1, le=200),
    include_titles: bool = Query(True),
    debug: bool = Query(False),
    split: str = Query("test", pattern="^(test|val)$"),
):
    svc = get_v3_service()
    return svc.recommend(
        user_idx=user_idx,
        k=k,
        include_titles=include_titles,
        debug=debug,
        split=split,
    )


@app.post("/feedback")
def feedback(payload: FeedbackIn):
    svc = get_v3_service()
    return svc.record_feedback(
        user_idx=payload.user_idx,
        item_idx=payload.item_idx,
        action=payload.action,
    )


@app.get("/user_state")
def user_state(
    user_idx: int = Query(..., ge=0),
):
    svc = get_v3_service()
    return {
        "user_idx": int(user_idx),
        **svc.get_user_state(user_idx),
    }