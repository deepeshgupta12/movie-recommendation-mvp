from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional  # noqa: UP035

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.service.reco_service_v4 import V4RecommenderService, V4ServiceConfig

app = FastAPI(title="Movie Recommendation API", version="v4")


# -----------------------------
# Models (lightweight)
# -----------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "v4"


class FeedbackEvent(BaseModel):
    user_idx: int
    item_idx: int
    event: str = Field(..., description="like | watched | watch_later | start | dislike")
    ts: Optional[int] = None
    split: Optional[str] = None


# -----------------------------
# Global exception handler
# -----------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Ensure we NEVER crash silently due to handler paths
    return JSONResponse(
        status_code=500,
        content={
            "error": type(exc).__name__,
            "detail": str(exc),
            "path": str(request.url.path),
        },
    )


# -----------------------------
# Service factory
# -----------------------------

def _normalize_split(split: str) -> str:
    s = (split or "val").lower().strip()
    return s if s in {"val", "test"} else "val"


@lru_cache(maxsize=4)
def get_v4_service(split: str) -> V4RecommenderService:
    split = _normalize_split(split)
    cfg = V4ServiceConfig(split=split)
    return V4RecommenderService(cfg)


# -----------------------------
# Routes
# -----------------------------

@app.get("/health")
def health() -> HealthResponse:
    return HealthResponse()


@app.get("/recommend")
def recommend(
    user_idx: int = Query(..., ge=0),
    k: int = Query(20, ge=1, le=200),
    include_titles: bool = Query(True),
    debug: bool = Query(False),
    split: str = Query("val"),
):
    split = _normalize_split(split)

    # We keep this handler extremely flat to reduce risk of reload crashes.
    svc = get_v4_service(split)

    out = svc.recommend(
        user_idx=user_idx,
        k=k,
        include_titles=include_titles,
        debug=debug,
    )

    # Accept both shapes to stay stable across internal service evolutions
    items = None
    if isinstance(out, dict):
        items = out.get("items")
        if items is None:
            items = out.get("recommendations")

    if items is None:
        items = []

    payload: Dict[str, Any] = {
        "user_idx": user_idx,
        "k": k,
        "split": split,
        "recommendations": items,
        "debug": out.get("debug") if isinstance(out, dict) and debug else None,
    }

    return JSONResponse(status_code=200, content=payload)


@app.post("/feedback")
def feedback(evt: FeedbackEvent):
    split = _normalize_split(evt.split or "val")
    svc = get_v4_service(split)

    if hasattr(svc, "record_feedback"):
        svc.record_feedback(
            user_idx=evt.user_idx,
            item_idx=evt.item_idx,
            event=evt.event,
            ts=evt.ts,
        )
    elif hasattr(svc, "feedback"):
        svc.feedback(
            user_idx=evt.user_idx,
            item_idx=evt.item_idx,
            event=evt.event,
            ts=evt.ts,
        )

    return JSONResponse(status_code=200, content={"status": "ok", "version": "v4"})