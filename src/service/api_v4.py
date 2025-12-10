from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional  # noqa: UP035

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.service.reco_service_v4 import V4RecommenderService, V4ServiceConfig

app = FastAPI(title="Movie Recommendation API", version="v4")


# -----------------------------
# Models (lightweight, no strict route validation)
# -----------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "v4"


class FeedbackEvent(BaseModel):
    user_idx: int
    item_idx: int
    event: str = Field(..., description="like | watched | watch_later | start | dislike etc.")
    ts: Optional[int] = None


# -----------------------------
# Service factory
# -----------------------------

@lru_cache(maxsize=4)
def _svc(split: str) -> V4RecommenderService:
    cfg = V4ServiceConfig(split=split)
    return V4RecommenderService(cfg)


def _normalize_split(split: str) -> str:
    s = (split or "val").lower().strip()
    return s if s in {"val", "test"} else "val"


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
    """
    V4 recommend endpoint.
    We intentionally avoid FastAPI response_model here to prevent any
    response validation edge-case from killing the reload worker and causing
    RemoteDisconnected.

    The service is expected to return:
      {
        "items": [...],
        "debug": {...},
        "split": "val"
      }
    or equivalently "recommendations" instead of "items".
    """
    split = _normalize_split(split)

    try:
        svc = _svc(split)

        out = svc.recommend(
            user_idx=user_idx,
            k=k,
            include_titles=include_titles,
            debug=debug,
        )

        if not isinstance(out, dict):
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Service returned non-dict payload",
                    "type": str(type(out)),
                    "split": split,
                },
            )

        items = out.get("items")
        if items is None:
            items = out.get("recommendations", [])

        if not isinstance(items, list):
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Service returned invalid items shape",
                    "items_type": str(type(items)),
                    "split": split,
                },
            )

        # Build minimal stable payload
        payload: Dict[str, Any] = {
            "user_idx": user_idx,
            "k": k,
            "split": split,
            "recommendations": items,
        }

        if debug:
            payload["debug"] = out.get("debug", {})
        else:
            payload["debug"] = None

        return JSONResponse(status_code=200, content=payload)

    except FileNotFoundError as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": "FileNotFoundError",
                "detail": str(e),
                "split": split,
            },
        )
    except Exception as e:
        # Critical: never raise here; always respond
        return JSONResponse(
            status_code=500,
            content={
                "error": type(e).__name__,
                "detail": str(e),
                "split": split,
                "hint": "Check reco_service_v4.recommend output shape and poster/session paths.",
            },
        )


@app.post("/feedback")
def feedback(evt: FeedbackEvent):
    """
    Lightweight feedback endpoint.
    If the service exposes record_feedback/feedback, invoke it.
    Otherwise acknowledge OK to keep UI stable.
    """
    try:
        svc = _svc("val")

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

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": type(e).__name__, "detail": str(e)},
        )