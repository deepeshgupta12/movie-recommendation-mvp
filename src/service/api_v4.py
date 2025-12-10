"""
V4 API (Session-aware ranker)

Provides FastAPI endpoints for V4 recommendation service.

Design goals:
- Split-aware, cached service instances.
- No dependency on get_v4_service exported from reco_service_v4.
- Mirrors V3 API surface to keep Streamlit wiring simple.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Optional  # noqa: UP035

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.service.reco_service_v4 import V4RecommenderService, V4ServiceConfig

app = FastAPI(
    title="Movie Recommendation API - V4",
    version="4.0.0",
    description="Session-aware hybrid recommender (Two-Tower ANN + GRU signals + V2 prior + Session features).",
)

# Simple in-process cache (good enough for local dev)
_SERVICE_CACHE: Dict[str, V4RecommenderService] = {}


def _get_service(split: str) -> V4RecommenderService:
    split = (split or "val").strip().lower()
    if split not in {"val", "test"}:
        raise HTTPException(status_code=400, detail=f"Invalid split='{split}'. Use val|test.")

    if split in _SERVICE_CACHE:
        return _SERVICE_CACHE[split]

    cfg = V4ServiceConfig(split=split)
    svc = V4RecommenderService(cfg)
    _SERVICE_CACHE[split] = svc
    return svc


class RecommendResponse(BaseModel):
    user_idx: int
    split: str
    k: int
    recommendations: list
    debug: Optional[dict] = None


@app.get("/health")
def health():
    return {"status": "ok", "version": "v4"}


@app.get("/config")
def config(split: str = Query("val")):
    try:
        svc = _get_service(split)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # We don't assume cfg is a pydantic model; safest is dataclass-like
    cfg = getattr(svc, "cfg", None)
    if cfg is None:
        return {"split": split, "note": "service config not exposed"}
    try:
        return asdict(cfg)
    except Exception:
        return {"split": getattr(cfg, "split", split)}


@app.get("/recommend", response_model=RecommendResponse)
def recommend(
    user_idx: int = Query(..., ge=0),
    k: int = Query(20, ge=1, le=200),
    include_titles: bool = Query(True),
    debug: bool = Query(False),
    split: str = Query("val"),
):
    """
    Recommend top-k items for a given user_idx using V4 session-aware ranker.
    """
    try:
        svc = _get_service(split)
        out = svc.recommend(
            user_idx=user_idx,
            k=k,
            include_titles=include_titles,
            debug=debug,
        )
        # Expected shape from service:
        # {
        #   "user_idx": int,
        #   "split": str,
        #   "k": int,
        #   "recommendations": [...],
        #   "debug": {... optional ...}
        # }
        if not isinstance(out, dict):
            # Backward compatibility if service returns list
            out = {
                "user_idx": user_idx,
                "split": split,
                "k": k,
                "recommendations": out,
                "debug": None,
            }

        out.setdefault("user_idx", user_idx)
        out.setdefault("split", split)
        out.setdefault("k", k)
        out.setdefault("debug", None)

        return out

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"V4 recommend failed: {e}")