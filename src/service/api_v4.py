"""
V4 API (Session-aware ranker)

Robust FastAPI layer for V4 recommendations.

Key guarantees:
- Split-aware, cached service instances.
- Does NOT rely on get_v4_service exported from reco_service_v4.
- Normalizes service outputs so API always returns:
    {
      user_idx,
      split,
      k,
      recommendations: [...]
      debug: {... optional ...}
    }

Backward compatibility:
- If service returns "items", API maps it to "recommendations".
- If service returns list, wraps it into "recommendations".
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional  # noqa: UP035

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.service.reco_service_v4 import V4RecommenderService, V4ServiceConfig

app = FastAPI(
    title="Movie Recommendation API - V4",
    version="4.0.1",
    description="Session-aware hybrid recommender (Two-Tower ANN + GRU + V2 prior + Session features).",
)

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

    # Accept both keys from upstream via alias, but we will also normalize in code.
    recommendations: List[Any] = Field(default_factory=list, alias="items")

    debug: Optional[dict] = None

    # Pydantic v2 config
    model_config = {"populate_by_name": True, "extra": "allow"}

    # Pydantic v1 fallback
    class Config:
        allow_population_by_field_name = True
        extra = "allow"


def _normalize_out(
    *,
    raw: Any,
    user_idx: int,
    split: str,
    k: int,
) -> dict:
    """
    Normalize service output into the API contract.

    Supported shapes:
    1) dict with "recommendations"
    2) dict with "items"
    3) list of items
    """
    if isinstance(raw, list):
        return {
            "user_idx": user_idx,
            "split": split,
            "k": k,
            "recommendations": raw,
            "debug": None,
        }

    if not isinstance(raw, dict):
        return {
            "user_idx": user_idx,
            "split": split,
            "k": k,
            "recommendations": [],
            "debug": {"warning": f"Unexpected service output type: {type(raw)}"},
        }

    # Prefer explicit recommendations
    if "recommendations" in raw and isinstance(raw["recommendations"], list):
        recs = raw["recommendations"]
    # Fallback to items
    elif "items" in raw and isinstance(raw["items"], list):
        recs = raw["items"]
    else:
        recs = []

    out = {
        "user_idx": raw.get("user_idx", user_idx),
        "split": raw.get("split", split),
        "k": raw.get("k", k),
        "recommendations": recs,
        "debug": raw.get("debug"),
    }

    # Preserve extra keys (non-breaking)
    for key, val in raw.items():
        if key not in out and key not in {"items"}:
            out[key] = val

    return out


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
        raw = svc.recommend(
            user_idx=user_idx,
            k=k,
            include_titles=include_titles,
            debug=debug,
        )

        out = _normalize_out(raw=raw, user_idx=user_idx, split=split, k=k)
        return out

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"V4 recommend failed: {e}")