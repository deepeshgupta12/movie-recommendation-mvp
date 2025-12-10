"""
V4 API (Session-aware ranker)

FastAPI layer for V4 recommendations.

Guarantees:
- Split-aware cached service instances.
- API response contract is stable and ALWAYS returns:
    {
      user_idx,
      split,
      k,
      recommendations: [...]
      debug: {... optional ...}
    }

Compatibility:
- If service returns "items", API maps it to "recommendations".
- If service returns a list, wraps it into "recommendations".
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional  # noqa: UP035

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.service.reco_service_v4 import V4RecommenderService, V4ServiceConfig

app = FastAPI(
    title="Movie Recommendation API - V4",
    version="4.0.2",
    description="Session-aware hybrid recommender (ANN + GRU + V2 prior + Session features).",
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
    user_idx: int = Field(..., ge=0)
    split: str
    k: int = Field(..., ge=1, le=200)

    # We intentionally DO NOT use alias here.
    # Normalization happens in code, so the response model is strict & stable.
    recommendations: List[Any] = Field(default_factory=list)

    debug: Optional[dict] = None

    model_config = {"extra": "allow"}


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

    # Determine recommendations payload
    if isinstance(raw.get("recommendations"), list):
        recs = raw["recommendations"]
    elif isinstance(raw.get("items"), list):
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

    # Preserve extra keys if any (non-breaking)
    for key, val in raw.items():
        if key not in out and key != "items":
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
        # dataclass might not be used; safe fallback
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