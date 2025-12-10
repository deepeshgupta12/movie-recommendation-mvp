from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional  # noqa: UP035

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# Service imports
# We assume these exist based on your V4 work so far.
from src.service.reco_service_v4 import V4RecommenderService, V4ServiceConfig

app = FastAPI(title="Movie Recommendation API", version="v4")


# -----------------------------
# Response / Request Models
# -----------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "v4"


class RecommendItem(BaseModel):
    item_idx: int
    movieId: Optional[int] = None
    title: Optional[str] = None
    poster_url: Optional[str] = None

    score: float = 0.0
    reason: Optional[str] = None

    # explainability + session flags
    has_tt: int = 0
    has_seq: int = 0
    has_v2: int = 0

    short_term_boost: float = 0.0
    sess_hot: int = 0
    sess_warm: int = 0
    sess_cold: int = 0


class RecommendDebug(BaseModel):
    split: str
    project_root: Optional[str] = None
    candidates_path: Optional[str] = None
    session_features_path: Optional[str] = None
    ranker_path: Optional[str] = None
    ranker_meta_path: Optional[str] = None
    model_auc: Optional[float] = None
    feature_order: Optional[List[str]] = None
    candidate_count: Optional[int] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class RecommendResponse(BaseModel):
    user_idx: int
    k: int
    split: str
    recommendations: List[RecommendItem]
    debug: Optional[RecommendDebug] = None


class FeedbackEvent(BaseModel):
    user_idx: int
    item_idx: int
    event: str = Field(..., description="like | watched | watch_later | start | dislike etc.")
    ts: Optional[int] = Field(None, description="optional unix ts")


class FeedbackResponse(BaseModel):
    status: str = "ok"
    version: str = "v4"


# -----------------------------
# Service Factory
# -----------------------------

@dataclass(frozen=True)
class _SvcKey:
    split: str


@lru_cache(maxsize=4)
def _get_service_cached(split: str) -> V4RecommenderService:
    cfg = V4ServiceConfig(split=split)
    return V4RecommenderService(cfg)


def get_service(split: str) -> V4RecommenderService:
    split = (split or "val").lower().strip()
    if split not in {"val", "test"}:
        raise HTTPException(status_code=400, detail=f"Invalid split: {split}")
    return _get_service_cached(split)


# -----------------------------
# Routes
# -----------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.get("/recommend", response_model=RecommendResponse)
def recommend(
    user_idx: int = Query(..., ge=0),
    k: int = Query(20, ge=1, le=200),
    include_titles: bool = Query(True),
    debug: bool = Query(False),
    split: str = Query("val"),
) -> RecommendResponse:
    """
    V4 recommendations using:
    - V3 blended candidates
    - V4 session features
    - V4 ranker
    - Lightweight explanation reasons
    - Poster resolution from cache-first pipeline (handled inside service)
    """
    try:
        svc = get_service(split)

        out = svc.recommend(
            user_idx=user_idx,
            k=k,
            include_titles=include_titles,
            debug=debug,
        )

        # We expect service to return a dict-like payload.
        # Normalize into API response safely.

        items = out.get("items") or out.get("recommendations") or []
        if not isinstance(items, list):
            raise RuntimeError("Service returned invalid items shape.")

        recs: List[RecommendItem] = []
        for r in items:
            # tolerate dicts with extra keys
            recs.append(RecommendItem(**r))

        dbg_obj = None
        if debug:
            dbg = out.get("debug", {}) or {}
            if isinstance(dbg, dict):
                dbg_obj = RecommendDebug(
                    split=split,
                    project_root=dbg.get("project_root"),
                    candidates_path=dbg.get("candidates_path"),
                    session_features_path=dbg.get("session_features_path"),
                    ranker_path=dbg.get("ranker_path"),
                    ranker_meta_path=dbg.get("ranker_meta_path"),
                    model_auc=dbg.get("model_auc"),
                    feature_order=dbg.get("feature_order"),
                    candidate_count=dbg.get("candidate_count"),
                    extra={k: v for k, v in dbg.items() if k not in {
                        "project_root",
                        "candidates_path",
                        "session_features_path",
                        "ranker_path",
                        "ranker_meta_path",
                        "model_auc",
                        "feature_order",
                        "candidate_count",
                    }},
                )

        return RecommendResponse(
            user_idx=user_idx,
            k=k,
            split=split,
            recommendations=recs,
            debug=dbg_obj,
        )

    except HTTPException:
        raise
    except FileNotFoundError as e:
        # Missing model/features/candidates
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # Critical: convert to HTTPException
        # so connection doesn't hard-drop.
        raise HTTPException(
            status_code=500,
            detail=f"/recommend failed: {type(e).__name__}: {e}",
        )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(evt: FeedbackEvent) -> FeedbackResponse:
    """
    Simple feedback endpoint.
    The service should update in-memory/session store or write to a lightweight log.
    If the service doesn't implement persistence yet, we still acknowledge safely.
    """
    try:
        # Default split handling:
        # feedback is not split-specific in UX,
        # but we keep parity with service constructor.
        svc = get_service("val")

        # If your service implements feedback(), call it.
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
        # else: no-op, still OK

        return FeedbackResponse()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"/feedback failed: {type(e).__name__}: {e}",
        )