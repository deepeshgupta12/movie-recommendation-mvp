# src/service/reco_service_v4.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple  # noqa: UP035

import joblib
import numpy as np
import polars as pl

# Optional project settings (safe access only)
try:
    from src.config import settings  # type: ignore
except Exception:
    settings = None  # type: ignore


__all__ = [
    "V4ServiceConfig",
    "V4RecommenderService",
    "get_v4_service",
]


# ---------------------------
# Project root + safe dirs
# ---------------------------
# file: <root>/src/service/reco_service_v4.py
# parents[0] = .../src/service
# parents[1] = .../src
# parents[2] = .../<root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _safe_reports_dir() -> Path:
    """
    Prefer settings.REPORTS_DIR if present.
    Otherwise anchor to project root to avoid CWD issues.
    """
    if settings is not None and hasattr(settings, "REPORTS_DIR"):
        try:
            return Path(getattr(settings, "REPORTS_DIR"))
        except Exception:
            pass
    return PROJECT_ROOT / "reports"


def _safe_data_dir() -> Path:
    """
    Prefer settings.DATA_DIR if present.
    Otherwise anchor to project root to avoid CWD issues.
    """
    if settings is not None and hasattr(settings, "DATA_DIR"):
        try:
            return Path(getattr(settings, "DATA_DIR"))
        except Exception:
            pass
    return PROJECT_ROOT / "data"


def _first_existing(paths: List[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    return paths[0]


# ---------------------------
# Config
# ---------------------------
@dataclass
class V4ServiceConfig:
    """
    Online inference config for V4 session-aware ranker.
    Candidate universe remains aligned to V3 blended candidates.

    V4 ranker adds session features:
    - short_term_boost
    - sess_hot / sess_warm / sess_cold
    """
    split: str = "val"  # "val" or "test"
    cap_users: int = 50_000

    # Inputs (override if needed)
    candidates_path: Optional[str] = None
    session_features_path: Optional[str] = None
    dim_items_path: Optional[str] = None
    train_conf_path: Optional[str] = None

    # Ranker
    ranker_path: Optional[str] = None
    ranker_meta_path: Optional[str] = None

    # Behavior
    default_k: int = 20
    max_candidate_k: int = 200


# ---------------------------
# Service
# ---------------------------
class V4RecommenderService:
    """
    V4 session-aware online recommender.

    Inputs:
    - V3 blended candidates per user: v3_candidates_{split}.parquet
    - V4 session features per user: session_features_v4_{split}.parquet
    - Train history for user/item aggregates: train_conf.parquet
    - V4 ranker: ranker_hgb_v4_{split}.pkl (+ meta)
    - dim_items for titles/posters

    Output:
    - Ranked items with reason + session debug scaffold.
    """

    def __init__(self, cfg: Optional[V4ServiceConfig] = None):
        self.cfg = cfg or V4ServiceConfig()

        self.reports_dir = _safe_reports_dir()
        self.data_dir = _safe_data_dir()

        self._resolve_paths()
        self._load_dim_items()
        self._load_train_conf_aggregates()
        self._load_ranker()
        self._load_candidates()
        self._load_session_features()

    # ---------------------------
    # Path resolution
    # ---------------------------
    def _resolve_paths(self) -> None:
        split = self.cfg.split

        # Candidates: V4 reuses V3 blended candidates
        default_candidates = self.data_dir / "processed" / f"v3_candidates_{split}.parquet"
        fallback_candidates = self.data_dir / "processed" / "v3_candidates_val.parquet"

        self.candidates_path = Path(
            self.cfg.candidates_path
            or str(_first_existing([default_candidates, fallback_candidates]))
        )

        # Session features
        default_sess = self.data_dir / "processed" / f"session_features_v4_{split}.parquet"
        fallback_sess = self.data_dir / "processed" / "session_features_v4_val.parquet"

        self.session_features_path = Path(
            self.cfg.session_features_path
            or str(_first_existing([default_sess, fallback_sess]))
        )

        # dim_items
        default_dim = self.data_dir / "processed" / "dim_items.parquet"
        self.dim_items_path = Path(self.cfg.dim_items_path or str(default_dim))

        # train_conf
        default_train = self.data_dir / "processed" / "train_conf.parquet"
        self.train_conf_path = Path(self.cfg.train_conf_path or str(default_train))

        # Ranker + meta
        default_ranker = self.reports_dir / "models" / f"ranker_hgb_v4_{split}.pkl"
        fallback_ranker = self.reports_dir / "models" / "ranker_hgb_v4_val.pkl"

        self.ranker_path = Path(
            self.cfg.ranker_path
            or str(_first_existing([default_ranker, fallback_ranker]))
        )

        default_meta = self.reports_dir / "models" / f"ranker_hgb_v4_{split}.meta.json"
        fallback_meta = self.reports_dir / "models" / "ranker_hgb_v4_val.meta.json"

        self.ranker_meta_path = Path(
            self.cfg.ranker_meta_path
            or str(_first_existing([default_meta, fallback_meta]))
        )

    # ---------------------------
    # Loaders
    # ---------------------------
    def _load_dim_items(self) -> None:
        if not self.dim_items_path.exists():
            raise FileNotFoundError(
                f"dim_items not found at {self.dim_items_path}."
            )

        dim = pl.read_parquet(self.dim_items_path)
        want = [c for c in ["item_idx", "movieId", "title", "poster_url"] if c in dim.columns]
        dim = dim.select(want)

        if "poster_url" not in dim.columns:
            dim = dim.with_columns(pl.lit(None).alias("poster_url"))

        self.dim_items = dim

    def _load_train_conf_aggregates(self) -> None:
        if not self.train_conf_path.exists():
            raise FileNotFoundError(
                f"train_conf not found at {self.train_conf_path}."
            )

        df = pl.scan_parquet(self.train_conf_path)
        schema = df.collect_schema()
        cols = set(schema.names())

        conf_col = "confidence" if "confidence" in cols else ("conf" if "conf" in cols else None)
        decay_col = "conf_decay" if "conf_decay" in cols else None
        days_col = "days_since_last" if "days_since_last" in cols else (
            "days_since" if "days_since" in cols else None
        )

        if conf_col is None:
            df = df.with_columns(pl.lit(1.0).alias("confidence"))
            conf_col = "confidence"

        if decay_col is None:
            df = df.with_columns(pl.col(conf_col).alias("conf_decay"))
            decay_col = "conf_decay"

        if days_col is None:
            df = df.with_columns(pl.lit(0).alias("days_since_last"))
            days_col = "days_since_last"

        self.user_agg = (
            df.group_by("user_idx")
            .agg(
                pl.len().alias("user_interactions"),
                pl.col(conf_col).sum().alias("user_conf_sum"),
                pl.col(decay_col).sum().alias("user_conf_decay_sum"),
                pl.col(days_col).max().alias("user_days_since_last"),
            )
            .collect(streaming=True)
        )

        self.item_agg = (
            df.group_by("item_idx")
            .agg(
                pl.len().alias("item_interactions"),
                pl.col(conf_col).sum().alias("item_conf_sum"),
                pl.col(decay_col).sum().alias("item_conf_decay_sum"),
                pl.col(days_col).max().alias("item_days_since_last"),
            )
            .collect(streaming=True)
        )

    def _load_ranker(self) -> None:
        tried_ranker = [self.ranker_path]
        tried_meta = [self.ranker_meta_path]

        if not self.ranker_path.exists():
            raise FileNotFoundError(
                "V4 ranker not found.\n"
                f"Split={self.cfg.split}\n"
                f"Tried: {', '.join(str(p) for p in tried_ranker)}\n"
                "Train it first via:\n"
                "  python -m src.ranking.train_ranker_v4_val\n"
                "or ensure reports/ is present in this branch."
            )

        if not self.ranker_meta_path.exists():
            raise FileNotFoundError(
                "V4 ranker meta not found.\n"
                f"Split={self.cfg.split}\n"
                f"Tried: {', '.join(str(p) for p in tried_meta)}"
            )

        self.ranker = joblib.load(self.ranker_path)

        meta = json.loads(self.ranker_meta_path.read_text())
        self.feature_order: List[str] = meta.get("features", [])
        self.model_auc = meta.get("auc")

        if not self.feature_order:
            raise RuntimeError("Ranker meta missing locked 'features' list.")

    def _load_candidates(self) -> None:
        if not self.candidates_path.exists():
            raise FileNotFoundError(
                f"Candidates not found at {self.candidates_path}."
            )
        self.candidates_df = pl.read_parquet(self.candidates_path)

    def _load_session_features(self) -> None:
        if not self.session_features_path.exists():
            raise FileNotFoundError(
                f"Session features not found at {self.session_features_path}."
            )
        self.session_df = pl.read_parquet(self.session_features_path)

    # ---------------------------
    # Internal utilities
    # ---------------------------
    def _get_user_candidate_row(self, user_idx: int) -> Optional[pl.DataFrame]:
        row = self.candidates_df.filter(pl.col("user_idx") == user_idx)
        return None if row.height == 0 else row

    def _parse_source_flags(self, sources: List[str], n: int) -> Tuple[List[int], List[int], List[int]]:
        has_tt, has_seq, has_v2 = [], [], []
        for s in sources[:n]:
            st = (s or "").lower()
            tt = int(("two_tower" in st) or ("ann" in st))
            seq = int(("sequence" in st) or ("gru" in st))
            v2 = int(("v2" in st) or ("prior" in st))
            has_tt.append(tt)
            has_seq.append(seq)
            has_v2.append(v2)

        while len(has_tt) < n:
            has_tt.append(0)
            has_seq.append(0)
            has_v2.append(0)

        return has_tt, has_seq, has_v2

    def _session_for_user(self, user_idx: int) -> Dict[str, object]:
        row = self.session_df.filter(pl.col("user_idx") == user_idx)
        if row.height == 0:
            return {
                "short_term_boost": 0.0,
                "sess_hot": 0,
                "sess_warm": 0,
                "sess_cold": 0,
                "last_title": None,
            }

        r = row.row(0, named=True)
        bucket = str(r.get("session_recency_bucket") or "")

        return {
            "short_term_boost": float(r.get("short_term_boost") or 0.0),
            "sess_hot": 1 if bucket == "hot" else 0,
            "sess_warm": 1 if bucket == "warm" else 0,
            "sess_cold": 1 if bucket == "cold" else 0,
            "last_title": r.get("last_title"),
        }

    def _build_candidate_frame(self, user_idx: int) -> pl.DataFrame:
        row = self._get_user_candidate_row(user_idx)
        if row is None:
            return pl.DataFrame({"user_idx": [], "item_idx": []})

        cols = set(row.columns)

        candidates = row["candidates"][0]
        n = len(candidates)

        # Score source
        if "blend_score" in cols:
            blend_score = row["blend_score"][0]
        elif "tt_scores" in cols:
            blend_score = row["tt_scores"][0]
        else:
            blend_score = [0.0] * n

        # Sources
        if "blend_sources" in cols:
            blend_sources = row["blend_sources"][0]
        else:
            blend_sources = [""] * n

        has_tt, has_seq, has_v2 = self._parse_source_flags(blend_sources, n)

        base = pl.DataFrame(
            {
                "user_idx": [user_idx] * n,
                "item_idx": candidates,
                "blend_score": blend_score,
                "blend_sources": blend_sources,
                "has_tt": has_tt,
                "has_seq": has_seq,
                "has_v2": has_v2,
            }
        )

        # Aggregates
        base = base.join(self.user_agg, on="user_idx", how="left")
        base = base.join(self.item_agg, on="item_idx", how="left")

        # Session flags
        sess = self._session_for_user(user_idx)
        base = base.with_columns(
            pl.lit(sess["short_term_boost"]).alias("short_term_boost"),
            pl.lit(sess["sess_hot"]).cast(pl.Int8).alias("sess_hot"),
            pl.lit(sess["sess_warm"]).cast(pl.Int8).alias("sess_warm"),
            pl.lit(sess["sess_cold"]).cast(pl.Int8).alias("sess_cold"),
        )

        # Fill null aggregates
        for c in [
            "user_interactions",
            "user_conf_sum",
            "user_conf_decay_sum",
            "user_days_since_last",
            "item_interactions",
            "item_conf_sum",
            "item_conf_decay_sum",
            "item_days_since_last",
        ]:
            if c in base.columns:
                base = base.with_columns(pl.col(c).fill_null(0))

        base = base.with_columns(
            pl.col("has_tt").cast(pl.Int8),
            pl.col("has_seq").cast(pl.Int8),
            pl.col("has_v2").cast(pl.Int8),
        )

        return base

    def _ensure_feature_columns(self, feats: pl.DataFrame) -> pl.DataFrame:
        for col in self.feature_order:
            if col not in feats.columns:
                feats = feats.with_columns(pl.lit(0).alias(col))
        return feats

    def _vectorize(self, feats: pl.DataFrame) -> np.ndarray:
        feats = self._ensure_feature_columns(feats)
        return feats.select(self.feature_order).to_numpy()

    def _score(self, feats: pl.DataFrame) -> np.ndarray:
        x = self._vectorize(feats)
        if hasattr(self.ranker, "predict_proba"):
            p = self.ranker.predict_proba(x)
            return p[:, 1]
        if hasattr(self.ranker, "predict"):
            return self.ranker.predict(x)
        raise RuntimeError("Ranker model missing predict/predict_proba.")

    def _attach_reasons(self, df: pl.DataFrame, user_idx: int) -> pl.DataFrame:
        sess = self._session_for_user(user_idx)
        last_title = sess.get("last_title")
        is_hot = int(sess.get("sess_hot", 0) or 0)

        def pick_reason(row: Dict[str, object]) -> str:
            has_seq = int(row.get("has_seq", 0) or 0)
            has_tt = int(row.get("has_tt", 0) or 0)
            has_v2 = int(row.get("has_v2", 0) or 0)

            if is_hot == 1 and last_title:
                return f"Because you watched {last_title} recently"
            if has_seq == 1:
                return "Because you watched similar movies recently"
            if has_tt == 1:
                return "Similar to your taste"
            if has_v2 == 1:
                return "Popular among similar users"
            return "Recommended for you"

        reasons = [
            pick_reason(r)
            for r in df.select(["has_seq", "has_tt", "has_v2"]).to_dicts()
        ]
        return df.with_columns(pl.Series("reason", reasons))

    # ---------------------------
    # Public API
    # ---------------------------
    def recommend(
        self,
        user_idx: int,
        k: Optional[int] = None,
        include_titles: bool = True,
        debug: bool = False,
    ) -> Dict[str, object]:
        k = k or self.cfg.default_k

        feats = self._build_candidate_frame(user_idx)
        if feats.height == 0:
            out = {"user_idx": user_idx, "k": k, "items": []}
            if debug:
                out["debug"] = {"note": "no candidates for user"}
            return out

        scores = self._score(feats)
        feats = feats.with_columns(pl.Series("rank_score", scores))

        ranked = feats.sort("rank_score", descending=True).head(self.cfg.max_candidate_k)
        ranked = self._attach_reasons(ranked, user_idx)

        if include_titles:
            ranked = ranked.join(self.dim_items, on="item_idx", how="left")

        top = ranked.head(k)

        items = []
        for r in top.to_dicts():
            items.append(
                {
                    "item_idx": int(r["item_idx"]),
                    "movieId": r.get("movieId"),
                    "title": r.get("title"),
                    "poster_url": r.get("poster_url"),
                    "score": float(r["rank_score"]),
                    "reason": r.get("reason"),
                    "has_tt": int(r.get("has_tt", 0) or 0),
                    "has_seq": int(r.get("has_seq", 0) or 0),
                    "has_v2": int(r.get("has_v2", 0) or 0),
                    "short_term_boost": float(r.get("short_term_boost", 0.0) or 0.0),
                    "sess_hot": int(r.get("sess_hot", 0) or 0),
                    "sess_warm": int(r.get("sess_warm", 0) or 0),
                    "sess_cold": int(r.get("sess_cold", 0) or 0),
                }
            )

        payload: Dict[str, object] = {
            "user_idx": user_idx,
            "k": k,
            "items": items,
        }

        if debug:
            payload["debug"] = {
                "split": self.cfg.split,
                "project_root": str(PROJECT_ROOT),
                "candidates_path": str(self.candidates_path),
                "session_features_path": str(self.session_features_path),
                "ranker_path": str(self.ranker_path),
                "ranker_meta_path": str(self.ranker_meta_path),
                "model_auc": self.model_auc,
                "feature_order": self.feature_order,
                "candidate_count": int(feats.height),
            }

        return payload


# ---------------------------
# Singleton getter
# ---------------------------
_service_cache: Dict[str, V4RecommenderService] = {}


def get_v4_service(split: str = "val") -> V4RecommenderService:
    if split not in _service_cache:
        _service_cache[split] = V4RecommenderService(V4ServiceConfig(split=split))
    return _service_cache[split]