from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple  # noqa: UP035

import joblib
import numpy as np
import polars as pl

from src.service.poster_cache import PosterCache, load_item_meta


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / ".git").exists() or (p / "pyproject.toml").exists():
            return p
    return here.parents[2]


# -------------------------
# Feedback store (LIVE)
# -------------------------

class FeedbackStoreV4:
    """
    Lightweight local feedback log to make the V4 loop visible in UI.

    Storage:
      data/processed/feedback_events_v4_live.json

    Structure:
      {
        "<user_idx>": [
          {"ts": 1700000000, "item_idx": 123, "event": "like"},
          ...
        ]
      }
    """

    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, List[Dict[str, Any]]] = {}

    @classmethod
    def default(cls) -> "FeedbackStoreV4":
        root = _project_root()
        p = root / "data" / "processed" / "feedback_events_v4_live.json"
        store = cls(p)
        store.load()
        return store

    def load(self) -> None:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
                if isinstance(raw, dict):
                    self.data = raw
            except Exception:
                self.data = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2))

    def add_event(self, user_idx: int, item_idx: int, event: str) -> None:
        ev = {"ts": int(time.time()), "item_idx": int(item_idx), "event": str(event)}
        key = str(user_idx)
        self.data.setdefault(key, [])
        self.data[key].append(ev)

        # Keep latest 500 events per user to avoid bloat
        if len(self.data[key]) > 500:
            self.data[key] = self.data[key][-500:]

        self.save()

    def recent_events(self, user_idx: int, window_sec: int = 60 * 60 * 24) -> List[Dict[str, Any]]:
        key = str(user_idx)
        now = int(time.time())
        out = []
        for ev in self.data.get(key, []):
            try:
                if now - int(ev.get("ts", 0)) <= window_sec:
                    out.append(ev)
            except Exception:
                pass
        return out

    def latest_event_by_item(self, user_idx: int) -> Dict[int, str]:
        """
        Returns item_idx -> last_event_type
        """
        key = str(user_idx)
        mapping: Dict[int, Tuple[int, str]] = {}
        for ev in self.data.get(key, []):
            try:
                ts = int(ev.get("ts", 0))
                item = int(ev.get("item_idx"))
                et = str(ev.get("event"))
                prev = mapping.get(item)
                if prev is None or ts >= prev[0]:
                    mapping[item] = (ts, et)
            except Exception:
                continue
        return {k: v[1] for k, v in mapping.items()}


# -------------------------
# Config
# -------------------------

@dataclass
class V4ServiceConfig:
    split: str = "val"  # "val" or "test"
    candidate_k: int = 200
    default_k: int = 20
    feedback_window_sec: int = 60 * 60 * 24 * 7  # 7 days for "session-ish" recency
    enable_diversity: bool = True
    diversity_lambda: float = 0.15  # higher = more diversity penalty on repeated sources
    feedback_alpha: float = 0.25    # post-ranker boost scale


# -------------------------
# Service
# -------------------------

class V4RecommenderService:
    """
    V4 = V3 hybrid candidates + session features + diversity-aware + live feedback loop.
    """

    def __init__(self, cfg: V4ServiceConfig):
        self.cfg = cfg
        self.root = _project_root()

        self.processed = self.root / "data" / "processed"
        self.reports = self.root / "reports" / "models"

        self.candidates_path = self.processed / f"v3_candidates_{cfg.split}.parquet"
        self.session_path = self.processed / f"session_features_v4_{cfg.split}.parquet"
        self.ranker_path = self.reports / f"ranker_hgb_v4_{cfg.split}.pkl"
        self.ranker_meta_path = self.reports / f"ranker_hgb_v4_{cfg.split}.meta.json"

        self.user_feat_path = self.processed / "user_features.parquet"
        self.item_feat_path = self.processed / "item_features.parquet"

        self.poster_cache = PosterCache.load()
        self.item_meta = load_item_meta()

        self.feedback_store = FeedbackStoreV4.default()

        self._load_ranker()
        self._load_inputs()

    # -------------------------
    # Loading
    # -------------------------

    def _load_ranker(self) -> None:
        if not self.ranker_path.exists():
            raise FileNotFoundError(
                f"V4 ranker not found.\n"
                f"Split={self.cfg.split}\n"
                f"Tried: {self.ranker_path}\n"
                f"Train it first via:\n"
                f"  python -m src.ranking.train_ranker_v4_{self.cfg.split}"
            )
        self.ranker = joblib.load(self.ranker_path)

        if self.ranker_meta_path.exists():
            meta = json.loads(self.ranker_meta_path.read_text())
            self.feature_order = meta.get("features", [])
            self.model_auc = meta.get("auc", None)
        else:
            self.feature_order = []
            self.model_auc = None

    def _load_inputs(self) -> None:
        if not self.candidates_path.exists():
            raise FileNotFoundError(f"Candidates not found at {self.candidates_path}")

        self.cand_df = pl.read_parquet(self.candidates_path)

        # Normalize expected columns
        cols = set(self.cand_df.columns)
        if "blend_score" not in cols:
            # Some earlier versions might store tt_scores instead.
            # Fall back to zeros to keep pipeline alive.
            self.cand_df = self.cand_df.with_columns(
                pl.lit([0.0]).repeat_by(pl.len()).alias("blend_score")
            )

        if "blend_sources" not in cols:
            # Create a simple placeholder sources list
            self.cand_df = self.cand_df.with_columns(
                pl.lit([]).cast(pl.List(pl.Utf8)).alias("blend_sources")
            )

        # Session features optional
        if self.session_path.exists():
            self.session_df = pl.read_parquet(self.session_path)
        else:
            self.session_df = pl.DataFrame(
                {
                    "user_idx": [],
                    "short_term_boost": [],
                    "session_recency_bucket": [],
                }
            )

        # User/item features optional
        self.user_feat_df = pl.read_parquet(self.user_feat_path) if self.user_feat_path.exists() else None
        self.item_feat_df = pl.read_parquet(self.item_feat_path) if self.item_feat_path.exists() else None

    # -------------------------
    # Feedback API
    # -------------------------

    def record_feedback(self, user_idx: int, item_idx: int, event: str) -> None:
        self.feedback_store.add_event(user_idx=user_idx, item_idx=item_idx, event=event)

    # -------------------------
    # Feature engineering
    # -------------------------

    def _explode_user_candidates(self, user_idx: int) -> pl.DataFrame:
        row = self.cand_df.filter(pl.col("user_idx") == user_idx)
        if row.height == 0:
            return pl.DataFrame({"user_idx": [], "item_idx": [], "blend_score": [], "blend_sources": []})

        # Ensure list columns exist
        row = row.select(["user_idx", "candidates", "blend_score", "blend_sources"])

        # We expect:
        # candidates: list[i64]
        # blend_score: list[f64] OR scalar fallback
        # blend_sources: list[str]
        #
        # If blend_score isn't a list, create zero list aligned with candidates length.
        schema = row.schema
        bs_dtype = schema.get("blend_score")

        row = row.with_columns(
            pl.col("candidates").cast(pl.List(pl.Int64)),
            pl.col("blend_sources").cast(pl.List(pl.Utf8)),
        )

        # Harmonize blend_score into a list matching candidates length
        if bs_dtype is None or str(bs_dtype).lower().startswith("list") is False:
            # create zeros list same length as candidates
            row = row.with_columns(
                pl.col("candidates").list.eval(pl.lit(0.0)).alias("blend_score")
            )
        else:
            row = row.with_columns(pl.col("blend_score").cast(pl.List(pl.Float64)))

        out = row.explode(["candidates", "blend_score", "blend_sources"]).rename(
            {"candidates": "item_idx"}
        )

        # Source flags from blend_sources string values
        # Some pipelines store comma-separated single strings per item in list.
        out = out.with_columns(
            [
                pl.col("blend_sources").fill_null("").cast(pl.Utf8),
                pl.col("blend_score").fill_null(0.0).cast(pl.Float64),
            ]
        )

        has_tt = pl.col("blend_sources").str.contains("two_tower").cast(pl.Int8)
        has_seq = pl.col("blend_sources").str.contains("seq").cast(pl.Int8)
        has_v2 = pl.col("blend_sources").str.contains("v2").cast(pl.Int8)

        out = out.with_columns(
            [
                has_tt.alias("has_tt"),
                has_seq.alias("has_seq"),
                has_v2.alias("has_v2"),
            ]
        )

        return out

    def _attach_session_flags(self, base: pl.DataFrame) -> pl.DataFrame:
        if self.session_df.height == 0:
            return base.with_columns(
                [
                    pl.lit(0.0).alias("short_term_boost"),
                    pl.lit(0).alias("sess_hot"),
                    pl.lit(0).alias("sess_warm"),
                    pl.lit(0).alias("sess_cold"),
                ]
            )

        sess = self.session_df.select(
            ["user_idx", "short_term_boost", "session_recency_bucket"]
        )

        out = base.join(sess, on="user_idx", how="left")

        out = out.with_columns(
            [
                pl.col("short_term_boost").fill_null(0.0).cast(pl.Float64),
                (pl.col("session_recency_bucket") == "hot").cast(pl.Int8).alias("sess_hot"),
                (pl.col("session_recency_bucket") == "warm").cast(pl.Int8).alias("sess_warm"),
                (pl.col("session_recency_bucket") == "cold").cast(pl.Int8).alias("sess_cold"),
            ]
        )

        return out.drop(["session_recency_bucket"])

    def _attach_user_item_features(self, base: pl.DataFrame) -> pl.DataFrame:
        out = base

        # User features
        if self.user_feat_df is not None and self.user_feat_df.height > 0:
            uf = self.user_feat_df
            out = out.join(uf, on="user_idx", how="left")
        else:
            out = out.with_columns(
                [
                    pl.lit(0).alias("user_interactions"),
                    pl.lit(0.0).alias("user_conf_sum"),
                    pl.lit(0.0).alias("user_conf_decay_sum"),
                    pl.lit(9999).alias("user_days_since_last"),
                ]
            )

        # Item features
        if self.item_feat_df is not None and self.item_feat_df.height > 0:
            itf = self.item_feat_df
            out = out.join(itf, on="item_idx", how="left")
        else:
            out = out.with_columns(
                [
                    pl.lit(0).alias("item_interactions"),
                    pl.lit(0.0).alias("item_conf_sum"),
                    pl.lit(0.0).alias("item_conf_decay_sum"),
                    pl.lit(9999).alias("item_days_since_last"),
                ]
            )

        # Fill any missing numeric columns safely
        for c in [
            "user_interactions", "user_conf_sum", "user_conf_decay_sum", "user_days_since_last",
            "item_interactions", "item_conf_sum", "item_conf_decay_sum", "item_days_since_last"
        ]:
            if c in out.columns:
                out = out.with_columns(pl.col(c).fill_null(0))

        return out

    def _build_feature_frame(self, user_idx: int) -> pl.DataFrame:
        base = self._explode_user_candidates(user_idx)

        if base.height == 0:
            return base

        base = self._attach_session_flags(base)
        base = self._attach_user_item_features(base)

        # Ensure dtypes
        base = base.with_columns(
            [
                pl.col("item_idx").cast(pl.Int64),
                pl.col("user_idx").cast(pl.Int64),
                pl.col("has_tt").cast(pl.Int8),
                pl.col("has_seq").cast(pl.Int8),
                pl.col("has_v2").cast(pl.Int8),
                pl.col("blend_score").cast(pl.Float64),
                pl.col("short_term_boost").cast(pl.Float64),
                pl.col("sess_hot").cast(pl.Int8),
                pl.col("sess_warm").cast(pl.Int8),
                pl.col("sess_cold").cast(pl.Int8),
            ]
        )

        return base

    # -------------------------
    # Scoring
    # -------------------------

    def _predict_ranker_scores(self, feats: pl.DataFrame) -> np.ndarray:
        if feats.height == 0:
            return np.array([])

        # If meta feature order exists, follow it.
        # Else attempt a safe default order.
        if self.feature_order:
            cols = self.feature_order
        else:
            cols = [
                "blend_score", "has_tt", "has_seq", "has_v2",
                "short_term_boost", "sess_hot", "sess_warm", "sess_cold",
                "user_interactions", "user_conf_sum", "user_conf_decay_sum", "user_days_since_last",
                "item_interactions", "item_conf_sum", "item_conf_decay_sum", "item_days_since_last",
            ]

        for c in cols:
            if c not in feats.columns:
                feats = feats.with_columns(pl.lit(0).alias(c))

        X = feats.select(cols).to_numpy()

        if hasattr(self.ranker, "predict_proba"):
            proba = self.ranker.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1]
            return proba.reshape(-1)
        else:
            pred = self.ranker.predict(X)
            return np.asarray(pred).reshape(-1)

    def _feedback_boost_vector(self, user_idx: int, item_idxs: np.ndarray) -> np.ndarray:
        """
        Translate last feedback into a post-ranker adjustment.
        This is intentionally strong enough to be visible in UI.
        """
        last_ev = self.feedback_store.latest_event_by_item(user_idx)
        boosts = np.zeros(len(item_idxs), dtype=np.float64)

        for i, it in enumerate(item_idxs):
            ev = last_ev.get(int(it))
            if not ev:
                continue
            if ev == "like":
                boosts[i] += 0.15
            elif ev == "watch_later":
                boosts[i] += 0.08
            elif ev == "start":
                boosts[i] += 0.05
            elif ev == "watched":
                boosts[i] -= 0.20
            elif ev == "skip":
                boosts[i] -= 0.10

        return boosts * float(self.cfg.feedback_alpha)

    def _diversity_rerank(
        self,
        items: List[Dict[str, Any]],
        lambda_penalty: float = 0.15,
    ) -> List[Dict[str, Any]]:
        """
        Simple greedy diversity reranker using source repetition penalty.
        We penalize over-represented primary categories:
          - Because you watched
          - Similar to your taste
          - Popular among similar users
        """

        if not items:
            return items

        counts: Dict[str, int] = {}
        reranked: List[Dict[str, Any]] = []

        # Greedy selection by adjusted score
        remaining = items[:]
        for _ in range(len(items)):
            best_idx = None
            best_adj = -1e9

            for i, it in enumerate(remaining):
                cat = it.get("category", "Recommended")
                base = float(it.get("score", 0.0))
                penalty = counts.get(cat, 0) * lambda_penalty
                adj = base - penalty

                if adj > best_adj:
                    best_adj = adj
                    best_idx = i

            if best_idx is None:
                break

            chosen = remaining.pop(best_idx)
            cat = chosen.get("category", "Recommended")
            counts[cat] = counts.get(cat, 0) + 1
            chosen["diversity_adjusted_score"] = best_adj
            reranked.append(chosen)

        return reranked

    # -------------------------
    # Reasons + categories
    # -------------------------

    def _reason_and_category(self, row: Dict[str, Any], last_title: Optional[str]) -> Tuple[str, str]:
        has_seq = int(row.get("has_seq", 0))
        has_tt = int(row.get("has_tt", 0))
        has_v2 = int(row.get("has_v2", 0))

        if has_seq == 1 and last_title:
            return f"Because you watched {last_title} recently", "Because you watched"
        if has_tt == 1:
            return "Similar to your taste", "Similar to your taste"
        if has_v2 == 1:
            return "Popular among similar users", "Popular among similar users"

        return "Recommended for you", "Recommended"

    # -------------------------
    # Public API
    # -------------------------

    def recommend(
        self,
        user_idx: int,
        k: int = 20,
        include_titles: bool = True,
        debug: bool = False,
    ) -> Dict[str, Any]:
        k = int(k or self.cfg.default_k)

        feats = self._build_feature_frame(user_idx)
        if feats.height == 0:
            return {
                "user_idx": user_idx,
                "k": k,
                "recommendations": [],
                "debug": {"split": self.cfg.split} if debug else None,
                "split": self.cfg.split,
            }

        scores = self._predict_ranker_scores(feats)

        item_idxs = feats.select("item_idx").to_numpy().reshape(-1)
        fb_boost = self._feedback_boost_vector(user_idx, item_idxs)

        final_scores = scores + fb_boost

        feats = feats.with_columns(
            [
                pl.Series("ranker_score", scores),
                pl.Series("feedback_boost", fb_boost),
                pl.Series("score", final_scores),
            ]
        )

        # Attach meta
        last_title = None
        if self.session_df.height > 0:
            lt = self.session_df.filter(pl.col("user_idx") == user_idx).select("last_title")
            if lt.height > 0:
                last_title = lt.item()

        # title + movieId mapping
        if self.item_meta is not None and self.item_meta.height > 0:
            feats = feats.join(self.item_meta, on="item_idx", how="left")
        else:
            feats = feats.with_columns(
                [
                    pl.lit(None).alias("movieId"),
                    pl.lit(None).alias("title"),
                ]
            )

        # Convert to python list for reason/category logic
        rows = feats.select(
            [
                "item_idx", "movieId", "title",
                "score", "ranker_score", "feedback_boost",
                "blend_score", "blend_sources",
                "has_tt", "has_seq", "has_v2",
                "short_term_boost", "sess_hot", "sess_warm", "sess_cold",
            ]
        ).to_dicts()

        items: List[Dict[str, Any]] = []
        for r in rows:
            reason, category = self._reason_and_category(r, last_title)

            movie_id = r.get("movieId")
            poster_url = None
            if movie_id is not None:
                try:
                    poster_url = self.poster_cache.get_or_placeholder(int(movie_id))
                except Exception:
                    poster_url = self.poster_cache.get_or_placeholder(str(movie_id))
            else:
                poster_url = self.poster_cache.get_or_placeholder("unknown")

            items.append(
                {
                    "item_idx": int(r["item_idx"]),
                    "movieId": int(movie_id) if movie_id is not None else None,
                    "title": r.get("title") if include_titles else None,
                    "poster_url": poster_url,
                    "score": float(r.get("score", 0.0)),
                    "reason": reason,
                    "category": category,
                    "has_tt": int(r.get("has_tt", 0)),
                    "has_seq": int(r.get("has_seq", 0)),
                    "has_v2": int(r.get("has_v2", 0)),
                    "short_term_boost": float(r.get("short_term_boost", 0.0) or 0.0),
                    "sess_hot": int(r.get("sess_hot", 0) or 0),
                    "sess_warm": int(r.get("sess_warm", 0) or 0),
                    "sess_cold": int(r.get("sess_cold", 0) or 0),
                    "blend_score": float(r.get("blend_score", 0.0) or 0.0),
                    "ranker_score": float(r.get("ranker_score", 0.0) or 0.0),
                    "feedback_boost": float(r.get("feedback_boost", 0.0) or 0.0),
                }
            )

        # Base sort by score desc
        items.sort(key=lambda x: x["score"], reverse=True)

        # Diversity rerank
        if self.cfg.enable_diversity:
            items = self._diversity_rerank(items, lambda_penalty=self.cfg.diversity_lambda)

        # Remove duplicates by item_idx while preserving order
        seen = set()
        uniq: List[Dict[str, Any]] = []
        for it in items:
            ii = it["item_idx"]
            if ii in seen:
                continue
            seen.add(ii)
            uniq.append(it)
            if len(uniq) >= k:
                break

        payload = {
            "user_idx": user_idx,
            "k": k,
            "recommendations": uniq,
            "split": self.cfg.split,
        }

        if debug:
            payload["debug"] = {
                "split": self.cfg.split,
                "project_root": str(self.root),
                "candidates_path": str(self.candidates_path),
                "session_features_path": str(self.session_path),
                "ranker_path": str(self.ranker_path),
                "ranker_meta_path": str(self.ranker_meta_path),
                "model_auc": self.model_auc,
                "feature_order": self.feature_order,
                "candidate_count": int(self.cfg.candidate_k),
                "enable_diversity": self.cfg.enable_diversity,
                "diversity_lambda": self.cfg.diversity_lambda,
                "feedback_alpha": self.cfg.feedback_alpha,
            }

        return payload


# -------------------------
# Singleton helper
# -------------------------

_SERVICE_CACHE: Dict[str, V4RecommenderService] = {}


def get_v4_service(split: str = "val") -> V4RecommenderService:
    split = split or "val"
    if split not in _SERVICE_CACHE:
        _SERVICE_CACHE[split] = V4RecommenderService(V4ServiceConfig(split=split))
    return _SERVICE_CACHE[split]