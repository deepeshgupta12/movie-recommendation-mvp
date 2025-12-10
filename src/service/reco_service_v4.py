from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple  # noqa: UP035

import joblib
import numpy as np
import polars as pl

from src.service.feedback_store_v4 import FeedbackStoreV4
from src.service.poster_cache import PosterCache, get_or_fetch_poster


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / ".git").exists() or (p / "pyproject.toml").exists():
            return p
    return here.parents[2]


@dataclass
class V4ServiceConfig:
    split: str = "val"
    k_default: int = 20
    apply_feedback_overlay: bool = True
    like_boost: float = 0.08  # additive boost to rank score
    watch_later_boost: float = 0.04
    start_boost: float = 0.10  # helps Continue Watching items bubble up


class V4RecommenderService:
    """
    V4 = Session-aware ranker on top of V3 hybrid candidates,
    plus an online overlay for immediate Netflix-like reactivity.
    """

    def __init__(self, cfg: Optional[V4ServiceConfig] = None) -> None:
        self.cfg = cfg or V4ServiceConfig()
        self.root = _project_root()

        self.data_dir = self.root / "data"
        self.processed_dir = self.data_dir / "processed"
        self.reports_dir = self.root / "reports"
        self.models_dir = self.reports_dir / "models"

        self.feedback = FeedbackStoreV4()
        self.poster_cache = PosterCache.load()

        self._load_meta()
        self._load_candidates()
        self._load_session_features()
        self._load_ranker()

    # ---------- Loaders ----------

    def _load_meta(self) -> None:
        # Expected columns: item_idx, movieId, title
        meta_path = self.processed_dir / "item_meta.parquet"
        if not meta_path.exists():
            # Fallback to csv if present
            meta_csv = self.processed_dir / "item_meta.csv"
            if meta_csv.exists():
                self.item_meta = pl.read_csv(meta_csv)
            else:
                raise FileNotFoundError(
                    "Item metadata not found. Expected data/processed/item_meta.parquet "
                    "with columns: item_idx, movieId, title."
                )
        else:
            self.item_meta = pl.read_parquet(meta_path)

        # Build fast lookup frame
        self.item_meta = self.item_meta.select(
            [pl.col("item_idx").cast(pl.Int64), pl.col("movieId").cast(pl.Int64), pl.col("title")]
        ).unique(subset=["item_idx"])

    def _load_candidates(self) -> None:
        # Using V3 blended candidates as base for V4 ranker.
        p = self.processed_dir / f"v3_candidates_{self.cfg.split}.parquet"
        if not p.exists():
            raise FileNotFoundError(
                f"V3 candidates not found at {p}. Run v3 candidate pipeline first."
            )
        self.cand = pl.read_parquet(p)
        # Expected columns: user_idx, candidates (list), blend_sources (list), maybe tt_scores
        # We standardize names.
        if "blend_sources" not in self.cand.columns and "sources" in self.cand.columns:
            self.cand = self.cand.rename({"sources": "blend_sources"})
        if "blend_score" not in self.cand.columns:
            # If you stored scores separately, we will compute a placeholder 0.0 per item later.
            pass

    def _load_session_features(self) -> None:
        p = self.processed_dir / f"session_features_v4_{self.cfg.split}.parquet"
        if not p.exists():
            raise FileNotFoundError(
                f"Session features not found at {p}. Run build_session_features_v4."
            )
        self.sess = pl.read_parquet(p).select(
            [
                pl.col("user_idx").cast(pl.Int64),
                pl.col("last_item_idx").cast(pl.Int64),
                pl.col("last_title"),
                pl.col("last_seq_len"),
                pl.col("short_term_boost").cast(pl.Float64),
                pl.col("session_recency_bucket"),
            ]
        )

    def _load_ranker(self) -> None:
        ranker_path = self.models_dir / f"ranker_hgb_v4_{self.cfg.split}.pkl"
        meta_path = self.models_dir / f"ranker_hgb_v4_{self.cfg.split}.meta.json"

        if not ranker_path.exists():
            raise FileNotFoundError(
                f"V4 ranker not found.\n"
                f"Split={self.cfg.split}\n"
                f"Tried: {ranker_path}\n"
                f"Train it first via:\n"
                f"  python -m src.ranking.train_ranker_v4_{self.cfg.split}"
            )

        self.ranker = joblib.load(ranker_path)
        self.ranker_path = ranker_path
        self.ranker_meta_path = meta_path

        if meta_path.exists():
            m = json.loads(meta_path.read_text())
            self.feature_order = m.get("features") or []
            self.model_auc = m.get("auc")
        else:
            # Safe fallback order (must match training)
            self.feature_order = [
                "blend_score", "has_tt", "has_seq", "has_v2",
                "short_term_boost", "sess_hot", "sess_warm", "sess_cold",
                "user_interactions", "user_conf_sum", "user_conf_decay_sum", "user_days_since_last",
                "item_interactions", "item_conf_sum", "item_conf_decay_sum", "item_days_since_last",
            ]
            self.model_auc = None

    # ---------- Core frame builders ----------

    def _explode_user_candidates(self, user_idx: int) -> pl.DataFrame:
        row = self.cand.filter(pl.col("user_idx") == user_idx)
        if row.is_empty():
            return pl.DataFrame({"user_idx": [user_idx], "item_idx": [], "blend_sources": []})

        # We expect candidates list and blend_sources list aligned.
        # If blend_sources missing, create filler.
        candidates = row.select("candidates").item()
        sources = row.select("blend_sources").item() if "blend_sources" in row.columns else None

        if sources is None:
            sources = ["unknown"] * len(candidates)

        base = pl.DataFrame(
            {
                "user_idx": [user_idx] * len(candidates),
                "item_idx": candidates,
                "blend_sources": sources,
            }
        )

        # Compute source flags
        base = base.with_columns(
            [
                pl.col("blend_sources").cast(pl.Utf8),
                pl.col("item_idx").cast(pl.Int64),
                (pl.col("blend_sources").str.contains("two_tower") | pl.col("blend_sources").str.contains("tt"))
                    .cast(pl.Int8).alias("has_tt"),
                (pl.col("blend_sources").str.contains("seq"))
                    .cast(pl.Int8).alias("has_seq"),
                (pl.col("blend_sources").str.contains("v2"))
                    .cast(pl.Int8).alias("has_v2"),
            ]
        )

        # Provide a base blend_score if not present in candidate file.
        # We keep a stable placeholder; ranker will still incorporate session + history features.
        base = base.with_columns(pl.lit(0.0).alias("blend_score"))

        return base

    def _attach_session(self, df: pl.DataFrame) -> pl.DataFrame:
        # Join session features for this user; may be missing for many users
        sess_row = self.sess.filter(pl.col("user_idx") == df["user_idx"][0]) if df.height else self.sess.head(0)

        if sess_row.is_empty():
            return df.with_columns(
                [
                    pl.lit(0.0).alias("short_term_boost"),
                    pl.lit(0).cast(pl.Int8).alias("sess_hot"),
                    pl.lit(0).cast(pl.Int8).alias("sess_warm"),
                    pl.lit(0).cast(pl.Int8).alias("sess_cold"),
                    pl.lit(None).cast(pl.Int64).alias("last_item_idx"),
                    pl.lit(None).cast(pl.Utf8).alias("last_title"),
                ]
            )

        bucket = sess_row.select("session_recency_bucket").item()
        boost = float(sess_row.select("short_term_boost").item())
        last_item = int(sess_row.select("last_item_idx").item())
        last_title = str(sess_row.select("last_title").item())

        sess_hot = 1 if bucket == "hot" else 0
        sess_warm = 1 if bucket == "warm" else 0
        sess_cold = 1 if bucket == "cold" else 0

        return df.with_columns(
            [
                pl.lit(boost).alias("short_term_boost"),
                pl.lit(sess_hot).cast(pl.Int8).alias("sess_hot"),
                pl.lit(sess_warm).cast(pl.Int8).alias("sess_warm"),
                pl.lit(sess_cold).cast(pl.Int8).alias("sess_cold"),
                pl.lit(last_item).cast(pl.Int64).alias("last_item_idx"),
                pl.lit(last_title).cast(pl.Utf8).alias("last_title"),
            ]
        )

    def _attach_user_item_history_features(self, df: pl.DataFrame) -> pl.DataFrame:
        # MVP-friendly fallback:
        # We assume these columns were already engineered in rank pointwise generation,
        # but for service-time scoring we keep safe defaults if not present.
        # This keeps API robust even without heavy joins.

        defaults = {
            "user_interactions": 0,
            "user_conf_sum": 0.0,
            "user_conf_decay_sum": 0.0,
            "user_days_since_last": 9999,
            "item_interactions": 0,
            "item_conf_sum": 0.0,
            "item_conf_decay_sum": 0.0,
            "item_days_since_last": 9999,
        }

        for col, val in defaults.items():
            if col not in df.columns:
                lit = pl.lit(val)
                if isinstance(val, int):
                    df = df.with_columns(lit.cast(pl.Int64).alias(col))
                else:
                    df = df.with_columns(lit.cast(pl.Float64).alias(col))

        return df

    def _build_feature_frame(self, user_idx: int) -> pl.DataFrame:
        base = self._explode_user_candidates(user_idx)
        if base.is_empty():
            return base

        base = self._attach_session(base)
        base = self._attach_user_item_history_features(base)

        # Ensure all ranker features exist
        for f in self.feature_order:
            if f not in base.columns:
                base = base.with_columns(pl.lit(0).alias(f))

        return base

    # ---------- Online overlay ----------

    def _apply_feedback_overlay(self, user_idx: int, scored: pl.DataFrame) -> pl.DataFrame:
        state = self.feedback.get_state(user_idx)

        if scored.is_empty():
            return scored

        # Remove watched
        if state.watched:
            scored = scored.filter(~pl.col("item_idx").is_in(list(state.watched)))

        # Boost liked / watch_later / started (additive)
        def boost_expr(items: set[int], amount: float):
            if not items:
                return pl.lit(0.0)
            return pl.when(pl.col("item_idx").is_in(list(items))).then(amount).otherwise(0.0)

        scored = scored.with_columns(
            [
                boost_expr(state.liked, self.cfg.like_boost).alias("_like_boost"),
                boost_expr(state.watch_later, self.cfg.watch_later_boost).alias("_wl_boost"),
                boost_expr(state.started, self.cfg.start_boost).alias("_start_boost"),
            ]
        ).with_columns(
            (pl.col("score") + pl.col("_like_boost") + pl.col("_wl_boost") + pl.col("_start_boost")).alias("score")
        ).drop(["_like_boost", "_wl_boost", "_start_boost"])

        return scored

    # ---------- Reason + enrichment ----------

    def _why_this(self, row: Dict[str, Any], last_title: Optional[str]) -> str:
        # Priority: seq > tt > v2
        if int(row.get("has_seq", 0)) == 1 and last_title:
            return f"Because you watched {last_title} recently"
        if int(row.get("has_tt", 0)) == 1:
            return "Similar to your taste"
        if int(row.get("has_v2", 0)) == 1:
            return "Popular among similar users"
        return "Recommended for you"

    def _enrich_with_meta(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return df

        df = df.join(self.item_meta, on="item_idx", how="left")

        # Attach poster url from cache (and lazy fetch if TMDB key exists)
        # We do this row-wise to keep logic simple for MVP.
        movie_ids = df.select("movieId").to_series().to_list()
        titles = df.select("title").to_series().to_list()

        urls: List[Optional[str]] = []
        for mid, t in zip(movie_ids, titles):
            if mid is None or t is None:
                urls.append(None)
                continue
            url = self.poster_cache.get(int(mid))
            if not url:
                url = get_or_fetch_poster(int(mid), str(t), self.poster_cache)
            urls.append(url)

        df = df.with_columns(pl.Series("poster_url", urls))

        return df

    def _sectionize(self, df: pl.DataFrame, k: int) -> Dict[str, pl.DataFrame]:
        # Create logical UI rows based on reason/source flags.
        # Continue Watching is derived from feedback started-state.
        # The rest are grouped by primary reason class.
        if df.is_empty():
            return {
                "continue_watching": df,
                "because_you_watched": df,
                "similar_to_your_taste": df,
                "popular_among_similar_users": df,
                "more_for_you": df,
            }

        # We allow duplicate items across sections? No: we will allocate with priority.
        # Priority order for section assignment:
        # 1) started
        # 2) has_seq
        # 3) has_tt
        # 4) has_v2
        # 5) fallback

        return {}  # Will be built in recommend() with state context.

    # ---------- Public API ----------

    def recommend(
        self,
        user_idx: int,
        k: int = 20,
        include_titles: bool = True,
        debug: bool = False,
        apply_feedback: Optional[bool] = None,
    ) -> Dict[str, Any]:
        apply_feedback = self.cfg.apply_feedback_overlay if apply_feedback is None else apply_feedback

        feats = self._build_feature_frame(user_idx)
        if feats.is_empty():
            out = {
                "user_idx": user_idx,
                "k": k,
                "items": [],
                "sections": {},
                "split": self.cfg.split,
            }
            if debug:
                out["debug"] = self._debug_block(candidate_count=0)
            return out

        # Model scoring
        X = feats.select(self.feature_order).to_numpy()
        # HistGradientBoostingClassifier supports predict_proba
        proba = self.ranker.predict_proba(X)[:, 1]
        scored = feats.with_columns(pl.Series("score", proba))

        # Apply session boost already inside features; we don't post-adjust that here.

        # Online feedback overlay for instant UI reactivity
        if apply_feedback:
            scored = self._apply_feedback_overlay(user_idx, scored)

        # Sort by score desc
        scored = scored.sort("score", descending=True)

        # Join session row for reason text
        sess_row = self.sess.filter(pl.col("user_idx") == user_idx)
        last_title = None
        if not sess_row.is_empty():
            last_title = sess_row.select("last_title").item()

        # Enrich with meta + posters
        scored = self._enrich_with_meta(scored)

        # Build state-aware sections
        state = self.feedback.get_state(user_idx)
        started_set = set(state.started)

        # Allocate items into unique sections
        continue_rows = scored.filter(pl.col("item_idx").is_in(list(started_set))) if started_set else scored.head(0)
        rest = scored.filter(~pl.col("item_idx").is_in(list(started_set))) if started_set else scored

        because_you = rest.filter(pl.col("has_seq") == 1)
        rest2 = rest.filter(pl.col("has_seq") != 1)

        similar = rest2.filter(pl.col("has_tt") == 1)
        rest3 = rest2.filter(pl.col("has_tt") != 1)

        popular = rest3.filter(pl.col("has_v2") == 1)
        more = rest3.filter(pl.col("has_v2") != 1)

        # Trim section sizes; keep overall union roughly <= k*2 for nicer UI.
        def head_or_empty(d: pl.DataFrame, n: int) -> pl.DataFrame:
            return d.head(n) if d.height else d

        continue_rows = head_or_empty(continue_rows, min(10, k))
        because_you = head_or_empty(because_you, k)
        similar = head_or_empty(similar, k)
        popular = head_or_empty(popular, k)
        more = head_or_empty(more, k)

        # Compose final flat top-k from the section-prioritized order
        combined = pl.concat([continue_rows, because_you, similar, popular, more], how="vertical_relaxed")
        combined = combined.unique(subset=["item_idx"], keep="first").head(k)

        # Build reasons
        items: List[Dict[str, Any]] = []
        for row in combined.iter_rows(named=True):
            reason = self._why_this(row, last_title)
            items.append(
                {
                    "item_idx": int(row["item_idx"]),
                    "movieId": int(row["movieId"]) if row.get("movieId") is not None else None,
                    "title": row.get("title") if include_titles else None,
                    "poster_url": row.get("poster_url"),
                    "score": float(row["score"]),
                    "reason": reason,
                    "has_tt": int(row.get("has_tt", 0) or 0),
                    "has_seq": int(row.get("has_seq", 0) or 0),
                    "has_v2": int(row.get("has_v2", 0) or 0),
                    "short_term_boost": float(row.get("short_term_boost", 0.0) or 0.0),
                    "sess_hot": int(row.get("sess_hot", 0) or 0),
                    "sess_warm": int(row.get("sess_warm", 0) or 0),
                    "sess_cold": int(row.get("sess_cold", 0) or 0),
                }
            )

        # Section payloads for UI rows
        def to_items(d: pl.DataFrame, limit: int = 20) -> List[Dict[str, Any]]:
            if d.is_empty():
                return []
            d = d.sort("score", descending=True).head(limit)
            out_list = []
            for row in d.iter_rows(named=True):
                out_list.append(
                    {
                        "item_idx": int(row["item_idx"]),
                        "movieId": int(row["movieId"]) if row.get("movieId") is not None else None,
                        "title": row.get("title") if include_titles else None,
                        "poster_url": row.get("poster_url"),
                        "score": float(row["score"]),
                        "reason": self._why_this(row, last_title),
                        "has_tt": int(row.get("has_tt", 0) or 0),
                        "has_seq": int(row.get("has_seq", 0) or 0),
                        "has_v2": int(row.get("has_v2", 0) or 0),
                    }
                )
            return out_list

        sections = {
            "continue_watching": to_items(continue_rows, limit=10),
            "because_you_watched": to_items(because_you, limit=k),
            "similar_to_your_taste": to_items(similar, limit=k),
            "popular_among_similar_users": to_items(popular, limit=k),
            "more_for_you": to_items(more, limit=k),
        }

        out = {
            "user_idx": user_idx,
            "k": k,
            "items": items,
            "recommendations": items,  # backward compatibility
            "sections": sections,
            "split": self.cfg.split,
        }

        if debug:
            out["debug"] = self._debug_block(candidate_count=200)

        return out

    def record_feedback(self, user_idx: int, item_idx: int, event: str) -> None:
        self.feedback.record(user_idx=user_idx, item_idx=item_idx, event=event)

    def get_user_state(self, user_idx: int) -> Dict[str, List[int]]:
        s = self.feedback.get_state(user_idx)
        return {
            "liked": sorted(list(s.liked)),
            "watched": sorted(list(s.watched)),
            "watch_later": sorted(list(s.watch_later)),
            "started": sorted(list(s.started)),
        }

    def _debug_block(self, candidate_count: int) -> Dict[str, Any]:
        return {
            "split": self.cfg.split,
            "project_root": str(self.root),
            "candidates_path": str(self.processed_dir / f"v3_candidates_{self.cfg.split}.parquet"),
            "session_features_path": str(self.processed_dir / f"session_features_v4_{self.cfg.split}.parquet"),
            "ranker_path": str(self.ranker_path),
            "ranker_meta_path": str(self.ranker_meta_path),
            "model_auc": self.model_auc,
            "feature_order": self.feature_order,
            "candidate_count": candidate_count,
            "feedback_path": str(self.feedback.path),
            "poster_cache_path": str(self.poster_cache.path),
        }


# Singleton helper for API
_v4_singletons: Dict[str, V4RecommenderService] = {}


def get_v4_service(split: str = "val") -> V4RecommenderService:
    split = split.lower().strip()
    if split not in _v4_singletons:
        _v4_singletons[split] = V4RecommenderService(V4ServiceConfig(split=split))
    return _v4_singletons[split]