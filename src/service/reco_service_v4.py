# src/service/reco_service_v4.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple  # noqa: UP035

import numpy as np
import polars as pl

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None


# ---------------------------
# Helpers
# ---------------------------

def _project_root() -> Path:
    # src/service/reco_service_v4.py -> src/service -> src -> project root
    return Path(__file__).resolve().parents[2]


def _first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


# ---------------------------
# Poster resolver
# ---------------------------

class PosterResolver:
    """
    Resolve posters with strict priority:
      1) poster_cache_v4.json
      2) item_posters.json
      3) None
    """

    def __init__(self, root: Path):
        self.root = root
        self.cache_path = root / "data" / "processed" / "poster_cache_v4.json"
        self.legacy_path = root / "data" / "processed" / "item_posters.json"

        self.cache = _load_json(self.cache_path)
        self.legacy = _load_json(self.legacy_path)

        # Normalize keys to str for movieId
        self.cache = {str(k): v for k, v in self.cache.items()} if self.cache else {}
        self.legacy = {str(k): v for k, v in self.legacy.items()} if self.legacy else {}

    def get(self, movie_id: Optional[int]) -> Optional[str]:
        if movie_id is None:
            return None
        key = str(movie_id)
        if key in self.cache:
            return self.cache[key]
        if key in self.legacy:
            return self.legacy[key]
        return None


# ---------------------------
# Config
# ---------------------------

@dataclass
class V4ServiceConfig:
    split: str = "val"  # "val" | "test"
    project_root: Optional[Path] = None
    k_default: int = 20
    candidates_version: str = "v3"  # candidates still from V3 blend


# ---------------------------
# Service
# ---------------------------

class V4RecommenderService:
    """
    V4 = V3 hybrid candidates + session-aware features + lightweight diversity
         + *live feedback-aware reranking* (local file-backed).

    Key robustness choice:
    - Session features are injected as per-user scalar literals to avoid
      lazy join schema issues (fixes the _pos dtype crash).
    """

    def __init__(self, config: V4ServiceConfig):
        self.config = config
        self.root = config.project_root or _project_root()

        self.data_dir = self.root / "data"
        self.proc = self.data_dir / "processed"
        self.reports = self.root / "reports"

        self.poster_resolver = PosterResolver(self.root)

        self._load_candidates()
        self._load_session_features()
        self._load_item_meta()
        self._load_user_item_features()
        self._load_ranker()

        # Feedback store (lightweight)
        self.feedback_path = self.proc / "feedback_events_v4.jsonl"
        self._feedback_state = {
            "liked": {},
            "watched": {},
            "watch_later": {},
            "started": {},
            "disliked": {},
        }
        self._load_feedback_state()

    # ---------------------------
    # Loaders
    # ---------------------------

    def _load_candidates(self) -> None:
        split = self.config.split
        cand_path = self.proc / f"{self.config.candidates_version}_candidates_{split}.parquet"
        if not cand_path.exists():
            # also try v3_candidates_{split}.parquet (most likely)
            cand_path = self.proc / f"v3_candidates_{split}.parquet"

        if not cand_path.exists():
            raise FileNotFoundError(
                f"Candidates not found for split={split}. "
                f"Tried: {cand_path}"
            )

        self.candidates_path = cand_path
        self.cand_df = pl.read_parquet(cand_path)

        # We expect columns: user_idx, candidates, blend_score, blend_sources
        # Be defensive about missing blend_score
        cols = set(self.cand_df.columns)
        if "blend_score" not in cols:
            # create empty blend_score list with zeros later per user
            pass

    def _load_session_features(self) -> None:
        split = self.config.split
        sess_path = self.proc / f"session_features_v4_{split}.parquet"

        # Your run showed test uses val sequences; still file should exist now
        if not sess_path.exists():
            raise FileNotFoundError(
                f"Session features not found for split={split}: {sess_path}"
            )

        self.session_path = sess_path
        self.sess_df = pl.read_parquet(sess_path).select(
            ["user_idx", "last_item_idx", "last_title", "last_seq_len",
             "short_term_boost", "session_recency_bucket"]
        )

        # Build a tiny dict for instant per-user lookup
        # Avoid joins inside recommend
        self.sess_map: Dict[int, Dict[str, Any]] = {}
        for row in self.sess_df.iter_rows(named=True):
            uid = _safe_int(row["user_idx"])
            self.sess_map[uid] = row

    def _load_item_meta(self) -> None:
        # We rely on the file you confirmed exists
        meta_path = self.proc / "item_meta.parquet"
        if not meta_path.exists():
            # fallback candidates
            meta_path = _first_existing([
                self.proc / "movies_enriched.parquet",
                self.proc / "items.parquet",
                self.data_dir / "raw" / "movies.csv",
            ]) or meta_path

        if not meta_path.exists():
            raise FileNotFoundError(
                "Could not locate item metadata for titles/posters. "
                "Expected data/processed/item_meta.parquet."
            )

        self.item_meta_path = meta_path
        df = pl.read_parquet(meta_path) if meta_path.suffix == ".parquet" else pl.read_csv(meta_path)

        # Normalize expected column names
        cols = set(df.columns)

        # Must have item_idx
        if "item_idx" not in cols:
            # try infer from "itemId" or "idx"
            for alt in ["itemId", "idx"]:
                if alt in cols:
                    df = df.rename({alt: "item_idx"})
                    cols = set(df.columns)
                    break

        # Must have title
        if "title" not in cols:
            for alt in ["movie_title", "name"]:
                if alt in cols:
                    df = df.rename({alt: "title"})
                    cols = set(df.columns)
                    break

        # Must have movieId if available
        if "movieId" not in cols:
            for alt in ["movie_id", "tmdb_movie_id", "movielens_id"]:
                if alt in cols:
                    df = df.rename({alt: "movieId"})
                    cols = set(df.columns)
                    break

        # Keep minimal set
        keep = [c for c in ["item_idx", "movieId", "title"] if c in df.columns]
        self.item_meta = df.select(keep).unique(subset=["item_idx"])

        # Build dict for quick title/movieId lookup
        self.item_meta_map: Dict[int, Dict[str, Any]] = {}
        for row in self.item_meta.iter_rows(named=True):
            self.item_meta_map[int(row["item_idx"])] = row

    def _load_user_item_features(self) -> None:
        # These files should already exist from earlier pipelines.
        # We keep robust fallbacks.
        user_feat_path = _first_existing([
            self.proc / "user_features.parquet",
            self.proc / "user_feat.parquet",
        ])
        item_feat_path = _first_existing([
            self.proc / "item_features.parquet",
            self.proc / "item_feat.parquet",
        ])

        self.user_feat = pl.read_parquet(user_feat_path) if user_feat_path else None
        self.item_feat = pl.read_parquet(item_feat_path) if item_feat_path else None

    def _load_ranker(self) -> None:
        if joblib is None:
            raise RuntimeError("joblib not available in environment.")

        split = self.config.split
        model_path = self.reports / "models" / f"ranker_hgb_v4_{split}.pkl"
        meta_path = self.reports / "models" / f"ranker_hgb_v4_{split}.meta.json"

        if not model_path.exists():
            raise FileNotFoundError(
                "V4 ranker not found.\n"
                f"Split={split}\n"
                f"Tried: {model_path}\n"
                "Train it first via:\n"
                f"  python -m src.ranking.train_ranker_v4_{split}"
            )

        self.ranker_path = model_path
        self.ranker_meta_path = meta_path if meta_path.exists() else None

        self.model = joblib.load(model_path)

        self.model_auc = None
        self.feature_order: List[str] = []
        if self.ranker_meta_path and self.ranker_meta_path.exists():
            meta = json.loads(self.ranker_meta_path.read_text())
            self.model_auc = meta.get("auc")
            self.feature_order = meta.get("features", [])
        else:
            # fallback if meta missing
            self.feature_order = [
                "blend_score", "has_tt", "has_seq", "has_v2",
                "short_term_boost", "sess_hot", "sess_warm", "sess_cold",
                "user_interactions", "user_conf_sum", "user_conf_decay_sum",
                "user_days_since_last",
                "item_interactions", "item_conf_sum", "item_conf_decay_sum",
                "item_days_since_last",
            ]

    # ---------------------------
    # Feedback state
    # ---------------------------

    def _load_feedback_state(self) -> None:
        if not self.feedback_path.exists():
            return

        # Load historical events and fold into sets
        for line in self.feedback_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue

            uid = _safe_int(ev.get("user_idx"))
            it = _safe_int(ev.get("item_idx"))
            et = ev.get("event_type")

            if uid <= 0 and uid != 0:
                continue
            if it < 0:
                continue

            self._apply_event_to_state(uid, it, et)

    def _apply_event_to_state(self, user_idx: int, item_idx: int, event_type: str) -> None:
        if event_type not in self._feedback_state:
            return

        bucket = self._feedback_state[event_type]
        if user_idx not in bucket:
            bucket[user_idx] = set()
        bucket[user_idx].add(item_idx)

    def record_feedback(self, user_idx: int, item_idx: int, event_type: str) -> Dict[str, Any]:
        event_type = event_type.strip().lower()
        if event_type not in self._feedback_state:
            return {"ok": False, "error": "unknown_event_type", "event_type": event_type}

        ev = {
            "user_idx": int(user_idx),
            "item_idx": int(item_idx),
            "event_type": event_type,
        }

        # Append to file
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with self.feedback_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ev) + "\n")

        # Update in-memory
        self._apply_event_to_state(int(user_idx), int(item_idx), event_type)

        return {"ok": True, "event": ev}

    # ---------------------------
    # Core recommend
    # ---------------------------

    def _get_user_candidate_row(self, user_idx: int) -> Optional[Dict[str, Any]]:
        df = self.cand_df.filter(pl.col("user_idx") == int(user_idx))
        if df.height == 0:
            return None
        return df.row(0, named=True)

    def _derive_source_flags(self, sources: List[str]) -> Tuple[List[int], List[int], List[int]]:
        has_tt, has_seq, has_v2 = [], [], []
        for s in sources:
            s = (s or "").lower()
            has_tt.append(1 if "two_tower" in s or "tt" in s else 0)
            has_seq.append(1 if "seq" in s or "gru" in s else 0)
            has_v2.append(1 if "v2" in s or "prior" in s else 0)
        return has_tt, has_seq, has_v2

    def _session_literals(self, user_idx: int, n: int) -> Dict[str, Any]:
        s = self.sess_map.get(int(user_idx))
        if not s:
            # cold defaults
            return {
                "short_term_boost": [0.0] * n,
                "sess_hot": [0] * n,
                "sess_warm": [0] * n,
                "sess_cold": [1] * n,
                "last_item_idx": None,
                "last_title": None,
                "session_recency_bucket": "cold",
            }

        bucket = (s.get("session_recency_bucket") or "cold").lower()
        hot = 1 if bucket == "hot" else 0
        warm = 1 if bucket == "warm" else 0
        cold = 1 if bucket == "cold" else 0

        stb = float(s.get("short_term_boost") or 0.0)

        return {
            "short_term_boost": [stb] * n,
            "sess_hot": [hot] * n,
            "sess_warm": [warm] * n,
            "sess_cold": [cold] * n,
            "last_item_idx": s.get("last_item_idx"),
            "last_title": s.get("last_title"),
            "session_recency_bucket": bucket,
        }

    def _why_this(self, has_seq: int, has_tt: int, has_v2: int, last_title: Optional[str]) -> str:
        if has_seq == 1 and last_title:
            return f"Because you watched {last_title} recently"
        if has_tt == 1:
            return "Similar to your taste"
        if has_v2 == 1:
            return "Popular among similar users"
        return "Recommended for you"

    def _attach_user_item_features(self, base: pl.DataFrame) -> pl.DataFrame:
        out = base

        if self.user_feat is not None:
            out = out.join(self.user_feat, on="user_idx", how="left")
        if self.item_feat is not None:
            out = out.join(self.item_feat, on="item_idx", how="left")

        # Fill numeric nulls
        for c in out.columns:
            if c in {"user_idx", "item_idx", "blend_sources"}:
                continue
            if out[c].dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
                out = out.with_columns(pl.col(c).fill_null(0))
        return out

    def _score_with_ranker(self, feats: pl.DataFrame) -> np.ndarray:
        # Ensure feature order exists
        missing = [c for c in self.feature_order if c not in feats.columns]
        if missing:
            # Create missing numeric columns as zeros
            feats = feats.with_columns([pl.lit(0).alias(c) for c in missing])

        X = feats.select(self.feature_order).to_numpy()

        # HGB supports predict_proba
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        # fallback
        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        return np.zeros((X.shape[0],), dtype=np.float32)

    def _apply_feedback_adjustments(
        self, user_idx: int, item_ids: List[int], scores: np.ndarray
    ) -> np.ndarray:
        uid = int(user_idx)

        liked = self._feedback_state["liked"].get(uid, set())
        watched = self._feedback_state["watched"].get(uid, set())
        watch_later = self._feedback_state["watch_later"].get(uid, set())
        started = self._feedback_state["started"].get(uid, set())
        disliked = self._feedback_state["disliked"].get(uid, set())

        adj = scores.astype(np.float64).copy()

        for i, it in enumerate(item_ids):
            if it in disliked:
                adj[i] -= 1.0
            if it in watched:
                adj[i] -= 0.5
            if it in liked:
                adj[i] += 0.05
            if it in started:
                adj[i] += 0.03
            if it in watch_later:
                adj[i] += 0.02

        return adj

    def _filter_watched(self, user_idx: int, item_ids: List[int], scores: np.ndarray) -> Tuple[List[int], np.ndarray]:
        uid = int(user_idx)
        watched = self._feedback_state["watched"].get(uid, set())
        if not watched:
            return item_ids, scores

        keep_items, keep_scores = [], []
        for it, sc in zip(item_ids, scores):
            if it not in watched:
                keep_items.append(it)
                keep_scores.append(sc)
        return keep_items, np.asarray(keep_scores)

    def _simple_source_diversity(
        self,
        rows: List[Dict[str, Any]],
        k: int,
        max_streak: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Greedy reorder to avoid long streaks of identical primary reason bucket.
        This is lightweight and doesn't need genre metadata.
        """
        if len(rows) <= 1:
            return rows[:k]

        def bucket(r: Dict[str, Any]) -> str:
            # Use 'reason' string as bucket
            return (r.get("reason") or "other").lower()

        remaining = rows[:]
        out: List[Dict[str, Any]] = []
        streak = 0
        last_b = None

        while remaining and len(out) < k:
            # Try pick best item that won't violate streak
            picked_idx = None
            for i, r in enumerate(remaining):
                b = bucket(r)
                if last_b is None or b != last_b or streak < max_streak:
                    picked_idx = i
                    break

            if picked_idx is None:
                picked_idx = 0

            r = remaining.pop(picked_idx)
            b = bucket(r)

            if last_b is None or b != last_b:
                last_b = b
                streak = 1
            else:
                streak += 1

            out.append(r)

        return out

    def recommend(
        self,
        user_idx: int,
        k: Optional[int] = None,
        include_titles: bool = True,
        debug: bool = False,
        apply_diversity: bool = True,
    ) -> Dict[str, Any]:
        k = int(k or self.config.k_default)

        row = self._get_user_candidate_row(user_idx)
        if row is None:
            return {
                "user_idx": int(user_idx),
                "k": k,
                "split": self.config.split,
                "recommendations": [],
                "debug": {"reason": "user_not_in_candidates"} if debug else None,
            }

        candidates: List[int] = list(row.get("candidates") or [])
        sources: List[str] = list(row.get("blend_sources") or [])

        # blend_score list may be absent in some older files
        blend_scores = row.get("blend_score")
        if blend_scores is None:
            blend_scores = [0.0] * len(candidates)
        else:
            blend_scores = list(blend_scores)

        # Align lengths defensively
        n = min(len(candidates), len(sources), len(blend_scores))
        candidates = candidates[:n]
        sources = sources[:n]
        blend_scores = blend_scores[:n]

        if n == 0:
            return {
                "user_idx": int(user_idx),
                "k": k,
                "split": self.config.split,
                "recommendations": [],
                "debug": {"reason": "empty_candidate_list"} if debug else None,
            }

        has_tt, has_seq, has_v2 = self._derive_source_flags(sources)

        sess_lit = self._session_literals(user_idx, n)

        base = pl.DataFrame({
            "user_idx": [int(user_idx)] * n,
            "item_idx": candidates,
            "blend_score": blend_scores,
            "blend_sources": sources,
            "has_tt": has_tt,
            "has_seq": has_seq,
            "has_v2": has_v2,
            "short_term_boost": sess_lit["short_term_boost"],
            "sess_hot": sess_lit["sess_hot"],
            "sess_warm": sess_lit["sess_warm"],
            "sess_cold": sess_lit["sess_cold"],
        })

        feats = self._attach_user_item_features(base)
        scores = self._score_with_ranker(feats)

        # Apply feedback adjustments + drop watched
        scores = self._apply_feedback_adjustments(user_idx, candidates, scores)
        candidates, scores = self._filter_watched(user_idx, candidates, scores)

        if len(candidates) == 0:
            return {
                "user_idx": int(user_idx),
                "k": k,
                "split": self.config.split,
                "recommendations": [],
                "debug": {"reason": "all_filtered_as_watched"} if debug else None,
            }

        # Rebuild small aligned structures after filtering
        # For simplicity, recompute sources/flags by item_idx lookup from original base
        base_map = {it: i for i, it in enumerate(base["item_idx"].to_list())}
        rows_out: List[Dict[str, Any]] = []

        last_title = sess_lit.get("last_title")

        for it, sc in sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True):
            i = base_map.get(it)
            src = sources[i] if i is not None and i < len(sources) else ""
            tt = has_tt[i] if i is not None and i < len(has_tt) else 0
            sq = has_seq[i] if i is not None and i < len(has_seq) else 0
            v2 = has_v2[i] if i is not None and i < len(has_v2) else 0

            meta = self.item_meta_map.get(int(it), {})
            title = meta.get("title")
            movie_id = meta.get("movieId")

            poster_url = self.poster_resolver.get(movie_id)

            reason = self._why_this(sq, tt, v2, last_title)

            rows_out.append({
                "item_idx": int(it),
                "movieId": int(movie_id) if movie_id is not None else None,
                "title": title if include_titles else None,
                "poster_url": poster_url,
                "score": float(sc),
                "reason": reason,
                "has_tt": int(tt),
                "has_seq": int(sq),
                "has_v2": int(v2),
                "short_term_boost": float(sess_lit["short_term_boost"][0]) if sess_lit["short_term_boost"] else 0.0,
                "sess_hot": int(sess_lit["sess_hot"][0]) if sess_lit["sess_hot"] else 0,
                "sess_warm": int(sess_lit["sess_warm"][0]) if sess_lit["sess_warm"] else 0,
                "sess_cold": int(sess_lit["sess_cold"][0]) if sess_lit["sess_cold"] else 1,
                "blend_source_raw": src,
            })

        if apply_diversity:
            rows_out = self._simple_source_diversity(rows_out, k=k, max_streak=3)
        else:
            rows_out = rows_out[:k]

        payload = {
            "user_idx": int(user_idx),
            "k": k,
            "split": self.config.split,
            "recommendations": rows_out,
        }

        if debug:
            payload["debug"] = {
                "split": self.config.split,
                "project_root": str(self.root),
                "candidates_path": str(self.candidates_path),
                "session_features_path": str(self.session_path),
                "ranker_path": str(self.ranker_path),
                "ranker_meta_path": str(self.ranker_meta_path) if self.ranker_meta_path else None,
                "model_auc": self.model_auc,
                "feature_order": self.feature_order,
                "candidate_count": n,
                "session_bucket": sess_lit.get("session_recency_bucket"),
                "last_title": last_title,
                "poster_cache_path": str(self.poster_resolver.cache_path),
            }

        return payload


# ---------------------------
# Factory
# ---------------------------

_SERVICE_CACHE: Dict[str, V4RecommenderService] = {}


def get_v4_service(split: str = "val") -> V4RecommenderService:
    split = (split or "val").lower()
    if split not in _SERVICE_CACHE:
        _SERVICE_CACHE[split] = V4RecommenderService(V4ServiceConfig(split=split))
    return _SERVICE_CACHE[split]