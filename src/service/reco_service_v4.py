# src/service/reco_service_v4.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple  # noqa: UP035

import joblib
import numpy as np
import polars as pl

Split = Literal["val", "test"]


# ---------------------------
# Config
# ---------------------------

@dataclass
class V4ServiceConfig:
    split: Split = "val"
    apply_diversity: bool = True
    project_root: Optional[Path] = None

    # Paths (relative to root)
    candidates_val: str = "data/processed/v3_candidates_val.parquet"
    candidates_test: str = "data/processed/v3_candidates_test.parquet"

    session_val: str = "data/processed/session_features_v4_val.parquet"
    session_test: str = "data/processed/session_features_v4_test.parquet"

    ranker_val: str = "reports/models/ranker_hgb_v4_val.pkl"
    ranker_val_meta: str = "reports/models/ranker_hgb_v4_val.meta.json"

    ranker_test: str = "reports/models/ranker_hgb_v4_test.pkl"
    ranker_test_meta: str = "reports/models/ranker_hgb_v4_test.meta.json"

    # Meta / posters
    item_meta: str = "data/processed/item_meta.parquet"
    movies_raw: str = "data/raw/movies.csv"

    poster_cache_v4: str = "data/processed/poster_cache_v4.json"
    item_posters_legacy: str = "data/processed/item_posters.json"

    # Feedback state (lightweight local)
    feedback_state_path: str = "data/processed/feedback_state_v4.json"


# ---------------------------
# Service singleton helper
# ---------------------------

_V4_SERVICE_SINGLETON: Optional["V4RecommenderService"] = None


def get_v4_service(cfg: Optional[V4ServiceConfig] = None) -> "V4RecommenderService":
    global _V4_SERVICE_SINGLETON
    if cfg is None:
        cfg = V4ServiceConfig()
    if _V4_SERVICE_SINGLETON is None or _V4_SERVICE_SINGLETON.config.split != cfg.split:
        _V4_SERVICE_SINGLETON = V4RecommenderService(cfg)
    return _V4_SERVICE_SINGLETON


# ---------------------------
# Core Service
# ---------------------------

class V4RecommenderService:
    """
    V4 = V3 blended candidates + session-aware features + optional diversity + live feedback filtering.

    Key stability choice:
    - We DO NOT use Polars joins with synthetic `_pos` keys for diversity.
      Diversity is applied on Python lists to avoid dtype Null join crashes.
    """

    def __init__(self, config: V4ServiceConfig):
        self.config = config
        self.root = config.project_root or self._infer_root()

        # Paths resolved
        self.candidates_path = self._resolve_candidates_path()
        self.session_path = self._resolve_session_path()
        self.ranker_path, self.ranker_meta_path = self._resolve_ranker_paths()

        # Load artifacts
        self._load_item_meta()
        self._load_posters()
        self._load_feedback_state()
        self._load_ranker_meta_and_model()

        # Session features lazy load (small)
        self.session_df = self._load_session_features()

    # ---------------------------
    # Root + paths
    # ---------------------------

    def _infer_root(self) -> Path:
        # assumes file lives at src/service/reco_service_v4.py
        here = Path(__file__).resolve()
        return here.parents[2]

    def _resolve_candidates_path(self) -> Path:
        rel = self.config.candidates_val if self.config.split == "val" else self.config.candidates_test
        return (self.root / rel).resolve()

    def _resolve_session_path(self) -> Path:
        rel = self.config.session_val if self.config.split == "val" else self.config.session_test
        return (self.root / rel).resolve()

    def _resolve_ranker_paths(self) -> Tuple[Path, Path]:
        if self.config.split == "val":
            return (self.root / self.config.ranker_val).resolve(), (self.root / self.config.ranker_val_meta).resolve()
        return (self.root / self.config.ranker_test).resolve(), (self.root / self.config.ranker_test_meta).resolve()

    # ---------------------------
    # Load meta / posters
    # ---------------------------

    def _load_item_meta(self) -> None:
        meta_path = (self.root / self.config.item_meta).resolve()
        if meta_path.exists():
            df = pl.read_parquet(meta_path)
        else:
            # fallback
            raw_path = (self.root / self.config.movies_raw).resolve()
            if not raw_path.exists():
                raise FileNotFoundError(
                    f"Could not locate item metadata.\n"
                    f"Tried: {meta_path}\n"
                    f"Fallback raw missing: {raw_path}"
                )
            df = pl.read_csv(raw_path)

            # best-effort normalize
            # Expect at least: movieId, title, genres
            if "movieId" not in df.columns or "title" not in df.columns:
                raise FileNotFoundError("movies.csv exists but missing required columns movieId/title.")

            # create naive item_idx = row index
            df = df.with_row_count("item_idx")

        # Build lookup dicts
        self.item_meta_df = df
        # Ensure item_idx exists
        if "item_idx" not in self.item_meta_df.columns:
            self.item_meta_df = self.item_meta_df.with_row_count("item_idx")

        # Keep compact mapping for fast service use
        self.item_lookup: Dict[int, Dict[str, Optional[str]]] = {}
        cols = set(self.item_meta_df.columns)

        has_movieId = "movieId" in cols
        has_title = "title" in cols
        has_genres = "genres" in cols

        for row in self.item_meta_df.select(
            ["item_idx"] + (["movieId"] if has_movieId else []) + (["title"] if has_title else []) + (["genres"] if has_genres else [])
        ).iter_rows(named=True):
            idx = int(row["item_idx"])
            self.item_lookup[idx] = {
                "movieId": str(row["movieId"]) if has_movieId and row.get("movieId") is not None else None,
                "title": row.get("title") if has_title else None,
                "genres": row.get("genres") if has_genres else None,
            }

    def _load_posters(self) -> None:
        self.poster_map: Dict[str, str] = {}

        cache_path = (self.root / self.config.poster_cache_v4).resolve()
        legacy_path = (self.root / self.config.item_posters_legacy).resolve()

        if cache_path.exists():
            try:
                self.poster_map.update(json.loads(cache_path.read_text()))
            except Exception:
                pass

        if legacy_path.exists():
            try:
                legacy = json.loads(legacy_path.read_text())
                # legacy might be keyed by movieId or item_idx; merge without overwrite
                for k, v in legacy.items():
                    if k not in self.poster_map and v:
                        self.poster_map[k] = v
            except Exception:
                pass

    # ---------------------------
    # Feedback state
    # ---------------------------

    def _load_feedback_state(self) -> None:
        self.feedback_state_path = (self.root / self.config.feedback_state_path).resolve()
        self.feedback_state: Dict[str, Dict[str, List[int]]] = {}

        if self.feedback_state_path.exists():
            try:
                self.feedback_state = json.loads(self.feedback_state_path.read_text())
            except Exception:
                self.feedback_state = {}

    def _save_feedback_state(self) -> None:
        self.feedback_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.feedback_state_path.write_text(json.dumps(self.feedback_state, indent=2))

    def record_feedback(self, user_idx: int, item_idx: int, event: str) -> Dict[str, str]:
        """
        Lightweight file-backed feedback.
        We primarily use feedback to:
        - filter explicit seen/watched items in next recommend call
        - allow UI to show state
        """
        u = str(user_idx)
        st = self.feedback_state.get(u, {"liked": [], "watched": [], "watch_later": [], "skipped": []})

        def _add(key: str):
            if item_idx not in st[key]:
                st[key].append(item_idx)

        def _remove(key: str):
            if item_idx in st[key]:
                st[key].remove(item_idx)

        if event == "like":
            _add("liked")
        elif event == "remove_like":
            _remove("liked")
        elif event == "watched":
            _add("watched")
        elif event == "watch_later":
            _add("watch_later")
        elif event == "remove_watch_later":
            _remove("watch_later")
        elif event == "skip":
            _add("skipped")
        else:
            return {"status": "ignored", "detail": f"unknown event: {event}"}

        self.feedback_state[u] = st
        self._save_feedback_state()
        return {"status": "ok"}

    # ---------------------------
    # Ranker
    # ---------------------------

    def _load_ranker_meta_and_model(self) -> None:
        if not self.ranker_meta_path.exists():
            raise FileNotFoundError(
                f"V4 ranker meta not found at {self.ranker_meta_path}. "
                f"Train it first."
            )
        if not self.ranker_path.exists():
            raise FileNotFoundError(
                f"V4 ranker not found.\n"
                f"Split={self.config.split}\n"
                f"Tried: {self.ranker_path}\n"
                f"Train it first via:\n"
                f"  python -m src.ranking.train_ranker_v4_{self.config.split}"
            )

        meta = json.loads(self.ranker_meta_path.read_text())
        self.model_auc = float(meta.get("auc", 0.0))
        self.feature_order: List[str] = list(meta.get("features", []))

        self.ranker = joblib.load(self.ranker_path)

    # ---------------------------
    # Session features
    # ---------------------------

    def _load_session_features(self) -> pl.DataFrame:
        if self.session_path.exists():
            df = pl.read_parquet(self.session_path)
            return df
        # empty fallback with expected cols
        return pl.DataFrame(
            {
                "user_idx": pl.Series([], dtype=pl.Int64),
                "last_item_idx": pl.Series([], dtype=pl.Int64),
                "last_title": pl.Series([], dtype=pl.Utf8),
                "last_seq_len": pl.Series([], dtype=pl.Int64),
                "short_term_boost": pl.Series([], dtype=pl.Float64),
                "session_recency_bucket": pl.Series([], dtype=pl.Utf8),
            }
        )

    def _get_session_row(self, user_idx: int) -> Dict[str, object]:
        if self.session_df.is_empty():
            return {
                "short_term_boost": 0.0,
                "sess_hot": 0,
                "sess_warm": 0,
                "sess_cold": 0,
                "last_item_idx": None,
                "last_title": None,
            }

        row = self.session_df.filter(pl.col("user_idx") == user_idx)
        if row.height == 0:
            return {
                "short_term_boost": 0.0,
                "sess_hot": 0,
                "sess_warm": 0,
                "sess_cold": 0,
                "last_item_idx": None,
                "last_title": None,
            }

        r = row.row(0, named=True)
        bucket = r.get("session_recency_bucket")
        sess_hot = 1 if bucket == "hot" else 0
        sess_warm = 1 if bucket == "warm" else 0
        sess_cold = 1 if bucket == "cold" else 0

        return {
            "short_term_boost": float(r.get("short_term_boost") or 0.0),
            "sess_hot": sess_hot,
            "sess_warm": sess_warm,
            "sess_cold": sess_cold,
            "last_item_idx": r.get("last_item_idx"),
            "last_title": r.get("last_title"),
        }

    # ---------------------------
    # Candidates
    # ---------------------------

    def _load_user_candidate_row(self, user_idx: int) -> Optional[pl.DataFrame]:
        if not self.candidates_path.exists():
            raise FileNotFoundError(f"Candidates not found at: {self.candidates_path}")

        # Scan + filter for single user
        lf = pl.scan_parquet(self.candidates_path).filter(pl.col("user_idx") == user_idx)
        df = lf.collect()
        if df.height == 0:
            return None
        return df

    def _explode_user_candidates(self, user_idx: int) -> pl.DataFrame:
        """
        Safer than Polars explode: convert to Python lists, rebuild a flat DF.
        Avoids dtype explode errors and list/float issues.
        """
        row_df = self._load_user_candidate_row(user_idx)
        if row_df is None:
            return pl.DataFrame({"user_idx": [], "item_idx": [], "blend_score": [], "has_tt": [], "has_seq": [], "has_v2": [], "blend_sources": []})

        cols = row_df.columns

        cand_list = row_df["candidates"][0] if "candidates" in cols else []
        # blend_score may not exist in some earlier versions
        score_list = row_df["blend_score"][0] if "blend_score" in cols else None
        tt_scores = row_df["tt_scores"][0] if "tt_scores" in cols else None

        # Prefer blend_score if present else normalize tt_scores else zeros
        if score_list is None:
            if tt_scores is not None:
                score_list = tt_scores
            else:
                score_list = [0.0] * len(cand_list)

        sources_list = row_df["blend_sources"][0] if "blend_sources" in cols else None
        if sources_list is None:
            sources_list = ["two_tower_ann"] * len(cand_list)

        # Defensive length alignment
        n = len(cand_list)
        if len(score_list) != n:
            score_list = (list(score_list) + [0.0] * n)[:n]
        if len(sources_list) != n:
            sources_list = (list(sources_list) + ["unknown"] * n)[:n]

        has_tt = []
        has_seq = []
        has_v2 = []

        for s in sources_list:
            s_str = str(s)
            has_tt.append(1 if "two_tower" in s_str or "tt" in s_str else 0)
            has_seq.append(1 if "seq" in s_str else 0)
            has_v2.append(1 if "v2" in s_str else 0)

        df = pl.DataFrame(
            {
                "user_idx": [user_idx] * n,
                "item_idx": [int(x) for x in cand_list],
                "blend_score": [float(x) for x in score_list],
                "blend_sources": [str(x) for x in sources_list],
                "has_tt": has_tt,
                "has_seq": has_seq,
                "has_v2": has_v2,
            }
        )
        return df

    # ---------------------------
    # Feature building
    # ---------------------------

    def _load_user_features_table(self) -> pl.DataFrame:
        candidates = [
            "data/processed/user_features.parquet",
            "data/processed/user_agg.parquet",
            "data/processed/user_stats.parquet",
        ]
        for rel in candidates:
            p = (self.root / rel).resolve()
            if p.exists():
                try:
                    return pl.read_parquet(p)
                except Exception:
                    continue

        # empty fallback with likely cols
        return pl.DataFrame(
            {
                "user_idx": pl.Series([], dtype=pl.Int64),
                "user_interactions": pl.Series([], dtype=pl.Int64),
                "user_conf_sum": pl.Series([], dtype=pl.Float64),
                "user_conf_decay_sum": pl.Series([], dtype=pl.Float64),
                "user_days_since_last": pl.Series([], dtype=pl.Float64),
            }
        )

    def _load_item_features_table(self) -> pl.DataFrame:
        candidates = [
            "data/processed/item_features.parquet",
            "data/processed/item_agg.parquet",
            "data/processed/item_stats.parquet",
        ]
        for rel in candidates:
            p = (self.root / rel).resolve()
            if p.exists():
                try:
                    return pl.read_parquet(p)
                except Exception:
                    continue

        return pl.DataFrame(
            {
                "item_idx": pl.Series([], dtype=pl.Int64),
                "item_interactions": pl.Series([], dtype=pl.Int64),
                "item_conf_sum": pl.Series([], dtype=pl.Float64),
                "item_conf_decay_sum": pl.Series([], dtype=pl.Float64),
                "item_days_since_last": pl.Series([], dtype=pl.Float64),
            }
        )

    def _build_feature_frame(self, user_idx: int) -> Tuple[pl.DataFrame, Dict[str, object]]:
        base = self._explode_user_candidates(user_idx)

        sess = self._get_session_row(user_idx)

        # Add session columns as constants
        base = base.with_columns(
            [
                pl.lit(float(sess["short_term_boost"])).alias("short_term_boost"),
                pl.lit(int(sess["sess_hot"])).alias("sess_hot"),
                pl.lit(int(sess["sess_warm"])).alias("sess_warm"),
                pl.lit(int(sess["sess_cold"])).alias("sess_cold"),
            ]
        )

        # Join user/item features if present
        uf = self._load_user_features_table()
        itf = self._load_item_features_table()

        if uf.height > 0:
            base = base.join(uf, on="user_idx", how="left")
        if itf.height > 0:
            base = base.join(itf, on="item_idx", how="left")

        # Fill nulls for known numeric feature set
        numeric_fallbacks = {
            "user_interactions": 0,
            "user_conf_sum": 0.0,
            "user_conf_decay_sum": 0.0,
            "user_days_since_last": 0.0,
            "item_interactions": 0,
            "item_conf_sum": 0.0,
            "item_conf_decay_sum": 0.0,
            "item_days_since_last": 0.0,
            "short_term_boost": 0.0,
            "sess_hot": 0,
            "sess_warm": 0,
            "sess_cold": 0,
        }

        for col, default in numeric_fallbacks.items():
            if col not in base.columns:
                base = base.with_columns(pl.lit(default).alias(col))
            else:
                base = base.with_columns(pl.col(col).fill_null(default))

        debug_sess = {
            "short_term_boost": sess["short_term_boost"],
            "sess_hot": sess["sess_hot"],
            "sess_warm": sess["sess_warm"],
            "sess_cold": sess["sess_cold"],
            "last_item_idx": sess["last_item_idx"],
            "last_title": sess["last_title"],
        }

        return base, debug_sess

    # ---------------------------
    # Posters + item safe getters
    # ---------------------------

    def _poster_for(self, item_idx: int) -> Optional[str]:
        meta = self.item_lookup.get(int(item_idx), {})
        movie_id = meta.get("movieId")

        # Poster cache keys could be movieId or item_idx str
        if movie_id and str(movie_id) in self.poster_map:
            return self.poster_map[str(movie_id)]
        if str(item_idx) in self.poster_map:
            return self.poster_map[str(item_idx)]
        return None

    def _title_for(self, item_idx: int) -> str:
        meta = self.item_lookup.get(int(item_idx), {})
        return meta.get("title") or f"Item {item_idx}"

    def _movie_id_for(self, item_idx: int) -> Optional[int]:
        meta = self.item_lookup.get(int(item_idx), {})
        mid = meta.get("movieId")
        try:
            return int(mid) if mid is not None else None
        except Exception:
            return None

    # ---------------------------
    # Reasons + buckets
    # ---------------------------

    def _why_this(self, has_seq: int, has_tt: int, has_v2: int, last_title: Optional[str]) -> str:
        if has_seq == 1 and last_title:
            return f"Because you watched {last_title} recently"
        if has_seq == 1:
            return "Because you watched something similar recently"
        if has_tt == 1:
            return "Similar to your taste"
        if has_v2 == 1:
            return "Popular among similar users"
        return "Recommended for you"

    def _bucket_for(self, has_seq: int, has_tt: int, has_v2: int) -> str:
        if has_seq == 1:
            return "Because you watched"
        if has_tt == 1:
            return "Similar to your taste"
        if has_v2 == 1:
            return "Popular among similar users"
        return "More picks"

    # ---------------------------
    # Diversity (Python-only, safe)
    # ---------------------------

    def _apply_diversity_round_robin(self, items: List[Dict], k: int) -> List[Dict]:
        """
        Interleave buckets to reduce monotony.
        Uses only Python lists to avoid Polars join dtype issues.
        """
        # group by bucket
        buckets: Dict[str, List[Dict]] = {}
        for it in items:
            b = it.get("bucket", "More picks")
            buckets.setdefault(b, []).append(it)

        # sort each bucket by score desc
        for b in buckets:
            buckets[b].sort(key=lambda x: x.get("score", 0.0), reverse=True)

        # priority order
        order = ["Because you watched", "Similar to your taste", "Popular among similar users", "More picks"]
        order = [b for b in order if b in buckets] + [b for b in buckets.keys() if b not in order]

        out: List[Dict] = []
        pointers = {b: 0 for b in order}

        while len(out) < k:
            progressed = False
            for b in order:
                i = pointers[b]
                if i < len(buckets[b]):
                    out.append(buckets[b][i])
                    pointers[b] += 1
                    progressed = True
                    if len(out) >= k:
                        break
            if not progressed:
                break

        return out

    # ---------------------------
    # Scoring + feedback
    # ---------------------------

    def _score_with_ranker(self, feats: pl.DataFrame) -> np.ndarray:
        # Ensure feature order columns exist
        for f in self.feature_order:
            if f not in feats.columns:
                feats = feats.with_columns(pl.lit(0).alias(f))

        X = feats.select(self.feature_order).to_numpy()

        # HistGradientBoostingClassifier predict_proba
        proba = self.ranker.predict_proba(X)
        # positive class
        return proba[:, 1]

    def _apply_session_multiplier(self, feats: pl.DataFrame, scores: np.ndarray) -> np.ndarray:
        """
        Light, visible session effect:
        - hot: boost seq-sourced items
        - warm: mild boost
        - cold: neutral
        """
        if feats.height == 0:
            return scores

        sess_hot = int(feats["sess_hot"][0]) if "sess_hot" in feats.columns else 0
        sess_warm = int(feats["sess_warm"][0]) if "sess_warm" in feats.columns else 0

        has_seq = feats["has_seq"].to_numpy() if "has_seq" in feats.columns else np.zeros(len(scores))
        mult = np.ones_like(scores, dtype=float)

        if sess_hot == 1:
            mult = np.where(has_seq == 1, 1.20, mult)
        elif sess_warm == 1:
            mult = np.where(has_seq == 1, 1.10, mult)

        return scores * mult

    def _filter_seen_by_feedback(self, user_idx: int, items: List[Dict]) -> List[Dict]:
        st = self.feedback_state.get(str(user_idx))
        if not st:
            return items

        seen = set(st.get("watched", [])) | set(st.get("skipped", []))
        # keep liked/watch_later items (they are preferences), but we still may not want to re-recommend them
        seen |= set(st.get("liked", []))

        return [it for it in items if int(it["item_idx"]) not in seen]

    # ---------------------------
    # Public API
    # ---------------------------

    def recommend(
        self,
        user_idx: int,
        k: int = 20,
        include_titles: bool = True,
        debug: bool = False,
        apply_diversity: Optional[bool] = None,
    ) -> Dict:
        if apply_diversity is None:
            apply_diversity = self.config.apply_diversity

        feats, sess_debug = self._build_feature_frame(user_idx)
        if feats.height == 0:
            out = {
                "user_idx": user_idx,
                "k": k,
                "items": [],
                "split": self.config.split,
            }
            if debug:
                out["debug"] = {
                    "split": self.config.split,
                    "candidates_path": str(self.candidates_path),
                    "session_features_path": str(self.session_path),
                    "ranker_path": str(self.ranker_path),
                    "model_auc": self.model_auc,
                    "feature_order": self.feature_order,
                    "candidate_count": 0,
                    "session": sess_debug,
                }
            return out

        raw_scores = self._score_with_ranker(feats)
        scores = self._apply_session_multiplier(feats, raw_scores)

        # Build python items
        last_title = sess_debug.get("last_title")

        items: List[Dict] = []
        for i in range(feats.height):
            item_idx = int(feats["item_idx"][i])
            has_tt = int(feats["has_tt"][i])
            has_seq = int(feats["has_seq"][i])
            has_v2 = int(feats["has_v2"][i])

            reason = self._why_this(has_seq, has_tt, has_v2, last_title)
            bucket = self._bucket_for(has_seq, has_tt, has_v2)

            items.append(
                {
                    "item_idx": item_idx,
                    "movieId": self._movie_id_for(item_idx),
                    "title": self._title_for(item_idx) if include_titles else None,
                    "poster_url": self._poster_for(item_idx),
                    "score": float(scores[i]),
                    "blend_score": float(feats["blend_score"][i]) if "blend_score" in feats.columns else 0.0,
                    "reason": reason,
                    "bucket": bucket,
                    "has_tt": has_tt,
                    "has_seq": has_seq,
                    "has_v2": has_v2,
                    "short_term_boost": float(feats["short_term_boost"][i]) if "short_term_boost" in feats.columns else 0.0,
                    "sess_hot": int(feats["sess_hot"][i]) if "sess_hot" in feats.columns else 0,
                    "sess_warm": int(feats["sess_warm"][i]) if "sess_warm" in feats.columns else 0,
                    "sess_cold": int(feats["sess_cold"][i]) if "sess_cold" in feats.columns else 0,
                }
            )

        # Sort by score desc
        items.sort(key=lambda x: x["score"], reverse=True)

        # Apply feedback filtering
        items = self._filter_seen_by_feedback(user_idx, items)

        # Apply diversity
        if apply_diversity:
            items = self._apply_diversity_round_robin(items, k=max(k, 1))
        else:
            items = items[:k]

        # Trim to k
        items = items[:k]

        # Build sections for UI if needed
        sections: Dict[str, List[Dict]] = {}
        for it in items:
            sections.setdefault(it["bucket"], []).append(it)

        out = {
            "user_idx": user_idx,
            "k": k,
            "items": items,
            "sections": sections,
            "split": self.config.split,
        }

        if debug:
            out["debug"] = {
                "split": self.config.split,
                "project_root": str(self.root),
                "candidates_path": str(self.candidates_path),
                "session_features_path": str(self.session_path),
                "ranker_path": str(self.ranker_path),
                "ranker_meta_path": str(self.ranker_meta_path),
                "model_auc": self.model_auc,
                "feature_order": self.feature_order,
                "candidate_count": int(feats.height),
                "session": sess_debug,
                "apply_diversity": bool(apply_diversity),
                "feedback_state_path": str(self.feedback_state_path),
            }

        return out