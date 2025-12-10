# src/service/reco_service_v4.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple  # noqa: UP035

import joblib
import polars as pl

from src.service.poster_cache import PosterCache

# -----------------------------
# Feedback store (lightweight)
# -----------------------------

class FeedbackStoreV4:
    """
    Very lightweight feedback store.
    Keeps an in-memory state per user and appends events to a JSON file.
    This is designed to make the V4 loop *visibly* change results
    without retraining.

    Event types we support:
      - like
      - watched
      - watch_later
      - dislike
      - unwatch / unlike / remove_watch_later (optional)
    """
    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = self.root / "data" / "processed" / "feedback_events_v4_live.json"
        self.state: Dict[int, Dict[str, set[int]]] = {}

        self._load_existing()

    def _load_existing(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
            # Accept either:
            # 1) {"events":[...]} style
            # 2) [{...},{...}] style
            if isinstance(raw, dict) and "events" in raw:
                events = raw["events"]
            else:
                events = raw

            if not isinstance(events, list):
                return

            for e in events:
                try:
                    self.record_event(
                        user_idx=int(e["user_idx"]),
                        item_idx=int(e["item_idx"]),
                        event=str(e["event"]),
                        persist=False,
                        ts=float(e.get("ts", time.time()))
                    )
                except Exception:
                    continue
        except Exception:
            # If file is corrupt, ignore to keep service alive
            return

    def _ensure_user(self, user_idx: int) -> Dict[str, set[int]]:
        if user_idx not in self.state:
            self.state[user_idx] = {
                "like": set(),
                "watched": set(),
                "watch_later": set(),
                "dislike": set(),
            }
        return self.state[user_idx]

    def record_event(
        self,
        user_idx: int,
        item_idx: int,
        event: str,
        persist: bool = True,
        ts: Optional[float] = None
    ) -> None:
        ts = float(ts or time.time())
        s = self._ensure_user(user_idx)

        # Normalize event
        ev = event.strip().lower()

        if ev in ("like", "liked"):
            s["like"].add(item_idx)
            s["dislike"].discard(item_idx)

        elif ev in ("watched", "watch"):
            s["watched"].add(item_idx)
            s["watch_later"].discard(item_idx)

        elif ev in ("watch_later", "later"):
            s["watch_later"].add(item_idx)

        elif ev in ("dislike", "skip"):
            s["dislike"].add(item_idx)
            s["like"].discard(item_idx)
            s["watch_later"].discard(item_idx)

        elif ev in ("unlike", "remove_like"):
            s["like"].discard(item_idx)

        elif ev in ("unwatch", "remove_watched"):
            s["watched"].discard(item_idx)

        elif ev in ("remove_watch_later", "unlater"):
            s["watch_later"].discard(item_idx)

        # Persist as a simple list file
        if persist:
            self._append_event_file(
                {"user_idx": user_idx, "item_idx": item_idx, "event": ev, "ts": ts}
            )

    def _append_event_file(self, event: Dict[str, Any]) -> None:
        # Keep it simple: store as {"events":[...]} to avoid JSONL complexity.
        # Small scale MVP.
        try:
            if self.path.exists():
                raw = json.loads(self.path.read_text())
                if isinstance(raw, dict) and isinstance(raw.get("events"), list):
                    raw["events"].append(event)
                elif isinstance(raw, list):
                    raw.append(event)
                    raw = {"events": raw}
                else:
                    raw = {"events": [event]}
            else:
                raw = {"events": [event]}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(raw, ensure_ascii=False, indent=2))
        except Exception:
            # Never crash the recommender because feedback persistence failed
            pass

    def get_state(self, user_idx: int) -> Dict[str, List[int]]:
        s = self._ensure_user(user_idx)
        return {
            "like": sorted(list(s["like"])),
            "watched": sorted(list(s["watched"])),
            "watch_later": sorted(list(s["watch_later"])),
            "dislike": sorted(list(s["dislike"])),
        }


# -----------------------------
# Config
# -----------------------------

@dataclass
class V4ServiceConfig:
    split: str = "val"  # val | test
    k_default: int = 20
    project_root: Optional[Path] = None

    # Optional overrides if you want:
    candidates_path: Optional[Path] = None
    session_features_path: Optional[Path] = None
    item_meta_path: Optional[Path] = None
    ranker_path: Optional[Path] = None
    ranker_meta_path: Optional[Path] = None

    def resolve_root(self) -> Path:
        if self.project_root:
            return self.project_root
        # src/service/reco_service_v4.py -> src/service -> src -> project root
        return Path(__file__).resolve().parents[2]


# -----------------------------
# Service
# -----------------------------

class V4RecommenderService:
    """
    V4 objective:
      - Use V3 blended candidate pool
      - Add session-aware features (short-term boost + hot/warm/cold flags)
      - Keep the V3 live feedback loop experience
      - Provide stable API outputs for UI
    """

    def __init__(self, config: V4ServiceConfig) -> None:
        self.config = config
        self.root = config.resolve_root()

        self.split = (config.split or "val").lower().strip()
        if self.split not in ("val", "test"):
            self.split = "val"

        # Paths
        self.candidates_path = (
            config.candidates_path
            or self.root / "data" / "processed" / f"v3_candidates_{self.split}.parquet"
        )
        self.session_features_path = (
            config.session_features_path
            or self.root / "data" / "processed" / f"session_features_v4_{self.split}.parquet"
        )
        self.item_meta_path = (
            config.item_meta_path
            or self.root / "data" / "processed" / "item_meta.parquet"
        )

        self.ranker_path = (
            config.ranker_path
            or self.root / "reports" / "models" / f"ranker_hgb_v4_{self.split}.pkl"
        )
        self.ranker_meta_path = (
            config.ranker_meta_path
            or self.root / "reports" / "models" / f"ranker_hgb_v4_{self.split}.meta.json"
        )

        # Core data
        self.poster_cache = PosterCache(self.root)
        self.feedback = FeedbackStoreV4(self.root)

        self._item_meta = self._load_item_meta()
        self._session_feats = self._load_session_features()

        self._ranker, self._feature_order, self._model_auc = self._load_ranker()

        # We keep candidates lazy to avoid repeated huge loads.
        self._cand_lf = self._load_candidates_lazy()

    # -----------------------------
    # Loaders
    # -----------------------------

    def _load_item_meta(self) -> pl.DataFrame:
        if not self.item_meta_path.exists():
            # Minimal fallback to keep UI alive
            return pl.DataFrame(
                {"item_idx": [], "movieId": [], "title": []},
                schema={"item_idx": pl.Int64, "movieId": pl.Int64, "title": pl.Utf8},
            )
        df = pl.read_parquet(self.item_meta_path)
        # normalize cols
        cols = set(df.columns)
        if "item_idx" not in cols:
            # attempt to infer
            if "itemId" in cols:
                df = df.rename({"itemId": "item_idx"})
        if "movieId" not in cols:
            df = df.with_columns(pl.lit(None).cast(pl.Int64).alias("movieId"))
        if "title" not in cols:
            df = df.with_columns(pl.lit("").alias("title"))
        return df.select([c for c in ["item_idx", "movieId", "title"] if c in df.columns])

    def _load_session_features(self) -> pl.DataFrame:
        if not self.session_features_path.exists():
            return pl.DataFrame(
                {
                    "user_idx": [],
                    "last_item_idx": [],
                    "last_title": [],
                    "last_seq_len": [],
                    "short_term_boost": [],
                    "session_recency_bucket": [],
                }
            )
        return pl.read_parquet(self.session_features_path)

    def _load_ranker(self):
        if not self.ranker_path.exists() or not self.ranker_meta_path.exists():
            raise FileNotFoundError(
                "V4 ranker not found.\n"
                f"Split={self.split}\n"
                f"Tried: {self.ranker_path}\n"
                "Train it first via:\n"
                "  python -m src.ranking.train_ranker_v4_val\n"
                "  python -m src.ranking.train_ranker_v4_test\n"
            )

        ranker = joblib.load(self.ranker_path)
        meta = json.loads(self.ranker_meta_path.read_text())
        feature_order = meta.get("features", [])
        model_auc = float(meta.get("auc", 0.0))
        return ranker, feature_order, model_auc

    def _load_candidates_lazy(self) -> pl.LazyFrame:
        if not self.candidates_path.exists():
            # keep empty LF
            return pl.LazyFrame(
                schema={
                    "user_idx": pl.Int64,
                    "candidates": pl.List(pl.Int64),
                }
            )
        return pl.scan_parquet(self.candidates_path)

    # -----------------------------
    # Helpers
    # -----------------------------

    def _get_user_candidate_row(self, user_idx: int) -> pl.DataFrame:
        lf = self._cand_lf.filter(pl.col("user_idx") == user_idx)
        df = lf.collect()
        return df

    def _get_session_row(self, user_idx: int) -> Optional[dict]:
        if self._session_feats.is_empty():
            return None
        try:
            row = (
                self._session_feats
                .filter(pl.col("user_idx") == user_idx)
                .head(1)
            )
            if row.height == 0:
                return None
            return row.to_dicts()[0]
        except Exception:
            return None

    def _resolve_title(self, item_idx: int) -> str:
        if self._item_meta.is_empty():
            return ""
        try:
            r = self._item_meta.filter(pl.col("item_idx") == item_idx).head(1)
            if r.height == 0:
                return ""
            return str(r["title"][0])
        except Exception:
            return ""

    def _resolve_movie_id(self, item_idx: int) -> Optional[int]:
        if self._item_meta.is_empty():
            return None
        try:
            r = self._item_meta.filter(pl.col("item_idx") == item_idx).head(1)
            if r.height == 0:
                return None
            v = r["movieId"][0]
            return int(v) if v is not None else None
        except Exception:
            return None

    def _resolve_poster_url(self, item_idx: int) -> Optional[str]:
        # Cache-first:
        mid = self._resolve_movie_id(item_idx)
        if mid is not None:
            url = self.poster_cache.get_by_movie_id(mid)
            if url:
                return url

        # Fallback to item_idx keyed caches (if any)
        url = self.poster_cache.get_by_item_idx(item_idx)
        if url:
            return url

        return None

    def _derive_source_flags(self, blend_sources: Optional[List[str]]) -> Tuple[int, int, int]:
        if not blend_sources:
            return 0, 0, 0
        # blend_sources entries might be like "two_tower_ann,v2_prior"
        joined = ",".join(blend_sources).lower()
        has_tt = 1 if ("two_tower" in joined or "tt" in joined or "ann" in joined) else 0
        has_seq = 1 if ("seq" in joined or "gru" in joined) else 0
        has_v2 = 1 if ("v2" in joined or "prior" in joined) else 0
        return has_tt, has_seq, has_v2

    def _ensure_feature_order(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add any missing features in meta with safe defaults.
        """
        if not self._feature_order:
            return df

        existing = set(df.columns)
        to_add = []
        for f in self._feature_order:
            if f not in existing:
                # default numeric
                to_add.append(pl.lit(0.0).alias(f))
        if to_add:
            df = df.with_columns(to_add)

        # Cast ints/bools to numeric if needed
        # Keep ordering strict
        cols = [c for c in self._feature_order if c in df.columns]
        return df.select(cols)

    # -----------------------------
    # Core V4 scoring
    # -----------------------------

    def _build_base_frame(self, user_idx: int) -> pl.DataFrame:
        cand_row = self._get_user_candidate_row(user_idx)
        if cand_row.height == 0:
            return pl.DataFrame(
                {"user_idx": [], "item_idx": [], "blend_score": [], "blend_sources": []}
            )

        # Normalize columns
        cols = cand_row.columns
        if "candidates" not in cols:
            # defensive: some earlier variants use candidates_v3
            for alt in ("candidates_v3", "items", "item_idx_list"):
                if alt in cols:
                    cand_row = cand_row.rename({alt: "candidates"})
                    break

        if "blend_score" not in cols:
            # might be tt_scores or something else
            if "tt_scores" in cols:
                cand_row = cand_row.rename({"tt_scores": "blend_score"})
            else:
                cand_row = cand_row.with_columns(
                    pl.lit([]).cast(pl.List(pl.Float64)).alias("blend_score")
                )

        if "blend_sources" not in cols:
            # may not exist in early blends
            cand_row = cand_row.with_columns(
                pl.lit([]).cast(pl.List(pl.Utf8)).alias("blend_sources")
            )

        # Ensure list lengths align by safest strategy:
        # explode candidates alone, then left-attach scores/sources by index if present.
        # But in our V3 files, explode(["candidates","blend_score","blend_sources"]) usually works.
        # We'll handle mismatch gracefully using list.get with row_count.

        # Add row_count for safe alignment
        cand_row = cand_row.with_columns(pl.lit(user_idx).alias("user_idx"))

        # Explode candidates
        exploded = cand_row.select(["user_idx", "candidates", "blend_score", "blend_sources"]).explode("candidates")
        exploded = exploded.rename({"candidates": "item_idx"})

        # If blend_score is a list and same length, we can try a parallel explode
        # but that can fail on dtype mismatch. We'll rebuild safely:
        # create an index per user row
        exploded = exploded.with_columns(pl.int_range(0, pl.len()).alias("_pos"))

        # Fetch original lists
        orig = cand_row.select(["blend_score", "blend_sources"]).to_dicts()[0]
        score_list = orig.get("blend_score") or []
        src_list = orig.get("blend_sources") or []

        # Create small helper frames from python lists for alignment
        s_df = pl.DataFrame({"_pos": list(range(len(score_list))), "blend_score": score_list})
        b_df = pl.DataFrame({"_pos": list(range(len(src_list))), "blend_sources": src_list})

        exploded = exploded.join(s_df, on="_pos", how="left").join(b_df, on="_pos", how="left")

        # Fill missing blend_score
        exploded = exploded.with_columns(
            pl.col("blend_score").fill_null(0.0).cast(pl.Float64)
        )

        return exploded.drop("_pos")

    def _attach_session_features(self, base: pl.DataFrame, user_idx: int) -> pl.DataFrame:
        sess = self._get_session_row(user_idx)
        if not sess:
            return base.with_columns(
                [
                    pl.lit(0.0).alias("short_term_boost"),
                    pl.lit(0).alias("sess_hot"),
                    pl.lit(0).alias("sess_warm"),
                    pl.lit(0).alias("sess_cold"),
                ]
            )

        bucket = str(sess.get("session_recency_bucket", "") or "").lower()
        hot = 1 if bucket == "hot" else 0
        warm = 1 if bucket == "warm" else 0
        cold = 1 if bucket == "cold" else 0
        stb = float(sess.get("short_term_boost", 0.0) or 0.0)

        return base.with_columns(
            [
                pl.lit(stb).alias("short_term_boost"),
                pl.lit(hot).alias("sess_hot"),
                pl.lit(warm).alias("sess_warm"),
                pl.lit(cold).alias("sess_cold"),
            ]
        )

    def _attach_source_flags(self, base: pl.DataFrame) -> pl.DataFrame:
        # blend_sources might be null scalar per row; convert to list[str] or empty
        # We'll derive flags row-wise using a Python apply for robustness.
        def _flags(x):
            if x is None:
                return (0, 0, 0)
            if isinstance(x, str):
                lst = [x]
            else:
                lst = list(x) if isinstance(x, (list, tuple)) else []
            return self._derive_source_flags(lst)

        return base.with_columns(
            [
                pl.col("blend_sources").map_elements(lambda x: _flags(x)[0], return_dtype=pl.Int8).alias("has_tt"),
                pl.col("blend_sources").map_elements(lambda x: _flags(x)[1], return_dtype=pl.Int8).alias("has_seq"),
                pl.col("blend_sources").map_elements(lambda x: _flags(x)[2], return_dtype=pl.Int8).alias("has_v2"),
            ]
        )

    def _score_with_ranker(self, feat_df: pl.DataFrame) -> List[float]:
        # Ensure strict feature order and defaults
        X = self._ensure_feature_order(feat_df)

        # Convert to pandas for sklearn
        pd_df = X.to_pandas()
        scores = self._ranker.predict_proba(pd_df)[:, 1].tolist()
        return [float(s) for s in scores]

    def _build_feature_frame(self, user_idx: int) -> pl.DataFrame:
        base = self._build_base_frame(user_idx)
        if base.is_empty():
            return base

        base = self._attach_session_features(base, user_idx)
        base = self._attach_source_flags(base)

        # If user/item features exist from earlier steps, try to load them.
        # Otherwise leave zeros (we'll add missing columns later).
        # We keep this ultra-defensive to avoid breaking your branch.
        user_feat_path = self.root / "data" / "processed" / "user_features.parquet"
        item_feat_path = self.root / "data" / "processed" / "item_features.parquet"

        if user_feat_path.exists():
            try:
                uf = pl.read_parquet(user_feat_path)
                base = base.join(uf, on="user_idx", how="left")
            except Exception:
                pass

        if item_feat_path.exists():
            try:
                itf = pl.read_parquet(item_feat_path)
                base = base.join(itf, on="item_idx", how="left")
            except Exception:
                pass

        # Fill known numeric nulls
        for col in [
            "user_interactions", "user_conf_sum", "user_conf_decay_sum", "user_days_since_last",
            "item_interactions", "item_conf_sum", "item_conf_decay_sum", "item_days_since_last",
        ]:
            if col in base.columns:
                base = base.with_columns(
                    pl.col(col).fill_null(0.0)
                )

        return base

    # -----------------------------
    # Why-this generator
    # -----------------------------

    def _why_this(self, row: dict, last_title: Optional[str]) -> str:
        # Session-aware reason first if seq context exists
        has_seq = int(row.get("has_seq", 0) or 0)
        has_tt = int(row.get("has_tt", 0) or 0)
        has_v2 = int(row.get("has_v2", 0) or 0)

        if has_seq == 1 and last_title:
            return f"Because you watched {last_title} recently"
        if has_tt == 1:
            return "Similar to your taste"
        if has_v2 == 1:
            return "Popular among similar users"
        return "Recommended for you"

    # -----------------------------
    # Simple diversity interleaver
    # -----------------------------

    def _interleave_by_reason(self, items: List[dict], k: int) -> List[dict]:
        buckets: Dict[str, List[dict]] = {}
        order_pref = [
            "Because you watched",  # prefix match
            "Similar to your taste",
            "Popular among similar users",
            "Recommended for you",
        ]

        for it in items:
            r = str(it.get("reason") or "Recommended for you")
            key = r
            # normalize the seq reason prefix
            if r.startswith("Because you watched"):
                key = "Because you watched"
            buckets.setdefault(key, []).append(it)

        out: List[dict] = []
        idx = 0
        # round-robin across preferred order
        while len(out) < k:
            progressed = False
            for pref in order_pref:
                if pref in buckets and buckets[pref]:
                    out.append(buckets[pref].pop(0))
                    progressed = True
                    if len(out) >= k:
                        break
            if not progressed:
                # dump remaining
                for kk in list(buckets.keys()):
                    while buckets[kk] and len(out) < k:
                        out.append(buckets[kk].pop(0))
                break
            idx += 1

        return out[:k]

    # -----------------------------
    # Public API
    # -----------------------------

    def record_feedback(self, user_idx: int, item_idx: int, event: str) -> Dict[str, Any]:
        self.feedback.record_event(user_idx=user_idx, item_idx=item_idx, event=event)
        return {"ok": True, "user_idx": int(user_idx), "item_idx": int(item_idx), "event": event}

    def get_user_state(self, user_idx: int) -> Dict[str, Any]:
        return self.feedback.get_state(user_idx)

    def recommend(
        self,
        user_idx: int,
        k: int = 20,
        include_titles: bool = True,
        debug: bool = False
    ) -> Dict[str, Any]:
        user_idx = int(user_idx)
        k = int(k)

        base = self._build_feature_frame(user_idx)
        if base.is_empty():
            out = {"user_idx": user_idx, "k": k, "items": [], "split": self.split}
            if debug:
                out["debug"] = {"split": self.split, "candidates_path": str(self.candidates_path)}
            return out

        # Get last title for why-this
        sess_row = self._get_session_row(user_idx)
        last_title = None
        if sess_row:
            last_title = str(sess_row.get("last_title") or "") or None

        # Score with ranker
        scores = self._score_with_ranker(base)

        # Build item dicts
        rows = base.to_dicts()
        items: List[dict] = []
        state = self.feedback.get_state(user_idx)
        watched_set = set(state["watched"])
        dislike_set = set(state["dislike"])

        for row, sc in zip(rows, scores):
            item_idx = int(row.get("item_idx"))
            if item_idx in watched_set or item_idx in dislike_set:
                continue

            # Session boost in post-score for visible effect
            stb = float(row.get("short_term_boost") or 0.0)
            sess_hot = int(row.get("sess_hot") or 0)

            # Mild bump if hot
            final_sc = float(sc + (0.02 * stb) + (0.01 * sess_hot))

            title = self._resolve_title(item_idx) if include_titles else ""
            movie_id = self._resolve_movie_id(item_idx)

            reason = self._why_this(row, last_title)

            items.append(
                {
                    "item_idx": item_idx,
                    "movieId": int(movie_id) if movie_id is not None else None,
                    "title": title,
                    "poster_url": self._resolve_poster_url(item_idx),
                    "score": float(final_sc),
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

        # Sort by score
        items.sort(key=lambda x: x["score"], reverse=True)

        # Interleave by reason for "Netflix-like" grouping feel
        items = self._interleave_by_reason(items, k)

        out: Dict[str, Any] = {
            "user_idx": user_idx,
            "k": k,
            "items": items,
            "split": self.split,
        }

        if debug:
            out["debug"] = {
                "split": self.split,
                "project_root": str(self.root),
                "candidates_path": str(self.candidates_path),
                "session_features_path": str(self.session_features_path),
                "ranker_path": str(self.ranker_path),
                "ranker_meta_path": str(self.ranker_meta_path),
                "model_auc": float(self._model_auc),
                "feature_order": list(self._feature_order),
                "candidate_count": int(len(rows)),
                "poster_cache_v4_path": str(self.poster_cache.cache_v4_path),
                "poster_cache_has_any": bool(self.poster_cache.has_any()),
            }

        return out


# -----------------------------
# Singleton helper
# -----------------------------

_SERVICE_SINGLETON: Dict[str, V4RecommenderService] = {}


def get_v4_service(split: str = "val") -> V4RecommenderService:
    split = (split or "val").lower().strip()
    if split not in ("val", "test"):
        split = "val"

    if split in _SERVICE_SINGLETON:
        return _SERVICE_SINGLETON[split]

    svc = V4RecommenderService(V4ServiceConfig(split=split))
    _SERVICE_SINGLETON[split] = svc
    return svc