# src/service/reco_service_v3.py
from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple  # noqa: UP035

import polars as pl

# -------------------------
# Safe path helpers
# -------------------------

def _project_root() -> Path:
    # assumes this file is src/service/reco_service_v3.py
    return Path(__file__).resolve().parents[2]


def _data_dir() -> Path:
    # Prefer env override, else repo-relative
    env = os.getenv("DATA_DIR")
    if env:
        return Path(env)
    return _project_root() / "data"


def _processed_dir() -> Path:
    env = os.getenv("PROCESSED_DIR")
    if env:
        return Path(env)
    return _data_dir() / "processed"


def _reports_dir() -> Path:
    env = os.getenv("REPORTS_DIR")
    if env:
        return Path(env)
    return _project_root() / "reports"


# -------------------------
# Config
# -------------------------

@dataclass
class V3ServiceConfig:
    split: str = "test"  # "test" or "val"
    default_k: int = 20

    # candidate files
    candidates_test_path: Path = _processed_dir() / "v3_candidates_test.parquet"
    candidates_val_path: Path = _processed_dir() / "v3_candidates_val.parquet"

    # optional helpers
    dim_items_path: Path = _processed_dir() / "dim_items.parquet"
    user_seq_val_path: Path = _processed_dir() / "user_seq_val.parquet"
    user_seq_test_path: Path = _processed_dir() / "user_seq_test.parquet"  # may not exist

    # feedback store
    feedback_db_path: Path = _processed_dir() / "v3_feedback.db"

    # UI niceties
    poster_placeholder: str = "https://via.placeholder.com/342x513.png?text=Poster+Unavailable"


# -------------------------
# Feedback Store
# -------------------------

class V3FeedbackStore:
    """
    Lightweight persistent feedback store using sqlite.
    One row per event. We derive latest state by action precedence + latest timestamp.
    """
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init(self):
        with self._conn() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    user_idx INTEGER NOT NULL,
                    item_idx INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    ts INTEGER NOT NULL
                )
                """
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback(user_idx)"
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_user_item ON feedback(user_idx, item_idx)"
            )

    def add(self, user_idx: int, item_idx: int, action: str) -> None:
        ts = int(time.time())
        with self._conn() as con:
            con.execute(
                "INSERT INTO feedback(user_idx, item_idx, action, ts) VALUES (?, ?, ?, ?)",
                (int(user_idx), int(item_idx), str(action), ts),
            )

    def get_events(self, user_idx: int) -> List[Tuple[int, str, int]]:
        with self._conn() as con:
            cur = con.execute(
                "SELECT item_idx, action, ts FROM feedback WHERE user_idx=? ORDER BY ts ASC",
                (int(user_idx),),
            )
            return [(int(r[0]), str(r[1]), int(r[2])) for r in cur.fetchall()]

    def get_state(self, user_idx: int) -> Dict[str, List[int]]:
        """
        Derive latest state by reading events in time order.
        Supported actions:
          - like
          - watched
          - start
          - unlike
          - unwatch
          - unstart
        """
        liked = set()
        watched = set()
        started = set()

        for item_idx, action, _ in self.get_events(user_idx):
            if action == "like":
                liked.add(item_idx)
            elif action == "unlike":
                liked.discard(item_idx)
            elif action == "watched":
                watched.add(item_idx)
            elif action == "unwatch":
                watched.discard(item_idx)
            elif action == "start":
                started.add(item_idx)
            elif action == "unstart":
                started.discard(item_idx)

        # Watched implies not "started" in display semantics
        started = started - watched

        return {
            "liked": sorted(liked),
            "watched": sorted(watched),
            "started": sorted(started),
        }


# -------------------------
# Recommender Service
# -------------------------

class V3RecommenderService:
    """
    V3 online-ish service for UI:
    - reads blended candidates file
    - attaches flags + lightweight reasons
    - applies feedback filters/boosts
    """

    def __init__(self, cfg: Optional[V3ServiceConfig] = None):
        self.cfg = cfg or V3ServiceConfig()
        self.feedback = V3FeedbackStore(self.cfg.feedback_db_path)

        self._dim_items = None
        self._cand_test = None
        self._cand_val = None
        self._user_seq_val = None
        self._user_seq_test = None

    # ------------- loaders -------------

    def _load_dim_items(self) -> pl.DataFrame:
        if self._dim_items is None:
            if self.cfg.dim_items_path.exists():
                self._dim_items = pl.read_parquet(self.cfg.dim_items_path)
            else:
                # fallback minimal dim
                self._dim_items = pl.DataFrame(
                    {"item_idx": [], "title": [], "movieId": []}
                )
        return self._dim_items

    def _load_candidates(self, split: str) -> pl.DataFrame:
        split = split.lower()
        if split == "val":
            if self._cand_val is None:
                self._cand_val = pl.read_parquet(self.cfg.candidates_val_path)
            return self._cand_val
        # default test
        if self._cand_test is None:
            self._cand_test = pl.read_parquet(self.cfg.candidates_test_path)
        return self._cand_test

    def _load_user_seq(self, split: str) -> Optional[pl.DataFrame]:
        split = split.lower()
        if split == "test":
            if self._user_seq_test is None:
                if self.cfg.user_seq_test_path.exists():
                    self._user_seq_test = pl.read_parquet(self.cfg.user_seq_test_path)
                else:
                    self._user_seq_test = None
            return self._user_seq_test

        # val
        if self._user_seq_val is None:
            if self.cfg.user_seq_val_path.exists():
                self._user_seq_val = pl.read_parquet(self.cfg.user_seq_val_path)
            else:
                self._user_seq_val = None
        return self._user_seq_val

    # ------------- reasons + flags -------------

    def _extract_flags_from_sources(self, sources: List[str]) -> Tuple[int, int, int]:
        """
        sources entry may look like:
          "two_tower_ann,sequence_gru,v2_prior"
        """
        if not sources:
            return 0, 0, 0
        s = ",".join(sources).lower()
        has_tt = 1 if "two_tower" in s or "ann" in s else 0
        has_seq = 1 if "sequence" in s or "gru" in s else 0
        has_v2 = 1 if "v2" in s or "prior" in s else 0
        return has_tt, has_seq, has_v2

    def _get_last_watched_title_from_seq(self, user_idx: int, split: str) -> Optional[str]:
        seq_df = self._load_user_seq(split)
        if seq_df is None or seq_df.is_empty():
            return None

        try:
            row = (
                seq_df.filter(pl.col("user_idx") == int(user_idx))
                .select(["seq"])
                .head(1)
            )
            if row.height == 0:
                return None
            seq_list = row["seq"][0]
            if not seq_list:
                return None
            last_item = int(seq_list[-1])

            dim = self._load_dim_items()
            if "title" in dim.columns:
                trow = (
                    dim.filter(pl.col("item_idx") == last_item)
                    .select(["title"])
                    .head(1)
                )
                if trow.height:
                    return str(trow["title"][0])
            return None
        except Exception:
            return None

    def _lightweight_reason(self, has_seq: int, has_tt: int, has_v2: int, last_title: Optional[str]) -> Optional[str]:
        if has_seq and last_title:
            return f"Because you watched {last_title} recently"
        if has_tt:
            return "Similar to your taste"
        if has_v2:
            return "Popular among similar users"
        return None

    # ------------- candidate explode -------------

    def _get_user_candidate_row(self, user_idx: int, split: str) -> pl.DataFrame:
        cdf = self._load_candidates(split)
        row = cdf.filter(pl.col("user_idx") == int(user_idx)).head(1)
        if row.height == 0:
            # No candidates: return empty schema-safe row
            return pl.DataFrame({"user_idx": [int(user_idx)], "candidates": [[]], "blend_sources": [[]], "blend_score": [[]]})
        return row

    def _explode_user_candidates(self, user_idx: int, split: str) -> pl.DataFrame:
        """
        Returns long frame:
          user_idx, item_idx, rank_pos, source_raw
        Handles missing columns gracefully.
        """
        row = self._get_user_candidate_row(user_idx, split)

        # Normalize expected columns
        if "candidates" not in row.columns:
            # try legacy alt names
            for alt in ["items", "recommendations"]:
                if alt in row.columns:
                    row = row.rename({alt: "candidates"})
                    break

        if "blend_sources" not in row.columns:
            # attempt derive from other column names
            for alt in ["sources", "source_list"]:
                if alt in row.columns:
                    row = row.rename({alt: "blend_sources"})
                    break

        # Create fallback blend_sources if missing
        if "blend_sources" not in row.columns:
            row = row.with_columns(pl.lit([[]]).alias("blend_sources"))

        # Optional score list column used only if present
        score_list_col = None
        for cand_score_col in ["blend_score", "blend_scores", "tt_scores"]:
            if cand_score_col in row.columns:
                score_list_col = cand_score_col
                break
        if score_list_col and score_list_col != "blend_score":
            row = row.rename({score_list_col: "blend_score"})
        if "blend_score" not in row.columns:
            row = row.with_columns(pl.lit([[]]).alias("blend_score"))

        # Explode safely
        out = row.select(["user_idx", "candidates", "blend_sources", "blend_score"])

        # make sure lists are aligned in length where possible
        # If sources list is empty, we will fill later per item
        # If score list is empty, we will compute a positional base score
        out = out.explode("candidates").rename({"candidates": "item_idx"})

        # Create rank position (1-based)
        out = out.with_row_index("rank_pos", offset=1)

        # If blend_sources is a list-of-strings per user but not list per item,
        # we cannot explode it directly alongside candidates.
        # We'll carry raw user-level lists and expand later in python logic.
        # So keep as a single repeated value per row:
        sources_list = row["blend_sources"][0] if row.height else []
        scores_list = row["blend_score"][0] if row.height else []

        # attach user-level lists as columns
        out = out.with_columns(
            pl.lit(sources_list).alias("blend_sources_user"),
            pl.lit(scores_list).alias("blend_score_user"),
        )

        return out

    # ------------- scoring -------------

    def _base_score_from_position(self, rank_pos: int) -> float:
        # simple monotonically decreasing score
        return 1.0 / float(max(rank_pos, 1))

    def _apply_feedback(self, df: pl.DataFrame, user_idx: int) -> pl.DataFrame:
        state = self.feedback.get_state(user_idx)
        watched = set(state["watched"])
        liked = set(state["liked"])

        if df.is_empty():
            return df

        # Filter watched
        df = df.filter(~pl.col("item_idx").is_in(list(watched)))

        # Apply liked boost
        # We'll compute final_score = base_score * (1 + boost)
        # We'll attach a boolean liked flag
        df = df.with_columns(
            pl.col("item_idx").is_in(list(liked)).cast(pl.Int8).alias("is_liked")
        )

        df = df.with_columns(
            (
                pl.col("base_score")
                * (1.0 + (pl.col("is_liked") * 0.25))
            ).alias("final_score")
        )
        return df

    # ------------- public API -------------

    def record_feedback(self, user_idx: int, item_idx: int, action: str) -> Dict[str, Any]:
        action = action.lower().strip()
        allowed = {
            "like", "unlike",
            "watched", "unwatch",
            "start", "unstart",
        }
        if action not in allowed:
            raise ValueError(f"Unsupported action: {action}")

        self.feedback.add(int(user_idx), int(item_idx), action)
        return {"ok": True, "user_idx": int(user_idx), "item_idx": int(item_idx), "action": action}

    def get_user_state(self, user_idx: int) -> Dict[str, Any]:
        return self.feedback.get_state(int(user_idx))

    def recommend(
        self,
        user_idx: int,
        k: int = 20,
        include_titles: bool = True,
        debug: bool = False,
        split: Optional[str] = None,
    ) -> Dict[str, Any]:

        split = (split or self.cfg.split).lower()
        k = int(k)

        # Explode candidates
        base = self._explode_user_candidates(user_idx, split)

        # attach base score from position
        if base.is_empty():
            return {
                "user_idx": int(user_idx),
                "split": split,
                "k": k,
                "items": [],
                "debug": {"note": "no candidates found"} if debug else None,
            }

        base = base.with_columns(
            pl.col("rank_pos").map_elements(self._base_score_from_position, return_dtype=pl.Float64).alias("base_score")
        )

        # Compute flags from sources list at user-level
        sources_list = []
        try:
            sources_list = base["blend_sources_user"][0] if base.height else []
        except Exception:
            sources_list = []

        has_tt, has_seq, has_v2 = self._extract_flags_from_sources(
            sources_list if isinstance(sources_list, list) else []
        )

        # Attach flags as constants per row (lightweight)
        base = base.with_columns(
            pl.lit(int(has_tt)).alias("has_tt"),
            pl.lit(int(has_seq)).alias("has_seq"),
            pl.lit(int(has_v2)).alias("has_v2"),
        )

        # Apply feedback filter + boost
        scored = self._apply_feedback(base, user_idx)

        # Sort by final_score desc
        if "final_score" not in scored.columns:
            scored = scored.with_columns(pl.col("base_score").alias("final_score"))

        scored = scored.sort("final_score", descending=True).head(k)

        # Titles join
        titles = {}
        if include_titles:
            dim = self._load_dim_items()
            if dim.height and "item_idx" in dim.columns and "title" in dim.columns:
                tdf = dim.select(["item_idx", "title"])
                joined = scored.join(tdf, on="item_idx", how="left")
                scored = joined
            else:
                scored = scored.with_columns(pl.lit(None).alias("title"))

        # Reason generation
        last_title = self._get_last_watched_title_from_seq(user_idx, split)
        reason = self._lightweight_reason(has_seq, has_tt, has_v2, last_title)

        # Build response items
        items_out: List[Dict[str, Any]] = []
        for row in scored.to_dicts():
            items_out.append(
                {
                    "item_idx": int(row.get("item_idx")),
                    "title": row.get("title") or "Unknown Title",
                    "score": float(row.get("final_score", row.get("base_score", 0.0))),
                    "rank_pos": int(row.get("rank_pos", 0)),
                    "has_tt": int(row.get("has_tt", 0)),
                    "has_seq": int(row.get("has_seq", 0)),
                    "has_v2": int(row.get("has_v2", 0)),
                    "reason": reason,
                    # Allow UI to resolve posters; service can add later
                    "poster_url": row.get("poster_url"),
                }
            )

        out = {
            "user_idx": int(user_idx),
            "split": split,
            "k": k,
            "items": items_out,
        }

        if debug:
            out["debug"] = {
                "candidates_file": str(self.cfg.candidates_test_path if split == "test" else self.cfg.candidates_val_path),
                "has_tt": has_tt,
                "has_seq": has_seq,
                "has_v2": has_v2,
                "last_seq_title": last_title,
                "feedback_state": self.get_user_state(user_idx),
                "raw_sources_user": sources_list,
            }

        return out


# -------------------------
# Singleton accessor
# -------------------------

_SERVICE_SINGLETON: Optional[V3RecommenderService] = None


def get_v3_service(cfg: Optional[V3ServiceConfig] = None) -> V3RecommenderService:
    global _SERVICE_SINGLETON
    if cfg is not None:
        # If caller provides a cfg, return a new instance
        return V3RecommenderService(cfg)
    if _SERVICE_SINGLETON is None:
        _SERVICE_SINGLETON = V3RecommenderService()
    return _SERVICE_SINGLETON