"""
V4 TEST pointwise builder.

Goal
- Build a pointwise training/eval table for V4 TEST ranking.
- Keep symmetry with V4 VAL builder.
- Add session features (short_term_boost + hot/warm/cold flags).
- Use V3 blended candidate pool as base.

Expected inputs
- data/processed/v3_candidates_test.parquet
    columns expected:
      user_idx: int
      candidates: list[int]
      blend_score: list[float] OR blend_score scalar after explode
      blend_sources: list[str] OR blend_sources scalar after explode

- data/processed/session_features_v4_test.parquet
    columns:
      user_idx, short_term_boost, session_recency_bucket, last_item_idx, last_title, last_seq_len

- train_conf / test_conf
    We attempt common locations:
      data/processed/train_conf.parquet
      data/processed/test_conf.parquet

Output
- data/processed/rank_v4_test_pointwise.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple  # noqa: UP035

import polars as pl

DATA_DIR = Path("data")
PROCESSED = DATA_DIR / "processed"


def _first_existing(paths: List[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(
        "None of the expected files exist. Tried:\n" + "\n".join(str(p) for p in paths)
    )


def _load_candidates_test() -> pl.LazyFrame:
    p = PROCESSED / "v3_candidates_test.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing V3 test candidates: {p}")
    return pl.scan_parquet(p)


def _load_session_test() -> pl.LazyFrame:
    p = PROCESSED / "session_features_v4_test.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing V4 session features (test): {p}")
    return pl.scan_parquet(p)


def _load_train_conf() -> pl.LazyFrame:
    p = _first_existing(
        [
            PROCESSED / "train_conf.parquet",
            PROCESSED / "confidence_train.parquet",
            PROCESSED / "train_confidence.parquet",
        ]
    )
    return pl.scan_parquet(p)


def _load_test_conf() -> pl.LazyFrame:
    p = _first_existing(
        [
            PROCESSED / "test_conf.parquet",
            PROCESSED / "confidence_test.parquet",
            PROCESSED / "test_confidence.parquet",
        ]
    )
    return pl.scan_parquet(p)


def _infer_weight_col(df: pl.LazyFrame) -> str:
    # Try common weight/confidence column names.
    schema = df.schema
    for c in ["confidence", "conf", "weight", "rating"]:
        if c in schema:
            return c
    # If nothing obvious exists, we'll synthesize weight=1.0 later.
    return ""


def _infer_time_col(df: pl.LazyFrame) -> str:
    schema = df.schema
    for c in ["ts", "timestamp", "created_at", "event_time"]:
        if c in schema:
            return c
    return ""


def _build_user_item_features(train_conf: pl.LazyFrame) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    wcol = _infer_weight_col(train_conf)
    tcol = _infer_time_col(train_conf)

    base = train_conf
    if wcol == "":
        base = base.with_columns(pl.lit(1.0).alias("_w"))
        wcol = "_w"

    # Basic aggregates
    user_agg = (
        base.group_by("user_idx")
        .agg(
            pl.len().alias("user_interactions"),
            pl.col(wcol).sum().alias("user_conf_sum"),
        )
    )

    item_agg = (
        base.group_by("item_idx")
        .agg(
            pl.len().alias("item_interactions"),
            pl.col(wcol).sum().alias("item_conf_sum"),
        )
    )

    # Decay + recency fallbacks
    # If time column exists, we can compute days since last event.
    if tcol:
        # Convert to datetime if needed; we avoid strict parsing assumptions.
        tmp = base.with_columns(
            pl.col(tcol).cast(pl.Datetime, strict=False).alias("_dt")
        )

        user_last = tmp.group_by("user_idx").agg(pl.col("_dt").max().alias("_user_last"))
        item_last = tmp.group_by("item_idx").agg(pl.col("_dt").max().alias("_item_last"))

        # We can't compute "days since last" without a reference "now" in offline eval.
        # So we store 0 for now; consistent with earlier V3 fallback logic.
        user_last = user_last.with_columns(pl.lit(0.0).alias("user_days_since_last")).drop("_user_last")
        item_last = item_last.with_columns(pl.lit(0.0).alias("item_days_since_last")).drop("_item_last")

        user_agg = user_agg.join(user_last, on="user_idx", how="left")
        item_agg = item_agg.join(item_last, on="item_idx", how="left")
    else:
        user_agg = user_agg.with_columns(pl.lit(0.0).alias("user_days_since_last"))
        item_agg = item_agg.with_columns(pl.lit(0.0).alias("item_days_since_last"))

    # For simplicity in MVP, decay sum mirrors conf sum.
    user_agg = user_agg.with_columns(
        pl.col("user_conf_sum").alias("user_conf_decay_sum")
    )
    item_agg = item_agg.with_columns(
        pl.col("item_conf_sum").alias("item_conf_decay_sum")
    )

    return user_agg, item_agg


def _build_truth_map(test_conf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Build a single target item per user for labeling.

    We pick the highest-confidence row per user if confidence-like column exists.
    Otherwise pick the last row by natural order (stable enough for MVP).
    """
    wcol = _infer_weight_col(test_conf)

    if wcol:
        truth = (
            test_conf
            .group_by("user_idx")
            .agg(
                pl.col("item_idx")
                .sort_by(pl.col(wcol), descending=True)
                .first()
                .alias("truth_item_idx")
            )
        )
    else:
        truth = (
            test_conf
            .group_by("user_idx")
            .agg(
                pl.col("item_idx").first().alias("truth_item_idx")
            )
        )

    return truth


def _explode_candidates(cand: pl.LazyFrame) -> pl.LazyFrame:
    """
    Supports both:
    - blend_score as list aligned with candidates
    - or blend_score already scalar

    Same for blend_sources.
    """
    schema = cand.schema

    # Ensure expected columns exist
    if "candidates" not in schema:
        raise ValueError("v3_candidates_test.parquet must contain 'candidates' list column")

    has_score = "blend_score" in schema
    has_sources = "blend_sources" in schema

    # If missing, create safe defaults
    if not has_score:
        cand = cand.with_columns(pl.lit([]).alias("blend_score"))
    if not has_sources:
        cand = cand.with_columns(pl.lit([]).alias("blend_sources"))

    # Explode lists if they are list dtypes
    # We do a best-effort approach based on dtype string.
    def _is_list(col: str) -> bool:
        dt = str(schema.get(col, ""))
        return "List" in dt

    cols_to_explode = ["candidates"]
    if _is_list("blend_score"):
        cols_to_explode.append("blend_score")
    if _is_list("blend_sources"):
        cols_to_explode.append("blend_sources")

    exploded = cand.explode(cols_to_explode)

    # Rename to item_idx for downstream symmetry
    exploded = exploded.with_columns(
        pl.col("candidates").cast(pl.Int64).alias("item_idx"),
    ).drop("candidates")

    # If blend_score was missing or empty, fill zeros
    if "blend_score" in exploded.schema:
        exploded = exploded.with_columns(
            pl.col("blend_score").cast(pl.Float64).fill_null(0.0)
        )
    else:
        exploded = exploded.with_columns(pl.lit(0.0).alias("blend_score"))

    # Normalize sources to string
    if "blend_sources" in exploded.schema:
        exploded = exploded.with_columns(
            pl.col("blend_sources").cast(pl.Utf8, strict=False).fill_null("")
        )
    else:
        exploded = exploded.with_columns(pl.lit("").alias("blend_sources"))

    return exploded


def _derive_source_flags(df: pl.LazyFrame) -> pl.LazyFrame:
    src = pl.col("blend_sources")

    return df.with_columns(
        (src.str.contains("two_tower", literal=False) | src.str.contains("ann", literal=False))
        .cast(pl.Int8)
        .alias("has_tt"),
        (src.str.contains("sequence", literal=False) | src.str.contains("gru", literal=False))
        .cast(pl.Int8)
        .alias("has_seq"),
        (src.str.contains("v2", literal=False) | src.str.contains("prior", literal=False))
        .cast(pl.Int8)
        .alias("has_v2"),
    )


def _one_hot_session(sess: pl.LazyFrame) -> pl.LazyFrame:
    # session_recency_bucket expected: hot/warm/cold
    return sess.with_columns(
        (pl.col("session_recency_bucket") == "hot").cast(pl.Int8).alias("sess_hot"),
        (pl.col("session_recency_bucket") == "warm").cast(pl.Int8).alias("sess_warm"),
        (pl.col("session_recency_bucket") == "cold").cast(pl.Int8).alias("sess_cold"),
    )


def build_v4_test_pointwise(cap_users: int = 50000) -> Path:
    out_path = PROCESSED / "rank_v4_test_pointwise.parquet"

    cand = _load_candidates_test()

    # Cap users deterministically
    cand = cand.filter(pl.col("user_idx") < cap_users)

    test_conf = _load_test_conf()
    train_conf = _load_train_conf()
    sess = _load_session_test()

    truth = _build_truth_map(test_conf)

    user_feats, item_feats = _build_user_item_features(train_conf)
    sess = _one_hot_session(sess)

    exploded = _explode_candidates(cand)
    exploded = _derive_source_flags(exploded)

    # Join truth in for labels
    exploded = exploded.join(truth, on="user_idx", how="left")

    exploded = exploded.with_columns(
        (pl.col("item_idx") == pl.col("truth_item_idx"))
        .cast(pl.Int8)
        .fill_null(0)
        .alias("label")
    ).drop("truth_item_idx")

    # Join session + long-term features
    exploded = (
        exploded.join(sess.select(["user_idx", "short_term_boost", "sess_hot", "sess_warm", "sess_cold"]),
                      on="user_idx", how="left")
        .join(user_feats, on="user_idx", how="left")
        .join(item_feats, on="item_idx", how="left")
    )

    # Null safety
    exploded = exploded.with_columns(
        pl.col("short_term_boost").fill_null(0.0),
        pl.col("sess_hot").fill_null(0),
        pl.col("sess_warm").fill_null(0),
        pl.col("sess_cold").fill_null(0),
        pl.col("user_interactions").fill_null(0),
        pl.col("user_conf_sum").fill_null(0.0),
        pl.col("user_conf_decay_sum").fill_null(0.0),
        pl.col("user_days_since_last").fill_null(0.0),
        pl.col("item_interactions").fill_null(0),
        pl.col("item_conf_sum").fill_null(0.0),
        pl.col("item_conf_decay_sum").fill_null(0.0),
        pl.col("item_days_since_last").fill_null(0.0),
    )

    # Order + persist with streaming
    exploded = exploded.select(
        [
            "user_idx",
            "item_idx",
            "label",
            "blend_score",
            "has_tt",
            "has_seq",
            "has_v2",
            "short_term_boost",
            "sess_hot",
            "sess_warm",
            "sess_cold",
            "user_interactions",
            "user_conf_sum",
            "user_conf_decay_sum",
            "user_days_since_last",
            "item_interactions",
            "item_conf_sum",
            "item_conf_decay_sum",
            "item_days_since_last",
        ]
    )

    # Write (lazy -> streaming)
    exploded.sink_parquet(out_path, compression="zstd")

    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap-users", type=int, default=50000)
    args = parser.parse_args()

    print("[START] Building V4 TEST pointwise table (lazy+streaming)...")
    p = build_v4_test_pointwise(cap_users=args.cap_users)
    print("[DONE] V4 TEST pointwise table saved (streamed).")
    print(f"[PATH] {p}")


if __name__ == "__main__":
    main()