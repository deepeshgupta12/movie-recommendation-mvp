"""
V4 TEST pointwise builder.

Goal
- Build a pointwise training/eval table for V4 TEST ranking.
- Symmetric with V4 VAL builder.
- Add session features:
    short_term_boost + sess_hot/sess_warm/sess_cold
- Use V3 blended candidate pool as base.
- Polars-safe lazy+streaming (avoid OOM / zsh killed).

Expected inputs
- data/processed/v3_candidates_test.parquet
    expected columns (flexible):
      user_idx: int
      candidates: list[int]
      blend_score: list[float] (if available)
      OR tt_scores: list[float] (fallback)
      blend_sources: list[str] (if available)

- data/processed/session_features_v4_test.parquet
    columns:
      user_idx, short_term_boost, session_recency_bucket, last_item_idx, last_title, last_seq_len

- data/processed/train_conf.parquet
- data/processed/test_conf.parquet

Output
- data/processed/rank_v4_test_pointwise.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple  # noqa: UP035

import polars as pl

DATA_DIR = Path("data")
PROCESSED = DATA_DIR / "processed"


# -----------------------------
# Utilities
# -----------------------------

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
    schema = df.collect_schema()
    for c in ["confidence", "conf", "weight", "rating"]:
        if c in schema:
            return c
    return ""


def _infer_time_col(df: pl.LazyFrame) -> str:
    schema = df.collect_schema()
    for c in ["ts", "timestamp", "created_at", "event_time"]:
        if c in schema:
            return c
    return ""


# -----------------------------
# Feature builders
# -----------------------------

def _build_user_item_features(train_conf: pl.LazyFrame) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    wcol = _infer_weight_col(train_conf)
    tcol = _infer_time_col(train_conf)

    base = train_conf
    if wcol == "":
        base = base.with_columns(pl.lit(1.0).alias("_w"))
        wcol = "_w"

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

    # MVP decay mirrors sum
    user_agg = user_agg.with_columns(
        pl.col("user_conf_sum").alias("user_conf_decay_sum")
    )
    item_agg = item_agg.with_columns(
        pl.col("item_conf_sum").alias("item_conf_decay_sum")
    )

    # Recency fallback
    if tcol:
        user_agg = user_agg.with_columns(pl.lit(0.0).alias("user_days_since_last"))
        item_agg = item_agg.with_columns(pl.lit(0.0).alias("item_days_since_last"))
    else:
        user_agg = user_agg.with_columns(pl.lit(0.0).alias("user_days_since_last"))
        item_agg = item_agg.with_columns(pl.lit(0.0).alias("item_days_since_last"))

    return user_agg, item_agg


def _build_truth_map(test_conf: pl.LazyFrame) -> pl.LazyFrame:
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
            .agg(pl.col("item_idx").first().alias("truth_item_idx"))
        )

    return truth


def _one_hot_session(sess: pl.LazyFrame) -> pl.LazyFrame:
    return sess.with_columns(
        (pl.col("session_recency_bucket") == "hot").cast(pl.Int8).alias("sess_hot"),
        (pl.col("session_recency_bucket") == "warm").cast(pl.Int8).alias("sess_warm"),
        (pl.col("session_recency_bucket") == "cold").cast(pl.Int8).alias("sess_cold"),
    )


def _explode_candidates(cand: pl.LazyFrame) -> pl.LazyFrame:
    """
    Robust explode for test candidates.

    We DO NOT create empty list placeholders.
    Instead:
    - detect which score column exists
    - explode only real list columns
    - after explode, derive scalar blend_score safely
    """
    schema = cand.collect_schema()

    if "candidates" not in schema:
        raise ValueError("v3_candidates_test.parquet must contain 'candidates' list column")

    has_blend_score = "blend_score" in schema
    has_tt_scores = "tt_scores" in schema
    has_sources = "blend_sources" in schema

    cols_to_explode = ["candidates"]
    if has_blend_score:
        cols_to_explode.append("blend_score")
    elif has_tt_scores:
        cols_to_explode.append("tt_scores")

    if has_sources:
        cols_to_explode.append("blend_sources")

    exploded = cand.explode(cols_to_explode)

    # Rename candidates -> item_idx
    exploded = exploded.with_columns(
        pl.col("candidates").cast(pl.Int64).alias("item_idx")
    ).drop("candidates")

    # Build scalar blend_score
    if has_blend_score:
        exploded = exploded.with_columns(
            pl.col("blend_score").cast(pl.Float64, strict=False).fill_null(0.0)
        )
    elif has_tt_scores:
        exploded = exploded.with_columns(
            pl.col("tt_scores").cast(pl.Float64, strict=False).fill_null(0.0).alias("blend_score")
        ).drop("tt_scores")
    else:
        exploded = exploded.with_columns(pl.lit(0.0).alias("blend_score"))

    # Normalize sources
    if has_sources:
        exploded = exploded.with_columns(
            pl.col("blend_sources").cast(pl.Utf8, strict=False).fill_null("")
        )
    else:
        exploded = exploded.with_columns(pl.lit("").alias("blend_sources"))

    return exploded


def _derive_source_flags(df: pl.LazyFrame) -> pl.LazyFrame:
    src = pl.col("blend_sources")

    return df.with_columns(
        (src.str.contains("two_tower") | src.str.contains("ann"))
        .cast(pl.Int8)
        .alias("has_tt"),
        (src.str.contains("sequence") | src.str.contains("gru"))
        .cast(pl.Int8)
        .alias("has_seq"),
        (src.str.contains("v2") | src.str.contains("prior"))
        .cast(pl.Int8)
        .alias("has_v2"),
    )


# -----------------------------
# Main builder
# -----------------------------

def build_v4_test_pointwise(cap_users: int = 50000) -> Path:
    out_path = PROCESSED / "rank_v4_test_pointwise.parquet"

    cand = _load_candidates_test().filter(pl.col("user_idx") < cap_users)

    test_conf = _load_test_conf()
    train_conf = _load_train_conf()
    sess = _load_session_test()

    truth = _build_truth_map(test_conf)
    user_feats, item_feats = _build_user_item_features(train_conf)
    sess = _one_hot_session(sess)

    exploded = _explode_candidates(cand)
    exploded = _derive_source_flags(exploded)

    exploded = exploded.join(truth, on="user_idx", how="left")

    exploded = exploded.with_columns(
        (pl.col("item_idx") == pl.col("truth_item_idx"))
        .cast(pl.Int8)
        .fill_null(0)
        .alias("label")
    ).drop("truth_item_idx")

    exploded = (
        exploded.join(
            sess.select(["user_idx", "short_term_boost", "sess_hot", "sess_warm", "sess_cold"]),
            on="user_idx",
            how="left",
        )
        .join(user_feats, on="user_idx", how="left")
        .join(item_feats, on="item_idx", how="left")
    )

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
        pl.col("blend_score").fill_null(0.0),
        pl.col("has_tt").fill_null(0),
        pl.col("has_seq").fill_null(0),
        pl.col("has_v2").fill_null(0),
    )

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

    # Lazy + streaming write
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