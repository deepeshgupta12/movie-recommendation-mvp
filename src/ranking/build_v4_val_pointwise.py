# src/ranking/build_v4_val_pointwise.py

from __future__ import annotations

from pathlib import Path

import polars as pl

PROCESSED = Path("data/processed")

CANDS_VAL = PROCESSED / "v3_candidates_val.parquet"
TRUTH_VAL = PROCESSED / "val_conf.parquet"
SESSION_VAL = PROCESSED / "session_features_v4_val.parquet"

USER_FEATS = PROCESSED / "user_features.parquet"
ITEM_FEATS = PROCESSED / "item_features.parquet"

OUT_DEFAULT = PROCESSED / "rank_v4_val_pointwise.parquet"


USER_FEATURE_COLS = [
    "user_idx",
    "user_interactions",
    "user_conf_sum",
    "user_conf_decay_sum",
    "user_days_since_last",
]

ITEM_FEATURE_COLS = [
    "item_idx",
    "item_interactions",
    "item_conf_sum",
    "item_conf_decay_sum",
    "item_days_since_last",
]


def _schema(path: Path) -> dict:
    return pl.read_parquet_schema(str(path))


def build_v4_val_pointwise(out_path: Path = OUT_DEFAULT) -> Path:
    if not CANDS_VAL.exists():
        raise FileNotFoundError(f"Missing candidates file: {CANDS_VAL}")
    if not TRUTH_VAL.exists():
        raise FileNotFoundError(f"Missing truth file: {TRUTH_VAL}")
    if not SESSION_VAL.exists():
        raise FileNotFoundError(f"Missing session features file: {SESSION_VAL}")

    print("[START] Building V4 VAL pointwise table (lazy+streaming)...")

    cand_schema = _schema(CANDS_VAL)

    # We expect these from your V3 blend pipeline
    # candidates: list[int]
    # blend_score: list[float] OR tt_scores
    # blend_sources: list[str]
    has_candidates = "candidates" in cand_schema
    has_blend_score = "blend_score" in cand_schema
    has_tt_scores = "tt_scores" in cand_schema
    has_blend_sources = "blend_sources" in cand_schema

    if not has_candidates:
        raise RuntimeError(f"{CANDS_VAL.name} missing 'candidates' column")

    # Build lazy candidates frame
    cands = pl.scan_parquet(str(CANDS_VAL))

    # Normalize score column
    if has_blend_score:
        pass
    elif has_tt_scores:
        cands = cands.rename({"tt_scores": "blend_score"})
    else:
        # Create a zero score list aligned to candidates
        cands = cands.with_columns(
            pl.col("candidates").list.eval(pl.lit(0.0)).alias("blend_score")
        )

    # Normalize sources
    if not has_blend_sources:
        # Create a default sources list aligned to candidates
        cands = cands.with_columns(
            pl.col("candidates").list.eval(pl.lit("two_tower_ann")).alias("blend_sources")
        )

    cands = cands.select(["user_idx", "candidates", "blend_score", "blend_sources"])

    # Truth (val_conf) usually contains the held-out positive(s)
    truth = (
        pl.scan_parquet(str(TRUTH_VAL))
        .select(["user_idx", "item_idx"])
        .unique()
        .with_columns(pl.lit(1).alias("label"))
    )

    # Session features
    sess = (
        pl.scan_parquet(str(SESSION_VAL))
        .select(["user_idx", "short_term_boost", "session_recency_bucket"])
    )

    # Optional user/item features: only load minimal cols
    join_user_feats = USER_FEATS.exists()
    join_item_feats = ITEM_FEATS.exists()

    uf = None
    if join_user_feats:
        uf_schema = _schema(USER_FEATS)
        needed = [c for c in USER_FEATURE_COLS if c in uf_schema]
        if "user_idx" in needed and len(needed) > 1:
            uf = pl.scan_parquet(str(USER_FEATS)).select(needed)
        else:
            uf = None

    itf = None
    if join_item_feats:
        it_schema = _schema(ITEM_FEATS)
        needed = [c for c in ITEM_FEATURE_COLS if c in it_schema]
        if "item_idx" in needed and len(needed) > 1:
            itf = pl.scan_parquet(str(ITEM_FEATS)).select(needed)
        else:
            itf = None

    # Explode ALL list columns together (crucial for memory + alignment)
    exploded = (
        cands
        .explode(["candidates", "blend_score", "blend_sources"])
        .rename({"candidates": "item_idx"})
    )

    # Join truth for labels
    exploded = (
        exploded
        .join(truth, on=["user_idx", "item_idx"], how="left")
        .with_columns(
            pl.col("label").fill_null(0).cast(pl.Int8)
        )
    )

    # Source flags
    exploded = exploded.with_columns(
        pl.col("blend_sources").cast(pl.Utf8).alias("blend_source_raw"),
        pl.col("blend_score").cast(pl.Float64),
    ).with_columns(
        pl.col("blend_source_raw").str.contains("two_tower").cast(pl.Int8).alias("has_tt"),
        pl.col("blend_source_raw").str.contains("sequence").cast(pl.Int8).alias("has_seq"),
        pl.col("blend_source_raw").str.contains("v2").cast(pl.Int8).alias("has_v2"),
    )

    # Join session features
    exploded = exploded.join(sess, on="user_idx", how="left")

    # Bucket one-hot + null guards
    exploded = exploded.with_columns(
        pl.col("short_term_boost").fill_null(0.0).cast(pl.Float64),
        (pl.col("session_recency_bucket") == "hot").cast(pl.Int8).alias("sess_hot"),
        (pl.col("session_recency_bucket") == "warm").cast(pl.Int8).alias("sess_warm"),
        (pl.col("session_recency_bucket") == "cold").cast(pl.Int8).alias("sess_cold"),
    )

    # Join optional user/item features
    if uf is not None:
        exploded = exploded.join(uf, on="user_idx", how="left")
    if itf is not None:
        exploded = exploded.join(itf, on="item_idx", how="left")

    # If some base features are absent, create safe defaults lazily
    base_defaults = {  # noqa: F841
        "user_interactions": pl.lit(0).cast(pl.Int32),
        "user_conf_sum": pl.lit(0).cast(pl.Int32),
        "user_conf_decay_sum": pl.lit(0).cast(pl.Int32),
        "user_days_since_last": pl.lit(0).cast(pl.Int32),
        "item_interactions": pl.lit(0).cast(pl.Int32),
        "item_conf_sum": pl.lit(0).cast(pl.Int32),
        "item_conf_decay_sum": pl.lit(0).cast(pl.Int32),
        "item_days_since_last": pl.lit(0).cast(pl.Int32),
    }

    # Apply missing-col guards based on the evolving schema
    # (We can't "if col not in lazyframe" directly, so we just add them;
    # Polars will overwrite only if duplicates are allowed? To avoid conflicts,
    # we add with "when-null" patterns at select-time.)
    final = exploded.select([
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

        pl.col("user_interactions").fill_null(0).cast(pl.Int32).alias("user_interactions"),
        pl.col("user_conf_sum").fill_null(0).cast(pl.Int32).alias("user_conf_sum"),
        pl.col("user_conf_decay_sum").fill_null(0).cast(pl.Int32).alias("user_conf_decay_sum"),
        pl.col("user_days_since_last").fill_null(0).cast(pl.Int32).alias("user_days_since_last"),

        pl.col("item_interactions").fill_null(0).cast(pl.Int32).alias("item_interactions"),
        pl.col("item_conf_sum").fill_null(0).cast(pl.Int32).alias("item_conf_sum"),
        pl.col("item_conf_decay_sum").fill_null(0).cast(pl.Int32).alias("item_conf_decay_sum"),
        pl.col("item_days_since_last").fill_null(0).cast(pl.Int32).alias("item_days_since_last"),
    ])

    # Streaming write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.sink_parquet(str(out_path))

    print("[DONE] V4 VAL pointwise table saved (streamed).")
    print(f"[PATH] {out_path}")

    return out_path


def main():
    build_v4_val_pointwise()


if __name__ == "__main__":
    main()