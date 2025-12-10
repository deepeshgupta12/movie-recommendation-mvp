# src/ranking/build_v4_val_pointwise.py

from __future__ import annotations

from pathlib import Path

import polars as pl

DATA_DIR = Path("data")
PROCESSED = DATA_DIR / "processed"
REPORTS = Path("reports")


def _load_candidates_val() -> pl.DataFrame:
    p = PROCESSED / "v3_candidates_val.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing V3 candidates file: {p}")
    df = pl.read_parquet(p)

    # Expect at least these
    required = {"user_idx", "candidates"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"v3_candidates_val missing columns: {missing} | cols={df.columns}")

    # Normalize expected optional columns
    if "blend_score" not in df.columns:
        # fallback to tt_scores if present
        if "tt_scores" in df.columns:
            df = df.rename({"tt_scores": "blend_score"})
        else:
            df = df.with_columns(pl.col("candidates").list.eval(pl.lit(0.0)).alias("blend_score"))

    if "blend_sources" not in df.columns:
        df = df.with_columns(
            pl.col("candidates").list.eval(pl.lit("two_tower_ann")).alias("blend_sources")
        )

    return df.select(["user_idx", "candidates", "blend_score", "blend_sources"])


def _load_truth_val() -> pl.DataFrame:
    # V3 used val_conf as truth source
    p = PROCESSED / "val_conf.parquet"
    if not p.exists():
        # fallback: try legacy naming if any
        raise FileNotFoundError(f"Missing truth file for val: {p}")
    df = pl.read_parquet(p)

    # Expect: user_idx, item_idx, confidence (or similar)
    if "user_idx" not in df.columns or "item_idx" not in df.columns:
        raise RuntimeError(f"val_conf missing required columns | cols={df.columns}")

    return df.select(["user_idx", "item_idx"]).unique()


def _load_user_features() -> pl.DataFrame | None:
    p = PROCESSED / "user_features.parquet"
    if p.exists():
        return pl.read_parquet(p)
    return None


def _load_item_features() -> pl.DataFrame | None:
    p = PROCESSED / "item_features.parquet"
    if p.exists():
        return pl.read_parquet(p)
    return None


def _load_session_features_val() -> pl.DataFrame:
    p = PROCESSED / "session_features_v4_val.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing session features: {p}")
    return pl.read_parquet(p)


def build_v4_val_pointwise(out_path: Path | None = None) -> Path:
    print("[START] Building V4 VAL pointwise table...")

    cands = _load_candidates_val()
    truth = _load_truth_val()
    sess = _load_session_features_val()

    print(f"[OK] candidate users: {cands.select('user_idx').n_unique()}")
    print(f"[OK] truth users: {truth.select('user_idx').n_unique()}")
    print(f"[OK] session users: {sess.select('user_idx').n_unique()}")

    # Explode candidate lists
    # Important: explode only list columns.
    exploded = (
        cands
        .explode("candidates")
        .explode("blend_score")
        .explode("blend_sources")
        .rename({
            "candidates": "item_idx",
        })
    )

    print(f"[OK] exploded rows: {exploded.height}")

    # Label creation
    exploded = exploded.join(
        truth.with_columns(pl.lit(1).alias("label")),
        on=["user_idx", "item_idx"],
        how="left",
    ).with_columns(
        pl.col("label").fill_null(0).cast(pl.Int8)
    )

    # Derive source flags from blend_sources
    # blend_sources is a string per row after explode
    exploded = exploded.with_columns(
        pl.col("blend_sources").cast(pl.Utf8).alias("blend_source_raw")
    )

    exploded = exploded.with_columns(
        pl.col("blend_source_raw").str.contains("two_tower").cast(pl.Int8).alias("has_tt"),
        pl.col("blend_source_raw").str.contains("sequence").cast(pl.Int8).alias("has_seq"),
        pl.col("blend_source_raw").str.contains("v2").cast(pl.Int8).alias("has_v2"),
        pl.col("blend_score").cast(pl.Float64),
    )

    # Join session features
    exploded = exploded.join(
        sess.select([
            "user_idx",
            "short_term_boost",
            "session_recency_bucket",
        ]),
        on="user_idx",
        how="left",
    )

    # One-hot for buckets (stable for tree models)
    exploded = exploded.with_columns(
        (pl.col("session_recency_bucket") == "hot").cast(pl.Int8).alias("sess_hot"),
        (pl.col("session_recency_bucket") == "warm").cast(pl.Int8).alias("sess_warm"),
        (pl.col("session_recency_bucket") == "cold").cast(pl.Int8).alias("sess_cold"),
        pl.col("short_term_boost").fill_null(0.0).cast(pl.Float64),
    )

    # Join existing user/item features if available
    uf = _load_user_features()
    if uf is not None and "user_idx" in uf.columns:
        exploded = exploded.join(uf, on="user_idx", how="left")

    itf = _load_item_features()
    if itf is not None and "item_idx" in itf.columns:
        exploded = exploded.join(itf, on="item_idx", how="left")

    # Minimal fallback columns if user/item features don't exist
    # This avoids downstream crashes in the trainer.
    for col_name in [
        "user_interactions",
        "user_conf_sum",
        "user_conf_decay_sum",
        "user_days_since_last",
        "item_interactions",
        "item_conf_sum",
        "item_conf_decay_sum",
        "item_days_since_last",
    ]:
        if col_name not in exploded.columns:
            exploded = exploded.with_columns(pl.lit(0).cast(pl.Int32).alias(col_name))

    # Final V4 feature frame
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
        "user_interactions",
        "user_conf_sum",
        "user_conf_decay_sum",
        "user_days_since_last",
        "item_interactions",
        "item_conf_sum",
        "item_conf_decay_sum",
        "item_days_since_last",
    ])

    if out_path is None:
        out_path = PROCESSED / "rank_v4_val_pointwise.parquet"

    final.write_parquet(out_path)

    print("[DONE] V4 VAL pointwise table saved.")
    print(f"[PATH] {out_path}")
    print(f"[OK] rows: {final.height}")

    return out_path


def main():
    build_v4_val_pointwise()


if __name__ == "__main__":
    main()