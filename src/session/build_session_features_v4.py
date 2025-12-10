# src/session/build_session_features_v4.py

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

DATA_DIR = Path("data")
PROCESSED = DATA_DIR / "processed"


def _load_dim_items() -> pl.DataFrame | None:
    """
    Optional dimension table to map item_idx -> title.
    """
    p = PROCESSED / "dim_items.parquet"
    if p.exists():
        df = pl.read_parquet(p)
        cols = df.columns
        if "item_idx" in cols and "title" in cols:
            return df.select(["item_idx", "title"])
    return None


def _load_user_seq_for_split(split: str) -> tuple[pl.DataFrame, str]:
    """
    Load user sequences for the given split.

    For TEST:
      - prefer user_seq_test.parquet
      - fallback to user_seq_val.parquet (consistent with V3 behavior)
    For VAL:
      - use user_seq_val.parquet
    """
    if split == "test":
        p_test = PROCESSED / "user_seq_test.parquet"
        if p_test.exists():
            return pl.read_parquet(p_test), "user_seq_test.parquet"

        p_val = PROCESSED / "user_seq_val.parquet"
        if p_val.exists():
            return pl.read_parquet(p_val), "user_seq_val.parquet"

        raise FileNotFoundError("No user_seq_test.parquet or user_seq_val.parquet found.")

    p_val = PROCESSED / "user_seq_val.parquet"
    if p_val.exists():
        return pl.read_parquet(p_val), "user_seq_val.parquet"

    raise FileNotFoundError("user_seq_val.parquet not found.")


def build_session_features(split: str, cap_users: int = 50000) -> Path:
    seq_df, source_name = _load_user_seq_for_split(split)

    # Expected columns: user_idx, seq (list)
    keep = ["user_idx", "seq"]
    missing = [c for c in keep if c not in seq_df.columns]
    if missing:
        raise RuntimeError(f"Sequence file missing columns: {missing} | cols={seq_df.columns}")

    df = seq_df.select(keep)

    # Cap to local demo users (consistent with your 50k policy)
    df = df.filter(pl.col("user_idx") < cap_users)

    # Derive basic session signals
    df = df.with_columns(
        pl.col("seq").list.len().alias("last_seq_len"),
        pl.col("seq").list.get(-1).alias("last_item_idx"),
    )

    # Lightweight short-term boost heuristic
    df = df.with_columns(
        pl.when(pl.col("last_seq_len") >= 20).then(pl.lit(1.0))
          .when(pl.col("last_seq_len") >= 10).then(pl.lit(0.6))
          .when(pl.col("last_seq_len") >= 5).then(pl.lit(0.3))
          .otherwise(pl.lit(0.0))
          .alias("short_term_boost")
    )

    # Offline-friendly session "recency" bucket
    df = df.with_columns(
        pl.when(pl.col("last_seq_len") >= 15).then(pl.lit("hot"))
          .when(pl.col("last_seq_len") >= 8).then(pl.lit("warm"))
          .otherwise(pl.lit("cold"))
          .alias("session_recency_bucket")
    )

    # Attach last title if dim_items exists
    dim_items = _load_dim_items()
    if dim_items is not None:
        df = df.join(dim_items, left_on="last_item_idx", right_on="item_idx", how="left")
        df = df.rename({"title": "last_title"})
    else:
        df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("last_title"))

    out = PROCESSED / f"session_features_v4_{split}.parquet"

    df.select([
        "user_idx",
        "last_item_idx",
        "last_title",
        "last_seq_len",
        "short_term_boost",
        "session_recency_bucket",
    ]).write_parquet(out)

    print(f"[DONE] V4 session features built for {split}.")
    print(f"[PATH] {out}")
    print(f"[OK] source={source_name} | rows={df.height}")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--cap_users", type=int, default=50000)
    args = ap.parse_args()

    build_session_features(args.split, cap_users=args.cap_users)


if __name__ == "__main__":
    main()