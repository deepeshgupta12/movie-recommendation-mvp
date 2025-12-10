# src/ranking/build_v3_test_pointwise.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple  # noqa: UP035

import polars as pl


@dataclass
class BuildV3TestPointwiseConfig:
    candidates_path: str = "data/processed/v3_candidates_test.parquet"
    test_conf_path: str = "data/processed/test_conf.parquet"
    user_features_path: str = "data/processed/user_features.parquet"
    item_features_path: str = "data/processed/item_features.parquet"

    out_pointwise_path: str = "data/processed/rank_v3_test_pointwise.parquet"

    max_users: int = 50000
    max_candidates_per_user: int = 200


def _ensure_parents(path_str: str) -> None:
    p = Path(path_str)
    p.parent.mkdir(parents=True, exist_ok=True)


def _parse_source_flags(src: Optional[str]) -> Tuple[int, int, int]:
    if not src:
        return 0, 0, 0
    s = str(src).lower()
    has_tt = 1 if ("two_tower" in s or "tt" in s or "ann" in s) else 0
    has_seq = 1 if ("sequence" in s or "gru" in s or "seq" in s) else 0
    has_v2 = 1 if ("v2" in s or "prior" in s or "pop" in s or "genre" in s) else 0
    return has_tt, has_seq, has_v2


def _explode_candidates(df: pl.DataFrame, max_k: int) -> pl.DataFrame:
    cols = df.columns

    if "candidates" not in cols:
        raise ValueError("v3_candidates_test.parquet must contain 'candidates' list column.")

    if "blend_scores" not in cols:
        df = df.with_columns(pl.lit(None).alias("blend_scores"))
    if "blend_sources" not in cols:
        df = df.with_columns(pl.lit(None).alias("blend_sources"))

    df = df.with_columns(
        pl.col("candidates").list.head(max_k).alias("candidates"),
        pl.col("blend_scores").list.head(max_k).alias("blend_scores"),
        pl.col("blend_sources").list.head(max_k).alias("blend_sources"),
    )

    exploded = df.explode(["candidates", "blend_scores", "blend_sources"]).rename(
        {
            "candidates": "item_idx",
            "blend_scores": "blend_score",
            "blend_sources": "blend_source_raw",
        }
    )

    exploded = exploded.with_columns(
        pl.col("blend_score").fill_null(0.0).cast(pl.Float64)
    )

    return exploded


def build_pointwise(cfg: BuildV3TestPointwiseConfig) -> pl.DataFrame:
    print("\n[START] Building V3 TEST pointwise table (decoupled)...")

    cand = pl.read_parquet(cfg.candidates_path)
    test_conf = pl.read_parquet(cfg.test_conf_path)

    print(f"[OK] v3 candidate users (raw): {cand.select('user_idx').n_unique()}")
    print(f"[OK] test_conf rows: {test_conf.height}")

    cand = cand.sort("user_idx").head(cfg.max_users)

    truth = (
        test_conf
        .select(["user_idx", "item_idx"])
        .unique()
        .with_columns(pl.lit(1).alias("label"))
    )

    print(f"[OK] truth users: {truth.select('user_idx').n_unique()}")

    print("[START] Exploding candidate lists...")
    exploded = _explode_candidates(cand, max_k=cfg.max_candidates_per_user)
    print(f"[OK] exploded rows: {exploded.height}")

    print("[START] Creating labels...")
    exploded = exploded.join(truth, on=["user_idx", "item_idx"], how="left").with_columns(
        pl.col("label").fill_null(0).cast(pl.Int8)
    )

    print("[START] Deriving source flags...")
    src_py: List[Optional[str]] = exploded.select("blend_source_raw").to_series().to_list()
    flags = [_parse_source_flags(s) for s in src_py]

    exploded = exploded.with_columns(
        pl.Series("has_tt", [f[0] for f in flags]).cast(pl.Int8),
        pl.Series("has_seq", [f[1] for f in flags]).cast(pl.Int8),
        pl.Series("has_v2", [f[2] for f in flags]).cast(pl.Int8),
    )

    print("[START] Joining user/item features...")
    user_feat = pl.read_parquet(cfg.user_features_path)
    item_feat = pl.read_parquet(cfg.item_features_path)

    df = (
        exploded
        .join(user_feat, on="user_idx", how="left")
        .join(item_feat, on="item_idx", how="left")
    )

    # Fill null numeric cols with 0
    for c in df.columns:
        if c in ("user_idx", "item_idx", "blend_source_raw"):
            continue
        if df[c].dtype.is_numeric():
            df = df.with_columns(pl.col(c).fill_null(0))

    _ensure_parents(cfg.out_pointwise_path)
    df.write_parquet(cfg.out_pointwise_path)

    print("[DONE] V3 TEST pointwise table saved.")
    print(f"[PATH] {Path(cfg.out_pointwise_path).resolve()}")
    print(f"[OK] rows: {df.height}")

    return df


def main():
    cfg = BuildV3TestPointwiseConfig()
    build_pointwise(cfg)


if __name__ == "__main__":
    main()