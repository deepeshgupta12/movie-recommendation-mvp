# src/ranking/train_ranker_v3_test.py

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple  # noqa: UP035

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# -------------------------
# Config / Paths
# -------------------------

@dataclass
class RankerV3TestConfig:
    # Inputs
    candidates_path: str = "data/processed/v3_candidates_test.parquet"
    test_conf_path: str = "data/processed/test_conf.parquet"
    user_features_path: str = "data/processed/user_features.parquet"
    item_features_path: str = "data/processed/item_features.parquet"

    # Outputs
    out_pointwise_path: str = "data/processed/rank_v3_test_pointwise.parquet"
    model_path: str = "reports/models/ranker_hgb_v3_test.pkl"
    meta_path: str = "reports/models/ranker_hgb_v3_test.meta.json"

    # Training limits
    max_users: int = 50000
    max_candidates_per_user: int = 200

    # Train/val split
    val_frac: float = 0.2
    seed: int = 42


# Locked feature order for V3 ranker
FEATURE_ORDER_V3: List[str] = [
    "blend_score",
    "has_tt",
    "has_seq",
    "has_v2",
    "user_interactions",
    "user_conf_sum",
    "user_conf_decay_sum",
    "user_days_since_last",
    "item_interactions",
    "item_conf_sum",
    "item_conf_decay_sum",
    "item_days_since_last",
]


# -------------------------
# Helpers
# -------------------------

def _ensure_parents(path_str: str) -> None:
    p = Path(path_str)
    p.parent.mkdir(parents=True, exist_ok=True)


def _parse_source_flags(src_list: Optional[List[str]]) -> Tuple[int, int, int]:
    """
    blend_sources is a list[str] aligned to candidates.
    Each entry may contain comma-separated sources like:
      "two_tower_ann,sequence_gru,v2_prior"
    Return flags (has_tt, has_seq, has_v2) for a single candidate row.
    """
    if not src_list:
        return 0, 0, 0

    # We'll accept either a single string or a list of strings.
    # In exploded context we'll pass a single raw string token.
    s = ",".join(src_list) if isinstance(src_list, list) else str(src_list)
    s_low = s.lower()

    has_tt = 1 if ("two_tower" in s_low or "tt" in s_low or "ann" in s_low) else 0
    has_seq = 1 if ("sequence" in s_low or "gru" in s_low or "seq" in s_low) else 0
    has_v2 = 1 if ("v2" in s_low or "prior" in s_low or "pop" in s_low or "genre" in s_low) else 0

    return has_tt, has_seq, has_v2


def _explode_candidates(df: pl.DataFrame, max_k: int) -> pl.DataFrame:
    """
    Input format expected:
      user_idx
      candidates: list[int]
      blend_scores: list[float] (optional)
      blend_sources: list[str] (optional)
    """
    cols = df.columns

    if "candidates" not in cols:
        raise ValueError("v3_candidates_test.parquet must contain 'candidates' list column.")

    # Normalize optional cols
    if "blend_scores" not in cols:
        # create dummy scores aligned to candidates
        df = df.with_columns(
            pl.col("candidates").list.len().alias("_cand_len")
        ).with_columns(
            pl.arange(0, pl.col("_cand_len")).cast(pl.Float64).alias("_tmp_idx")
        )
        # We can't easily build list of zeros per row without udf; simplest:
        # just set a scalar 0.0 then expand after explode.
        df = df.drop(["_cand_len", "_tmp_idx"])
        df = df.with_columns(pl.lit(None).alias("blend_scores"))

    if "blend_sources" not in cols:
        df = df.with_columns(pl.lit(None).alias("blend_sources"))

    # Truncate per-user to max_k before explode if possible
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

    # If blend_score is null (because missing earlier), set 0
    exploded = exploded.with_columns(
        pl.col("blend_score").fill_null(0.0).cast(pl.Float64)
    )

    return exploded


def _build_pointwise_table(cfg: RankerV3TestConfig) -> pl.DataFrame:
    print("\n[START] Building V3 TEST training table...")

    cand = pl.read_parquet(cfg.candidates_path)
    test_conf = pl.read_parquet(cfg.test_conf_path)

    print(f"[OK] v3 candidate users (raw): {cand.select('user_idx').n_unique()}")
    print(f"[OK] test_conf rows: {test_conf.height}")

    # Cap users for local
    cand = cand.sort("user_idx").head(cfg.max_users)

    # Build truth per user for labeling
    # We'll only label items that appear in test_conf
    # Join on user_idx, item_idx later.
    truth = test_conf.select(["user_idx", "item_idx"]).unique()

    print(f"[OK] truth users: {truth.select('user_idx').n_unique()}")

    print("[START] Exploding candidate lists...")
    exploded = _explode_candidates(cand, max_k=cfg.max_candidates_per_user)

    print(f"[OK] exploded rows: {exploded.height}")

    print("[START] Creating labels...")
    exploded = exploded.join(
        truth.with_columns(pl.lit(1).alias("label")),
        on=["user_idx", "item_idx"],
        how="left",
    ).with_columns(
        pl.col("label").fill_null(0).cast(pl.Int8)
    )

    print("[START] Deriving source flags...")
    # Use a Python-side map for robustness across Polars versions
    # We collect blend_source_raw as a python list of strings
    src_py = exploded.select("blend_source_raw").to_series().to_list()
    flags = [_parse_source_flags([s] if s is not None else None) for s in src_py]
    has_tt = [f[0] for f in flags]
    has_seq = [f[1] for f in flags]
    has_v2 = [f[2] for f in flags]

    exploded = exploded.with_columns(
        pl.Series("has_tt", has_tt).cast(pl.Int8),
        pl.Series("has_seq", has_seq).cast(pl.Int8),
        pl.Series("has_v2", has_v2).cast(pl.Int8),
    )

    print("[START] Joining user/item features...")
    user_feat = pl.read_parquet(cfg.user_features_path)
    item_feat = pl.read_parquet(cfg.item_features_path)

    df = (
        exploded
        .join(user_feat, on="user_idx", how="left")
        .join(item_feat, on="item_idx", how="left")
    )

    # Fill null numeric
    for c in df.columns:
        if c in ("user_idx", "item_idx", "blend_source_raw"):
            continue
        if df[c].dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64):
            df = df.with_columns(pl.col(c).fill_null(0))

    print(f"[OK] final training rows: {df.height}")

    _ensure_parents(cfg.out_pointwise_path)
    df.write_parquet(cfg.out_pointwise_path)

    print("[DONE] V3 TEST training table saved.")
    print(f"[PATH] {Path(cfg.out_pointwise_path).resolve()}")

    return df


def _train_hgb(cfg: RankerV3TestConfig, df: pl.DataFrame) -> None:
    print("\n[START] Training V3 TEST ranker...")

    # Shuffle and split
    rng = np.random.default_rng(cfg.seed)
    idx = np.arange(df.height)
    rng.shuffle(idx)

    n_val = int(df.height * cfg.val_frac)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    # Build numpy arrays with locked feature order
    # Ensure missing columns get filled with 0
    for col in FEATURE_ORDER_V3 + ["label"]:
        if col not in df.columns:
            df = df.with_columns(pl.lit(0).alias(col))

    train_df = df[train_idx]
    val_df = df[val_idx]

    X_train = train_df.select(FEATURE_ORDER_V3).to_numpy()
    y_train = train_df.select("label").to_numpy().ravel()

    X_val = val_df.select(FEATURE_ORDER_V3).to_numpy()
    y_val = val_df.select("label").to_numpy().ravel()

    pos_rate = float(y_train.mean()) if len(y_train) else 0.0

    print(f"[OK] Train rows: {len(train_idx)} | Val rows: {len(val_idx)}")
    print(f"[OK] Train positive rate: {pos_rate:.4f}")

    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=250,
        random_state=cfg.seed,
    )

    model.fit(X_train, y_train)

    # AUC
    try:
        p_val = model.predict_proba(X_val)[:, 1]
        auc = float(roc_auc_score(y_val, p_val))
    except Exception:
        auc = float("nan")

    _ensure_parents(cfg.model_path)
    _ensure_parents(cfg.meta_path)

    # Persist model
    import joblib
    joblib.dump(model, cfg.model_path)

    meta = {
        "split": "test",
        "features": FEATURE_ORDER_V3,
        "auc": auc,
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "train_pos_rate": pos_rate,
        "config": asdict(cfg),
        "model_type": "HistGradientBoostingClassifier",
    }

    Path(cfg.meta_path).write_text(json.dumps(meta, indent=2))

    print("[DONE] V3 TEST ranker trained.")
    print(f"[OK] AUC={auc:.4f}")
    print(f"[PATH] {Path(cfg.model_path).resolve()}")
    print(f"[PATH] {Path(cfg.meta_path).resolve()}")
    print(f"[OK] Features used (locked order): {FEATURE_ORDER_V3}")


def main():
    cfg = RankerV3TestConfig()

    df = _build_pointwise_table(cfg)
    _train_hgb(cfg, df)


if __name__ == "__main__":
    main()