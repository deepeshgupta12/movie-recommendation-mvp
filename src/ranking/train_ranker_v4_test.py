"""
Train V4 TEST ranker.

Design
- If pointwise exists, skip rebuild.
- Else call V4 TEST builder.
- Train HistGradientBoostingClassifier for pointwise ranking proxy.
- Lock feature order in meta to keep service inference stable.

Output
- reports/models/ranker_hgb_v4_test.pkl
- reports/models/ranker_hgb_v4_test.meta.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple  # noqa: UP035

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.ranking.build_v4_test_pointwise import build_v4_test_pointwise

PROCESSED = Path("data/processed")
REPORTS_MODELS = Path("reports/models")


FEATURES_V4: List[str] = [
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


@dataclass
class TrainConfig:
    cap_users: int = 50000
    val_frac: float = 0.2
    random_state: int = 42
    max_iter: int = 200
    learning_rate: float = 0.05
    max_depth: int = 6


def _ensure_pointwise(cfg: TrainConfig) -> Path:
    p = PROCESSED / "rank_v4_test_pointwise.parquet"
    if p.exists():
        print("[SKIP] V4 TEST pointwise already exists:")
        print(f"  [PATH] {p}")
        return p

    print("[START] Building V4 TEST pointwise table...")
    out = build_v4_test_pointwise(cap_users=cfg.cap_users)
    return out


def _load_pointwise(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path)


def _train_val_split(df: pl.DataFrame, cfg: TrainConfig) -> Tuple[pl.DataFrame, pl.DataFrame]:
    # Deterministic split by hashing user_idx
    # Keeps memory stable and avoids full shuffle.
    h = (pl.col("user_idx") * 2654435761) % 100
    df = df.with_columns(h.alias("_h"))

    val_cut = int(cfg.val_frac * 100)

    val_df = df.filter(pl.col("_h") < val_cut).drop("_h")
    train_df = df.filter(pl.col("_h") >= val_cut).drop("_h")

    return train_df, val_df


def _to_xy(df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = df.select(FEATURES_V4).to_numpy()
    y = df.select("label").to_numpy().reshape(-1)
    return X, y


def train_ranker_v4_test(cfg: TrainConfig) -> None:
    REPORTS_MODELS.mkdir(parents=True, exist_ok=True)

    pointwise_path = _ensure_pointwise(cfg)
    df = _load_pointwise(pointwise_path)

    print(f"[OK] final training rows: {df.height}")
    pos_rate = df.select(pl.col("label").mean()).item()
    print(f"[OK] Train positive rate: {pos_rate:.6f}")
    print(f"[PATH] {pointwise_path}")

    train_df, val_df = _train_val_split(df, cfg)

    print("\n[START] Training V4 TEST ranker...")
    X_train, y_train = _to_xy(train_df)
    X_val, y_val = _to_xy(val_df)

    model = HistGradientBoostingClassifier(
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        max_iter=cfg.max_iter,
        random_state=cfg.random_state,
        verbose=0,
    )

    model.fit(X_train, y_train)

    # AUC
    val_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_proba)

    model_path = REPORTS_MODELS / "ranker_hgb_v4_test.pkl"
    meta_path = REPORTS_MODELS / "ranker_hgb_v4_test.meta.json"

    joblib.dump(model, model_path)
    meta = {
        "version": "v4_test",
        "auc": float(auc),
        "features": FEATURES_V4,
        "pointwise_path": str(pointwise_path),
        "val_frac": cfg.val_frac,
        "cap_users": cfg.cap_users,
        "model_class": "HistGradientBoostingClassifier",
        "params": {
            "max_depth": cfg.max_depth,
            "learning_rate": cfg.learning_rate,
            "max_iter": cfg.max_iter,
            "random_state": cfg.random_state,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print("[DONE] V4 TEST ranker trained.")
    print(f"[OK] AUC={auc:.4f}")
    print(f"[PATH] {model_path}")
    print(f"[PATH] {meta_path}")
    print(f"[OK] Features used (locked order): {FEATURES_V4}")


def main():
    cfg = TrainConfig()
    print("[START] Building V4 TEST training table...")
    train_ranker_v4_test(cfg)


if __name__ == "__main__":
    main()