# src/ranking/train_ranker_v4_val.py

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

PROCESSED = Path("data/processed")
REPORTS_DIR = Path("reports")
MODELS_DIR = REPORTS_DIR / "models"

POINTWISE_PATH = PROCESSED / "rank_v4_val_pointwise.parquet"

MODEL_PATH = MODELS_DIR / "ranker_hgb_v4_val.pkl"
META_PATH = MODELS_DIR / "ranker_hgb_v4_val.meta.json"


# Locked order: extend V3 with session signals
FEATURES = [
    # blend-level
    "blend_score",
    "has_tt",
    "has_seq",
    "has_v2",

    # session-level (new)
    "short_term_boost",
    "sess_hot",
    "sess_warm",
    "sess_cold",

    # user-level
    "user_interactions",
    "user_conf_sum",
    "user_conf_decay_sum",
    "user_days_since_last",

    # item-level
    "item_interactions",
    "item_conf_sum",
    "item_conf_decay_sum",
    "item_days_since_last",
]


@dataclass
class V4RankerMeta:
    model_name: str
    auc: float
    features: list[str]
    rows: int
    pos_rate: float
    train_rows: int
    val_rows: int
    split_seed: int


def _ensure_pointwise():
    if POINTWISE_PATH.exists():
        print("[SKIP] V4 VAL pointwise already exists:")
        print(f"  [PATH] {POINTWISE_PATH}")
        return

    print("[INFO] V4 VAL pointwise not found. Building now...")
    from src.ranking.build_v4_val_pointwise import build_v4_val_pointwise
    build_v4_val_pointwise(out_path=POINTWISE_PATH)


def _load_training_frame() -> pl.DataFrame:
    df = pl.read_parquet(str(POINTWISE_PATH))

    # Fill any null session flags/values safely
    for c in ["short_term_boost", "sess_hot", "sess_warm", "sess_cold"]:
        if c in df.columns:
            if c == "short_term_boost":
                df = df.with_columns(pl.col(c).fill_null(0.0))
            else:
                df = df.with_columns(pl.col(c).fill_null(0))

    # Some features might be missing depending on earlier pipelines
    # Create safe defaults if absent
    defaults_float = {"blend_score": 0.0, "short_term_boost": 0.0}
    defaults_int = {
        "has_tt": 0, "has_seq": 0, "has_v2": 0,
        "sess_hot": 0, "sess_warm": 0, "sess_cold": 0,
        "user_interactions": 0, "user_conf_sum": 0, "user_conf_decay_sum": 0, "user_days_since_last": 0,
        "item_interactions": 0, "item_conf_sum": 0, "item_conf_decay_sum": 0, "item_days_since_last": 0,
    }

    for k, v in defaults_float.items():
        if k not in df.columns:
            df = df.with_columns(pl.lit(v).alias(k))

    for k, v in defaults_int.items():
        if k not in df.columns:
            df = df.with_columns(pl.lit(v).alias(k))

    # Ensure correct dtypes
    df = df.with_columns(
        pl.col("label").cast(pl.Int8),
        pl.col("blend_score").cast(pl.Float64),
        pl.col("short_term_boost").cast(pl.Float64),
    )

    for c in FEATURES:
        if c not in ["blend_score", "short_term_boost"]:
            df = df.with_columns(pl.col(c).cast(pl.Int8) if c.startswith(("has_", "sess_")) else pl.col(c))

    return df.select(FEATURES + ["label"])


def train_ranker_v4_val(seed: int = 42) -> V4RankerMeta:
    print("\n[START] Building V4 VAL training table...")
    _ensure_pointwise()

    df = _load_training_frame()
    rows = df.height

    pos_rate = float(df.select(pl.col("label").mean()).item())

    print("[OK] final training rows:", rows)
    print("[OK] Train positive rate:", round(pos_rate, 6))
    print("[PATH]", POINTWISE_PATH)

    # Convert to numpy for sklearn
    X = df.select(FEATURES).to_numpy()
    y = df.select("label").to_numpy().reshape(-1)

    # Stratified split to stabilize AUC under extreme imbalance
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=seed,
        stratify=y
    )

    print("\n[START] Training V4 VAL ranker...")
    clf = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_depth=6,
        max_iter=250,
        min_samples_leaf=50,
        random_state=seed,
    )

    clf.fit(X_train, y_train)

    # Evaluate
    val_pred = clf.predict_proba(X_val)[:, 1]
    auc = float(roc_auc_score(y_val, val_pred))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)

    meta = V4RankerMeta(
        model_name="ranker_hgb_v4_val",
        auc=auc,
        features=FEATURES,
        rows=rows,
        pos_rate=pos_rate,
        train_rows=int(len(y_train)),
        val_rows=int(len(y_val)),
        split_seed=seed,
    )

    with open(META_PATH, "w") as f:
        json.dump(asdict(meta), f, indent=2)

    print("[DONE] V4 VAL ranker trained.")
    print(f"[OK] AUC={auc:.4f}")
    print(f"[PATH] {MODEL_PATH}")
    print(f"[PATH] {META_PATH}")
    print(f"[OK] Features used (locked order): {FEATURES}")

    return meta


def main():
    train_ranker_v4_val()


if __name__ == "__main__":
    main()