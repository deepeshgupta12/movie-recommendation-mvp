# src/ranking/train_ranker_v3_test.py

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple  # noqa: UP035

import numpy as np
import polars as pl
from joblib import dump
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.config.settings import settings

# -----------------------------
# Robust path resolution
# -----------------------------
# File is: <repo>/src/ranking/train_ranker_v3_test.py
# parents[0]=ranking, [1]=src, [2]=repo root
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = Path(getattr(settings, "DATA_DIR", BASE_DIR / "data"))
PROCESSED_DIR = DATA_DIR / "processed"

REPORTS_DIR = Path(getattr(settings, "REPORTS_DIR", BASE_DIR / "reports"))
MODELS_DIR = REPORTS_DIR / "models"


# -----------------------------
# Paths
# -----------------------------
POINTWISE_PATH = PROCESSED_DIR / "rank_v3_test_pointwise.parquet"
CANDIDATES_PATH = PROCESSED_DIR / "v3_candidates_test.parquet"
TEST_CONF_PATH = PROCESSED_DIR / "test_conf.parquet"

MODEL_PATH = MODELS_DIR / "ranker_hgb_v3_test.pkl"
META_PATH = MODELS_DIR / "ranker_hgb_v3_test.meta.json"


# -----------------------------
# Locked feature order
# -----------------------------
FEATURES_LOCKED: List[str] = [
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


def _ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _build_pointwise_if_missing() -> None:
    if POINTWISE_PATH.exists():
        print(f"[SKIP] Pointwise table already exists:\n  [PATH] {POINTWISE_PATH}")
        return

    print("[INFO] Pointwise table missing. Building via src.ranking.build_v3_test_pointwise...")
    from src.ranking.build_v3_test_pointwise import build_v3_test_pointwise

    build_v3_test_pointwise(
        candidates_path=str(CANDIDATES_PATH),
        truth_path=str(TEST_CONF_PATH),
        out_path=str(POINTWISE_PATH),
    )


def _load_pointwise() -> pl.DataFrame:
    if not POINTWISE_PATH.exists():
        raise FileNotFoundError(f"Pointwise file not found: {POINTWISE_PATH}")

    df = pl.read_parquet(str(POINTWISE_PATH))

    missing = [c for c in FEATURES_LOCKED + ["label"] if c not in df.columns]
    if missing:
        raise RuntimeError(
            "Pointwise table missing required columns: "
            + ", ".join(missing)
            + "\nRebuild the pointwise file."
        )

    return df


def _prepare_xy(df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = df.select(FEATURES_LOCKED).to_numpy()
    y = df.select("label").to_numpy().reshape(-1)
    return X, y


def _train_model(X: np.ndarray, y: np.ndarray) -> Tuple[HistGradientBoostingClassifier, float]:
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    pos_rate = float(np.mean(y_train)) if len(y_train) else 0.0
    print(f"[OK] Train rows: {X_train.shape[0]} | Val rows: {X_val.shape[0]}")
    print(f"[OK] Train positive rate: {pos_rate:.4f}")

    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=200,
        random_state=42,
        verbose=0,
    )

    model.fit(X_train, y_train)

    if len(np.unique(y_val)) > 1:
        y_pred = model.predict_proba(X_val)[:, 1]
        auc = float(roc_auc_score(y_val, y_pred))
    else:
        auc = 0.0

    return model, auc


def main() -> None:
    _ensure_dirs()

    print("\n[START] Building V3 TEST training table...")
    _build_pointwise_if_missing()

    df = _load_pointwise()

    print(f"[OK] final training rows: {df.height}")
    print("[DONE] V3 TEST training table ready.")
    print(f"[PATH] {POINTWISE_PATH}")

    print("\n[START] Training V3 TEST ranker...")

    X, y = _prepare_xy(df)
    model, auc = _train_model(X, y)

    dump(model, str(MODEL_PATH))

    meta = {
        "model": "HistGradientBoostingClassifier",
        "auc": auc,
        "features_locked": FEATURES_LOCKED,
        "pointwise_path": str(POINTWISE_PATH),
        "candidates_path": str(CANDIDATES_PATH),
        "truth_path": str(TEST_CONF_PATH),
        "rows": int(df.height),
        "positive_rate": float(np.mean(y)) if len(y) else 0.0,
        "random_state": 42,
    }

    META_PATH.write_text(json.dumps(meta, indent=2))

    print("[DONE] V3 TEST ranker trained.")
    print(f"[OK] AUC={auc:.4f}")
    print(f"[PATH] {MODEL_PATH}")
    print(f"[PATH] {META_PATH}")
    print(f"[OK] Features used (locked order): {FEATURES_LOCKED}")


if __name__ == "__main__":
    main()