from __future__ import annotations

import time
from pathlib import Path

import joblib
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config.settings import settings
from src.ranking.features import build_feature_table


def main() -> None:
    pairs_path = settings.PROCESSED_DIR / "rank_train_pairs.parquet"
    if not pairs_path.exists():
        raise FileNotFoundError("rank_train_pairs.parquet not found. Run build_training_data first.")

    print("[START] Building feature table for ranker...")
    t0 = time.time()
    feat_df = build_feature_table(str(pairs_path))
    t1 = time.time()

    print(f"[OK] Feature table rows: {feat_df.height}")
    print(f"[OK] Feature build time: {t1 - t0:.2f}s")

    feature_cols = [
        "pop_bucket",
        "user_genre_aff",
        "item_pop_conf",
        "item_genre_count",
    ]

    print("[START] Preparing train/val split...")
    y = feat_df["label"].to_numpy()
    X = feat_df.select(feature_cols).to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"[OK] Train size: {len(y_train)} | Val size: {len(y_val)}")

    print("[START] Training HistGradientBoosting ranker...")

    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.06,
        max_iter=200,
        random_state=42,
    )

    # scikit doesn't expose per-iteration progress cleanly here,
    # so we provide step-level visibility.
    t2 = time.time()
    model.fit(X_train, y_train)
    t3 = time.time()

    print(f"[OK] Training time: {t3 - t2:.2f}s")

    print("[START] Validating...")
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)

    models_dir = settings.PROJECT_ROOT / "reports" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    out_path = models_dir / "ranker_hgb.pkl"
    joblib.dump(model, out_path)

    print(f"[DONE] Ranker trained. AUC={auc:.4f}")
    print(f"[PATH] {out_path}")


if __name__ == "__main__":
    main()