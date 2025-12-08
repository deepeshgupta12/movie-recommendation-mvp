from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.config.settings import settings
from src.ranking.features_v2 import build_feature_table_v2


def main() -> None:
    pairs_path = settings.PROCESSED_DIR / "rank_train_pairs.parquet"
    if not pairs_path.exists():
        raise FileNotFoundError("rank_train_pairs.parquet not found. Run build_training_data first.")

    print("[START] Building V2 feature table...")
    t0 = time.time()
    feat_df = build_feature_table_v2(str(pairs_path))
    t1 = time.time()

    print(f"[OK] Feature rows: {feat_df.height}")
    print(f"[OK] Build time: {t1 - t0:.2f}s")

    # -----------------------------
    # Define canonical feature order
    # -----------------------------
    feature_cols = [
        "user_interactions",
        "user_conf_sum",
        "user_conf_decay_sum",
        "user_days_since_last",
        "item_interactions",
        "item_conf_sum",
        "item_conf_decay_sum",
        "item_days_since_last",
    ]

    # Validate presence
    missing = [c for c in feature_cols if c not in feat_df.columns]
    if missing:
        raise ValueError(f"Missing expected V2 features: {missing}")

    y = feat_df["label"].to_numpy()
    X = feat_df.select(feature_cols).to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[OK] Train size: {len(y_train)} | Val size: {len(y_val)}")
    print("[START] Training V2 ranker...")

    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=250,
        random_state=42,
    )

    t2 = time.time()
    model.fit(X_train, y_train)
    t3 = time.time()

    print(f"[OK] Training time: {t3 - t2:.2f}s")

    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)

    models_dir = settings.PROJECT_ROOT / "reports" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    out_model = models_dir / "ranker_hgb_v2.pkl"
    out_meta = models_dir / "ranker_hgb_v2.features.json"

    joblib.dump(model, out_model)

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump({"feature_cols": feature_cols}, f, indent=2)

    print(f"[DONE] V2 Ranker trained. AUC={auc:.4f}")
    print(f"[PATH] {out_model}")
    print(f"[PATH] {out_meta}")
    print(f"[OK] Features used (locked order): {feature_cols}")


if __name__ == "__main__":
    main()