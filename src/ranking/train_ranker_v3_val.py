from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple  # noqa: UP035

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.config.settings import settings

PROCESSED = Path(settings.PROCESSED_DIR)
REPORTS = Path(getattr(settings, "REPORTS_DIR", "reports"))
MODELS_DIR = REPORTS / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _p(name: str) -> Path:
    return PROCESSED / name


def _load_df(name: str) -> pl.DataFrame:
    path = _p(name)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pl.read_parquet(path)


def _parse_sources_list(src_list: List[str]) -> Tuple[int, int, int]:
    """
    src strings look like:
      "two_tower_ann,sequence_gru,v2_prior"
      "two_tower_ann"
    We convert to 3 binary flags.
    """
    s = ",".join(src_list) if isinstance(src_list, list) else str(src_list)
    has_tt = 1 if "two_tower" in s else 0
    has_seq = 1 if "sequence" in s else 0
    has_v2 = 1 if "v2_prior" in s else 0
    return has_tt, has_seq, has_v2


def build_v3_val_training_table(
    out_path: Path,
    max_users: int = 50000,
    max_candidates: int = 200,
):
    print("\n[START] Building V3 VAL training table...")

    # Load V3 blended candidates
    v3 = _load_df("v3_candidates_val.parquet").select(
        ["user_idx", "candidates", "blend_scores", "blend_sources"]
    )
    print(f"[OK] v3 candidate users: {v3.height}")

    # Cap users (should already be 50k)
    if v3.height > max_users:
        v3 = v3.head(max_users)

    # Load val_conf for truth
    val_conf = _load_df("val_conf.parquet").select(["user_idx", "item_idx"])
    print(f"[OK] val_conf rows: {val_conf.height}")

    # Build truth set per user
    truth = (
        val_conf.group_by("user_idx")
        .agg(pl.col("item_idx").unique().alias("truth_items"))
    )
    print(f"[OK] truth users: {truth.height}")

    # Join truth onto candidate base
    base = v3.join(truth, on="user_idx", how="left").with_columns(
        pl.col("truth_items").fill_null(pl.lit([]))
    )

    # Explode candidates into pointwise rows
    print("[START] Exploding candidate lists...")
    exploded = (
        base.with_columns(
            [
                pl.col("candidates").list.head(max_candidates).alias("cand_cut"),
                pl.col("blend_scores").list.head(max_candidates).alias("bs_cut"),
                pl.col("blend_sources").list.head(max_candidates).alias("src_cut"),
            ]
        )
        .select(["user_idx", "cand_cut", "bs_cut", "src_cut", "truth_items"])
        .explode(["cand_cut", "bs_cut", "src_cut"])
        .rename(
            {
                "cand_cut": "item_idx",
                "bs_cut": "blend_score",
                "src_cut": "blend_source_raw",
            }
        )
    )

    print(f"[OK] exploded rows: {exploded.height}")

    # Add label
    print("[START] Creating labels...")
    exploded = exploded.with_columns(
        [
            pl.col("truth_items")
            .list.contains(pl.col("item_idx"))
            .cast(pl.Int8)
            .alias("label")
        ]
    ).drop("truth_items")

    # Add source flags using Python UDF (fast enough at this scale)
    print("[START] Deriving source flags...")
    src_py = exploded.select(["blend_source_raw"]).to_series().to_list()
    flags = [ _parse_sources_list(x) for x in src_py ]
    has_tt = [f[0] for f in flags]
    has_seq = [f[1] for f in flags]
    has_v2 = [f[2] for f in flags]

    exploded = exploded.with_columns(
        [
            pl.Series("has_tt", has_tt, dtype=pl.Int8),
            pl.Series("has_seq", has_seq, dtype=pl.Int8),
            pl.Series("has_v2", has_v2, dtype=pl.Int8),
        ]
    )

    # Load V2 feature store outputs
    user_feat = _load_df("user_features.parquet")
    item_feat = _load_df("item_features.parquet")

    # Join features
    print("[START] Joining user/item features...")
    feat = (
        exploded.join(user_feat, on="user_idx", how="left")
        .join(item_feat, on="item_idx", how="left")
    )

    # Fill null numeric features with 0
    numeric_cols = [c for c, t in zip(feat.columns, feat.dtypes) if t in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64)]
    fill_exprs = [pl.col(c).fill_null(0) for c in numeric_cols]
    feat = feat.with_columns(fill_exprs)

    print(f"[OK] final training rows: {feat.height}")

    feat.write_parquet(out_path)
    print("[DONE] V3 VAL training table saved.")
    print(f"[PATH] {out_path}")


def train_ranker_v3_val(
    pairs_path: Path,
    model_path: Path,
    meta_path: Path,
):
    print("\n[START] Training V3 VAL ranker...")

    df = pl.read_parquet(pairs_path)

    # Define locked feature order
    feature_cols = [
        # new V3-specific
        "blend_score",
        "has_tt",
        "has_seq",
        "has_v2",

        # user features (from feature store)
        "user_interactions",
        "user_conf_sum",
        "user_conf_decay_sum",
        "user_days_since_last",

        # item features (from feature store)
        "item_interactions",
        "item_conf_sum",
        "item_conf_decay_sum",
        "item_days_since_last",
    ]

    # Ensure columns exist
    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required feature column: {c}")

    X = df.select(feature_cols).to_numpy()
    y = df.select("label").to_numpy().ravel()

    # Train/val split (simple deterministic split)
    n = X.shape[0]
    split = int(n * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"[OK] Train rows: {len(y_train)} | Val rows: {len(y_val)}")
    pos_rate = float(y_train.mean()) if len(y_train) else 0.0
    print(f"[OK] Train positive rate: {pos_rate:.4f}")

    clf = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=250,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    prob = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, prob) if len(np.unique(y_val)) > 1 else 0.0

    # Save model
    import joblib
    joblib.dump(clf, model_path)

    meta = {
        "model": "HistGradientBoostingClassifier",
        "features_locked_order": feature_cols,
        "auc_val_pointwise": float(auc),
        "train_rows": int(len(y_train)),
        "val_rows": int(len(y_val)),
        "random_state": 42,
        "notes": "V3 ranker trained on V3 blended VAL candidates with V2 feature store + source flags.",
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print("[DONE] V3 ranker trained.")
    print(f"[OK] AUC={auc:.4f}")
    print(f"[PATH] {model_path}")
    print(f"[PATH] {meta_path}")
    print(f"[OK] Features used (locked order): {feature_cols}")


def main():
    pairs_path = _p("rank_v3_val_pointwise.parquet")
    model_path = MODELS_DIR / "ranker_hgb_v3_val.pkl"
    meta_path = MODELS_DIR / "ranker_hgb_v3_val.meta.json"

    # Build training table
    build_v3_val_training_table(out_path=pairs_path)

    # Train
    train_ranker_v3_val(
        pairs_path=pairs_path,
        model_path=model_path,
        meta_path=meta_path,
    )


if __name__ == "__main__":
    main()