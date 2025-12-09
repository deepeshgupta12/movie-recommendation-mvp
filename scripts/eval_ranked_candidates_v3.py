from __future__ import annotations

from pathlib import Path
from typing import Dict, List  # noqa: UP035

import numpy as np
import polars as pl

from src.config.settings import settings

PROCESSED = Path(settings.PROCESSED_DIR)
REPORTS = Path(getattr(settings, "REPORTS_DIR", "reports"))
MODELS_DIR = REPORTS / "models"


def _p(name: str) -> Path:
    return PROCESSED / name


def _load(name: str) -> pl.DataFrame:
    path = _p(name)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pl.read_parquet(path)


def _load_model():
    import joblib
    model_path = MODELS_DIR / "ranker_hgb_v3_val.pkl"
    meta_path = MODELS_DIR / "ranker_hgb_v3_val.meta.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta: {meta_path}")

    meta = __import__("json").loads(meta_path.read_text())
    clf = joblib.load(model_path)
    feature_cols = meta["features_locked_order"]
    return clf, feature_cols, meta


def _build_truth_map(val_conf: pl.DataFrame) -> Dict[int, set]:
    grouped = val_conf.group_by("user_idx").agg(pl.col("item_idx").unique().alias("items"))
    out: Dict[int, set] = {}
    for row in grouped.iter_rows(named=True):
        out[int(row["user_idx"])] = set(int(x) for x in (row["items"] or []))
    return out


def _ndcg_at_k(recs: List[int], truth: set, k: int) -> float:
    if not truth:
        return 0.0
    dcg = 0.0
    for idx, item in enumerate(recs[:k]):
        if item in truth:
            dcg += 1.0 / np.log2(idx + 2)
    # ideal dcg
    ideal_hits = min(len(truth), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return float(dcg / idcg) if idcg > 0 else 0.0


def _recall_at_k(recs: List[int], truth: set, k: int) -> float:
    if not truth:
        return 0.0
    hits = len(set(recs[:k]) & truth)
    return float(hits / len(truth))


def main():
    print("\n[START] Evaluating Ranked V3 candidates on VAL...")

    # Load data
    v3 = _load("v3_candidates_val.parquet")
    user_feat = _load("user_features.parquet")
    item_feat = _load("item_features.parquet")
    val_conf = _load("val_conf.parquet").select(["user_idx", "item_idx"])

    clf, feature_cols, meta = _load_model()

    truth_map = _build_truth_map(val_conf)

    print(f"[OK] V3 candidate users: {v3.height}")
    print(f"[OK] Truth users: {len(truth_map)}")
    print(f"[OK] Using model AUC meta: {meta.get('auc_val_pointwise', None)}")

    # Explode candidates
    base = v3.select(["user_idx", "candidates", "blend_scores", "blend_sources"])

    exploded = (
        base
        .with_columns(
            [
                pl.col("candidates").list.head(200).alias("cand_cut"),
                pl.col("blend_scores").list.head(200).alias("bs_cut"),
                pl.col("blend_sources").list.head(200).alias("src_cut"),
            ]
        )
        .select(["user_idx", "cand_cut", "bs_cut", "src_cut"])
        .explode(["cand_cut", "bs_cut", "src_cut"])
        .rename(
            {
                "cand_cut": "item_idx",
                "bs_cut": "blend_score",
                "src_cut": "blend_source_raw",
            }
        )
    )

    # Source flags (cheap python parse)
    src_py = exploded.select(["blend_source_raw"]).to_series().to_list()

    def parse_src(x):
        s = ",".join(x) if isinstance(x, list) else str(x)
        return (
            1 if "two_tower" in s else 0,
            1 if "sequence" in s else 0,
            1 if "v2_prior" in s else 0,
        )

    flags = [parse_src(x) for x in src_py]
    exploded = exploded.with_columns(
        [
            pl.Series("has_tt", [f[0] for f in flags], dtype=pl.Int8),
            pl.Series("has_seq", [f[1] for f in flags], dtype=pl.Int8),
            pl.Series("has_v2", [f[2] for f in flags], dtype=pl.Int8),
        ]
    )

    # Join features
    feat = (
        exploded
        .join(user_feat, on="user_idx", how="left")
        .join(item_feat, on="item_idx", how="left")
    )

    # Fill missing numeric
    for c in feat.columns:
        if c in ("blend_source_raw",):
            continue
        if feat[c].dtype in (
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64
        ):
            feat = feat.with_columns(pl.col(c).fill_null(0))

    # Score
    X = feat.select(feature_cols).to_numpy()
    scores = clf.predict_proba(X)[:, 1]
    feat = feat.with_columns(pl.Series("rank_score", scores))

    # Re-aggregate to ranked lists
    ranked = (
        feat.sort(["user_idx", "rank_score"], descending=[False, True])
        .group_by("user_idx")
        .agg(pl.col("item_idx").list().alias("ranked_items"))
    )

    # Compute metrics
    ks = [10, 20, 50, 100, 200]
    recalls = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}

    for row in ranked.iter_rows(named=True):
        u = int(row["user_idx"])
        recs = [int(x) for x in (row["ranked_items"] or [])]
        truth = truth_map.get(u, set())
        if not truth:
            continue

        for k in ks:
            recalls[k].append(_recall_at_k(recs, truth, k))
            ndcgs[k].append(_ndcg_at_k(recs, truth, k))

    users_evaluated = len(recalls[10]) if recalls[10] else 0

    print("\n[Ranked V3 Hybrid Results - VAL]")
    for k in ks:
        r = float(np.mean(recalls[k])) if recalls[k] else 0.0
        n = float(np.mean(ndcgs[k])) if ndcgs[k] else 0.0
        print(
            {
                f"recall@{k}": r,
                f"ndcg@{k}": n,
                "users_evaluated": float(users_evaluated),
            }
        )

    print("\n[DONE] Ranked V3 val eval complete.")


if __name__ == "__main__":
    main()