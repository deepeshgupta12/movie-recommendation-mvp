# scripts/eval_ranked_candidates_v3_test.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple  # noqa: UP035

import numpy as np
import polars as pl

DEFAULT_FEATURE_ORDER_V3 = [
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


def _parse_source_flags_single(src: Optional[str]) -> Tuple[int, int, int]:
    if not src:
        return 0, 0, 0
    s = str(src).lower()
    has_tt = 1 if ("two_tower" in s or "tt" in s or "ann" in s) else 0
    has_seq = 1 if ("sequence" in s or "gru" in s or "seq" in s) else 0
    has_v2 = 1 if ("v2" in s or "prior" in s or "pop" in s or "genre" in s) else 0
    return has_tt, has_seq, has_v2


def _load_ranker_and_meta() -> Tuple[object, Dict]:
    # Prefer test model
    test_model = Path("reports/models/ranker_hgb_v3_test.pkl")
    test_meta = Path("reports/models/ranker_hgb_v3_test.meta.json")

    val_model = Path("reports/models/ranker_hgb_v3_val.pkl")
    val_meta = Path("reports/models/ranker_hgb_v3_val.meta.json")

    model_path = test_model if test_model.exists() else val_model
    meta_path = test_meta if test_meta.exists() else val_meta

    if not model_path.exists():
        raise FileNotFoundError(
            "No V3 ranker model found. Expected one of:\n"
            " - reports/models/ranker_hgb_v3_test.pkl\n"
            " - reports/models/ranker_hgb_v3_val.pkl"
        )

    import joblib
    model = joblib.load(model_path)

    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    meta.setdefault("features", DEFAULT_FEATURE_ORDER_V3)
    meta.setdefault("auc", None)
    meta["model_path"] = str(model_path)
    meta["meta_path"] = str(meta_path)

    return model, meta


def _explode_candidates(cand: pl.DataFrame, max_k: int = 200) -> pl.DataFrame:
    cols = cand.columns
    if "candidates" not in cols:
        raise ValueError("v3_candidates_test.parquet must contain 'candidates'.")

    if "blend_scores" not in cols:
        cand = cand.with_columns(pl.lit(None).alias("blend_scores"))
    if "blend_sources" not in cols:
        cand = cand.with_columns(pl.lit(None).alias("blend_sources"))

    cand = cand.with_columns(
        pl.col("candidates").list.head(max_k).alias("candidates"),
        pl.col("blend_scores").list.head(max_k).alias("blend_scores"),
        pl.col("blend_sources").list.head(max_k).alias("blend_sources"),
    )

    exploded = cand.explode(["candidates", "blend_scores", "blend_sources"]).rename(
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


def _build_truth(test_conf: pl.DataFrame) -> Dict[int, set]:
    truth = {}
    for row in test_conf.select(["user_idx", "item_idx"]).iter_rows():
        u, i = row
        truth.setdefault(int(u), set()).add(int(i))
    return truth


def _recall_at_k(pred: List[int], truth: set, k: int) -> float:
    if not truth:
        return 0.0
    hits = 0
    for i in pred[:k]:
        if i in truth:
            hits += 1
    return hits / float(len(truth))


def _dcg(rels: List[int]) -> float:
    s = 0.0
    for idx, r in enumerate(rels):
        # log2(idx+2)
        denom = np.log2(idx + 2)
        s += (2 ** r - 1) / denom
    return float(s)


def _ndcg_at_k(pred: List[int], truth: set, k: int) -> float:
    if not truth:
        return 0.0
    rels = [1 if i in truth else 0 for i in pred[:k]]
    dcg = _dcg(rels)
    ideal = _dcg(sorted(rels, reverse=True))
    return 0.0 if ideal == 0 else dcg / ideal


def main():
    print("\n[START] Evaluating Ranked V3 candidates on TEST...")

    candidates_path = "data/processed/v3_candidates_test.parquet"
    test_conf_path = "data/processed/test_conf.parquet"
    user_features_path = "data/processed/user_features.parquet"
    item_features_path = "data/processed/item_features.parquet"

    if not Path(candidates_path).exists():
        raise FileNotFoundError(f"Missing {candidates_path}")

    cand = pl.read_parquet(candidates_path)
    test_conf = pl.read_parquet(test_conf_path)

    print(f"[OK] V3 candidate users: {cand.select('user_idx').n_unique()}")
    print(f"[OK] Truth users: {test_conf.select('user_idx').n_unique()}")

    # Cap to local 50k if your pipeline is consistent
    cand = cand.sort("user_idx").head(50000)

    model, meta = _load_ranker_and_meta()
    feature_order = meta.get("features", DEFAULT_FEATURE_ORDER_V3)

    if meta.get("auc") is not None:
        print(f"[OK] Using model AUC meta: {meta.get('auc')}")
    print(f"[OK] Model path: {meta.get('model_path')}")

    # Explode
    exploded = _explode_candidates(cand, max_k=200)

    # Source flags
    src_py = exploded.select("blend_source_raw").to_series().to_list()
    flags = [_parse_source_flags_single(s) for s in src_py]
    exploded = exploded.with_columns(
        pl.Series("has_tt", [f[0] for f in flags]).cast(pl.Int8),
        pl.Series("has_seq", [f[1] for f in flags]).cast(pl.Int8),
        pl.Series("has_v2", [f[2] for f in flags]).cast(pl.Int8),
    )

    # Join features
    user_feat = pl.read_parquet(user_features_path)
    item_feat = pl.read_parquet(item_features_path)

    df = (
        exploded
        .join(user_feat, on="user_idx", how="left")
        .join(item_feat, on="item_idx", how="left")
    )

    # Ensure feature cols exist
    for col in feature_order:
        if col not in df.columns:
            df = df.with_columns(pl.lit(0).alias(col))

    # Fill null numeric
    for c in feature_order:
        df = df.with_columns(pl.col(c).fill_null(0))

    # Score
    X = df.select(feature_order).to_numpy()

    # Robust proba extraction
    # Some sklearn versions may not expose predict_proba in identical ways
    try:
        proba = model.predict_proba(X)[:, 1]
    except Exception:
        # fallback to decision_function if available
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            # map to 0..1 with sigmoid
            proba = 1 / (1 + np.exp(-scores))
        else:
            proba = model.predict(X).astype(float)

    df = df.with_columns(pl.Series("score", proba))

    # Aggregate ranked lists per user
    # Sort within each user by score desc
    df = df.sort(["user_idx", "score"], descending=[False, True])

    grouped = df.group_by("user_idx").agg(
        pl.col("item_idx").alias("ranked_items"),
        pl.col("score").alias("ranked_scores"),
    )

    # Build truth dict
    truth_map = _build_truth(test_conf)

    # Evaluate only users that have ranked items and truth
    users = grouped.select("user_idx").to_series().to_list()

    ks = [10, 20, 50, 100, 200]
    recall_sums = {k: 0.0 for k in ks}
    ndcg_sums = {k: 0.0 for k in ks}
    evaluated = 0

    # Iterate rows
    for u, items in grouped.select(["user_idx", "ranked_items"]).iter_rows():
        u = int(u)
        pred_list = list(map(int, items)) if items is not None else []
        truth = truth_map.get(u, set())
        if not truth or not pred_list:
            continue

        evaluated += 1
        for k in ks:
            recall_sums[k] += _recall_at_k(pred_list, truth, k)
            ndcg_sums[k] += _ndcg_at_k(pred_list, truth, k)

    # Normalize
    results = []
    for k in ks:
        if evaluated == 0:
            r = 0.0
            n = 0.0
        else:
            r = recall_sums[k] / evaluated
            n = ndcg_sums[k] / evaluated
        results.append((k, r, n))

    print("\n[Ranked V3 Hybrid Results - TEST]")
    for k, r, n in results:
        print(
            {
                f"recall@{k}": r,
                f"ndcg@{k}": n,
                "users_evaluated": float(evaluated),
            }
        )

    print("\n[DONE] Ranked V3 test eval complete.")


if __name__ == "__main__":
    main()