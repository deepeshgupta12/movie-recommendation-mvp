"""
Evaluate Ranked V4 candidates on VAL.

Uses:
- data/processed/rank_v4_val_pointwise.parquet
- reports/models/ranker_hgb_v4_val.pkl
- reports/models/ranker_hgb_v4_val.meta.json

Computes:
- recall@{10,20,50,100,200}
- ndcg@{10,20,50,100,200}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List  # noqa: UP035

import joblib
import numpy as np
import polars as pl

PROCESSED = Path("data/processed")
MODELS = Path("reports/models")


def _load_meta() -> dict:
    p = MODELS / "ranker_hgb_v4_val.meta.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())


def _load_model():
    p = MODELS / "ranker_hgb_v4_val.pkl"
    if not p.exists():
        raise FileNotFoundError(p)
    return joblib.load(p), p


def _load_pointwise() -> pl.DataFrame:
    p = PROCESSED / "rank_v4_val_pointwise.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    return pl.read_parquet(p)


def _ndcg_at_k(rank: List[int], truth: int, k: int) -> float:
    if truth is None:
        return 0.0
    top = rank[:k]
    try:
        idx = top.index(truth)
    except ValueError:
        return 0.0
    # DCG with binary relevance
    return 1.0 / np.log2(idx + 2)


def _recall_at_k(rank: List[int], truth: int, k: int) -> float:
    if truth is None:
        return 0.0
    return 1.0 if truth in rank[:k] else 0.0


def main():
    print("[START] Evaluating Ranked V4 candidates on VAL...")

    meta = _load_meta()
    features = meta["features"]
    model, model_path = _load_model()

    df = _load_pointwise()

    print(f"[OK] Using model AUC meta: {meta.get('auc')}")
    print(f"[OK] Model path: {model_path}")

    # Build truth per user
    truth = (
        df.filter(pl.col("label") == 1)
        .group_by("user_idx")
        .agg(pl.col("item_idx").first().alias("truth_item"))
    )

    # Score all rows
    X = df.select(features).to_numpy()
    proba = model.predict_proba(X)[:, 1]

    scored = df.select(["user_idx", "item_idx"]).with_columns(
        pl.Series("score", proba)
    )

    # Rank per user
    ranked = (
        scored.join(truth, on="user_idx", how="left")
        .sort(["user_idx", "score"], descending=[False, True])
        .group_by("user_idx")
        .agg(
            pl.col("item_idx").head(200).implode().alias("ranked_items"),
            pl.col("truth_item").first().alias("truth_item"),
        )
    )

    # Collect python lists for metrics
    rows = ranked.select(["ranked_items", "truth_item"]).to_dicts()

    ks = [10, 20, 50, 100, 200]
    metrics: Dict[str, float] = {}
    n = len(rows)

    for k in ks:
        r = 0.0
        nd = 0.0
        for row in rows:
            rank_list = row["ranked_items"] or []
            truth_item = row["truth_item"]
            r += _recall_at_k(rank_list, truth_item, k)
            nd += _ndcg_at_k(rank_list, truth_item, k)

        metrics[f"recall@{k}"] = r / n if n else 0.0
        metrics[f"ndcg@{k}"] = nd / n if n else 0.0

    print("\n[Ranked V4 Hybrid Results - VAL]")
    for k in ks:
        print(
            {
                f"recall@{k}": metrics[f"recall@{k}"],
                f"ndcg@{k}": metrics[f"ndcg@{k}"],
                "users_evaluated": float(n),
            }
        )

    print("\n[DONE] Ranked V4 val eval complete.")


if __name__ == "__main__":
    main()