from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple  # noqa: UP035

import numpy as np
import polars as pl

from src.config.settings import settings


def _processed_dir() -> Path:
    return Path(settings.PROCESSED_DIR)


def _recall_at_k(truth: List[int], preds: List[int], k: int) -> float:
    if not truth:
        return 0.0
    topk = set(preds[:k])
    hits = sum(1 for t in truth if t in topk)
    return hits / len(truth)


def _ndcg_at_k(truth: List[int], preds: List[int], k: int) -> float:
    if not truth:
        return 0.0
    truth_set = set(truth)
    dcg = 0.0
    for i, p in enumerate(preds[:k], start=1):
        if p in truth_set:
            dcg += 1.0 / np.log2(i + 1)

    # ideal DCG
    ideal_hits = min(len(truth), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    print("\n[START] Evaluating V3 sequence candidates on val...")

    seq_val_path = _processed_dir() / "user_seq_val.parquet"
    cand_path = _processed_dir() / "seq_candidates_v3_val.parquet"

    if not seq_val_path.exists():
        raise FileNotFoundError("Missing user_seq_val.parquet")
    if not cand_path.exists():
        raise FileNotFoundError("Missing seq_candidates_v3_val.parquet. Run generator first.")

    truth_df = pl.read_parquet(seq_val_path).select(["user_idx", "target"])
    cand_df = pl.read_parquet(cand_path).select(["user_idx", "candidates"])

    df = truth_df.join(cand_df, on="user_idx", how="inner")
    print(f"[OK] users_evaluated: {df.height}")

    ks = [10, 20, 50, 100, 200]

    metrics = {f"recall@{k}": [] for k in ks}
    metrics.update({f"ndcg@{k}": [] for k in ks})

    for row in df.iter_rows(named=True):
        target = int(row["target"])
        preds = row["candidates"] or []
        truth = [target]

        for k in ks:
            metrics[f"recall@{k}"].append(_recall_at_k(truth, preds, k))
            metrics[f"ndcg@{k}"].append(_ndcg_at_k(truth, preds, k))

    out = {}
    for k in ks:
        out[f"recall@{k}"] = float(np.mean(metrics[f"recall@{k}"]))
        out[f"ndcg@{k}"] = float(np.mean(metrics[f"ndcg@{k}"]))

    out["users_evaluated"] = float(df.height)

    print("\n[V3 Sequence Candidate Results]")
    for k in ks:
        print(
            {
                f"recall@{k}": out[f"recall@{k}"],
                f"ndcg@{k}": out[f"ndcg@{k}"],
                "users_evaluated": out["users_evaluated"],
            }
        )

    print("\n[DONE] Sequence candidate eval complete.")


if __name__ == "__main__":
    main()