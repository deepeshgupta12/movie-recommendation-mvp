from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple  # noqa: UP035

import numpy as np
import polars as pl

from src.config.settings import settings
from src.neural.two_tower import TwoTowerModel


def _processed_dir() -> Path:
    return Path(settings.PROCESSED_DIR)


def _reports_models_dir() -> Path:
    base = getattr(settings, "BASE_DIR", None)
    p = Path(base) / "reports" / "models" if base else Path("reports") / "models"
    return p


def _load_val_users(cap: int = 50000) -> List[int]:
    val_p = _processed_dir() / "val_conf.parquet"
    if not val_p.exists():
        raise FileNotFoundError("Missing val_conf.parquet.")

    val_conf = pl.read_parquet(val_p)
    users = (
        val_conf.select("user_idx")
        .unique()
        .sort("user_idx")
        .head(cap)["user_idx"]
        .to_list()
    )
    return [int(u) for u in users]


def _load_item_embeddings():
    p = _processed_dir() / "item_emb_v3.parquet"
    if not p.exists():
        raise FileNotFoundError("Missing item_emb_v3.parquet. Run export_embeddings_v3 first.")
    df = pl.read_parquet(p)
    item_ids = df["item_idx"].to_numpy().astype(np.int64, copy=False)
    mat = np.asarray(df["embedding"].to_list(), dtype=np.float32)
    return item_ids, mat


def _build_item_lookup(item_ids: np.ndarray, item_mat: np.ndarray) -> Dict[int, np.ndarray]:
    return {int(i): item_mat[idx] for idx, i in enumerate(item_ids)}


def main(
    ann_k: int = 200,
    out_k: int = 200,
    user_cap: int = 50000,
):
    import hnswlib

    print("\n[START] Exporting ANN candidates for val (V3)...")

    users = _load_val_users(cap=user_cap)
    print(f"[OK] val users after cap: {len(users)}")

    model = TwoTowerModel.load("reports/models/two_tower_v3.pt")
    print("[OK] Two-tower loaded.")

    item_ids, item_mat = _load_item_embeddings()
    item_lookup = _build_item_lookup(item_ids, item_mat)
    print(f"[OK] item pool: {len(item_ids)} | dim={item_mat.shape[1]}")

    meta_path = _reports_models_dir() / "ann_hnsw_items_v3.meta.json"
    index_path = _reports_models_dir() / "ann_hnsw_items_v3.bin"

    dim = item_mat.shape[1]
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        dim = int(meta.get("dim", dim))

    index = hnswlib.Index(space="ip", dim=dim)
    index.load_index(str(index_path))
    index.set_ef(max(100, ann_k))

    # Batch user embedding computation for speed on MPS
    print("[START] Embedding users...")
    u_emb = model.embed_users(users).astype(np.float32)

    rows_user: List[int] = []
    rows_items: List[List[int]] = []
    rows_scores: List[List[float]] = []

    print(f"[START] ANN retrieval + dot-product rescoring (ann_k={ann_k}, out_k={out_k})...")

    # HNSW query in loop (hnswlib API is single-query oriented)
    for idx_u, u in enumerate(users):
        uvec = u_emb[idx_u]

        labels, _ = index.knn_query(uvec, k=ann_k)
        cand_ids = [int(x) for x in labels[0]]

        rescored = []
        for iid in cand_ids:
            ivec = item_lookup.get(iid)
            if ivec is None:
                continue
            rescored.append((iid, float(np.dot(uvec, ivec))))

        rescored.sort(key=lambda x: x[1], reverse=True)
        top = rescored[:out_k]

        rows_user.append(int(u))
        rows_items.append([int(i) for i, _ in top])
        rows_scores.append([float(s) for _, s in top])

    out = pl.DataFrame(
        {
            "user_idx": rows_user,
            "candidates": rows_items,
            "tt_scores": rows_scores,
        }
    )

    out_path = _processed_dir() / "ann_candidates_v3_val.parquet"
    out.write_parquet(out_path)

    print("[DONE] ANN candidates exported.")
    print(f"[PATH] {out_path}")


if __name__ == "__main__":
    main()