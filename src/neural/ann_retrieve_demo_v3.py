from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple  # noqa: UP035

import numpy as np
import polars as pl

from src.config.settings import settings
from src.neural.two_tower import TwoTowerModel


def _reports_models_dir() -> Path:
    base = getattr(settings, "BASE_DIR", None)
    if base:
        return Path(base) / "reports" / "models"
    return Path("reports") / "models"


def _processed_dir() -> Path:
    return Path(settings.PROCESSED_DIR)


def _load_item_embeddings() -> Tuple[np.ndarray, np.ndarray]:
    df = pl.read_parquet(_processed_dir() / "item_emb_v3.parquet")
    item_ids = df["item_idx"].to_numpy().astype(np.int64, copy=False)
    item_mat = np.array(df["embedding"].to_list(), dtype=np.float32)
    return item_ids, item_mat


def _build_item_lookup(item_ids: np.ndarray, item_mat: np.ndarray):
    return {int(i): item_mat[idx] for idx, i in enumerate(item_ids)}


def _load_dim_items_title_map() -> dict:
    """
    Preferred mapping:
    data/processed/dim_items.parquet with columns:
    [item_idx, movieId, title]
    """
    p = _processed_dir() / "dim_items.parquet"
    if not p.exists():
        return {}

    df = pl.read_parquet(p)
    cols = set(df.columns)
    if not {"item_idx", "title"}.issubset(cols):
        return {}

    return {int(r[0]): (r[1] if r[1] else None) for r in df.select(["item_idx", "title"]).iter_rows()}


def main(user_idx: int = 9764, k: int = 50, ann_k: int = 200):
    print("\n[START] ANN retrieval demo (V3)")

    model = TwoTowerModel.load("reports/models/two_tower_v3.pt")
    print("[OK] Two-tower loaded.")

    item_ids, item_mat = _load_item_embeddings()
    item_lookup = _build_item_lookup(item_ids, item_mat)
    print(f"[OK] Item embedding pool: {len(item_ids)}")

    import hnswlib

    meta_path = _reports_models_dir() / "ann_hnsw_items_v3.meta.json"
    index_path = _reports_models_dir() / "ann_hnsw_items_v3.bin"

    dim = item_mat.shape[1]
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        dim = int(meta.get("dim", dim))

    index = hnswlib.Index(space="ip", dim=dim)
    index.load_index(str(index_path))
    index.set_ef(max(100, ann_k))

    uvec = model.embed_users([user_idx])[0].astype(np.float32)

    labels, _ = index.knn_query(uvec, k=ann_k)
    cand_ids = [int(x) for x in labels[0]]

    rescored = []
    for iid in cand_ids:
        ivec = item_lookup.get(iid)
        if ivec is None:
            continue
        rescored.append((iid, float(np.dot(uvec, ivec))))

    rescored.sort(key=lambda x: x[1], reverse=True)
    top = rescored[:k]

    idx_to_title = _load_dim_items_title_map()

    print(f"\n[DEMO] Top-{k} ANN+rescored recommendations for user_idx={user_idx}\n")

    for rank, (iid, score) in enumerate(top, start=1):
        title = idx_to_title.get(iid)
        label = title if title else f"item_idx={iid}"
        print(f"{rank:02d}. {label} | score={score:.4f}")

    print("\n[DONE] ANN retrieval demo complete.")


if __name__ == "__main__":
    main()