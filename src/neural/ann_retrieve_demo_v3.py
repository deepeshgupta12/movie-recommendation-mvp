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
    p = Path(base) / "reports" / "models" if base else Path("reports") / "models"
    return p


def _processed_dir() -> Path:
    return Path(settings.PROCESSED_DIR)


def _load_item_embeddings() -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      item_ids: (N,)
      item_mat: (N, D) float32
    """
    df = pl.read_parquet(_processed_dir() / "item_emb_v3.parquet")
    item_ids = df["item_idx"].to_numpy().astype(np.int64, copy=False)
    item_mat = np.array(df["embedding"].to_list(), dtype=np.float32)
    return item_ids, item_mat


def _build_item_lookup(item_ids: np.ndarray, item_mat: np.ndarray):
    """
    Build lookup dict for fast dot rescoring.
    """
    return {int(i): item_mat[idx] for idx, i in enumerate(item_ids)}


def _load_movies_title_map() -> dict:
    """
    Uses movies parquet produced earlier in your pipeline.
    Falls back gracefully if file/path differs.
    """
    candidates = [
        _processed_dir() / "movies.parquet",
        Path("data/processed/movies.parquet"),
    ]

    for p in candidates:
        if p.exists():
            m = pl.read_parquet(p).select(["movieId", "title"])
            # item_idx != movieId; we likely have a mapping table elsewhere
            # For demo, we will try to load item mapping if exists.
            return {int(r[0]): r[1] for r in m.iter_rows()}
    return {}


def _load_item_index_map() -> dict:
    """
    If your pipeline created item_idx mapping parquet, use it:
    expected columns: [item_idx, movieId]
    """
    candidates = [
        _processed_dir() / "dim_items.parquet",
        _processed_dir() / "item_map.parquet",
        Path("data/processed/dim_items.parquet"),
        Path("data/processed/item_map.parquet"),
    ]
    for p in candidates:
        if p.exists():
            df = pl.read_parquet(p)
            cols = set(df.columns)
            if {"item_idx", "movieId"}.issubset(cols):
                return {int(r[0]): int(r[1]) for r in df.select(["item_idx", "movieId"]).iter_rows()}
    return {}


def main(user_idx: int = 9764, k: int = 50, ann_k: int = 200):
    print("\n[START] ANN retrieval demo (V3)")

    # Load two-tower model
    model = TwoTowerModel.load("reports/models/two_tower_v3.pt")
    print("[OK] Two-tower loaded.")

    # Load item embeddings for rescoring
    item_ids, item_mat = _load_item_embeddings()
    item_lookup = _build_item_lookup(item_ids, item_mat)
    print(f"[OK] Item embedding pool: {len(item_ids)}")

    # Load ANN index
    import hnswlib

    meta_path = _reports_models_dir() / "ann_hnsw_items_v3.meta.json"
    index_path = _reports_models_dir() / "ann_hnsw_items_v3.bin"

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        dim = int(meta["dim"])
    else:
        dim = item_mat.shape[1]

    index = hnswlib.Index(space="ip", dim=dim)
    index.load_index(str(index_path))

    # Increase ef for better recall in demo
    index.set_ef(max(100, ann_k))

    # Query user embedding
    uvec = model.embed_users([user_idx])[0].astype(np.float32)

    # ANN candidate fetch
    labels, distances = index.knn_query(uvec, k=ann_k)
    cand_ids = [int(x) for x in labels[0]]

    # True dot-product rescoring for correctness
    rescored = []
    for iid in cand_ids:
        ivec = item_lookup.get(iid)
        if ivec is None:
            continue
        score = float(np.dot(uvec, ivec))
        rescored.append((iid, score))

    rescored.sort(key=lambda x: x[1], reverse=True)
    top = rescored[:k]

    # Try to map to titles (best-effort)
    idx_to_movie = _load_item_index_map()
    movie_to_title = _load_movies_title_map()

    print(f"\n[DEMO] Top-{k} ANN+rescored recommendations for user_idx={user_idx}\n")

    for rank, (iid, score) in enumerate(top, start=1):
        mid = idx_to_movie.get(iid)
        title = movie_to_title.get(mid, None) if mid is not None else None
        label = title if title else f"item_idx={iid}"
        print(f"{rank:02d}. {label} | score={score:.4f}")

    print("\n[DONE] ANN retrieval demo complete.")


if __name__ == "__main__":
    # Default demo user based on your earlier service example
    main()