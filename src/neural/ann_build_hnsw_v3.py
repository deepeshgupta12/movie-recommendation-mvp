from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import polars as pl

from src.config.settings import settings


def _reports_models_dir() -> Path:
    # Mirror the two_tower safe convention: prefer BASE_DIR if present
    base = getattr(settings, "BASE_DIR", None)
    p = Path(base) / "reports" / "models" if base else Path("reports") / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _processed_dir() -> Path:
    return Path(settings.PROCESSED_DIR)


def main():
    print("\n[START] Loading item embeddings (V3)...")
    item_path = _processed_dir() / "item_emb_v3.parquet"
    df = pl.read_parquet(item_path)

    if "item_idx" not in df.columns or "embedding" not in df.columns:
        raise ValueError("item_emb_v3.parquet must contain [item_idx, embedding].")

    item_ids = df["item_idx"].to_numpy().astype(np.int64, copy=False)
    emb_list = df["embedding"].to_list()

    # Force float32 matrix
    mat = np.array(emb_list, dtype=np.float32)

    print(f"[OK] items={len(item_ids)} | dim={mat.shape[1]}")

    print("[START] Building HNSW index...")
    import hnswlib

    dim = mat.shape[1]
    num_items = mat.shape[0]

    # Use inner product space
    index = hnswlib.Index(space="ip", dim=dim)

    # HNSW params (local MVP)
    # Higher M/ef_construction improves recall but increases build time/memory.
    M = 48
    ef_construction = 200

    index.init_index(
        max_elements=num_items,
        ef_construction=ef_construction,
        M=M,
    )

    # Let the library use native threads
    # (Python-side is minimal)
    cpu_ct = os.cpu_count() or 8
    index.set_num_threads(min(8, cpu_ct))

    index.add_items(mat, item_ids)

    # Query-time parameter; can be overridden in retrieval script
    index.set_ef(100)

    out_dir = _reports_models_dir()
    index_path = out_dir / "ann_hnsw_items_v3.bin"
    meta_path = out_dir / "ann_hnsw_items_v3.meta.json"

    print("[START] Saving index...")
    index.save_index(str(index_path))

    meta = {
        "index_path": str(index_path),
        "meta_path": str(meta_path),
        "space": "ip",
        "dim": dim,
        "num_items": int(num_items),
        "M": M,
        "ef_construction": ef_construction,
        "ef_default": 100,
        "item_embedding_source": str(item_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[DONE] HNSW item index built.")
    print(f"[PATH] {index_path}")
    print(f"[PATH] {meta_path}")


if __name__ == "__main__":
    main()