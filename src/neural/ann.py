from __future__ import annotations

import json
from typing import Dict, Tuple  # noqa: UP035

import polars as pl

from src.neural.contracts import AnnIndexConfig
from src.neural.paths import HNSW_INDEX, HNSW_META, ITEM_EMB


def build_hnsw_index(
    item_emb_path: str = str(ITEM_EMB),
    config: AnnIndexConfig | None = None,
) -> Tuple[str, Dict]:
    """
    Contract-first placeholder.

    In Step 8.4:
    - load item embeddings
    - build HNSW index using hnswlib
    - persist index + meta

    Returns:
        (index_path, meta_dict)
    """
    raise NotImplementedError("ANN index builder will be implemented in Step 8.4.")


def load_hnsw_meta() -> Dict:
    if not HNSW_META.exists():
        return {}
    return json.loads(HNSW_META.read_text(encoding="utf-8"))