from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Dict, List, Tuple  # noqa: UP035

import numpy as np
import polars as pl
from tqdm import tqdm

from src.config.settings import settings

PROCESSED = Path(settings.PROCESSED_DIR)
REPORTS = Path(getattr(settings, "REPORTS_DIR", "reports"))
MODELS_DIR = REPORTS / "models"


def _p(name: str) -> Path:
    return PROCESSED / name


def _load_parquet(name: str) -> pl.DataFrame:
    path = _p(name)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pl.read_parquet(path)


def _load_hnsw_index():
    # Uses hnswlib; assumed already in deps for your ANN steps
    import hnswlib

    idx_path = MODELS_DIR / "ann_hnsw_items_v3.bin"
    meta_path = MODELS_DIR / "ann_hnsw_items_v3.meta.json"

    if not idx_path.exists():
        raise FileNotFoundError(f"Missing ANN index: {idx_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing ANN meta: {meta_path}")

    import json
    meta = json.loads(meta_path.read_text())

    dim = int(meta.get("dim", 64))
    space = meta.get("space", "ip")  # your builder likely used "ip"

    index = hnswlib.Index(space=space, dim=dim)
    index.load_index(str(idx_path))

    # Optional tuning
    ef = int(meta.get("ef_search", 100))
    with contextlib.suppress(Exception):
        index.set_ef(ef)

    return index, meta


def _to_vec_list(df: pl.DataFrame, id_col: str) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for row in df.select([id_col, "embedding"]).iter_rows(named=True):
        idx = int(row[id_col])
        emb = np.asarray(row["embedding"], dtype=np.float32)
        out[idx] = emb
    return out


def _build_seen_map(train_conf: pl.DataFrame, users: List[int]) -> Dict[int, set]:
    # Build a compact user->seen set for only capped users
    user_set = set(users)

    filt = train_conf.filter(pl.col("user_idx").is_in(list(user_set))).select(
        ["user_idx", "item_idx"]
    )

    grouped = filt.group_by("user_idx").agg(
        pl.col("item_idx").unique().alias("seen")
    )

    seen_map: Dict[int, set] = {}
    for row in grouped.iter_rows(named=True):
        u = int(row["user_idx"])
        seen_map[u] = set(int(x) for x in (row["seen"] or []))
    return seen_map


def main(
    topk: int = 200,
    ann_k: int = 200,
    user_cap: int = 50000,
    filter_seen: bool = True,
):
    print("\n[START] Exporting ANN candidates for test (V3)...")

    # --- Load test users ---
    test_conf = _load_parquet("test_conf.parquet").select(["user_idx", "item_idx"])
    users = (
        test_conf.select("user_idx")
        .unique()
        .sort("user_idx")
        .head(user_cap)
        .get_column("user_idx")
        .to_list()
    )
    users = [int(u) for u in users]

    print(f"[OK] test users after cap: {len(users)}")

    # --- Load embeddings ---
    print("[START] Loading user/item embeddings (V3)...")
    user_emb = _load_parquet("user_emb_v3.parquet")
    item_emb = _load_parquet("item_emb_v3.parquet")

    user_vecs = _to_vec_list(user_emb, "user_idx")
    item_vecs = _to_vec_list(item_emb, "item_idx")

    # Intersect cap with available exported embeddings
    users = [u for u in users if u in user_vecs]
    print(f"[OK] users with available embeddings: {len(users)}")
    print(f"[OK] item embedding pool: {len(item_vecs)}")

    # --- Load ANN index ---
    index, meta = _load_hnsw_index()
    dim = int(meta.get("dim", 64))
    print(f"[OK] ANN index loaded | dim={dim}")

    # --- Optional seen filter ---
    seen_map: Dict[int, set] = {}
    if filter_seen:
        print("[START] Building seen map from train_conf...")
        train_conf = _load_parquet("train_conf.parquet").select(["user_idx", "item_idx"])
        seen_map = _build_seen_map(train_conf, users)
        print(f"[OK] seen map users: {len(seen_map)}")

    # --- Batch ANN retrieval ---
    print(
        f"[START] ANN retrieval + dot-product rescoring (ann_k={ann_k}, out_k={topk}, filter_seen={filter_seen})..."
    )

    results_user: List[int] = []
    results_cands: List[List[int]] = []
    results_scores: List[List[float]] = []

    # Build a fast item lookup for rescoring
    # We will only rescore items that exist in item_vecs
    def rescore(uvec: np.ndarray, cand_ids: List[int]) -> List[float]:
        # Safe dot product
        vecs = []
        valid_ids = []
        for cid in cand_ids:
            v = item_vecs.get(cid)
            if v is not None:
                vecs.append(v)
                valid_ids.append(cid)
        if not vecs:
            return [], []
        mat = np.vstack(vecs)  # (n, d)
        s = mat @ uvec  # (n,)
        return valid_ids, s.astype(np.float32).tolist()

    # Small batch to reduce python overhead
    batch_size = 512
    for i in tqdm(range(0, len(users), batch_size), desc="ANN users (test)"):
        batch_users = users[i : i + batch_size]
        batch_vecs = np.vstack([user_vecs[u] for u in batch_users]).astype(np.float32)

        # hnswlib returns labels + distances
        labels, _ = index.knn_query(batch_vecs, k=ann_k)

        for u, cand_arr, uvec in zip(batch_users, labels, batch_vecs):
            cand_ids = [int(x) for x in cand_arr.tolist()]

            if filter_seen:
                seen = seen_map.get(u, set())
                if seen:
                    cand_ids = [c for c in cand_ids if c not in seen]

            cand_ids, cand_scores = rescore(uvec, cand_ids)

            # Truncate
            cand_ids = cand_ids[:topk]
            cand_scores = cand_scores[:topk]

            results_user.append(int(u))
            results_cands.append(cand_ids)
            results_scores.append([float(x) for x in cand_scores])

    out = pl.DataFrame(
        {
            "user_idx": results_user,
            "candidates": results_cands,
            "tt_scores": results_scores,
        }
    )

    out_path = _p("ann_candidates_v3_test.parquet")
    out.write_parquet(out_path)

    print("[DONE] ANN candidates exported.")
    print(f"[PATH] {out_path.resolve()}")
    print(f"[OK] users exported: {out.height}")
    print(f"[OK] topk={topk} | ann_k={ann_k} | filter_seen={filter_seen}")


if __name__ == "__main__":
    main()