from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple  # noqa: UP035

import numpy as np
import polars as pl
from tqdm import tqdm

try:
    import hnswlib
except Exception as e:
    raise RuntimeError(
        "hnswlib is required for ANN retrieval. "
        "Install it via your pyproject dependencies."
    ) from e


# -----------------------------
# Paths
# -----------------------------
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = Path("reports")
MODELS_DIR = REPORTS_DIR / "models"

USER_EMB_TEST_PATH = PROCESSED_DIR / "user_emb_v3_test.parquet"
ITEM_EMB_PATH = PROCESSED_DIR / "item_emb_v3.parquet"

TRAIN_CONF_PATH = PROCESSED_DIR / "train_conf.parquet"
TEST_CONF_PATH = PROCESSED_DIR / "test_conf.parquet"

ANN_INDEX_PATH = MODELS_DIR / "ann_hnsw_items_v3.bin"
ANN_META_PATH = MODELS_DIR / "ann_hnsw_items_v3.meta.json"

OUT_PATH = PROCESSED_DIR / "ann_candidates_v3_test.parquet"


# -----------------------------
# Config
# -----------------------------
DEFAULT_TOPK = 200
DEFAULT_ANN_K = 200
FILTER_SEEN = True


# -----------------------------
# Helpers: Loaders
# -----------------------------
def _load_item_embeddings() -> Tuple[np.ndarray, np.ndarray]:
    if not ITEM_EMB_PATH.exists():
        raise FileNotFoundError(f"Missing item embeddings: {ITEM_EMB_PATH}")

    df = pl.read_parquet(ITEM_EMB_PATH).select(["item_idx", "embedding"])

    item_idx = df["item_idx"].to_numpy()
    emb_list = df["embedding"].to_list()

    mat = np.asarray(emb_list, dtype=np.float32)
    if mat.ndim != 2:
        raise RuntimeError("Item embedding matrix is not 2D.")

    return item_idx.astype(np.int64), mat


def _load_user_embeddings_test() -> Tuple[np.ndarray, np.ndarray]:
    if not USER_EMB_TEST_PATH.exists():
        raise FileNotFoundError(f"Missing test user embeddings: {USER_EMB_TEST_PATH}")

    df = (
        pl.read_parquet(USER_EMB_TEST_PATH)
        .select(["user_idx", "embedding"])
        .sort("user_idx")
    )

    user_idx = df["user_idx"].to_numpy()
    emb_list = df["embedding"].to_list()

    mat = np.asarray(emb_list, dtype=np.float32)
    if mat.ndim != 2:
        raise RuntimeError("User embedding matrix is not 2D.")

    return user_idx.astype(np.int64), mat


def _load_test_users_from_conf(cap_users: int = 50000) -> np.ndarray:
    if not TEST_CONF_PATH.exists():
        raise FileNotFoundError(f"Missing: {TEST_CONF_PATH}")

    df = pl.read_parquet(TEST_CONF_PATH).select("user_idx").unique().sort("user_idx")
    users = df["user_idx"].to_list()

    if len(users) > cap_users:
        users = users[:cap_users]

    return np.asarray(users, dtype=np.int64)


def _build_seen_map() -> Dict[int, set]:
    if not FILTER_SEEN:
        return {}

    if not TRAIN_CONF_PATH.exists():
        print("[WARN] train_conf.parquet missing; cannot filter seen.")
        return {}

    print("[START] Building seen map from train_conf...")
    train = pl.read_parquet(TRAIN_CONF_PATH).select(["user_idx", "item_idx"])

    grouped = train.group_by("user_idx").agg(pl.col("item_idx").alias("seen"))

    seen_map: Dict[int, set] = {}
    for u, items in zip(grouped["user_idx"].to_list(), grouped["seen"].to_list()):
        seen_map[int(u)] = set(int(x) for x in items)

    print(f"[OK] seen map users: {len(seen_map)}")
    return seen_map


def _load_ann_index(dim: int) -> "hnswlib.Index":
    if not ANN_INDEX_PATH.exists():
        raise FileNotFoundError(f"Missing ANN index: {ANN_INDEX_PATH}")

    idx = hnswlib.Index(space="ip", dim=dim)
    idx.load_index(str(ANN_INDEX_PATH))
    return idx


# -----------------------------
# Helpers: Label mode handling
# -----------------------------
def _read_ann_label_mode_from_meta() -> Optional[str]:
    if not ANN_META_PATH.exists():
        return None
    try:
        meta = json.loads(ANN_META_PATH.read_text())
        mode = meta.get("id_mode") or meta.get("label_mode") or meta.get("ann_id_mode")
        if mode in {"row", "item_idx"}:
            return mode
    except Exception:
        return None
    return None


def _infer_label_mode(
    index: "hnswlib.Index",
    sample_user_vec: np.ndarray,
    item_idx: np.ndarray,
    idx_to_row: Dict[int, int],
    ann_k: int
) -> str:
    labels, _ = index.knn_query(sample_user_vec, k=min(ann_k, 200))
    labels = labels[0].astype(np.int64)

    # Heuristic 1: if all labels are small and fit row space
    if labels.max(initial=0) < len(item_idx):
        return "row"

    # Heuristic 2: if labels look like item_idx values
    # We check a small sample membership
    sample = labels[: min(len(labels), 20)]
    ok = all(int(x) in idx_to_row for x in sample.tolist())
    if ok:
        return "item_idx"

    raise RuntimeError(
        "Could not infer ANN label mode. "
        "Your ANN index likely doesn't match current item_emb_v3. "
        "Rebuild ann_hnsw_items_v3.bin from item_emb_v3.parquet."
    )


def _resolve_labels(
    labels: np.ndarray,
    item_idx: np.ndarray,
    idx_to_row: Dict[int, int],
    mode: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      cand_items: global item_idx values
      local_ids: row indices into item_mat for scoring
    """
    labels = labels.astype(np.int64)

    if mode == "row":
        # labels are row ids
        local_ids = labels
        cand_items = item_idx[local_ids]
        return cand_items, local_ids

    if mode == "item_idx":
        # labels are global item_idx values
        cand_items = labels
        local_ids = np.asarray([idx_to_row[int(i)] for i in cand_items.tolist()], dtype=np.int64)
        return cand_items, local_ids

    raise ValueError(f"Unknown label mode: {mode}")


# -----------------------------
# Helpers: Scoring
# -----------------------------
def _dot_scores(user_vec: np.ndarray, item_mat: np.ndarray, local_ids: np.ndarray) -> np.ndarray:
    cand_vecs = item_mat[local_ids]
    return cand_vecs @ user_vec


def _build_user_emb_lookup(emb_users: np.ndarray, emb_mat: np.ndarray) -> Dict[int, np.ndarray]:
    return {int(u): emb_mat[i] for i, u in enumerate(emb_users.tolist())}


def _validate_alignment(test_users: np.ndarray, emb_users: np.ndarray) -> np.ndarray:
    set_emb = set(int(u) for u in emb_users.tolist())
    missing = [int(u) for u in test_users.tolist() if int(u) not in set_emb]

    if missing:
        raise RuntimeError(
            f"Missing embeddings for {len(missing)} test users. "
            f"Example missing user_idx: {missing[:10]}. "
            "Your export_user_embeddings_test_v3 must cover the same 50k user cap."
        )
    return test_users


# -----------------------------
# Main
# -----------------------------
def main(topk: int = DEFAULT_TOPK, ann_k: int = DEFAULT_ANN_K):
    print("\n[START] Exporting ANN candidates for test (V3) - full 50k coverage...")

    if topk > 200:
        raise ValueError("topk should be <= 200 for this local V3 pipeline.")
    if ann_k < topk:
        ann_k = topk

    # Canonical test users
    test_users = _load_test_users_from_conf(cap_users=50000)
    print(f"[OK] test users after cap: {len(test_users)}")

    # Load embeddings
    print("[START] Loading user embeddings (TEST) ...")
    emb_users, user_mat = _load_user_embeddings_test()
    print(f"[OK] users with embeddings file: {len(emb_users)}")

    print("[START] Loading item embeddings (V3) ...")
    item_idx, item_mat = _load_item_embeddings()
    print(f"[OK] item embedding pool: {len(item_idx)}")

    # Alignment check
    test_users = _validate_alignment(test_users, emb_users)

    # Lookups
    user_lookup = _build_user_emb_lookup(emb_users, user_mat)
    idx_to_row = {int(ix): i for i, ix in enumerate(item_idx.tolist())}

    # Load ANN index
    dim = int(item_mat.shape[1])
    idx = _load_ann_index(dim=dim)
    print(f"[OK] ANN index loaded | dim={dim}")

    # Seen map
    seen_map = _build_seen_map() if FILTER_SEEN else {}

    # Detect label mode
    mode = _read_ann_label_mode_from_meta()
    if mode is None:
        # Infer using first test user
        probe_user = int(test_users[0])
        mode = _infer_label_mode(idx, user_lookup[probe_user], item_idx, idx_to_row, ann_k)
    print(f"[OK] ANN label mode: {mode}")

    out_users: List[int] = []
    out_cands: List[List[int]] = []
    out_scores: List[List[float]] = []

    print(
        f"[START] ANN retrieval + dot-product rescoring "
        f"(ann_k={ann_k}, out_k={topk}, filter_seen={FILTER_SEEN})..."
    )

    for u in tqdm(test_users.tolist(), desc="ANN users (test)"):
        u = int(u)
        uvec = user_lookup[u]

        labels, _ = idx.knn_query(uvec, k=ann_k)
        labels = labels[0].astype(np.int64)

        # Map labels -> (global item_idx, local row ids for scoring)
        cand_items, local_ids = _resolve_labels(labels, item_idx, idx_to_row, mode)

        # Optionally filter seen using global item_idx
        if FILTER_SEEN and u in seen_map:
            s = seen_map[u]
            mask = np.array([int(ci) not in s for ci in cand_items.tolist()], dtype=bool)
            cand_items = cand_items[mask]
            local_ids = local_ids[mask]

        if len(local_ids) == 0:
            # Defensive fallback: no filter for this user
            labels, _ = idx.knn_query(uvec, k=topk)
            labels = labels[0].astype(np.int64)
            cand_items, local_ids = _resolve_labels(labels, item_idx, idx_to_row, mode)

        # Rescore by dot product
        scores = _dot_scores(uvec, item_mat, local_ids)

        # Sort by score desc
        order = np.argsort(-scores)
        cand_items = cand_items[order]
        scores = scores[order]

        # Truncate
        cand_items = cand_items[:topk]
        scores = scores[:topk]

        out_users.append(u)
        out_cands.append([int(x) for x in cand_items.tolist()])
        out_scores.append([float(x) for x in scores.tolist()])

    df = pl.DataFrame(
        {"user_idx": out_users, "candidates": out_cands, "tt_scores": out_scores},
        orient="col",
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_PATH)

    # Write meta for reproducibility
    meta = {
        "version": "v3_test",
        "topk": topk,
        "ann_k": ann_k,
        "filter_seen": FILTER_SEEN,
        "users_cap": 50000,
        "users_exported": len(out_users),
        "item_pool": int(len(item_idx)),
        "embedding_dim": int(dim),
        "ann_index_path": str(ANN_INDEX_PATH),
        "user_emb_path": str(USER_EMB_TEST_PATH),
        "item_emb_path": str(ITEM_EMB_PATH),
        "id_mode": mode,
    }
    (OUT_PATH.parent / "ann_candidates_v3_test.meta.json").write_text(json.dumps(meta, indent=2))

    print("[DONE] ANN candidates exported.")
    print(f"[PATH] {OUT_PATH.resolve()}")
    print(f"[OK] users exported: {len(out_users)}")
    print(f"[OK] topk={topk} | ann_k={ann_k} | filter_seen={FILTER_SEEN} | id_mode={mode}")


if __name__ == "__main__":
    main()