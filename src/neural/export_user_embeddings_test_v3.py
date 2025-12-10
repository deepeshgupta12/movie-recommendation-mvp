from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple  # noqa: UP035

import polars as pl
import torch
from tqdm import tqdm

from src.config.settings import settings

# -----------------------------
# Paths
# -----------------------------
REPORTS_DIR = Path(getattr(settings, "REPORTS_DIR", "reports"))
MODELS_DIR = REPORTS_DIR / "models"

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"

PT_PATH = MODELS_DIR / "two_tower_v3.pt"
META_PATH = MODELS_DIR / "two_tower_v3.meta.json"

TEST_CONF_PATH = PROCESSED_DIR / "test_conf.parquet"
TRAIN_CONF_PATH = PROCESSED_DIR / "train_conf.parquet"

OUT_PATH = PROCESSED_DIR / "user_emb_v3_test.parquet"


# -----------------------------
# Meta helpers
# -----------------------------
def _load_json(p: Path) -> dict:
    if p.exists():
        return json.loads(p.read_text())
    return {}


# -----------------------------
# Checkpoint helpers
# -----------------------------
def _load_checkpoint(pt_path: Path) -> dict:
    if not pt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {pt_path}")
    obj = torch.load(pt_path, map_location="cpu")
    if isinstance(obj, dict):
        return obj
    # Extremely defensive fallback: wrap module-like objects
    if isinstance(obj, torch.nn.Module):
        return {"format": "module_raw", "state_dict": obj.state_dict()}
    raise RuntimeError("Unsupported checkpoint object type.")


def _extract_state_dict(ckpt: dict) -> Dict[str, torch.Tensor]:
    sd = ckpt.get("state_dict") or ckpt.get("model_state_dict")

    # Some older formats might still exist
    if sd is None and isinstance(ckpt.get("user_tower_state_dict"), dict) and isinstance(ckpt.get("item_tower_state_dict"), dict):
        # We cannot reliably merge these without the model structure,
        # so we fail with a clear message.
        raise RuntimeError(
            "Checkpoint contains separate tower dicts but no unified state_dict. "
            "Re-run the normalizer or update the export script to a model-based loader."
        )

    if not isinstance(sd, dict):
        raise RuntimeError(
            "Could not find a usable state_dict in two_tower_v3.pt. "
            "Your normalizer should have saved {format, state_dict}."
        )

    # Ensure tensors only
    out = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            out[k] = v
    return out


def _find_user_embedding_matrix(sd: Dict[str, torch.Tensor], max_user_idx: int) -> Tuple[str, torch.Tensor]:
    """
    Locate the user embedding weight matrix by heuristics.
    We search for 2D tensors whose key indicates 'user' + 'emb/embedding'.
    Pick the candidate that can cover max_user_idx.
    """
    candidates: List[Tuple[str, torch.Tensor]] = []

    for k, v in sd.items():
        if not isinstance(v, torch.Tensor) or v.ndim != 2:
            continue
        lk = k.lower()
        if "user" in lk and ("emb" in lk or "embedding" in lk):
            candidates.append((k, v))

    # Fallback heuristic: sometimes keys are generic but prefixed under user_tower.*
    if not candidates:
        for k, v in sd.items():
            if not isinstance(v, torch.Tensor) or v.ndim != 2:
                continue
            lk = k.lower()
            if "user_tower" in lk and ("emb" in lk or "embedding" in lk):
                candidates.append((k, v))

    if not candidates:
        # Last resort: pick the largest 2D tensor as a guess, but only if it's plausible.
        # We still require it can index max_user_idx.
        largest = None
        for k, v in sd.items():
            if isinstance(v, torch.Tensor) and v.ndim == 2:
                if largest is None or v.shape[0] > largest[1].shape[0]:
                    largest = (k, v)
        if largest and largest[1].shape[0] > max_user_idx:
            return largest[0], largest[1]
        raise RuntimeError(
            "Could not locate user embedding matrix inside state_dict. "
            "Expected keys containing 'user' and 'emb'."
        )

    # Prefer candidates that cover our index range
    valid = [c for c in candidates if c[1].shape[0] > max_user_idx]
    if valid:
        # Choose the one with the largest vocab size
        valid.sort(key=lambda x: x[1].shape[0], reverse=True)
        return valid[0]

    # If none cover max_user_idx, surface a precise error
    sizes = [(k, int(v.shape[0]), int(v.shape[1])) for k, v in candidates]
    raise RuntimeError(
        f"Found user embedding candidates but none cover max_user_idx={max_user_idx}. "
        f"Candidates: {sizes}"
    )


# -----------------------------
# Data helpers
# -----------------------------
def _load_test_users(cap_users: int = 50000) -> List[int]:
    if not TEST_CONF_PATH.exists():
        raise FileNotFoundError(
            f"Missing {TEST_CONF_PATH}. Make sure confidence splits exist in data/processed."
        )

    df = pl.read_parquet(TEST_CONF_PATH).select("user_idx").unique()

    # Deterministic ordering
    users = df.sort("user_idx")["user_idx"].to_list()

    # Full 50k coverage intent: take first 50k unique user_idx as per cap philosophy
    if len(users) > cap_users:
        users = users[:cap_users]

    return [int(u) for u in users]


def _ensure_train_conf_exists():
    if not TRAIN_CONF_PATH.exists():
        # Not strictly required for embedding extraction,
        # but good to keep explicit signals for the pipeline.
        print("[WARN] train_conf.parquet not found. Proceeding without seen-map validation.")


# -----------------------------
# Main
# -----------------------------
def main():
    print("\n[START] Exporting Two-Tower V3 user embeddings for TEST (full 50k target)...")

    _ensure_train_conf_exists()

    # Load users
    test_users = _load_test_users(cap_users=50000)
    if not test_users:
        raise RuntimeError("No test users found.")

    max_user_idx = max(test_users)

    print(f"[OK] test users selected: {len(test_users)}")
    print(f"[OK] max test user_idx: {max_user_idx}")

    # Load checkpoint
    print("[START] Loading normalized Two-Tower checkpoint...")
    ckpt = _load_checkpoint(PT_PATH)

    fmt = ckpt.get("format", "unknown")
    print(f"[OK] checkpoint format: {fmt}")

    # Extract state dict
    sd = _extract_state_dict(ckpt)

    # Locate user embedding matrix
    print("[START] Locating user embedding matrix in state_dict...")
    emb_key, user_W = _find_user_embedding_matrix(sd, max_user_idx=max_user_idx)

    dim = int(user_W.shape[1])
    vocab = int(user_W.shape[0])

    print(f"[OK] user embedding key: {emb_key}")
    print(f"[OK] user embedding vocab size: {vocab}")
    print(f"[OK] embedding dim: {dim}")

    # Export rows for our user set
    print("[START] Materializing embeddings...")

    # Index selection tensor
    idx_t = torch.tensor(test_users, dtype=torch.long)

    # Gather embeddings
    with torch.no_grad():
        selected = user_W.index_select(0, idx_t).cpu()

    # Convert to python lists efficiently
    # Avoid giant intermediate conversions where possible
    embeddings: List[List[float]] = []
    for row in tqdm(selected, desc="Converting embeddings"):
        embeddings.append([float(x) for x in row.tolist()])

    out_df = pl.DataFrame(
        {
            "user_idx": test_users,
            "embedding": embeddings,
        },
        orient="col",
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(OUT_PATH)

    print("[DONE] TEST user embeddings exported.")
    print(f"[PATH] {OUT_PATH.resolve()}")
    print(f"[OK] rows={out_df.shape[0]} | dim={dim}")


if __name__ == "__main__":
    main()