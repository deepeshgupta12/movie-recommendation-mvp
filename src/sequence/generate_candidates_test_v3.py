"""
Generate sequence-based candidates for TEST users (V3).

Design goals:
- Dedicated TEST script (no patching VAL logic).
- Safe defaults for Apple Silicon (num_workers=0).
- Robust dataset path handling:
    1) Prefer user_seq_test.parquet if present.
    2) Fallback to user_seq_val.parquet with warning.
- Filter seen items using train_conf history.
- Save to data/processed/seq_candidates_v3_test.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple  # noqa: UP035

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config.settings import settings

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = Path(settings.DATA_DIR)
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = Path(getattr(settings, "REPORTS_DIR", "reports"))

SEQ_TEST_PATH = PROCESSED_DIR / "user_seq_test.parquet"
SEQ_VAL_PATH = PROCESSED_DIR / "user_seq_val.parquet"
TRAIN_CONF_PATH = PROCESSED_DIR / "train_conf.parquet"

MODEL_PATH = REPORTS_DIR / "models" / "sequence_gru_v3.pt"
MODEL_META_PATH = REPORTS_DIR / "models" / "sequence_gru_v3.meta.json"

OUT_PATH = PROCESSED_DIR / "seq_candidates_v3_test.parquet"


# -----------------------------
# Dataset
# -----------------------------
class SeqDataset(Dataset):
    def __init__(self, df: pl.DataFrame):
        # Expect columns: user_idx, seq, target (target may exist but not needed)
        self.user_idx = df["user_idx"].to_numpy()
        self.seq = df["seq"].to_list()

    def __len__(self) -> int:
        return len(self.user_idx)

    def __getitem__(self, idx: int):
        return int(self.user_idx[idx]), self.seq[idx]


def collate_fn(batch):
    # batch = [(user_idx, seq_list), ...]
    users = []
    seqs = []
    lengths = []
    for u, s in batch:
        users.append(u)
        seqs.append(s)
        lengths.append(len(s))

    max_len = max(lengths) if lengths else 1
    padded = np.zeros((len(batch), max_len), dtype=np.int64)

    for i, s in enumerate(seqs):
        if len(s) > 0:
            padded[i, -len(s):] = np.asarray(s, dtype=np.int64)

    return (
        torch.tensor(users, dtype=torch.int64),
        torch.tensor(padded, dtype=torch.int64),
        torch.tensor(lengths, dtype=torch.int64),
    )


# -----------------------------
# Model loader
# -----------------------------
def _load_meta_num_items() -> int:
    if not MODEL_META_PATH.exists():
        raise FileNotFoundError(f"Missing meta: {MODEL_META_PATH}")
    meta = pl.read_json(str(MODEL_META_PATH))
    # Expect meta contains num_items
    # handle both dict-like json or single-row json
    if "num_items" in meta.columns:
        return int(meta["num_items"][0])
    raise RuntimeError("sequence_gru_v3.meta.json missing num_items")


class GRUSeqModel(torch.nn.Module):
    """
    Minimal-compatible loader for your saved GRU model.
    We assume the training script saved a standard torch state_dict
    with:
      - item embedding layer
      - GRU
      - linear head

    If your internal model class is already present (preferred),
    you can replace this with:
      from src.sequence.gru_model import GRUSequenceModel
    and load that instead.
    """

    def __init__(self, num_items: int, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.num_items = num_items
        self.embed = torch.nn.Embedding(num_items, embed_dim)
        self.gru = torch.nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, num_items)

    def forward(self, x, lengths):
        # x: [B, T]
        emb = self.embed(x)
        # pack for efficiency
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.gru(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        # take last valid hidden for each sequence
        idx = (lengths - 1).clamp(min=0)
        last = out[torch.arange(out.size(0)), idx]
        logits = self.fc(last)
        return logits


def load_gru_model(device: str) -> torch.nn.Module:
    num_items = _load_meta_num_items()

    # Try to infer dims from checkpoint if possible; else use safe defaults.
    state = torch.load(str(MODEL_PATH), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    elif isinstance(state, dict):
        state_dict = state
    else:
        raise RuntimeError("Unrecognized GRU checkpoint format.")

    # Heuristic dimension extraction
    embed_w = None
    for k in state_dict.keys():
        if "embed" in k and k.endswith("weight"):
            embed_w = state_dict[k]
            break
    embed_dim = int(embed_w.shape[1]) if embed_w is not None else 64

    model = GRUSeqModel(num_items=num_items, embed_dim=embed_dim)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


# -----------------------------
# Utilities
# -----------------------------
def load_sequences_for_test() -> Tuple[pl.DataFrame, Path]:
    if SEQ_TEST_PATH.exists():
        df = pl.read_parquet(str(SEQ_TEST_PATH))
        return df, SEQ_TEST_PATH

    # Fallback
    if SEQ_VAL_PATH.exists():
        print(
            f"[WARN] {SEQ_TEST_PATH.name} not found. "
            f"Falling back to {SEQ_VAL_PATH.name} for sequence candidates."
        )
        df = pl.read_parquet(str(SEQ_VAL_PATH))
        return df, SEQ_VAL_PATH

    raise FileNotFoundError(
        "No sequence dataset found. Expected one of:\n"
        f" - {SEQ_TEST_PATH}\n"
        f" - {SEQ_VAL_PATH}\n"
        "Create user_seq_test.parquet if you want strict TEST semantics."
    )


def build_seen_map(train_conf: pl.DataFrame) -> Dict[int, set]:
    # train_conf expected columns: user_idx, item_idx
    seen: Dict[int, set] = {}
    # group-by without .list() to avoid ExprListNameSpace issues
    gb = train_conf.group_by("user_idx").agg(pl.col("item_idx"))
    # gb['item_idx'] will be list type
    for u, items in zip(gb["user_idx"].to_list(), gb["item_idx"].to_list()):
        seen[int(u)] = set(int(x) for x in items)
    return seen


def topk_filter_seen(
    scores: np.ndarray,
    seen_items: set,
    topk: int,
) -> List[int]:
    # scores length = num_items
    # mask seen
    if seen_items:
        scores = scores.copy()
        idxs = np.fromiter(seen_items, dtype=np.int64, count=len(seen_items))
        idxs = idxs[(idxs >= 0) & (idxs < scores.shape[0])]
        scores[idxs] = -1e9

    # argpartition for speed
    k = min(topk, scores.shape[0])
    cand = np.argpartition(-scores, k - 1)[:k]
    cand = cand[np.argsort(-scores[cand])]
    return cand.astype(np.int64).tolist()


# -----------------------------
# Main generation
# -----------------------------
def generate_test_candidates(
    topk: int = 200,
    filter_seen: bool = True,
    batch_size: int = 512,
    num_workers: int = 0,
):
    print("[START] Loading user sequences for TEST...")
    seq_df, used_path = load_sequences_for_test()
    print(f"[OK] rows={seq_df.height} | source={used_path.name}")

    # Optional cap consistency: if your pipeline caps test users to 50k
    if "user_idx" in seq_df.columns:
        # keep stable order for reproducibility
        seq_df = seq_df.sort("user_idx")
        seq_df = seq_df.filter(pl.col("user_idx") < 50000)

    print(f"[OK] rows after cap filter (if applied): {seq_df.height}")

    print("[START] Loading train_conf for seen filtering...")
    if not TRAIN_CONF_PATH.exists():
        raise FileNotFoundError(f"Missing {TRAIN_CONF_PATH}")
    train_conf = pl.read_parquet(str(TRAIN_CONF_PATH)).select(["user_idx", "item_idx"])
    print(f"[OK] train_conf rows: {train_conf.height}")

    seen_map = build_seen_map(train_conf) if filter_seen else {}
    print(f"[OK] seen users: {len(seen_map)}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[OK] device={device}")

    print("[START] Loading GRU model...")
    model = load_gru_model(device=device)
    num_items = getattr(model, "num_items", _load_meta_num_items())
    print(f"[OK] num_items={num_items}")

    ds = SeqDataset(seq_df)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,  # keep 0 by default for M1 stability
        collate_fn=collate_fn,
        pin_memory=False,
    )

    out_users: List[int] = []
    out_items: List[List[int]] = []
    out_scores: List[List[float]] = []

    print(
        f"[START] Generating top-{topk} sequence candidates "
        f"(filter_seen={filter_seen}, num_workers={num_workers})..."
    )

    with torch.no_grad():
        for users, seqs, lengths in tqdm(
            loader, desc="Seq candidates (test)"
        ):
            users = users.to(device)
            seqs = seqs.to(device)
            lengths = lengths.to(device)

            logits = model(seqs, lengths)  # [B, num_items]
            scores = torch.softmax(logits, dim=-1).cpu().numpy()

            for b in range(scores.shape[0]):
                u = int(users[b].cpu().item())
                s = scores[b]

                seen = seen_map.get(u, set()) if filter_seen else set()
                cand = topk_filter_seen(s, seen, topk=topk)

                out_users.append(u)
                out_items.append(cand)
                out_scores.append([float(s[i]) for i in cand])

    result = pl.DataFrame(
        {
            "user_idx": out_users,
            "candidates_seq": out_items,
            "seq_scores": out_scores,
        }
    ).sort("user_idx")

    result.write_parquet(str(OUT_PATH))
    print("[DONE] Sequence TEST candidates saved.")
    print(f"[PATH] {OUT_PATH}")
    print(f"[OK] users exported: {result.select(pl.col('user_idx').n_unique()).item()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--filter_seen", action="store_true", default=True)
    parser.add_argument("--no_filter_seen", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()
    filter_seen = args.filter_seen and not args.no_filter_seen

    generate_test_candidates(
        topk=args.topk,
        filter_seen=filter_seen,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()