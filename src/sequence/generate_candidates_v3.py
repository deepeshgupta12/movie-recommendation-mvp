from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple  # noqa: UP035

import numpy as np
import polars as pl

from src.config.settings import settings


def _processed_dir() -> Path:
    return Path(settings.PROCESSED_DIR)


def _reports_models_dir() -> Path:
    base = getattr(settings, "BASE_DIR", None)
    p = Path(base) / "reports" / "models" if base else Path("reports") / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pad_sequences(seqs: List[List[int]], max_len: int) -> np.ndarray:
    """
    Left-pad with 0. Keep most recent max_len items.
    0 is reserved as padding idx.
    """
    out = np.zeros((len(seqs), max_len), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[-max_len:]
        if len(s) > 0:
            out[i, -len(s):] = np.asarray(s, dtype=np.int64)
    return out


def _load_model():
    import torch

    from src.sequence.model_gru_v3 import GRUSeqConfig, GRUSeqModel

    model_path = _reports_models_dir() / "sequence_gru_v3.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            "sequence_gru_v3.pt not found. Run src.sequence.train_gru_v3 first."
        )

    blob = torch.load(model_path, map_location="cpu")
    cfg = GRUSeqConfig(**blob["config"])

    m = GRUSeqModel(cfg)
    m.model.load_state_dict(blob["state_dict"])

    return m, cfg


def _infer_num_items() -> int:
    dim_items_path = _processed_dir() / "dim_items.parquet"
    if dim_items_path.exists():
        dim_items = pl.read_parquet(dim_items_path)
        return int(dim_items["item_idx"].max()) + 1

    # fallback
    seq_train = pl.read_parquet(_processed_dir() / "user_seq_train.parquet")
    max_seq = int(seq_train.select(pl.col("seq").list.max()).to_series().max())
    max_tgt = int(seq_train["target"].max())
    return max(max_seq, max_tgt) + 1


class SeqCandidateDataset:
    """
    Top-level dataset to support macOS spawn pickling.
    """

    def __init__(self, users_arr: np.ndarray, x_arr: np.ndarray, seq_list: List[List[int]]):
        self.users_arr = users_arr
        self.x_arr = x_arr
        self.seq_list = seq_list

    def __len__(self):
        return self.x_arr.shape[0]

    def __getitem__(self, idx):
        return self.users_arr[idx], self.x_arr[idx], self.seq_list[idx]


def generate_sequence_candidates(
    split: str = "val",
    topk: int = 200,
    filter_seen: bool = True,
    batch_size: int = 512,
    num_workers: int | None = None,
):
    """
    Generates next-item candidates per user from GRU sequence model.
    Saves:
      data/processed/seq_candidates_v3_<split>.parquet

    Schema:
      user_idx: u32
      candidates: list[i32]
      seq_scores: list[f32]
    """
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    print(f"\n[START] Loading user_seq_{split}...")
    seq_path = _processed_dir() / f"user_seq_{split}.parquet"
    if not seq_path.exists():
        raise FileNotFoundError(f"Missing {seq_path}. Run src.sequence.data_prep first.")

    df = pl.read_parquet(seq_path).select(["user_idx", "seq", "target"])
    print(f"[OK] rows={df.height}")

    model, cfg = _load_model()
    device = _device()
    model.to(device)
    model.eval_mode()

    inferred_num_items = _infer_num_items()
    if inferred_num_items != cfg.num_items:
        print(
            f"[WARN] num_items mismatch: cfg={cfg.num_items} inferred={inferred_num_items}. "
            "Using cfg value for model inference."
        )

    users = df["user_idx"].to_numpy().astype(np.int64)
    seqs = df["seq"].to_list()

    x = _pad_sequences(seqs, cfg.max_seq_len)

    # M1/MPS stability rule:
    # DataLoader multiprocessing can be flaky + slower for small local inference.
    if num_workers is None:
        num_workers = 0

    ds = SeqCandidateDataset(users, x, seqs)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=False,
    )

    out_users: List[int] = []
    out_cands: List[List[int]] = []
    out_scores: List[List[float]] = []

    num_items = cfg.num_items

    print(
        f"[START] Generating top-{topk} sequence candidates "
        f"(filter_seen={filter_seen}, num_workers={num_workers})..."
    )

    with torch.no_grad():
        for uids, xb, seq_list in tqdm(loader, desc=f"Seq candidates ({split})", leave=True):
            xb = xb.to(device)

            logits = model.model(xb)  # (B, num_items)

            raw_k = topk + 50 if filter_seen else topk
            raw_k = min(raw_k, num_items)

            vals, idxs = torch.topk(logits, k=raw_k, dim=1)

            idxs_np = idxs.cpu().numpy()
            vals_np = vals.cpu().numpy()

            for uid, cand_row, score_row, seen_seq in zip(
                uids.tolist(), idxs_np, vals_np, seq_list, strict=False
            ):
                seen_set = set(int(x) for x in seen_seq) if filter_seen else set()

                filtered_cands: List[int] = []
                filtered_scores: List[float] = []

                for i, s in zip(cand_row, score_row):
                    ii = int(i)
                    if ii == 0:
                        continue
                    if ii in seen_set:
                        continue
                    filtered_cands.append(ii)
                    filtered_scores.append(float(s))
                    if len(filtered_cands) >= topk:
                        break

                # If filtering was too aggressive, fill without seen constraint
                if len(filtered_cands) < topk:
                    for i, s in zip(cand_row, score_row):
                        ii = int(i)
                        if ii == 0:
                            continue
                        if ii in filtered_cands:
                            continue
                        filtered_cands.append(ii)
                        filtered_scores.append(float(s))
                        if len(filtered_cands) >= topk:
                            break

                out_users.append(int(uid))
                out_cands.append(filtered_cands)
                out_scores.append(filtered_scores)

    out_df = pl.DataFrame(
        {
            "user_idx": out_users,
            "candidates": out_cands,
            "seq_scores": out_scores,
        }
    )

    out_path = _processed_dir() / f"seq_candidates_v3_{split}.parquet"
    out_df.write_parquet(out_path)

    print("[DONE] Sequence candidates saved.")
    print(f"[PATH] {out_path}")


def main():
    generate_sequence_candidates(
        split="val",
        topk=200,
        filter_seen=True,
        batch_size=512,
        num_workers=0,  # explicit for M1 stability
    )


if __name__ == "__main__":
    main()