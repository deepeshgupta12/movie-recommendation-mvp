from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple  # noqa: UP035

import numpy as np
import polars as pl

from src.config.settings import settings
from src.sequence.model_gru_v3 import GRUSeqConfig, GRUSeqModel


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


def _set_seed(seed: int):
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_seq_df(kind: str) -> pl.DataFrame:
    p = _processed_dir() / f"user_seq_{kind}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run src.sequence.data_prep first.")
    return pl.read_parquet(p).select(["user_idx", "seq", "target"])


def _pad_sequences(seqs: List[List[int]], max_len: int) -> np.ndarray:
    """
    Pads with 0 on the left to keep the most recent events.
    0 is reserved as padding index.
    """
    out = np.zeros((len(seqs), max_len), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[-max_len:]
        out[i, -len(s):] = np.array(s, dtype=np.int64)
    return out


class _SeqDataset:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def main():
    print("\n[START] Loading sequence datasets...")
    train_df = _load_seq_df("train")
    val_df = _load_seq_df("val")

    print(f"[OK] train rows: {train_df.height}")
    print(f"[OK] val rows:   {val_df.height}")

    # Infer num_items from dim_items if available, else from seq+target max
    dim_items_path = _processed_dir() / "dim_items.parquet"
    if dim_items_path.exists():
        dim_items = pl.read_parquet(dim_items_path)
        num_items = int(dim_items["item_idx"].max()) + 1
    else:
        max_seq = int(train_df.select(pl.col("seq").list.max()).to_series().max())
        max_tgt = int(train_df["target"].max())
        num_items = max(max_seq, max_tgt) + 1

    config = GRUSeqConfig(
        num_items=num_items,
        embed_dim=64,
        hidden_dim=128,
        max_seq_len=50,
        batch_size=512,
        epochs=2,
        lr=1e-3,
        seed=42,
    )

    _set_seed(config.seed)

    print(f"[OK] num_items={config.num_items}")

    # Build numpy tensors
    train_seqs = train_df["seq"].to_list()
    train_tgts = train_df["target"].to_numpy().astype(np.int64)

    val_seqs = val_df["seq"].to_list()
    val_tgts = val_df["target"].to_numpy().astype(np.int64)

    x_train = _pad_sequences(train_seqs, config.max_seq_len)
    x_val = _pad_sequences(val_seqs, config.max_seq_len)

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    train_ds = _SeqDataset(x_train, train_tgts)
    val_ds = _SeqDataset(x_val, val_tgts)

    cpu_ct = os.cpu_count() or 8
    num_workers = min(2, max(1, cpu_ct // 4))

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    device = _device()
    model = GRUSeqModel(config).to(device)

    opt = torch.optim.Adam(model.model.parameters(), lr=config.lr)

    print(f"[OK] device={device}")
    print("[START] Training GRU sequence model...")

    best_val = 1e18

    for epoch in range(1, config.epochs + 1):
        model.train_mode()
        running = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc=f"Seq Epoch {epoch}/{config.epochs}", leave=True)
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model.model(xb)
            loss = F.cross_entropy(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.item())
            steps += 1

            if steps % 50 == 0:
                pbar.set_postfix({"loss": running / steps})

        train_loss = running / max(1, steps)

        # Validation
        model.eval_mode()
        v_running = 0.0
        v_steps = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model.model(xb)
                loss = F.cross_entropy(logits, yb)
                v_running += float(loss.item())
                v_steps += 1

        val_loss = v_running / max(1, v_steps)

        print(f"[OK] epoch={epoch} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss

    # Save model
    out_dir = _reports_models_dir()
    model_path = out_dir / "sequence_gru_v3.pt"
    meta_path = out_dir / "sequence_gru_v3.meta.json"

    torch.save(
        {
            "state_dict": model.model.state_dict(),
            "config": config.__dict__,
        },
        model_path,
    )

    meta = {
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "num_items": config.num_items,
        "embed_dim": config.embed_dim,
        "hidden_dim": config.hidden_dim,
        "max_seq_len": config.max_seq_len,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "lr": config.lr,
        "seed": config.seed,
        "best_val_loss": best_val,
        "train_source": str(_processed_dir() / "user_seq_train.parquet"),
        "val_source": str(_processed_dir() / "user_seq_val.parquet"),
    }

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[DONE] Sequence GRU model trained and saved.")
    print(f"[PATH] {model_path}")
    print(f"[PATH] {meta_path}")


if __name__ == "__main__":
    main()