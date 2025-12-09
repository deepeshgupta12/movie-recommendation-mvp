from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple  # noqa: UP035

import numpy as np
import polars as pl

from src.config.settings import settings
from src.neural.contracts import TwoTowerArtifacts, TwoTowerConfig

# =========================================================
# Safe reports directory resolution
# =========================================================

def _resolve_reports_dir() -> Path:
    """
    Robust resolver that avoids hard dependency on Settings.REPORTS_DIR.

    Priority order:
    1) settings.REPORTS_DIR (if defined)
    2) settings.BASE_DIR / "reports" (if BASE_DIR exists)
    3) Path("reports") relative to repo root
    """
    rep = getattr(settings, "REPORTS_DIR", None)
    if rep:
        return Path(rep)

    base = getattr(settings, "BASE_DIR", None)
    if base:
        return Path(base) / "reports"

    return Path("reports")


REPORTS_DIR = _resolve_reports_dir()
MODEL_DIR = REPORTS_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL_PATH = MODEL_DIR / "two_tower_v3.pt"
DEFAULT_META_PATH = MODEL_DIR / "two_tower_v3.meta.json"


# =========================================================
# Torch lazy import helpers
# =========================================================

def _lazy_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


def _device():
    torch, _, _ = _lazy_torch()
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_seed(seed: int):
    torch, _, _ = _lazy_torch()
    np.random.seed(seed)
    torch.manual_seed(seed)


# =========================================================
# Model definition
# =========================================================

class _TwoTowerNet:
    """
    Minimal two-tower embedding model.
    """

    def __init__(self, num_users: int, num_items: int, embed_dim: int):
        torch, nn, _ = _lazy_torch()

        class TT(nn.Module):
            def __init__(self, nu: int, ni: int, d: int):
                super().__init__()
                self.user_emb = nn.Embedding(nu, d)
                self.item_emb = nn.Embedding(ni, d)
                nn.init.xavier_uniform_(self.user_emb.weight)
                nn.init.xavier_uniform_(self.item_emb.weight)

        self.model = TT(num_users, num_items, embed_dim)

    def to(self, device):
        self.model.to(device)
        return self

    def parameters(self):
        return self.model.parameters()


# =========================================================
# Data utilities
# =========================================================

def _load_pairs(path: str) -> pl.DataFrame:
    df = pl.read_parquet(path)
    needed = {"user_idx", "item_idx"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Two-tower pairs missing columns: {missing}")

    if "conf" not in df.columns:
        df = df.with_columns(pl.lit(1.0).alias("conf"))
    if "ts" not in df.columns:
        df = df.with_columns(pl.lit(0).alias("ts"))

    return df.select(["user_idx", "item_idx", "conf", "ts"])


def _cap_positive_pairs_per_user(
    df: pl.DataFrame,
    max_pos_per_user: int,
) -> pl.DataFrame:
    """
    Local-safe cap:
    keep last N positives per user based on ts.
    """
    if max_pos_per_user <= 0:
        return df

    df = df.sort(["user_idx", "ts"])
    capped = df.group_by("user_idx", maintain_order=True).tail(max_pos_per_user)
    return capped


def _infer_vocab_sizes(train_df: pl.DataFrame) -> Tuple[int, int]:
    """
    Embedding sizes inferred from max index + 1.
    We assume indices are already normalized by earlier pipeline.
    """
    max_user = int(train_df["user_idx"].max())
    max_item = int(train_df["item_idx"].max())
    return max_user + 1, max_item + 1


# =========================================================
# Torch Dataset
# =========================================================

class _PairDataset:
    """
    Positive pairs dataset.
    Negatives are sampled during training.
    """

    def __init__(self, users: np.ndarray, items: np.ndarray, conf: np.ndarray):
        self.users = users.astype(np.int64, copy=False)
        self.items = items.astype(np.int64, copy=False)
        self.conf = conf.astype(np.float32, copy=False)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.conf[idx]


# =========================================================
# TwoTowerModel
# =========================================================

class TwoTowerModel:
    """
    V3 Two-Tower retrieval model (local MVP).

    - Dot-product of user/item embeddings
    - On-the-fly negative sampling
    - Per-user positive caps for M1 feasibility
    - MPS/CPU runtime support
    """

    def __init__(self, config: Optional[TwoTowerConfig] = None):
        self.config = config or TwoTowerConfig()
        self._net: Optional[_TwoTowerNet] = None
        self._num_users: Optional[int] = None
        self._num_items: Optional[int] = None

    def fit(
        self,
        train_df: pl.DataFrame,
        val_df: Optional[pl.DataFrame] = None,
        max_pos_per_user: Optional[int] = None,
        model_path: str = str(DEFAULT_MODEL_PATH),
        meta_path: str = str(DEFAULT_META_PATH),
    ) -> TwoTowerArtifacts:
        torch, _, F = _lazy_torch()
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        _set_seed(self.config.seed)

        print("\n[START] Two-Tower training (V3)")

        cap = max_pos_per_user if max_pos_per_user is not None else self.config.max_pos_per_user
        print(f"[START] Capping positives per user to {cap}...")
        train_df = _cap_positive_pairs_per_user(train_df, max_pos_per_user=cap)
        print(f"[OK] Train rows after per-user cap: {train_df.height}")

        num_users, num_items = _infer_vocab_sizes(train_df)
        self._num_users, self._num_items = num_users, num_items
        print(f"[OK] num_users={num_users} | num_items={num_items}")

        self._net = _TwoTowerNet(num_users, num_items, self.config.embed_dim)
        device = _device()
        self._net.to(device)
        print(f"[OK] device={device}")

        users = train_df["user_idx"].to_numpy()
        items = train_df["item_idx"].to_numpy()
        conf = train_df["conf"].to_numpy()

        dataset = _PairDataset(users, items, conf)

        cpu_ct = os.cpu_count() or 8
        num_workers = min(4, max(1, cpu_ct // 4))

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=True,
        )

        opt = torch.optim.Adam(self._net.parameters(), lr=self.config.lr)

        def sample_negatives(batch_size: int, num_negs: int) -> torch.Tensor:
            return torch.randint(0, num_items, (batch_size, num_negs), device=device)

        print("[START] Training loop...")

        for epoch in range(1, self.config.epochs + 1):
            self._net.model.train()
            running = 0.0
            steps = 0

            pbar = tqdm(loader, desc=f"Epoch {epoch}/{self.config.epochs}", leave=True)

            for u, pos_i, w in pbar:
                u = torch.tensor(u, device=device)
                pos_i = torch.tensor(pos_i, device=device)
                w = torch.tensor(w, device=device)

                batch_size = u.shape[0]
                neg_i = sample_negatives(batch_size, self.config.num_negatives)

                user_vec = self._net.model.user_emb(u)                     # (B, D)
                pos_vec = self._net.model.item_emb(pos_i)                  # (B, D)
                neg_vec = self._net.model.item_emb(neg_i)                  # (B, N, D)

                pos_scores = (user_vec * pos_vec).sum(dim=1)               # (B,)
                neg_scores = (user_vec.unsqueeze(1) * neg_vec).sum(dim=2)  # (B, N)

                # Implicit ranking loss
                pos_loss = F.logsigmoid(pos_scores).mean()
                neg_loss = F.logsigmoid(-neg_scores).mean()
                loss = -(pos_loss + neg_loss)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                running += float(loss.item())
                steps += 1

                if steps % 50 == 0:
                    pbar.set_postfix({"loss": running / steps})

            avg_loss = running / max(1, steps)
            print(f"[OK] Epoch {epoch} avg_loss={avg_loss:.6f}")

        self.save(model_path)

        meta = {
            "model_path": model_path,
            "embed_dim": self.config.embed_dim,
            "num_users": num_users,
            "num_items": num_items,
            "batch_size": self.config.batch_size,
            "epochs": self.config.epochs,
            "lr": self.config.lr,
            "num_negatives": self.config.num_negatives,
            "max_pos_per_user": cap,
            "seed": self.config.seed,
            "device_trained": str(device),
            "reports_dir_resolved": str(REPORTS_DIR),
        }
        Path(meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print("[DONE] Two-Tower model trained and saved.")
        print(f"[PATH] {model_path}")
        print(f"[PATH] {meta_path}")

        return TwoTowerArtifacts(
            model_path=model_path,
            meta_path=meta_path,
            config=self.config,
        )

    def save(self, path: str) -> str:
        torch, _, _ = _lazy_torch()
        if self._net is None or self._num_users is None or self._num_items is None:
            raise RuntimeError("Model not initialized.")

        obj = {
            "state_dict": self._net.model.state_dict(),
            "config": self.config.__dict__,
            "num_users": self._num_users,
            "num_items": self._num_items,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj, path)
        return path

    @classmethod
    def load(cls, path: str) -> TwoTowerModel:
        torch, _, _ = _lazy_torch()
        obj = torch.load(path, map_location="cpu")

        config = TwoTowerConfig(**obj.get("config", {}))
        model = cls(config=config)
        model._num_users = int(obj["num_users"])
        model._num_items = int(obj["num_items"])

        model._net = _TwoTowerNet(model._num_users, model._num_items, config.embed_dim)
        model._net.model.load_state_dict(obj["state_dict"])
        model._net.to(_device())
        model._net.model.eval()

        return model

    def embed_users(self, user_ids: List[int]) -> np.ndarray:
        torch, _, _ = _lazy_torch()
        if self._net is None:
            raise RuntimeError("Model not initialized.")

        device = _device()
        u = torch.tensor(user_ids, device=device, dtype=torch.long)

        with torch.no_grad():
            emb = self._net.model.user_emb(u).detach().cpu().numpy()

        return emb

    def embed_items(self, item_ids: List[int]) -> np.ndarray:
        torch, _, _ = _lazy_torch()
        if self._net is None:
            raise RuntimeError("Model not initialized.")

        device = _device()
        i = torch.tensor(item_ids, device=device, dtype=torch.long)

        with torch.no_grad():
            emb = self._net.model.item_emb(i).detach().cpu().numpy()

        return emb