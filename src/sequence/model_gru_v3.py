from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple  # noqa: UP035

import numpy as np


@dataclass
class GRUSeqConfig:
    """
    GRU-based next-item sequence model for V3 MVP.

    Local-first defaults for M1.
    """
    num_items: int
    embed_dim: int = 64
    hidden_dim: int = 128
    max_seq_len: int = 50

    batch_size: int = 512
    epochs: int = 2
    lr: float = 1e-3

    seed: int = 42


class GRUSeqModel:
    """
    Minimal GRU next-item predictor.

    Input: padded sequence of item_idx
    Output: logits over item vocabulary
    """

    def __init__(self, config: GRUSeqConfig):
        self.config = config
        self.model = self._build()

    def _build(self):
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self, cfg: GRUSeqConfig):
                super().__init__()
                self.cfg = cfg
                self.item_emb = nn.Embedding(cfg.num_items, cfg.embed_dim, padding_idx=0)
                self.gru = nn.GRU(
                    input_size=cfg.embed_dim,
                    hidden_size=cfg.hidden_dim,
                    batch_first=True,
                )
                self.out = nn.Linear(cfg.hidden_dim, cfg.num_items)

                nn.init.xavier_uniform_(self.item_emb.weight)
                nn.init.xavier_uniform_(self.out.weight)

            def forward(self, x):
                # x: (B, L)
                emb = self.item_emb(x)          # (B, L, D)
                h, _ = self.gru(emb)            # (B, L, H)
                last = h[:, -1, :]              # (B, H)
                logits = self.out(last)         # (B, num_items)
                return logits

        return Net(self.config)

    def to(self, device):
        self.model.to(device)
        return self

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()