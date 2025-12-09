from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional  # noqa: UP035

# =========================
# Two-Tower Retrieval
# =========================

@dataclass
class TwoTowerConfig:
    """
    Configuration contract for the V3 Two-Tower retrieval model.

    Local-first defaults tuned for Apple Silicon (M1).
    We keep this small and stable so that training, embedding export,
    and ANN indexing can rely on a consistent contract.

    Notes:
    - max_pos_per_user is critical for local feasibility.
    - num_negatives is used for on-the-fly negative sampling.
    """
    embed_dim: int = 64
    batch_size: int = 2048
    epochs: int = 2
    lr: float = 1e-3
    num_negatives: int = 10

    # Repro + local feasibility knobs
    seed: int = 42
    max_pos_per_user: int = 50


@dataclass
class TwoTowerArtifacts:
    """
    Training output contract for Two-Tower model.

    This meta is intentionally lightweight and file-path oriented
    so that downstream steps (embedding export, ANN build, evaluation)
    can proceed without importing training internals.
    """
    model_path: str
    meta_path: str
    config: TwoTowerConfig

    # Optional future outputs (safe for later expansion)
    user_embeddings_path: Optional[str] = None
    item_embeddings_path: Optional[str] = None
    ann_index_path: Optional[str] = None


# =========================
# Sequence Candidate Generator
# =========================

@dataclass
class SequenceConfig:
    """
    Configuration contract for the V3 sequence-based candidate generator.

    We keep this minimal in MVP:
    - next-item prediction from user history
    - compact embedding + GRU baseline

    This contract allows us to scale to transformers later
    without breaking the training entrypoint structure.
    """
    embed_dim: int = 64
    hidden_dim: int = 128
    max_seq_len: int = 50

    batch_size: int = 512
    epochs: int = 2
    lr: float = 1e-3

    # Repro + local feasibility knobs
    seed: int = 42
    max_users_cap: int = 50000


@dataclass
class SequenceArtifacts:
    """
    Training output contract for sequence model.
    """
    model_path: str
    meta_path: str
    config: SequenceConfig

    # Optional future outputs
    user_sequence_embeddings_path: Optional[str] = None


# =========================
# Public export
# =========================

__all__ = [
    "TwoTowerConfig",
    "TwoTowerArtifacts",
    "SequenceConfig",
    "SequenceArtifacts",
]