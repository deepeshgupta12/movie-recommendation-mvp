from __future__ import annotations

import polars as pl

from src.neural.paths import ITEM_EMB, USER_EMB


def export_user_item_embeddings(
    model_path: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Contract-first placeholder.

    In Step 8.4 we will:
    - load trained two-tower model
    - generate embeddings for capped universe
    - write USER_EMB / ITEM_EMB

    Returns:
        (user_emb_df, item_emb_df)
    """
    raise NotImplementedError("Embedding export will be implemented in Step 8.4.")