from __future__ import annotations

from pathlib import Path
from typing import List  # noqa: UP035

import numpy as np
import polars as pl

from src.config.settings import settings
from src.neural.two_tower import TwoTowerModel


def _processed_dir() -> Path:
    return Path(settings.PROCESSED_DIR)


def export_embeddings(
    model_path: str = "reports/models/two_tower_v3.pt",
    out_user: str = "user_emb_v3.parquet",
    out_item: str = "item_emb_v3.parquet",
):
    print("\n[START] Loading two-tower model...")
    model = TwoTowerModel.load(model_path)
    print("[OK] Model loaded.")

    # Load capped vocab from two-tower train parquet to ensure we export only relevant ids.
    tt_train_path = _processed_dir() / "two_tower_train.parquet"
    df = pl.read_parquet(tt_train_path).select(["user_idx", "item_idx"]).unique()

    users = df["user_idx"].unique().sort().to_list()
    items = df["item_idx"].unique().sort().to_list()

    print(f"[OK] Unique users to export: {len(users)}")
    print(f"[OK] Unique items to export: {len(items)}")

    # Batch embedding export to keep memory stable
    def batched(ids: List[int], batch_size: int = 4096):
        for i in range(0, len(ids), batch_size):
            yield ids[i : i + batch_size]

    print("[START] Exporting user embeddings...")
    user_rows = []
    for b in batched(users):
        emb = model.embed_users(b)  # (B, D)
        for uid, vec in zip(b, emb):
            user_rows.append((int(uid), vec.astype(np.float32)))

    print("[START] Exporting item embeddings...")
    item_rows = []
    for b in batched(items):
        emb = model.embed_items(b)
        for iid, vec in zip(b, emb):
            item_rows.append((int(iid), vec.astype(np.float32)))

    # Convert to Polars with list vectors
    user_df = pl.DataFrame(
        {
            "user_idx": [r[0] for r in user_rows],
            "embedding": [r[1].tolist() for r in user_rows],
        }
    )

    item_df = pl.DataFrame(
        {
            "item_idx": [r[0] for r in item_rows],
            "embedding": [r[1].tolist() for r in item_rows],
        }
    )

    out_user_path = _processed_dir() / out_user
    out_item_path = _processed_dir() / out_item

    user_df.write_parquet(out_user_path)
    item_df.write_parquet(out_item_path)

    print("[DONE] Embeddings exported.")
    print(f"[PATH] {out_user_path}")
    print(f"[PATH] {out_item_path}")


def main():
    export_embeddings()


if __name__ == "__main__":
    main()