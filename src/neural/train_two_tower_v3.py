from __future__ import annotations

import polars as pl

from src.neural.contracts import TwoTowerConfig
from src.neural.paths import TT_TRAIN, TT_VAL
from src.neural.two_tower import TwoTowerModel


def main():
    print("\n[START] Loading two-tower datasets...")
    train_df = pl.read_parquet(TT_TRAIN)
    val_df = pl.read_parquet(TT_VAL)

    print(f"[OK] TT train rows: {train_df.height}")
    print(f"[OK] TT val rows:   {val_df.height}")

    # Local-safe configuration
    config = TwoTowerConfig(
        embed_dim=64,
        batch_size=2048,
        epochs=2,
        lr=1e-3,
        num_negatives=10,
        seed=42,
        max_pos_per_user=50,
    )

    model = TwoTowerModel(config=config)

    model.fit(
        train_df=train_df,
        val_df=val_df,
        max_pos_per_user=config.max_pos_per_user,
    )


if __name__ == "__main__":
    main()