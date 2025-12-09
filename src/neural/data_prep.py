from __future__ import annotations

from typing import List, Tuple  # noqa: UP035

import polars as pl

from src.config.settings import settings
from src.neural.paths import TT_TRAIN, TT_VAL


def _path(name: str) -> str:
    return str(settings.PROCESSED_DIR / name)


def _load_conf_splits() -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Expected files from your earlier pipeline:
      - train_conf.parquet
      - val_conf.parquet

    Columns assumed (based on your confidence pipeline):
      - user_idx (int)
      - item_idx (int)
      - conf or confidence (float)
      - ts or timestamp (int)

    We load robustly and normalize column names.
    """
    train_path = _path("train_conf.parquet")
    val_path = _path("val_conf.parquet")

    train = pl.read_parquet(train_path)
    val = pl.read_parquet(val_path)

    # Normalize column names
    rename_map = {}
    cols = set(train.columns)

    if "confidence" in cols and "conf" not in cols:
        rename_map["confidence"] = "conf"
    if "timestamp" in cols and "ts" not in cols:
        rename_map["timestamp"] = "ts"

    if rename_map:
        train = train.rename(rename_map)

    cols_v = set(val.columns)
    rename_map_v = {}
    if "confidence" in cols_v and "conf" not in cols_v:
        rename_map_v["confidence"] = "conf"
    if "timestamp" in cols_v and "ts" not in cols_v:
        rename_map_v["timestamp"] = "ts"
    if rename_map_v:
        val = val.rename(rename_map_v)

    # Ensure required fields exist
    for df, tag in [(train, "train_conf"), (val, "val_conf")]:
        missing = [c for c in ["user_idx", "item_idx"] if c not in df.columns]
        if missing:
            raise ValueError(f"{tag} missing columns: {missing}")

        if "conf" not in df.columns:
            # fallback: implicit confidence may be stored as "weight"
            if "weight" in df.columns:
                df = df.rename({"weight": "conf"})
            else:
                df = df.with_columns(pl.lit(1.0).alias("conf"))

        if "ts" not in df.columns:
            df = df.with_columns(pl.lit(0).alias("ts"))

        if tag == "train_conf":
            train = df
        else:
            val = df

    return train, val


def _cap_users_items(
    df: pl.DataFrame,
    max_users_cap: int,
    max_items_cap: int,
) -> tuple[List[int], List[int]]:
    """
    Cap the universe for local training:
      - Top users by interaction count
      - Top items by interaction count within that user set
    """
    user_counts = (
        df.group_by("user_idx")
        .len()
        .sort("len", descending=True)
        .head(max_users_cap)
        .select("user_idx")
    )
    users = user_counts["user_idx"].to_list()

    df_u = df.filter(pl.col("user_idx").is_in(users))

    item_counts = (
        df_u.group_by("item_idx")
        .len()
        .sort("len", descending=True)
        .head(max_items_cap)
        .select("item_idx")
    )
    items = item_counts["item_idx"].to_list()

    return users, items


def build_two_tower_datasets(
    max_users_cap: int = 50000,
    max_items_cap: int = 30000,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Builds two-tower training and validation datasets.

    Output schema:
      - user_idx
      - item_idx
      - conf
      - ts

    Notes:
    - This is positive-only pair data.
    - Negatives will be sampled inside the training loop in Step 8.3.
    - We cap users/items for M1 feasibility.
    """

    print("\n[START] Loading confidence splits for Two-Tower...")
    train_conf, val_conf = _load_conf_splits()
    print(f"[OK] train_conf rows: {train_conf.height}")
    print(f"[OK] val_conf rows:   {val_conf.height}")

    print("[START] Capping users/items for local training...")
    users, items = _cap_users_items(train_conf, max_users_cap, max_items_cap)
    print(f"[OK] users after cap: {len(users)}")
    print(f"[OK] items after cap: {len(items)}")

    train_df = (
        train_conf.filter(pl.col("user_idx").is_in(users) & pl.col("item_idx").is_in(items))
        .select(["user_idx", "item_idx", "conf", "ts"])
    )

    val_df = (
        val_conf.filter(pl.col("user_idx").is_in(users) & pl.col("item_idx").is_in(items))
        .select(["user_idx", "item_idx", "conf", "ts"])
    )

    print(f"[OK] TT train rows (capped): {train_df.height}")
    print(f"[OK] TT val rows (capped):   {val_df.height}")

    TT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    train_df.write_parquet(TT_TRAIN)
    val_df.write_parquet(TT_VAL)

    print("[DONE] Two-tower datasets created.")
    print(f"[PATH] {TT_TRAIN}")
    print(f"[PATH] {TT_VAL}")

    return train_df, val_df


def main():
    build_two_tower_datasets()


if __name__ == "__main__":
    main()