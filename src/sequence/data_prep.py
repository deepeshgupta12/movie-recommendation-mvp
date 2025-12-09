from __future__ import annotations

from typing import List, Tuple  # noqa: UP035

import polars as pl

from src.config.settings import settings
from src.neural.paths import SEQ_TRAIN, SEQ_VAL


def _path(name: str) -> str:
    return str(settings.PROCESSED_DIR / name)


def _load_conf_splits() -> tuple[pl.DataFrame, pl.DataFrame]:
    train_path = _path("train_conf.parquet")
    val_path = _path("val_conf.parquet")

    train = pl.read_parquet(train_path)
    val = pl.read_parquet(val_path)

    # Normalize column names
    cols = set(train.columns)
    if "timestamp" in cols and "ts" not in cols:
        train = train.rename({"timestamp": "ts"})

    cols_v = set(val.columns)
    if "timestamp" in cols_v and "ts" not in cols_v:
        val = val.rename({"timestamp": "ts"})

    if "ts" not in train.columns:
        train = train.with_columns(pl.lit(0).alias("ts"))
    if "ts" not in val.columns:
        val = val.with_columns(pl.lit(0).alias("ts"))

    # Required columns
    for df, tag in [(train, "train_conf"), (val, "val_conf")]:
        missing = [c for c in ["user_idx", "item_idx"] if c not in df.columns]
        if missing:
            raise ValueError(f"{tag} missing columns: {missing}")

    return train, val


def _cap_users(df: pl.DataFrame, max_users_cap: int) -> List[int]:
    user_counts = (
        df.group_by("user_idx")
        .len()
        .sort("len", descending=True)
        .head(max_users_cap)
        .select("user_idx")
    )
    return user_counts["user_idx"].to_list()


def _build_last_item_examples(
    df: pl.DataFrame,
    max_seq_len: int,
) -> pl.DataFrame:
    """
    Builds one example per user:
      seq = previous items (up to max_seq_len)
      target = last item

    Output:
      - user_idx
      - seq (list[int32])
      - target (int32)
    """

    grouped = (
        df.sort(["user_idx", "ts"])
        .group_by("user_idx")
        .agg(pl.col("item_idx").implode().alias("items"))
        .with_columns(
            # enforce stable list dtype early
            pl.col("items").cast(pl.List(pl.Int32))
        )
    )

    def cut_seq(items: List[int]) -> List[int]:
        if items is None or len(items) <= 1:
            return []
        seq = items[:-1]
        if len(seq) > max_seq_len:
            seq = seq[-max_seq_len:]
        return seq

    def get_target(items: List[int]) -> int:
        if items is None or len(items) == 0:
            return -1
        return int(items[-1])

    out = (
        grouped.with_columns(
            pl.col("items").map_elements(
                cut_seq,
                return_dtype=pl.List(pl.Int32),
            ).alias("seq"),
            pl.col("items").map_elements(
                get_target,
                return_dtype=pl.Int32,
            ).alias("target"),
            pl.col("items").list.len().alias("len_items"),
        )
        .drop("items")
    )

    out = out.filter((pl.col("len_items") >= 2) & (pl.col("target") >= 0)).drop("len_items")
    return out


def build_sequence_datasets(
    max_seq_len: int = 50,
    max_users_cap: int = 50000,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Builds sequence datasets for GRU-style next-item modeling.

    Train examples:
      - Derived from train_conf sequences.

    Val examples:
      - Uses val_conf last items when possible,
        with sequence context from train_conf.

    Output schema:
      - user_idx
      - seq (list[int32])
      - target (int32)
    """

    print("\n[START] Loading confidence splits for Sequences...")
    train_conf, val_conf = _load_conf_splits()
    print(f"[OK] train_conf rows: {train_conf.height}")
    print(f"[OK] val_conf rows:   {val_conf.height}")

    print("[START] Capping users for local sequence build...")
    users = _cap_users(train_conf, max_users_cap)
    print(f"[OK] users after cap: {len(users)}")

    train_u = train_conf.filter(pl.col("user_idx").is_in(users))
    val_u = val_conf.filter(pl.col("user_idx").is_in(users))

    print("[START] Building per-user train sequences...")
    train_examples = _build_last_item_examples(train_u, max_seq_len=max_seq_len)
    print(f"[OK] train sequence rows: {train_examples.height}")

    # Validation:
    print("[START] Building per-user val sequences with train context...")

    train_group = (
        train_u.sort(["user_idx", "ts"])
        .group_by("user_idx")
        .agg(pl.col("item_idx").implode().alias("train_items"))
        .with_columns(pl.col("train_items").cast(pl.List(pl.Int32)))
    )

    val_last = (
        val_u.sort(["user_idx", "ts"])
        .group_by("user_idx")
        .agg(pl.col("item_idx").last().cast(pl.Int32).alias("val_target"))
    )

    merged = train_group.join(val_last, on="user_idx", how="inner")

    def build_val_seq(items: List[int]) -> List[int]:
        if items is None or len(items) == 0:
            return []
        seq = items
        if len(seq) > max_seq_len:
            seq = seq[-max_seq_len:]
        return seq

    val_examples = (
        merged.with_columns(
            pl.col("train_items").map_elements(
                build_val_seq,
                return_dtype=pl.List(pl.Int32),
            ).alias("seq"),
            pl.col("val_target").cast(pl.Int32).alias("target"),
            pl.col("train_items").list.len().alias("len_items"),
        )
        .drop("train_items")
    )

    val_examples = val_examples.filter(pl.col("len_items") >= 1).drop("len_items")
    print(f"[OK] val sequence rows: {val_examples.height}")

    SEQ_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    train_examples.write_parquet(SEQ_TRAIN)
    val_examples.write_parquet(SEQ_VAL)

    print("[DONE] Sequence datasets created.")
    print(f"[PATH] {SEQ_TRAIN}")
    print(f"[PATH] {SEQ_VAL}")

    return train_examples, val_examples


def main():
    build_sequence_datasets()


if __name__ == "__main__":
    main()