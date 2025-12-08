from __future__ import annotations

import polars as pl

from src.config.settings import settings


def main() -> None:
    path = settings.PROCESSED_DIR / "interactions.parquet"
    if not path.exists():
        raise FileNotFoundError("interactions.parquet not found. Run prepare_interactions first.")

    df = pl.read_parquet(path)

    # Sort by user and time
    df = df.sort(["user_idx", "timestamp"])

    # Last event per user -> test
    # Second last -> val
    # Rest -> train
    df = df.with_columns(
        pl.col("timestamp").rank("ordinal").over("user_idx").alias("user_time_rank"),
        pl.len().over("user_idx").alias("user_event_count")
    )

    test = df.filter(pl.col("user_time_rank") == pl.col("user_event_count"))
    val = df.filter(pl.col("user_time_rank") == (pl.col("user_event_count") - 1))
    train = df.filter(pl.col("user_time_rank") <= (pl.col("user_event_count") - 2))

    train_out = settings.PROCESSED_DIR / "train.parquet"
    val_out = settings.PROCESSED_DIR / "val.parquet"
    test_out = settings.PROCESSED_DIR / "test.parquet"

    train.write_parquet(train_out)
    val.write_parquet(val_out)
    test.write_parquet(test_out)

    print("[DONE] time-based split created:")
    print(f"  train: {train.height}")
    print(f"  val:   {val.height}")
    print(f"  test:  {test.height}")


if __name__ == "__main__":
    main()