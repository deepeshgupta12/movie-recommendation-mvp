from __future__ import annotations

import polars as pl

from src.config.settings import settings


def main() -> None:
    path = settings.PROCESSED_DIR / "implicit_confidence.parquet"
    if not path.exists():
        raise FileNotFoundError("implicit_confidence.parquet not found.")

    df = pl.read_parquet(path)

    df = df.sort(["user_idx", "timestamp"])

    df = df.with_columns(
        pl.col("timestamp").rank("ordinal").over("user_idx").alias("user_time_rank"),
        pl.len().over("user_idx").alias("user_event_count")
    )

    test = df.filter(pl.col("user_time_rank") == pl.col("user_event_count"))
    val = df.filter(pl.col("user_time_rank") == (pl.col("user_event_count") - 1))
    train = df.filter(pl.col("user_time_rank") <= (pl.col("user_event_count") - 2))

    train_out = settings.PROCESSED_DIR / "train_conf.parquet"
    val_out = settings.PROCESSED_DIR / "val_conf.parquet"
    test_out = settings.PROCESSED_DIR / "test_conf.parquet"

    train.write_parquet(train_out)
    val.write_parquet(val_out)
    test.write_parquet(test_out)

    print("[DONE] confidence time-based split created:")
    print(f"  train_conf: {train.height}")
    print(f"  val_conf:   {val.height}")
    print(f"  test_conf:  {test.height}")


if __name__ == "__main__":
    main()