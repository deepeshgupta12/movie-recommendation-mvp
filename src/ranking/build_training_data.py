from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple  # noqa: UP035

import numpy as np
import polars as pl
from tqdm import tqdm

from src.config.settings import settings


@dataclass
class RankingSampleConfig:
    negatives_per_positive: int = 4
    min_positive_conf: float = 0.1
    seed: int = 42
    top_pop_pool: int = 5000  # negative sampling pool size


def _load_movies() -> pl.DataFrame:
    return pl.read_parquet(settings.PROCESSED_DIR / "movies.parquet").select(
        pl.col("movieId"),
        pl.col("title"),
        pl.col("genres")
    )


def _load_items_map() -> pl.DataFrame:
    return pl.read_parquet(settings.PROCESSED_DIR / "items.parquet")


def _build_item_popularity(train_conf: pl.DataFrame) -> pl.DataFrame:
    return (
        train_conf.group_by("item_idx")
        .agg(pl.col("confidence").sum().alias("pop_conf_sum"))
        .sort("pop_conf_sum", descending=True)
    )


def main() -> None:
    cfg = RankingSampleConfig()

    train_path = settings.PROCESSED_DIR / "train_conf.parquet"
    if not train_path.exists():
        raise FileNotFoundError("train_conf.parquet not found.")

    print("[START] Loading train_conf...")
    train = pl.read_parquet(train_path).select(
        "user_idx", "item_idx", "confidence", "timestamp"
    )
    print(f"[OK] train_conf rows: {train.height}")

    print("[START] Filtering positives...")
    pos = train.filter(pl.col("confidence") >= cfg.min_positive_conf)
    print(f"[OK] positives rows: {pos.height}")

    print("[START] Building popularity prior...")
    pop = _build_item_popularity(train)
    top_pop_items = pop["item_idx"].head(cfg.top_pop_pool).to_list()
    print(f"[OK] popularity pool size: {len(top_pop_items)}")

    print("[START] Building user->positive map...")
    user_pos: Dict[int, Set[int]] = {}
    # This loop is large; show progress
    for u, i in tqdm(
        pos.select("user_idx", "item_idx").iter_rows(),
        total=pos.height,
        desc="Collecting user positives"
    ):
        user_pos.setdefault(int(u), set()).add(int(i))

    print(f"[OK] users with >=1 positive: {len(user_pos)}")

    rng = np.random.default_rng(cfg.seed)

    def pop_bucket_from_rank(rank: int) -> int:
        if rank <= 100:
            return 1
        if rank <= 500:
            return 2
        if rank <= 2000:
            return 3
        return 4

    # Precompute rank for bucket assignment
    pop_rank = {int(i): r for r, i in enumerate(pop["item_idx"].to_list(), start=1)}

    def pop_bucket(item_idx: int) -> int:
        r = pop_rank.get(item_idx, 999999)
        return pop_bucket_from_rank(r)

    print("[START] Creating ranking samples...")
    rows: List[Tuple[int, int, float, int]] = []

    # Show progress over users
    for u, items in tqdm(user_pos.items(), desc="Sampling negatives per user"):
        items_list = list(items)

        for i in items_list:
            # Positive row
            rows.append((u, i, 1.0, pop_bucket(i)))

            # Negatives
            negs: List[int] = []
            attempts = 0
            target = cfg.negatives_per_positive

            while len(negs) < target and attempts < target * 10:
                cand = int(rng.choice(top_pop_items))
                attempts += 1
                if cand not in items:
                    negs.append(cand)

            for n in negs:
                rows.append((u, n, 0.0, pop_bucket(n)))

    print(f"[OK] Total samples built in memory: {len(rows)}")

    samples = pl.DataFrame(
        rows, schema=["user_idx", "item_idx", "label", "pop_bucket"]
    )

    out = settings.PROCESSED_DIR / "rank_train_pairs.parquet"
    samples.write_parquet(out)

    print(f"[DONE] Ranking pairs created: {samples.height} rows")
    print(f"[PATH] {out}")


if __name__ == "__main__":
    main()