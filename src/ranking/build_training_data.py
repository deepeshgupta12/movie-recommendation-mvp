from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple  # noqa: UP035

import numpy as np
import polars as pl
from tqdm import tqdm

from src.config.settings import settings


@dataclass
class RankingSampleConfig:
    # Local-first defaults for M1 stability
    negatives_per_positive: int = 2
    min_positive_conf: float = 0.1
    seed: int = 42
    top_pop_pool: int = 5000

    # NEW: caps for local iteration
    max_users: int = 50000
    max_positives_per_user: int = 50


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
    for u, i in tqdm(
        pos.select("user_idx", "item_idx").iter_rows(),
        total=pos.height,
        desc="Collecting user positives"
    ):
        user_pos.setdefault(int(u), set()).add(int(i))

    all_users = list(user_pos.keys())
    print(f"[OK] users with >=1 positive (raw): {len(all_users)}")

    # Apply user cap
    if len(all_users) > cfg.max_users:
        all_users = all_users[:cfg.max_users]
        print(f"[OK] users after cap: {len(all_users)}")

    rng = np.random.default_rng(cfg.seed)

    # Precompute pop rank for bucket assignment
    pop_rank = {int(i): r for r, i in enumerate(pop["item_idx"].to_list(), start=1)}

    def pop_bucket(item_idx: int) -> int:
        r = pop_rank.get(item_idx, 999999)
        if r <= 100:
            return 1
        if r <= 500:
            return 2
        if r <= 2000:
            return 3
        return 4

    print("[START] Creating ranking samples (capped for local)...")
    rows: List[Tuple[int, int, float, int]] = []

    for u in tqdm(all_users, desc="Sampling negatives per user"):
        items = list(user_pos.get(u, set()))
        if not items:
            continue

        # Apply positives-per-user cap
        if len(items) > cfg.max_positives_per_user:
            items = items[:cfg.max_positives_per_user]

        for i in items:
            # Positive
            rows.append((u, i, 1.0, pop_bucket(i)))

            # Negatives
            negs: List[int] = []
            attempts = 0
            target = cfg.negatives_per_positive

            while len(negs) < target and attempts < target * 10:
                cand = int(rng.choice(top_pop_items))
                attempts += 1
                if cand not in user_pos[u]:
                    negs.append(cand)

            for n in negs:
                rows.append((u, n, 0.0, pop_bucket(n)))

    print(f"[OK] Total samples built in memory: {len(rows)}")

    samples = pl.DataFrame(
        rows,
        schema=["user_idx", "item_idx", "label", "pop_bucket"],
        orient="row",
    )

    out = settings.PROCESSED_DIR / "rank_train_pairs.parquet"
    samples.write_parquet(out)

    print(f"[DONE] Ranking pairs created: {samples.height} rows")
    print(f"[PATH] {out}")


if __name__ == "__main__":
    main()