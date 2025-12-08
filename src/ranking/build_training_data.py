from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple  # noqa: UP035

import numpy as np
import polars as pl

from src.config.settings import settings


@dataclass
class RankingSampleConfig:
    negatives_per_positive: int = 4
    min_positive_conf: float = 0.1
    seed: int = 42


def _load_movies() -> pl.DataFrame:
    return pl.read_parquet(settings.PROCESSED_DIR / "movies.parquet").select(
        pl.col("movieId"),
        pl.col("title"),
        pl.col("genres")
    )


def _load_items_map() -> pl.DataFrame:
    return pl.read_parquet(settings.PROCESSED_DIR / "items.parquet")


def _load_users_map() -> pl.DataFrame:
    return pl.read_parquet(settings.PROCESSED_DIR / "users.parquet")


def _build_item_popularity(train_conf: pl.DataFrame) -> pl.DataFrame:
    return (
        train_conf.group_by("item_idx")
        .agg(pl.col("confidence").sum().alias("pop_conf_sum"))
        .sort("pop_conf_sum", descending=True)
    )


def _parse_genres(genres: str) -> List[str]:
    if genres is None:
        return []
    return [g.strip() for g in genres.split("|") if g.strip()]


def main() -> None:
    cfg = RankingSampleConfig()

    train_path = settings.PROCESSED_DIR / "train_conf.parquet"
    if not train_path.exists():
        raise FileNotFoundError("train_conf.parquet not found.")

    rng = np.random.default_rng(cfg.seed)

    train = pl.read_parquet(train_path).select(
        "user_idx", "item_idx", "confidence", "timestamp"
    )

    # Positives
    pos = train.filter(pl.col("confidence") >= cfg.min_positive_conf)

    # Popularity prior for negative sampling
    pop = _build_item_popularity(train)
    top_pop_items = pop["item_idx"].head(5000).to_list()

    # Build user -> positives set
    user_pos: Dict[int, Set[int]] = {}
    for u, i in pos.select("user_idx", "item_idx").iter_rows():
        user_pos.setdefault(int(u), set()).add(int(i))

    # Load item metadata + stable maps
    items_map = _load_items_map()
    movies = _load_movies()

    # Join item_idx -> movieId -> genres
    item_meta = (
        items_map.join(movies, on="movieId", how="left")
        .select("item_idx", "movieId", "genres")
    )

    item_genres = {
        int(row[0]): _parse_genres(row[2]) if row[2] is not None else []
        for row in item_meta.iter_rows()
    }

    # Create samples
    rows: List[Tuple[int, int, float, int]] = []
    # (user_idx, item_idx, label, pop_rank_bucket)

    # Precompute popularity rank buckets
    pop_rank = {int(i): rank for rank, i in enumerate(pop["item_idx"].to_list(), start=1)}

    def pop_bucket(item_idx: int) -> int:
        r = pop_rank.get(item_idx, 999999)
        if r <= 100:
            return 1
        if r <= 500:
            return 2
        if r <= 2000:
            return 3
        return 4

    for u, items in user_pos.items():
        for i in items:
            rows.append((u, i, 1.0, pop_bucket(i)))

            # Negatives
            negs = []
            attempts = 0
            while len(negs) < cfg.negatives_per_positive and attempts < cfg.negatives_per_positive * 10:
                cand = int(rng.choice(top_pop_items))
                attempts += 1
                if cand not in items:
                    negs.append(cand)

            for n in negs:
                rows.append((u, n, 0.0, pop_bucket(n)))

    samples = pl.DataFrame(
        rows, schema=["user_idx", "item_idx", "label", "pop_bucket"]
    )

    out = settings.PROCESSED_DIR / "rank_train_pairs.parquet"
    samples.write_parquet(out)

    print(f"[DONE] Ranking pairs created: {samples.height} rows")
    print(f"[PATH] {out}")


if __name__ == "__main__":
    main()