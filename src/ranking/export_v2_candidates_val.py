from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple  # noqa: UP035

import polars as pl

from src.config.settings import settings


def _processed_dir() -> Path:
    return Path(settings.PROCESSED_DIR)


def _load_conf_splits():
    train_p = _processed_dir() / "train_conf.parquet"
    val_p = _processed_dir() / "val_conf.parquet"

    if not train_p.exists() or not val_p.exists():
        raise FileNotFoundError(
            "Missing confidence splits. Expected train_conf.parquet and val_conf.parquet "
            "from Step 3/4 pipeline."
        )

    train_conf = pl.read_parquet(train_p)
    val_conf = pl.read_parquet(val_p)

    return train_conf, val_conf


def _get_val_users(cap: int = 50000) -> List[int]:
    _, val_conf = _load_conf_splits()
    users = (
        val_conf.select("user_idx")
        .unique()
        .sort("user_idx")
        .head(cap)["user_idx"]
        .to_list()
    )
    return [int(u) for u in users]


def _safe_import_models():
    """
    We expect these to exist from earlier MVP steps.
    We keep the script robust in case an optional model isn't present.
    """
    from src.models.item_item import ItemItemSimilarityRecommender
    from src.models.popularity import PopularityRecommender

    genre_cls = None
    try:
        from src.models.genre_neighborhood import GenreNeighborhoodRecommender  # type: ignore
        genre_cls = GenreNeighborhoodRecommender
    except Exception:
        genre_cls = None

    return PopularityRecommender, ItemItemSimilarityRecommender, genre_cls


def main(max_k: int = 200, user_cap: int = 50000):
    print("\n[START] Exporting V2 candidates for val...")

    train_conf, _ = _load_conf_splits()
    users = _get_val_users(cap=user_cap)

    print(f"[OK] train_conf rows: {train_conf.height}")
    print(f"[OK] val users after cap: {len(users)}")

    PopularityRecommender, ItemItemSimilarityRecommender, GenreCls = _safe_import_models()

    print("[START] Fitting V2 candidate models...")
    pop = PopularityRecommender().fit()
    item_item = ItemItemSimilarityRecommender().fit()

    genre_nb = None
    if GenreCls is not None:
        try:
            genre_nb = GenreCls().fit()
            print("[OK] genre neighborhood model ready.")
        except Exception:
            genre_nb = None

    rows_user: List[int] = []
    rows_items: List[List[int]] = []
    rows_scores: List[List[float]] = []
    rows_source: List[List[str]] = []

    print("[START] Generating per-user V2 candidates...")

    # Conservative local loop; previous steps show this runs fine on M1.
    for u in users:
        u_items: List[int] = []
        u_scores: List[float] = []
        u_sources: List[str] = []

        # Popularity
        try:
            recs = pop.recommend(u, k=max_k)
            for r in recs:
                u_items.append(int(r))
                u_scores.append(1.0)  # popularity baseline score proxy
                u_sources.append("pop")
        except Exception:
            pass

        # Item-item
        try:
            recs = item_item.recommend(u, k=max_k)
            for r in recs:
                u_items.append(int(r))
                u_scores.append(1.0)
                u_sources.append("item_item")
        except Exception:
            pass

        # Genre neighborhood (optional)
        if genre_nb is not None:
            try:
                recs = genre_nb.recommend(u, k=max_k)
                for r in recs:
                    u_items.append(int(r))
                    u_scores.append(1.0)
                    u_sources.append("genre_nb")
            except Exception:
                pass

        # De-duplicate while preserving earliest source order
        seen = set()
        dedup_items: List[int] = []
        dedup_scores: List[float] = []
        dedup_sources: List[str] = []

        for i, s, src in zip(u_items, u_scores, u_sources):
            if i in seen:
                continue
            seen.add(i)
            dedup_items.append(i)
            dedup_scores.append(s)
            dedup_sources.append(src)
            if len(dedup_items) >= max_k:
                break

        rows_user.append(int(u))
        rows_items.append(dedup_items)
        rows_scores.append(dedup_scores)
        rows_source.append(dedup_sources)

    out = pl.DataFrame(
        {
            "user_idx": rows_user,
            "candidates": rows_items,
            "v2_scores": rows_scores,
            "v2_sources": rows_source,
        }
    )

    out_path = _processed_dir() / "v2_candidates_val.parquet"
    out.write_parquet(out_path)

    print("[DONE] V2 candidates exported.")
    print(f"[PATH] {out_path}")


if __name__ == "__main__":
    main()