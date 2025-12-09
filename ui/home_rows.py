from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple  # noqa: UP035

import polars as pl
import streamlit as st

from src.config.settings import settings


@st.cache_data(show_spinner=False)
def load_items_table() -> pl.DataFrame:
    items_path = settings.PROCESSED_DIR / "items.parquet"
    movies_path = settings.PROCESSED_DIR / "movies.parquet"

    items = pl.read_parquet(items_path).select(["item_idx", "movieId"])
    movies = pl.read_parquet(movies_path).select(["movieId", "title", "genres"])

    df = items.join(movies, on="movieId", how="left")
    return df


@st.cache_data(show_spinner=False)
def load_item_features() -> pl.DataFrame:
    path = settings.PROCESSED_DIR / "item_features.parquet"
    return pl.read_parquet(path)


def get_title_genres(item_idx: int) -> Tuple[str, str]:
    df = load_items_table()
    row = df.filter(pl.col("item_idx") == int(item_idx)).head(1)
    if row.height == 0:
        return ("", "")
    return (row["title"][0] or "", row["genres"][0] or "")


def get_trending_items(limit: int = 20) -> List[Dict[str, Any]]:
    """
    Local trending approximation using item_features.
    Uses item_interactions and item_conf_sum to rank.
    """
    feats = load_item_features()
    items = load_items_table()

    merged = feats.join(items, on="item_idx", how="left")

    ranked = (
        merged
        .with_columns(
            (pl.col("item_interactions") * 1.0 + pl.col("item_conf_sum") * 0.1).alias("trend_score")
        )
        .sort("trend_score", descending=True)
        .head(limit)
    )

    out = []
    for r in ranked.iter_rows(named=True):
        out.append(
            {
                "item_idx": int(r["item_idx"]),
                "title": r.get("title") or "",
                "genres": r.get("genres") or "",
                "score": float(r.get("trend_score") or 0.0),
                "reasons": ["Trending now"],
            }
        )
    return out


def filter_recs_by_genres(
    recs: List[Dict[str, Any]],
    seed_genres: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    if not seed_genres:
        return recs[:limit]

    seed_set = {g.strip() for g in seed_genres.split("|") if g.strip()}

    scored = []
    for item in recs:
        genres = str(item.get("genres", "") or "")
        gset = {g.strip() for g in genres.split("|") if g.strip()}
        overlap = len(seed_set.intersection(gset))
        if overlap > 0:
            scored.append(item)

    return scored[:limit]