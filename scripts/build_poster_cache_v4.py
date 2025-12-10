from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple  # noqa: UP035

import polars as pl

# We assume these exist from your earlier V3/V4 wiring.
# If the import path differs in your repo, we can adjust later.
from src.service.poster_cache import PosterCache, fetch_tmdb_poster_by_title


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / ".git").exists() or (p / "pyproject.toml").exists():
            return p
    # fallback
    return here.parents[1]


def _read_any(p: Path) -> pl.DataFrame:
    if p.suffix.lower() == ".parquet":
        return pl.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pl.read_csv(p)
    raise ValueError(f"Unsupported file type: {p}")


def _find_mapping_file(processed: Path) -> Optional[Path]:
    """
    Try to locate an internal item mapping that already knows item_idx <-> movieId.
    """
    mapping_candidates = [
        processed / "item_id_map.parquet",
        processed / "item_id_map.csv",
        processed / "item_map.parquet",
        processed / "item_map.csv",
        processed / "idx_to_movieid.parquet",
        processed / "idx_to_movieid.csv",
        processed / "movieid_to_idx.parquet",
        processed / "movieid_to_idx.csv",
    ]
    for p in mapping_candidates:
        if p.exists():
            return p
    return None


def _normalize_mapping(df: pl.DataFrame) -> Optional[pl.DataFrame]:
    cols = set(df.columns)

    # Common variants
    rename_map = {}
    if "movie_id" in cols and "movieId" not in cols:
        rename_map["movie_id"] = "movieId"
    if "item_id" in cols and "item_idx" not in cols:
        rename_map["item_id"] = "item_idx"
    if "idx" in cols and "item_idx" not in cols:
        rename_map["idx"] = "item_idx"

    if rename_map:
        df = df.rename(rename_map)
        cols = set(df.columns)

    if {"item_idx", "movieId"}.issubset(cols):
        return df.select(["item_idx", "movieId"]).unique()

    return None


def _find_movies_file(data_dir: Path) -> Optional[Path]:
    """
    Search aggressively for MovieLens-style movie metadata.
    """
    # Direct common locations
    direct = [
        data_dir / "raw" / "movies.csv",
        data_dir / "raw" / "movies.parquet",
        data_dir / "processed" / "movies.csv",
        data_dir / "processed" / "movies.parquet",
    ]
    for p in direct:
        if p.exists():
            return p

    # Recursive search for anything named movies.csv/movies.parquet
    patterns = ["**/movies.csv", "**/movies.parquet"]
    for pat in patterns:
        matches = sorted(data_dir.glob(pat))
        if matches:
            return matches[0]

    # Slightly broader fallback
    broader = sorted(data_dir.glob("**/*movies*.csv"))
    if broader:
        return broader[0]

    return None


def _normalize_movies(df: pl.DataFrame) -> Optional[pl.DataFrame]:
    """
    Ensure we can get movieId + title.
    """
    cols = set(df.columns)

    rename_map = {}
    if "movie_id" in cols and "movieId" not in cols:
        rename_map["movie_id"] = "movieId"

    if rename_map:
        df = df.rename(rename_map)
        cols = set(df.columns)

    if not {"movieId", "title"}.issubset(cols):
        return None

    return df.select(["movieId", "title"]).unique(subset=["movieId"])


def _build_item_meta_from_movies_only(movies: pl.DataFrame, processed: Path) -> pl.DataFrame:
    """
    If we don't have item_idx mapping, create a stable one.
    This won't perfectly match model item_idx if your training used a different mapping,
    but it's still useful for posters in UI (which key off movieId).
    """
    m = movies.sort("movieId").with_row_index(name="item_idx")
    out = processed / "item_meta.parquet"
    processed.mkdir(parents=True, exist_ok=True)
    m.select(["item_idx", "movieId", "title"]).write_parquet(out)
    return m.select(["item_idx", "movieId", "title"])


def _find_item_meta(root: Path) -> Tuple[pl.DataFrame, str]:
    """
    Returns (df, source_description)

    Expected output columns:
      - item_idx
      - movieId
      - title
    """

    data_dir = root / "data"
    processed = data_dir / "processed"

    # 1) If item_meta already exists, use it
    item_meta_candidates = [
        processed / "item_meta.parquet",
        processed / "item_meta.csv",
        processed / "items.parquet",
        processed / "items.csv",
        processed / "movies_enriched.parquet",
        processed / "movies_enriched.csv",
    ]
    for p in item_meta_candidates:
        if p.exists():
            df = _read_any(p)
            cols = set(df.columns)

            rename_map = {}
            if "movie_id" in cols and "movieId" not in cols:
                rename_map["movie_id"] = "movieId"
            if "item_id" in cols and "item_idx" not in cols:
                rename_map["item_id"] = "item_idx"

            if rename_map:
                df = df.rename(rename_map)
                cols = set(df.columns)

            if {"item_idx", "movieId", "title"}.issubset(cols):
                return df.select(["item_idx", "movieId", "title"]).unique(), f"processed:{p}"

    # 2) Try mapping + movies join
    mapping_path = _find_mapping_file(processed)
    movies_path = _find_movies_file(data_dir)

    mapping_df = None
    movies_df = None

    if mapping_path and mapping_path.exists():
        raw_map = _read_any(mapping_path)
        mapping_df = _normalize_mapping(raw_map)

    if movies_path and movies_path.exists():
        raw_movies = _read_any(movies_path)
        movies_df = _normalize_movies(raw_movies)

    if mapping_df is not None and movies_df is not None:
        joined = (
            mapping_df.join(movies_df, on="movieId", how="left")
            .with_columns(
                [
                    pl.col("item_idx").cast(pl.Int64),
                    pl.col("movieId").cast(pl.Int64),
                    pl.col("title").cast(pl.Utf8),
                ]
            )
        )

        # If some titles missing, we keep them but posters can still be fetched by title later
        # (the fetch helper may fail for null titles; we'll guard)
        out = processed / "item_meta.parquet"
        processed.mkdir(parents=True, exist_ok=True)
        joined.select(["item_idx", "movieId", "title"]).write_parquet(out)

        return joined.select(["item_idx", "movieId", "title"]), f"join:{mapping_path}+{movies_path}"

    # 3) Movies-only fallback (nested MovieLens folders supported)
    if movies_df is not None:
        meta = _build_item_meta_from_movies_only(movies_df, processed)
        return meta, f"movies-only:{movies_path}"

    raise FileNotFoundError(
        "Could not locate item metadata.\n"
        "Looked for processed meta/mapping and searched recursively under data/ for movies files.\n"
        "Expected at least one of:\n"
        "  - data/processed/item_meta.parquet\n"
        "  - data/processed/item_id_map.parquet (or similar)\n"
        "  - any movies.csv under data/**\n"
    )


def main(limit: int = 3000) -> None:
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        raise RuntimeError("TMDB_API_KEY not set. Export it before running cache warmup.")

    root = _project_root()
    meta, src = _find_item_meta(root)

    meta = (
        meta.with_columns(
            [
                pl.col("item_idx").cast(pl.Int64),
                pl.col("movieId").cast(pl.Int64),
                pl.col("title").cast(pl.Utf8),
            ]
        )
        .unique(subset=["movieId"])
        .filter(pl.col("title").is_not_null())
    )

    if limit and limit > 0:
        meta = meta.head(limit)

    cache = PosterCache.load()

    hit = 0
    miss = 0
    added = 0

    for row in meta.iter_rows(named=True):
        mid = int(row["movieId"])
        title = str(row["title"])

        existing = cache.get(mid)
        if existing:
            hit += 1
            continue

        url = fetch_tmdb_poster_by_title(title, api_key=api_key)
        if url:
            cache.set(mid, url)
            added += 1
        else:
            miss += 1

    cache.save()

    print("[DONE] Poster cache updated.")
    print(f"[SRC]  {src}")
    print(f"[PATH] {cache.path}")
    print(f"[OK] already_cached={hit} | added={added} | misses={miss} | total_cached={len(cache.cache)}")


if __name__ == "__main__":
    main()