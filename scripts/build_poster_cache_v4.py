"""
Builds a robust poster cache for V4, strictly keyed by movieId.

Output:
    data/processed/poster_cache_v4.json

Schema:
    {
        "<movieId>": "<poster_url or null>",
        ...
    }

Rules:
    - Only movieId is used as key (no dependence on item_idx ordering).
    - Primary source is data/processed/item_meta.parquet if available.
    - Fallback is data/raw/movies.csv from MovieLens.
    - Uses TMDB search API; requires TMDB_API_KEY in env.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple  # noqa: UP035

import polars as pl
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

ITEM_META_PARQUET = PROCESSED_DIR / "item_meta.parquet"
MOVIES_CSV = RAW_DIR / "movies.csv"
OUT_PATH = PROCESSED_DIR / "poster_cache_v4.json"

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def load_source_df() -> pl.DataFrame:
    """
    Try to load (movieId, title) from item_meta.parquet first,
    else fall back to movies.csv.
    """
    if ITEM_META_PARQUET.exists():
        _log(f"[SRC] Using {ITEM_META_PARQUET}")
        df = pl.read_parquet(ITEM_META_PARQUET)
        cols = set(df.columns)
        if "movieId" in cols and "title" in cols:
            return (
                df.select(["movieId", "title"])
                .unique("movieId")
                .sort("movieId")
            )
        else:
            _log("[WARN] item_meta.parquet missing movieId/title; falling back to movies.csv")

    if MOVIES_CSV.exists():
        _log(f"[SRC] Using {MOVIES_CSV}")
        df = pl.read_csv(MOVIES_CSV)
        if "movieId" not in df.columns or "title" not in df.columns:
            raise RuntimeError("movies.csv must contain movieId and title columns")
        return (
            df.select(["movieId", "title"])
            .unique("movieId")
            .sort("movieId")
        )

    raise FileNotFoundError(
        f"Could not find item_meta.parquet or movies.csv under {PROCESSED_DIR} / {RAW_DIR}"
    )


def parse_title_year(title: str) -> Tuple[str, Optional[int]]:
    """
    MovieLens titles often look like: 'Interstellar (2014)'.
    Extract base title + year if present.
    """
    title = title.strip()
    if title.endswith(")"):
        try:
            base, year_part = title.rsplit("(", 1)
            year = int(year_part.strip(") "))
            return base.strip(), year
        except Exception:
            return title, None
    return title, None


def tmdb_search_poster(title: str, year: Optional[int]) -> Optional[str]:
    """
    Search TMDB for the given title/year and return a full poster URL if found.
    Retries a few times on transient connection issues.
    """
    params = {"api_key": TMDB_API_KEY, "query": title}
    if year:
        params["year"] = year

    last_err = None
    for attempt in range(3):
        try:
            resp = requests.get(TMDB_SEARCH_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results") or []
            if not results:
                return None

            poster_path = results[0].get("poster_path")
            if not poster_path:
                return None

            return TMDB_IMG_BASE + poster_path
        except Exception as e:
            last_err = e
            time.sleep(1.0 + attempt * 1.0)

    # If all retries failed, bubble up the last error so caller logs it
    raise last_err


def main() -> None:
    if not TMDB_API_KEY:
        raise RuntimeError("TMDB_API_KEY not set. Export it before running this script.")

    _log("[START] Building poster_cache_v4.json (keyed by movieId)...")

    df = load_source_df()
    _log(f"[OK] Source rows: {df.height}")

    # If a previous cache exists, reuse what we can to avoid hammering TMDB
    cache: Dict[str, str] = {}
    if OUT_PATH.exists():
        try:
            cache = json.loads(OUT_PATH.read_text())
            _log(f"[OK] Loaded existing cache with {len(cache)} entries. Will reuse.")
        except Exception:
            _log("[WARN] Existing cache unreadable; starting fresh.")

    new_cache: Dict[str, Optional[str]] = {}

    rows = df.to_dicts()
    n = len(rows)
    for idx, row in enumerate(rows, start=1):
        movie_id = int(row["movieId"])
        title = str(row["title"])
        key = str(movie_id)

        if key in cache:
            new_cache[key] = cache[key]
            if idx % 200 == 0:
                _log(f"[SKIP] {idx}/{n} (cached) movieId={movie_id}")
            continue

        base_title, year = parse_title_year(title)
        try:
            poster_url = tmdb_search_poster(base_title, year)
            new_cache[key] = poster_url
            if idx % 100 == 0:
                _log(
                    f"[FETCH] {idx}/{n} movieId={movie_id} "
                    f"title='{base_title}' year={year} poster={bool(poster_url)}"
                )
            # Be polite with TMDB
            time.sleep(0.25)
        except Exception as e:
            _log(
                f"[ERR] movieId={movie_id} title='{base_title}': {e}"
            )
            new_cache[key] = None
            time.sleep(0.5)

    # Merge with old cache so we don't lose older entries
    merged: Dict[str, Optional[str]] = {**cache, **new_cache}

    # Clean out explicit null/None if you prefer, or keep as-is.
    OUT_PATH.write_text(json.dumps(merged, indent=2, sort_keys=True))
    _log(f"[DONE] poster_cache_v4.json written with {len(merged)} entries.")
    _log(f"[PATH] {OUT_PATH}")


if __name__ == "__main__":
    main()