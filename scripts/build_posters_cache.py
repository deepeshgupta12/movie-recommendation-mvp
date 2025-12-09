from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Tuple  # noqa: UP035

import polars as pl
import requests

from src.config.settings import settings

POSTER_CACHE_PATH = Path("data/processed/item_posters.json")


TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
TMDB_MOVIE_URL = "https://api.themoviedb.org/3/movie/{tmdb_id}"


def load_existing_cache() -> Dict[int, str]:
    if not POSTER_CACHE_PATH.exists():
        return {}
    try:
        raw = json.loads(POSTER_CACHE_PATH.read_text(encoding="utf-8"))
        out: Dict[int, str] = {}
        for k, v in raw.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
        return out
    except Exception:
        return {}


def save_cache(cache: Dict[int, str]) -> None:
    POSTER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # store as string keys for json stability
    payload = {str(k): v for k, v in sorted(cache.items(), key=lambda x: x[0])}
    POSTER_CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_item_to_tmdb_map() -> Dict[int, int]:
    items_path = settings.PROCESSED_DIR / "items.parquet"
    links_path = settings.PROCESSED_DIR / "links.parquet"

    items = pl.read_parquet(items_path).select(["item_idx", "movieId"])
    links = pl.read_parquet(links_path).select(["movieId", "tmdbId"])

    df = items.join(links, on="movieId", how="left").drop_nulls(["tmdbId"])

    mapping: Dict[int, int] = {}
    for r in df.iter_rows(named=True):
        try:
            mapping[int(r["item_idx"])] = int(r["tmdbId"])
        except Exception:
            continue

    return mapping


def fetch_tmdb_poster(tmdb_id: int, api_key: str, timeout: int = 10) -> Optional[str]:
    url = TMDB_MOVIE_URL.format(tmdb_id=tmdb_id)
    try:
        resp = requests.get(url, params={"api_key": api_key}, timeout=timeout)
        if resp.status_code != 200:
            return None
        data = resp.json()
        poster_path = data.get("poster_path")
        if not poster_path:
            return None
        return f"{TMDB_IMAGE_BASE}{poster_path}"
    except Exception:
        return None


def get_top_items_by_interactions(limit: int = 3000) -> list[int]:
    feats_path = settings.PROCESSED_DIR / "item_features.parquet"
    feats = pl.read_parquet(feats_path)

    ranked = (
        feats
        .select(["item_idx", "item_interactions", "item_conf_sum"])
        .with_columns(
            (pl.col("item_interactions") * 1.0 + pl.col("item_conf_sum") * 0.1).alias("score")
        )
        .sort("score", descending=True)
        .head(limit)
    )

    return [int(x) for x in ranked["item_idx"].to_list()]


def main():
    api_key = os.getenv("TMDB_API_KEY", "").strip()

    print("[START] Loading existing poster cache...")
    cache = load_existing_cache()
    print(f"[OK] Existing posters: {len(cache)}")

    print("[START] Building item -> tmdbId map...")
    item_to_tmdb = build_item_to_tmdb_map()
    print(f"[OK] Map size: {len(item_to_tmdb)}")

    top_items = get_top_items_by_interactions(limit=3000)
    missing = [i for i in top_items if i not in cache]

    print(f"[INFO] Top items targeted: {len(top_items)}")
    print(f"[INFO] Missing posters in top set: {len(missing)}")

    if not missing:
        print("[DONE] Cache already covers top items.")
        save_cache(cache)
        return

    if not api_key:
        print("[WARN] TMDB_API_KEY not found.")
        print("[WARN] Skipping external poster fetch. UI placeholders will cover missing posters.")
        save_cache(cache)
        print(f"[PATH] {POSTER_CACHE_PATH.resolve()}")
        return

    print("[START] Fetching posters from TMDB (parallel)...")

    def task(item_idx: int) -> Tuple[int, Optional[str]]:
        tmdb_id = item_to_tmdb.get(item_idx)
        if not tmdb_id:
            return item_idx, None
        url = fetch_tmdb_poster(tmdb_id, api_key=api_key)
        return item_idx, url

    added = 0
    with ThreadPoolExecutor(max_workers=12) as ex:
        futures = [ex.submit(task, i) for i in missing]
        for fut in as_completed(futures):
            item_idx, url = fut.result()
            if url:
                cache[item_idx] = url
                added += 1

    print(f"[OK] Posters added: {added}")
    save_cache(cache)
    print("[DONE] Poster cache updated.")
    print(f"[PATH] {POSTER_CACHE_PATH.resolve()}")


if __name__ == "__main__":
    main()