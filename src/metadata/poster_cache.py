from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple  # noqa: UP035

import polars as pl
from tqdm import tqdm

from src.config.settings import settings
from src.metadata.tmdb_client import TMDBClient

POSTER_CACHE_PATH = settings.PROCESSED_DIR / "item_posters.json"


def load_links() -> pl.DataFrame:
    links_path = settings.PROCESSED_DIR / "links.parquet"
    items_path = settings.PROCESSED_DIR / "items.parquet"

    if not links_path.exists():
        raise FileNotFoundError("links.parquet not found. Ensure MovieLens ingest wrote links.")
    if not items_path.exists():
        raise FileNotFoundError("items.parquet not found.")

    links = pl.read_parquet(links_path).select("movieId", "tmdbId")
    items = pl.read_parquet(items_path).select("item_idx", "movieId")

    df = (
        items.join(links, on="movieId", how="left")
        .with_columns(pl.col("tmdbId").cast(pl.Int64, strict=False))
        .filter(pl.col("tmdbId").is_not_null())
    )
    return df


def read_cache(path: Path = POSTER_CACHE_PATH) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_cache(cache: Dict[str, str], path: Path = POSTER_CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _fetch_one(client: TMDBClient, item_idx: int, tmdb_id: int) -> Optional[Tuple[int, str]]:
    data = client.get_movie(tmdb_id)
    if not data:
        return None
    poster_path = data.get("poster_path")
    url = client.poster_url(poster_path)
    if not url:
        return None
    return (item_idx, url)


def build_poster_cache(
    max_items: int = 20000,
    workers: int = 8,
) -> Dict[str, str]:
    """
    Multi-threaded poster cache builder.
    - Uses TMDB if TMDB_API_KEY exists.
    - Caches item_idx -> poster_url.
    """
    client = TMDBClient.from_env()
    if client is None:
        raise RuntimeError(
            "TMDB_API_KEY not set. Export it to enable poster enrichment."
        )

    df = load_links()
    if max_items:
        df = df.head(max_items)

    existing = read_cache()

    tasks = []
    for item_idx, tmdb_id in df.select("item_idx", "tmdbId").iter_rows():
        key = str(int(item_idx))
        if key in existing:
            continue
        tasks.append((int(item_idx), int(tmdb_id)))

    if not tasks:
        return existing

    start = time.time()
    new_entries: Dict[str, str] = {}

    # Cap workers to avoid oversubscription on M1
    workers = max(1, min(workers, 16))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_fetch_one, client, item_idx, tmdb_id): (item_idx, tmdb_id)
            for item_idx, tmdb_id in tasks
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Fetching posters"):
            res = fut.result()
            if res:
                item_idx, url = res
                new_entries[str(item_idx)] = url

    merged = {**existing, **new_entries}
    write_cache(merged)

    end = time.time()
    print(f"[DONE] Posters fetched: {len(new_entries)} | cache size: {len(merged)}")
    print(f"[PATH] {POSTER_CACHE_PATH}")
    print(f"[OK] Time: {end - start:.2f}s")

    return merged


def get_poster_map() -> Dict[int, str]:
    """
    Load the stored poster cache as int keys.
    """
    raw = read_cache()
    out: Dict[int, str] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = str(v)
        except Exception:
            continue
    return out