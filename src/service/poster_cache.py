from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional  # noqa: UP035

import polars as pl

# A tiny inline placeholder that Streamlit can render reliably.
# Simple SVG data URI to avoid external dependency.
PLACEHOLDER_POSTER_DATA_URI = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='300' height='450'>"
    "<rect width='100%' height='100%' fill='#111111'/>"
    "<rect x='20' y='20' width='260' height='410' fill='#1f1f1f' stroke='#333'/>"
    "<text x='50%' y='50%' fill='#bbbbbb' font-size='22' font-family='Arial' "
    "dominant-baseline='middle' text-anchor='middle'>No Poster</text>"
    "</svg>"
)


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / ".git").exists() or (p / "pyproject.toml").exists():
            return p
    return here.parents[2]


@dataclass
class PosterCache:
    """
    Poster cache keyed by movieId.
    Primary store: data/processed/poster_cache_v4.json
    Legacy fallback: data/processed/item_posters.json
    """

    path: Path
    legacy_path: Path
    cache: Dict[str, str]

    @classmethod
    def default_paths(cls) -> tuple[Path, Path]:
        root = _project_root()
        processed = root / "data" / "processed"
        return (
            processed / "poster_cache_v4.json",
            processed / "item_posters.json",
        )

    @classmethod
    def load(cls) -> "PosterCache":
        path, legacy = cls.default_paths()

        data: Dict[str, str] = {}
        if legacy.exists():
            try:
                legacy_data = json.loads(legacy.read_text())
                if isinstance(legacy_data, dict):
                    # Normalize keys to strings
                    data.update({str(k): v for k, v in legacy_data.items() if v})
            except Exception:
                pass

        if path.exists():
            try:
                new_data = json.loads(path.read_text())
                if isinstance(new_data, dict):
                    data.update({str(k): v for k, v in new_data.items() if v})
            except Exception:
                pass

        return cls(path=path, legacy_path=legacy, cache=data)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2))

    def get(self, movie_id: int | str) -> Optional[str]:
        return self.cache.get(str(movie_id))

    def get_or_placeholder(self, movie_id: int | str) -> str:
        return self.get(movie_id) or PLACEHOLDER_POSTER_DATA_URI

    def set(self, movie_id: int | str, url: str) -> None:
        if url:
            self.cache[str(movie_id)] = url


def fetch_tmdb_poster_by_title(title: str, api_key: str) -> Optional[str]:
    """
    Kept lightweight intentionally.
    This function is used only by cache warmup scripts.
    We do not do web calls here in the service runtime.
    """
    # We keep a stub-like design here to avoid hard dependency in service runtime code.
    # The actual implementation should exist in your script environment.
    # If your earlier version already implemented this with requests, keep that logic in
    # scripts/build_poster_cache_v4.py which imports this function.
    #
    # Returning None here is acceptable for runtime import safety.
    _ = (title, api_key)
    return None


def load_item_meta() -> Optional[pl.DataFrame]:
    """
    Optional helper used by services to map item_idx -> movieId -> title.
    """
    root = _project_root()
    p = root / "data" / "processed" / "item_meta.parquet"
    if not p.exists():
        return None
    df = pl.read_parquet(p)

    # Normalize cols
    cols = set(df.columns)
    if "movie_id" in cols and "movieId" not in cols:
        df = df.rename({"movie_id": "movieId"})
    if "item_id" in cols and "item_idx" not in cols:
        df = df.rename({"item_id": "item_idx"})

    needed = {"item_idx", "movieId", "title"}
    if not needed.issubset(set(df.columns)):
        return None

    return df.select(["item_idx", "movieId", "title"]).unique(subset=["item_idx"])