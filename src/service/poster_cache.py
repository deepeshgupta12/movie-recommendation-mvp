from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple  # noqa: UP035

import requests


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / ".git").exists() or (p / "pyproject.toml").exists():
            return p
    return here.parents[2]


TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


@dataclass
class PosterCache:
    path: Path
    cache: Dict[str, str]

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "PosterCache":
        root = _project_root()
        p = path or (root / "data" / "processed" / "poster_cache_v4.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return cls(path=p, cache={str(k): str(v) for k, v in data.items()})
            except Exception:
                pass
        return cls(path=p, cache={})

    def save(self) -> None:
        self.path.write_text(json.dumps(self.cache), encoding="utf-8")

    def get(self, movie_id: int) -> Optional[str]:
        return self.cache.get(str(movie_id))

    def set(self, movie_id: int, url: str) -> None:
        self.cache[str(movie_id)] = url


def _parse_title_year(title: str) -> Tuple[str, Optional[int]]:
    # "American Hustle (2013)" -> ("American Hustle", 2013)
    m = re.match(r"^(.*)\s+\((\d{4})\)\s*$", title.strip())
    if not m:
        return title.strip(), None
    return m.group(1).strip(), int(m.group(2))


def fetch_tmdb_poster_by_title(title: str, api_key: str) -> Optional[str]:
    name, year = _parse_title_year(title)
    params = {"api_key": api_key, "query": name}
    if year:
        params["year"] = year

    try:
        r = requests.get("https://api.themoviedb.org/3/search/movie", params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        results = data.get("results") or []
        if not results:
            return None
        poster_path = results[0].get("poster_path")
        if not poster_path:
            return None
        return f"{TMDB_IMAGE_BASE}{poster_path}"
    except Exception:
        return None


def get_or_fetch_poster(movie_id: int, title: str, cache: PosterCache) -> Optional[str]:
    existing = cache.get(movie_id)
    if existing:
        return existing

    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        return None

    url = fetch_tmdb_poster_by_title(title, api_key=api_key)
    if url:
        cache.set(movie_id, url)
        try:
            cache.save()
        except Exception:
            pass
    return url