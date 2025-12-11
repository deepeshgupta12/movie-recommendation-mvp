# src/service/poster_cache.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional  # noqa: UP035


@dataclass
class PosterCache:
    """
    Loads poster URLs from:
      1) poster_cache_v4.json (preferred)
      2) item_posters.json (legacy)
    Both expected to be dict-like:
      - movieId -> url
      OR
      - item_idx -> url
    The V4 service will resolve item_idx -> movieId using item_meta,
    then ask this cache.

    This class is intentionally lightweight and tolerant to schema drift.
    """
    root: Path

    def __post_init__(self) -> None:
        self.cache_v4_path = self.root / "data" / "processed" / "poster_cache_v4.json"
        self.legacy_path = self.root / "data" / "processed" / "item_posters.json"

        self._cache_v4: Dict[str, str] = {}
        self._legacy: Dict[str, str] = {}

        self._load()

    def _load_json_safely(self, p: Path) -> Dict[str, str]:
        if not p.exists():
            return {}
        try:
            obj = json.loads(p.read_text())
            if isinstance(obj, dict):
                # Coerce keys to str, values to str if possible
                out: Dict[str, str] = {}
                for k, v in obj.items():
                    if v is None:
                        continue
                    out[str(k)] = str(v)
                return out
            return {}
        except Exception:
            return {}

    def _load(self) -> None:
        self._cache_v4 = self._load_json_safely(self.cache_v4_path)
        self._legacy = self._load_json_safely(self.legacy_path)

    def has_any(self) -> bool:
        return bool(self._cache_v4) or bool(self._legacy)

    def get_by_movie_id(self, movie_id: int | str) -> Optional[str]:
        key = str(movie_id)
        if key in self._cache_v4:
            return self._cache_v4[key]
        if key in self._legacy:
            return self._legacy[key]
        return None

    def get_by_item_idx(self, item_idx: int | str) -> Optional[str]:
        """
        Only works if cache was built with item_idx keys.
        Kept for backwards compatibility.
        """
        key = str(item_idx)
        if key in self._cache_v4:
            return self._cache_v4[key]
        if key in self._legacy:
            return self._legacy[key]
        return None