from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional  # noqa: UP035

import requests


@dataclass
class TMDBConfig:
    api_key: str
    base_url: str = "https://api.themoviedb.org/3"
    image_base: str = "https://image.tmdb.org/t/p/w500"


class TMDBClient:
    def __init__(self, config: TMDBConfig):
        self.config = config

    @classmethod
    def from_env(cls) -> Optional["TMDBClient"]:
        key = os.getenv("TMDB_API_KEY", "").strip()
        if not key:
            return None
        return cls(TMDBConfig(api_key=key))

    def get_movie(self, tmdb_id: int, timeout: int = 15) -> Optional[Dict[str, Any]]:
        url = f"{self.config.base_url}/movie/{tmdb_id}"
        params = {"api_key": self.config.api_key}
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code != 200:
                return None
            return r.json()
        except Exception:
            return None

    def poster_url(self, poster_path: str | None) -> Optional[str]:
        if not poster_path:
            return None
        return f"{self.config.image_base}{poster_path}"