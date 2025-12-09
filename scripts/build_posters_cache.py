from __future__ import annotations

import os

from src.metadata.poster_cache import build_poster_cache


def main() -> None:
    max_items = int(os.getenv("POSTER_MAX_ITEMS", "20000"))
    workers = int(os.getenv("POSTER_WORKERS", "8"))

    print("[START] Building TMDB poster cache...")
    print(f"[OK] max_items={max_items} | workers={workers}")

    build_poster_cache(max_items=max_items, workers=workers)


if __name__ == "__main__":
    main()