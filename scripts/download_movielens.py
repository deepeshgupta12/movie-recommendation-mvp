from __future__ import annotations

import zipfile
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import settings


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with httpx.stream("GET", url, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)


def main() -> None:
    raw_dir = settings.RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / f"{settings.MOVIELENS_VARIANT}.zip"
    extract_dir = raw_dir / settings.MOVIELENS_VARIANT

    if extract_dir.exists():
        print(f"[OK] Dataset already extracted at: {extract_dir}")
        return

    if not zip_path.exists():
        print(f"[DL] Downloading MovieLens from {settings.MOVIELENS_URL}")
        _download_file(settings.MOVIELENS_URL, zip_path)
    else:
        print(f"[OK] Zip already present: {zip_path}")

    print(f"[EXTRACT] Extracting to {extract_dir}")
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # MovieLens zips usually contain an inner folder with same name.
    # We'll keep it as-is and resolve in ingestion.
    print("[DONE] Download + extract complete.")


if __name__ == "__main__":
    main()