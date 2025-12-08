from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Project root inferred from this file location:
    # repo/
    #   src/config/settings.py
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    SYNTHETIC_DIR: Path = DATA_DIR / "synthetic"

    # Local analytics store
    DUCKDB_PATH: Path = PROCESSED_DIR / "reco.duckdb"

    # Dataset config
    MOVIELENS_VARIANT: str = "ml-20m"  # keep stable for MVP
    MOVIELENS_URL: str = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"


settings = Settings()