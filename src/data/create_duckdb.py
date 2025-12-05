from __future__ import annotations

import duckdb

from src.config.settings import settings


def main() -> None:
    settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(settings.DUCKDB_PATH))

    con.execute("CREATE OR REPLACE TABLE ratings AS SELECT * FROM read_parquet(?)", [str(settings.PROCESSED_DIR / "ratings.parquet")])
    con.execute("CREATE OR REPLACE TABLE movies AS SELECT * FROM read_parquet(?)", [str(settings.PROCESSED_DIR / "movies.parquet")])

    tags_path = settings.PROCESSED_DIR / "tags.parquet"
    if tags_path.exists():
        con.execute("CREATE OR REPLACE TABLE tags AS SELECT * FROM read_parquet(?)", [str(tags_path)])

    links_path = settings.PROCESSED_DIR / "links.parquet"
    if links_path.exists():
        con.execute("CREATE OR REPLACE TABLE links AS SELECT * FROM read_parquet(?)", [str(links_path)])

    # Helpful joined view for exploration
    con.execute(
        """
        CREATE OR REPLACE VIEW v_ratings_enriched AS
        SELECT
            r.userId,
            r.movieId,
            r.rating,
            r.timestamp,
            m.title,
            m.genres
        FROM ratings r
        JOIN movies m
        USING(movieId)
        """
    )

    con.close()
    print(f"[DONE] DuckDB created at {settings.DUCKDB_PATH}")


if __name__ == "__main__":
    main()