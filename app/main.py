# app/main.py
from __future__ import annotations

from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.recommend import router as recommend_router
from app.deps import init_service


def create_app() -> FastAPI:
    app = FastAPI(
        title="Movie Recommendation MVP",
        version="0.2.0",
        description=(
            "V2 hybrid + ranking movie recommendation service built on the MovieLens 20M dataset.\n"
            "Implements a multi-source candidate generator (popularity, item-item, ALS, genre neighbors)\n"
            "and a time-aware, genre-cross-featured ranker."
        ),
    )

    # Routers
    app.include_router(health_router)
    app.include_router(recommend_router)

    @app.on_event("startup")
    async def on_startup() -> None:
        """
        Initialize the recommender service once at startup.
        """
        init_service(max_k=50)

    return app


app = create_app()