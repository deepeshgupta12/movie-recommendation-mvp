# app/api/routes/health.py
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    """
    Basic health check endpoint.
    """
    return {"status": "ok"}