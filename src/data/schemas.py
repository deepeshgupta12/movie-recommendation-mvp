from __future__ import annotations

from enum import Enum
from typing import Any, Optional, dict  # noqa: UP035

from pydantic import BaseModel, Field


class InteractionType(str, Enum):
    IMPRESSION = "impression"
    CLICK = "click"
    PLAY_START = "play_start"
    PLAY_STOP = "play_stop"
    COMPLETE = "complete"
    LIKE = "like"
    DISLIKE = "dislike"
    RATING = "rating"
    SEARCH = "search"
    ADD_TO_WATCHLIST = "add_to_watchlist"
    REMOVE_FROM_WATCHLIST = "remove_from_watchlist"
    NOT_INTERESTED = "not_interested"


class DeviceType(str, Enum):
    TV = "tv"
    MOBILE = "mobile"
    WEB = "web"
    TABLET = "tablet"
    OTHER = "other"


class InteractionEvent(BaseModel):
    """
    A Netflix/Amazon-style canonical event schema.
    MovieLens will populate only a subset; we will add synthetic/session events later.
    """

    user_id: str
    item_id: str
    interaction_type: InteractionType

    # Core time
    ts_unix: int = Field(..., description="Event timestamp in unix seconds")

    # Optional watch depth
    watch_ms: Optional[int] = None
    session_id: Optional[str] = None

    # Explicit signal
    rating_value: Optional[float] = None

    # Context
    device: Optional[DeviceType] = None
    locale: Optional[str] = None
    time_of_day_bucket: Optional[str] = None  # e.g., morning/evening

    # Flexible extension for experiments
    metadata: dict[str, Any] = Field(default_factory=dict)