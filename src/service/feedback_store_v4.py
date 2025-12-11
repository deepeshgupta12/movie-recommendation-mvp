from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple  # noqa: UP035


def _project_root() -> Path:
    # Robust root detection without relying on settings fields that may not exist.
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / ".git").exists() or (p / "pyproject.toml").exists():
            return p
    # Fallback: assume src/service/.. -> project root two levels up
    return here.parents[2]


@dataclass
class UserFeedbackState:
    liked: Set[int] = field(default_factory=set)
    watched: Set[int] = field(default_factory=set)
    watch_later: Set[int] = field(default_factory=set)
    started: Set[int] = field(default_factory=set)

    def apply(self, item_idx: int, event: str) -> None:
        # Canonical event types
        e = event.lower().strip()
        if e == "like":
            self.liked.add(item_idx)
            # If user likes something, it can still remain in watch_later/started.
        elif e == "unlike":
            self.liked.discard(item_idx)
        elif e == "watched":
            self.watched.add(item_idx)
            # Once watched, usually remove from watch_later/started
            self.watch_later.discard(item_idx)
            self.started.discard(item_idx)
        elif e == "watch_later":
            if item_idx not in self.watched:
                self.watch_later.add(item_idx)
        elif e == "remove_watch_later":
            self.watch_later.discard(item_idx)
        elif e == "start":
            if item_idx not in self.watched:
                self.started.add(item_idx)
        elif e == "remove_start":
            self.started.discard(item_idx)
        elif e == "reset":
            self.liked.clear()
            self.watched.clear()
            self.watch_later.clear()
            self.started.clear()


class FeedbackStoreV4:
    """
    Ultra-lightweight feedback store:
    - Appends events to JSONL file.
    - Maintains an in-memory map of user -> state.
    - Loads existing events on init.
    Good enough for local MVP + Streamlit.
    """
    def __init__(self, path: Optional[Path] = None) -> None:
        root = _project_root()
        self.path = path or (root / "data" / "processed" / "feedback_events_v4.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state: Dict[int, UserFeedbackState] = {}
        self._loaded = False
        self._load()

    def _load(self) -> None:
        if self._loaded:
            return
        if not self.path.exists():
            self._loaded = True
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    user_idx = int(obj["user_idx"])
                    item_idx = int(obj["item_idx"])
                    event = str(obj["event"])
                    self._state.setdefault(user_idx, UserFeedbackState()).apply(item_idx, event)
        except Exception:
            # Corrupt file shouldn't break the app.
            pass
        self._loaded = True

    def get_state(self, user_idx: int) -> UserFeedbackState:
        self._load()
        return self._state.setdefault(user_idx, UserFeedbackState())

    def record(self, user_idx: int, item_idx: int, event: str, meta: Optional[dict] = None) -> None:
        self._load()
        state = self._state.setdefault(user_idx, UserFeedbackState())
        state.apply(item_idx, event)

        payload = {
            "ts": datetime.utcnow().isoformat(),
            "user_idx": int(user_idx),
            "item_idx": int(item_idx),
            "event": event,
            "meta": meta or {},
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def reset_user(self, user_idx: int) -> None:
        # Soft reset in memory + append reset marker to log.
        self._state[user_idx] = UserFeedbackState()
        self.record(user_idx, -1, "reset")