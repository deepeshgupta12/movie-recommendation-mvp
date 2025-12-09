from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict  # noqa: UP035

DEFAULT_FEEDBACK_PATH = Path("data/processed/ui_feedback.jsonl")


def log_feedback(
    user_idx: int,
    item_idx: int,
    action: str,
    context: Dict[str, Any] | None = None,
    path: Path = DEFAULT_FEEDBACK_PATH,
) -> None:
    """
    Local-first feedback logger for Streamlit UI.
    Writes JSONL records to data/processed/ui_feedback.jsonl

    action: "like" | "dislike" | "save"
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "ts": int(time.time()),
        "user_idx": int(user_idx),
        "item_idx": int(item_idx),
        "action": str(action),
        "context": context or {},
    }

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")