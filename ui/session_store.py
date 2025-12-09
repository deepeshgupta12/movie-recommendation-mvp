from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional  # noqa: UP035

WATCH_PATH = Path("data/processed/ui_watch.jsonl")


def log_watch_event(
    user_idx: int,
    item_idx: int,
    event: str,
    context: Optional[Dict[str, Any]] = None,
    path: Path = WATCH_PATH,
) -> None:
    """
    Local-first watch event logger.
    event: "watch_start" | "watch_complete"
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "ts": int(time.time()),
        "user_idx": int(user_idx),
        "item_idx": int(item_idx),
        "event": str(event),
        "context": context or {},
    }

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _safe_load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def get_user_watch_state(user_idx: int, path: Path = WATCH_PATH) -> Dict[int, str]:
    """
    Returns latest state per item for the user:
    item_idx -> "watch_start" | "watch_complete"
    """
    rows = _safe_load_jsonl(path)
    if not rows:
        return {}

    latest: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        if int(r.get("user_idx", -1)) != int(user_idx):
            continue
        item = int(r.get("item_idx", -1))
        ts = int(r.get("ts", 0))
        if item < 0:
            continue
        if item not in latest or ts >= int(latest[item].get("ts", 0)):
            latest[item] = r

    return {i: str(v.get("event", "")) for i, v in latest.items()}


def get_continue_watching_items(user_idx: int, path: Path = WATCH_PATH) -> List[int]:
    """
    Simple heuristic:
    items that were started but not completed.
    """
    state = get_user_watch_state(user_idx, path=path)
    return [i for i, ev in state.items() if ev == "watch_start"]


def get_last_watched_item(user_idx: int, path: Path = WATCH_PATH) -> Optional[int]:
    rows = _safe_load_jsonl(path)
    rows = [r for r in rows if int(r.get("user_idx", -1)) == int(user_idx)]
    if not rows:
        return None
    rows.sort(key=lambda x: int(x.get("ts", 0)), reverse=True)
    return int(rows[0].get("item_idx", -1))