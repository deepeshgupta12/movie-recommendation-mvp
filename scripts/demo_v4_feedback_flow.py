from __future__ import annotations

from typing import Dict, List  # noqa: UP035

import requests

BASE = "http://127.0.0.1:8004"


def _get(user_idx: int, k: int = 10, split: str = "val"):
    r = requests.get(f"{BASE}/recommend", params={
        "user_idx": user_idx, "k": k, "include_titles": True, "debug": False, "split": split, "apply_feedback": True
    }, timeout=30)
    r.raise_for_status()
    return r.json()


def _post_feedback(user_idx: int, item_idx: int, event: str, split: str = "val"):
    r = requests.post(f"{BASE}/feedback", json={
        "user_idx": user_idx, "item_idx": item_idx, "event": event, "split": split
    }, timeout=30)
    r.raise_for_status()
    return r.json()


def _titles(resp) -> List[str]:
    return [it.get("title") or str(it.get("item_idx")) for it in resp.get("items", [])]


def main(user_idx: int = 9764, split: str = "val"):
    print("[DEMO] V4 feedback loop overlay")
    print("User:", user_idx, "| split:", split)

    before = _get(user_idx, k=10, split=split)
    before_titles = _titles(before)

    print("\n[BEFORE]")
    for i, t in enumerate(before_titles, 1):
        print(f"{i:02d}. {t}")

    if not before.get("items"):
        print("No items returned; aborting.")
        return

    # Pick 1st as start, 2nd as like, 3rd as watched
    items = before["items"]
    if len(items) >= 1:
        _post_feedback(user_idx, items[0]["item_idx"], "start", split=split)
    if len(items) >= 2:
        _post_feedback(user_idx, items[1]["item_idx"], "like", split=split)
    if len(items) >= 3:
        _post_feedback(user_idx, items[2]["item_idx"], "watched", split=split)

    after = _get(user_idx, k=10, split=split)
    after_titles = _titles(after)

    print("\n[AFTER]")
    for i, t in enumerate(after_titles, 1):
        print(f"{i:02d}. {t}")

    print("\n[DIFF]")
    removed = [t for t in before_titles if t not in after_titles]
    added = [t for t in after_titles if t not in before_titles]
    print("Removed:", removed)
    print("Added:", added)

    print("\n[OK] If overlay is working, the watched item should drop, "
          "and started/liked items should be prioritized in sections.")


if __name__ == "__main__":
    main()