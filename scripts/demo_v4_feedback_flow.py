# scripts/demo_v4_feedback_flow.py

from __future__ import annotations

import json
import urllib.request

BASE = "http://127.0.0.1:8004"


def _get(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(url: str, payload: dict):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _titles(recs):
    return [r.get("title") for r in recs if r.get("title")]


def main():
    user_idx = 9764
    k = 10

    print("[V4 FEEDBACK FLOW DEMO]")
    print("user_idx:", user_idx)

    url1 = f"{BASE}/recommend?user_idx={user_idx}&k={k}&include_titles=True&debug=False&split=val&apply_diversity=True"
    before = _get(url1)
    before_recs = before.get("recommendations", [])

    print("\n[BEFORE]")
    for i, r in enumerate(before_recs[:5], 1):
        print(f"{i:02d}. {r.get('title')} | {r.get('reason')} | score={r.get('score')}")

    # Pick two items from the current list to mark as watched + liked
    if len(before_recs) >= 2:
        item_a = before_recs[0]["item_idx"]
        item_b = before_recs[1]["item_idx"]

        print("\n[POST FEEDBACK]")
        print(_post(f"{BASE}/feedback", {"user_idx": user_idx, "item_idx": item_a, "event_type": "watched"}))
        print(_post(f"{BASE}/feedback", {"user_idx": user_idx, "item_idx": item_b, "event_type": "liked"}))

    after = _get(url1)
    after_recs = after.get("recommendations", [])

    print("\n[AFTER]")
    for i, r in enumerate(after_recs[:5], 1):
        print(f"{i:02d}. {r.get('title')} | {r.get('reason')} | score={r.get('score')}")

    # Simple diff view
    bt = _titles(before_recs)
    at = _titles(after_recs)

    print("\n[DIFF]")
    removed = [t for t in bt if t not in at]
    added = [t for t in at if t not in bt]

    print("Removed from top list:", removed[:10])
    print("Added to top list:", added[:10])


if __name__ == "__main__":
    main()