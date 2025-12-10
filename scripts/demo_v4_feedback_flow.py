# scripts/demo_v4_feedback_flow.py

import json
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8004"


def _get(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(url: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _top_titles(out, n=10):
    recs = out.get("recommendations", [])
    return [r.get("title") for r in recs[:n]]


def main():
    print("[V4 FEEDBACK FLOW DEMO]")
    user_idx = 9764
    print("user_idx:", user_idx)

    url1 = f"{BASE}/recommend?user_idx={user_idx}&k=20&include_titles=True&debug=False&split=val&apply_diversity=True"
    before = _get(url1)

    before_top = _top_titles(before, 10)
    print("\n[BEFORE TOP-10]")
    for i, t in enumerate(before_top, 1):
        print(f"{i:02d}. {t}")

    # Pick a couple of items from the current list and mark watched/like
    recs = before.get("recommendations", [])
    if len(recs) >= 2:
        item_a = recs[0]["item_idx"]
        item_b = recs[1]["item_idx"]
    else:
        print("Not enough recs to test feedback.")
        return

    fb_url = f"{BASE}/feedback"

    print("\n[POST FEEDBACK]")
    print("like:", item_a)
    print(_post(fb_url, {"user_idx": user_idx, "item_idx": item_a, "event": "like"}))

    print("watched:", item_b)
    print(_post(fb_url, {"user_idx": user_idx, "item_idx": item_b, "event": "watched"}))

    after = _get(url1)

    after_top = _top_titles(after, 10)
    print("\n[AFTER TOP-10]")
    for i, t in enumerate(after_top, 1):
        print(f"{i:02d}. {t}")

    # Simple diff
    removed = [t for t in before_top if t not in after_top]
    added = [t for t in after_top if t not in before_top]

    print("\n[DIFF]")
    print("removed_from_top10:", removed)
    print("added_to_top10:", added)


if __name__ == "__main__":
    main()