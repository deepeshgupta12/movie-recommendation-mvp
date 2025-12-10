# scripts/demo_v3_feedback_flow.py
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple  # noqa: UP035

import requests

DEFAULT_API = "http://127.0.0.1:8003"


def _api_base() -> str:
    return os.getenv("V3_API_BASE", DEFAULT_API).rstrip("/")


def _get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.get(url, params=params, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"GET {url} failed {r.status_code}: {r.text}")
    return r.json()


def _post(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(url, json=payload, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"POST {url} failed {r.status_code}: {r.text}")
    return r.json()


def fetch_recos(user_idx: int, k: int = 20, split: str = "test", debug: bool = False) -> List[Dict[str, Any]]:
    base = _api_base()
    url = f"{base}/recommend"
    payload = _get(
        url,
        {
            "user_idx": user_idx,
            "k": k,
            "include_titles": True,
            "debug": debug,
            "split": split,
        },
    )
    return payload.get("items", [])


def send_feedback(user_idx: int, item_idx: int, action: str) -> None:
    base = _api_base()
    url = f"{base}/feedback"
    _post(url, {"user_idx": user_idx, "item_idx": item_idx, "action": action})


def index_by_item(items: List[Dict[str, Any]]) -> Dict[int, Tuple[int, Dict[str, Any]]]:
    """
    Returns map item_idx -> (rank, item_dict)
    rank is 1-based index from list order.
    """
    out = {}
    for i, it in enumerate(items, start=1):
        try:
            out[int(it["item_idx"])] = (i, it)
        except Exception:
            continue
    return out


def title_of(it: Dict[str, Any]) -> str:
    return str(it.get("title") or f"item_idx={it.get('item_idx')}")


def print_list(label: str, items: List[Dict[str, Any]], max_n: int = 20) -> None:
    print(f"\n{label}")
    for i, it in enumerate(items[:max_n], start=1):
        t = title_of(it)
        s = it.get("score")
        print(f"{i:02d}. {t} | item_idx={it.get('item_idx')} | score={s}")


def diff_recos(before: List[Dict[str, Any]], after: List[Dict[str, Any]]) -> None:
    bmap = index_by_item(before)
    amap = index_by_item(after)

    b_ids = set(bmap.keys())
    a_ids = set(amap.keys())

    added = list(a_ids - b_ids)
    removed = list(b_ids - a_ids)
    common = list(a_ids & b_ids)

    # Rank movement summary
    moves = []
    for iid in common:
        br, bit = bmap[iid]
        ar, ait = amap[iid]
        delta = br - ar  # positive means moved up
        if delta != 0:
            moves.append((delta, iid, br, ar, title_of(ait)))

    moves.sort(reverse=True, key=lambda x: x[0])

    print("\n=== Diff Summary ===")
    print(f"Before count: {len(before)} | After count: {len(after)}")
    print(f"Added: {len(added)} | Removed: {len(removed)} | Common: {len(common)}")

    if removed:
        print("\nRemoved items (likely due to watched filter or reshuffle):")
        for iid in removed[:10]:
            br, bit = bmap[iid]
            print(f"- {title_of(bit)} | item_idx={iid} | prev_rank={br}")

    if added:
        print("\nAdded items:")
        for iid in added[:10]:
            ar, ait = amap[iid]
            print(f"+ {title_of(ait)} | item_idx={iid} | new_rank={ar}")

    if moves:
        print("\nBiggest rank moves (up first):")
        for delta, iid, br, ar, t in moves[:15]:
            arrow = "up" if delta > 0 else "down"
            print(f"* {t} | item_idx={iid} | {arrow} by {abs(delta)} | {br} -> {ar}")


def main():
    # CLI: python -m scripts.demo_v3_feedback_flow [user_idx] [split] [k]
    user_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 9764
    split = sys.argv[2] if len(sys.argv) > 2 else "test"
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    print(f"[DEMO] V3 Feedback Flow")
    print(f"[CFG] api_base={_api_base()} | user_idx={user_idx} | split={split} | k={k}")

    # 1) Before
    before = fetch_recos(user_idx=user_idx, k=k, split=split, debug=False)
    print_list("Before recommendations", before, max_n=k)

    if not before:
        print("\nNo recommendations returned. Exiting.")
        return

    # Pick two candidates from the top
    top1 = before[0]
    top2 = before[1] if len(before) > 1 else None

    top1_id = int(top1["item_idx"])
    top2_id = int(top2["item_idx"]) if top2 else None

    print("\n[STEP] Posting feedback events...")
    print(f"- like item_idx={top1_id} ({title_of(top1)})")
    send_feedback(user_idx, top1_id, "like")

    if top2_id is not None:
        print(f"- watched item_idx={top2_id} ({title_of(top2)})")
        send_feedback(user_idx, top2_id, "watched")

    # 2) After
    after = fetch_recos(user_idx=user_idx, k=k, split=split, debug=False)
    print_list("After recommendations", after, max_n=k)

    # 3) Diff
    diff_recos(before, after)

    print("\n[DONE] Feedback flow demo complete.")


if __name__ == "__main__":
    main()