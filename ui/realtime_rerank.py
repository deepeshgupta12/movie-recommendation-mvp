from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple  # noqa: UP035

DEFAULT_FEEDBACK_PATH = Path("data/processed/ui_feedback.jsonl")


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


def get_user_liked_genres(
    user_idx: int,
    path: Path = DEFAULT_FEEDBACK_PATH,
    top_n: int = 5,
) -> List[Tuple[str, int]]:
    """
    Returns top liked genres for a user from feedback logs.
    Requires UI to log genres in the feedback context.
    """
    rows = _safe_load_jsonl(path)
    if not rows:
        return []

    counts = Counter()
    for r in rows:
        if int(r.get("user_idx", -1)) != int(user_idx):
            continue
        if r.get("action") != "like":
            continue

        ctx = r.get("context") or {}
        genres = ctx.get("genres")
        if not genres or not isinstance(genres, str):
            continue

        for g in [x.strip() for x in genres.split("|") if x.strip()]:
            counts[g] += 1

    return counts.most_common(top_n)


def apply_feedback_rerank(
    recs: List[Dict[str, Any]],
    user_idx: int,
    path: Path = DEFAULT_FEEDBACK_PATH,
    boost: float = 0.08,
) -> List[Dict[str, Any]]:
    """
    UI-only re-ranking:
    - Adds a small score boost to items whose genres match user's liked genres.
    - Returns a NEW list sorted by boosted score.
    - Also appends a reason tag when boosted.
    """
    liked = get_user_liked_genres(user_idx, path=path, top_n=5)
    if not liked:
        return recs

    liked_genres = {g for g, _ in liked}

    enriched = []
    for item in recs:
        score = float(item.get("score", 0.0))
        genres = str(item.get("genres", "") or "")
        reasons = list(item.get("reasons", []) or [])

        item_genres = {x.strip() for x in genres.split("|") if x.strip()}
        overlap = liked_genres.intersection(item_genres)

        boosted_score = score
        if overlap:
            # Small additive boost; safe for demo
            boosted_score = score + boost

            if "Boosted by your likes" not in reasons:
                reasons = reasons + ["Boosted by your likes"]

        new_item = dict(item)
        new_item["score"] = float(boosted_score)
        new_item["reasons"] = reasons[:4]  # keep tidy
        enriched.append(new_item)

    enriched.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return enriched