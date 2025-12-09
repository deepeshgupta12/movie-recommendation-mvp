from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Set, Tuple  # noqa: UP035


def _genre_set(genres: str) -> Set[str]:
    if not genres:
        return set()
    return {g.strip() for g in genres.split("|") if g.strip()}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return (inter / union) if union else 0.0


def apply_genre_cap(
    items: List[Dict[str, Any]],
    max_per_genre: int = 3,
    fallback_keep_order: bool = True,
) -> List[Dict[str, Any]]:
    """
    Hard constraint layer:
    - Greedily picks items while ensuring no genre exceeds max_per_genre.
    - If an item has multiple genres, we count the 'primary' genre as the first tag.
    """
    if not items or max_per_genre <= 0:
        return items

    counts = Counter()
    selected = []
    skipped = []

    for it in items:
        genres = str(it.get("genres", "") or "")
        gs = [g.strip() for g in genres.split("|") if g.strip()]
        primary = gs[0] if gs else "__unknown__"

        if counts[primary] < max_per_genre:
            counts[primary] += 1
            selected.append(it)
        else:
            skipped.append(it)

    if fallback_keep_order:
        # append leftovers in original order
        return selected + skipped

    return selected


def mmr_rerank_by_genre(
    items: List[Dict[str, Any]],
    k: int,
    lambda_relevance: float = 0.75,
) -> List[Dict[str, Any]]:
    """
    Lightweight MMR using genre-set similarity.
    Score = lambda * relevance - (1-lambda) * max_sim_to_selected
    Relevance uses existing 'score' from ranker.
    """
    if not items:
        return []

    k = max(1, min(k, len(items)))
    lam = max(0.0, min(1.0, lambda_relevance))

    pool = []
    for it in items:
        new_it = dict(it)
        new_it["_gset"] = _genre_set(str(it.get("genres", "") or ""))
        new_it["_base_score"] = float(it.get("score", 0.0) or 0.0)
        pool.append(new_it)

    selected: List[Dict[str, Any]] = []

    # Start with top relevance
    pool.sort(key=lambda x: x["_base_score"], reverse=True)
    selected.append(pool.pop(0))

    while pool and len(selected) < k:
        best_idx = 0
        best_score = -1e18

        for i, cand in enumerate(pool):
            max_sim = 0.0
            for s in selected:
                sim = _jaccard(cand["_gset"], s["_gset"])
                if sim > max_sim:
                    max_sim = sim

            mmr_score = lam * cand["_base_score"] - (1.0 - lam) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        selected.append(pool.pop(best_idx))

    # Clean temp keys
    out = []
    for it in selected:
        it.pop("_gset", None)
        it.pop("_base_score", None)
        out.append(it)

    return out


def apply_diversity_pipeline(
    items: List[Dict[str, Any]],
    k: int,
    enable_genre_cap: bool = True,
    max_per_genre: int = 3,
    enable_mmr: bool = True,
    lambda_relevance: float = 0.75,
) -> List[Dict[str, Any]]:
    """
    Orchestrates diversity:
    1) optional genre cap (hard constraint)
    2) optional MMR (soft constraint)
    Returns top-k diversified items.
    """
    if not items:
        return []

    working = items

    if enable_genre_cap:
        working = apply_genre_cap(working, max_per_genre=max_per_genre, fallback_keep_order=True)

    if enable_mmr:
        diversified = mmr_rerank_by_genre(
            working,
            k=k,
            lambda_relevance=lambda_relevance,
        )
    else:
        diversified = working[:k]

    # Add a reason tag when diversity is enabled
    out = []
    for it in diversified:
        new_it = dict(it)
        reasons = list(new_it.get("reasons", []) or [])
        tag = "Diversity-adjusted"
        if tag not in reasons:
            reasons = reasons + [tag]
        new_it["reasons"] = reasons[:4]
        out.append(new_it)

    return out