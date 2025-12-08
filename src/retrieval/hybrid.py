from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence  # noqa: UP035


@dataclass
class SourceWeight:
    name: str
    weight: float


class HybridCandidateBlender:
    """
    Combine candidate lists from multiple sources using weighted voting.

    Strategy:
    - Each source contributes a ranked list.
    - Items accumulate score = weight * (1 / rank)
    - Final list is sorted by score and de-duplicated.
    """

    def __init__(self, sources: Sequence[SourceWeight]) -> None:
        self.sources = list(sources)

    def blend(
        self,
        candidates_by_source: Dict[str, List[int]],
        k: int = 200
    ) -> List[int]:
        scores: Dict[int, float] = {}

        for src in self.sources:
            items = candidates_by_source.get(src.name, [])
            for rank, item in enumerate(items, start=1):
                scores[item] = scores.get(item, 0.0) + src.weight * (1.0 / rank)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in ranked[:k]]