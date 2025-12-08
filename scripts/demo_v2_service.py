from __future__ import annotations

import random

from src.service.recommender_service import V2RecommenderService


def main() -> None:
    service = V2RecommenderService(max_k=50).load()

    # Pick a random user from a reasonable range
    # This is a lightweight demo default.
    user_idx = random.randint(0, 50000)

    print(f"\n[DEMO] Recommendations for user_idx={user_idx}\n")

    recs = service.recommend(user_idx, k=10)
    if not recs:
        print("No recommendations found.")
        return

    for r in recs:
        print(f"- {r.title} | {r.genres} | score={r.score:.4f} | reasons={'; '.join(r.reasons)}")

    print("\n[DEMO] Debug payload (top-5)\n")
    dbg = service.recommend_debug(user_idx, k=5)
    for row in dbg["ranked_top"]:
        print(
            f"* {row['title']} | score={row['score']:.4f} | sources={row['sources']} | reasons={row['reasons']}"
        )


if __name__ == "__main__":
    main()