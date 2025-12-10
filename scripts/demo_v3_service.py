# scripts/demo_v3_service.py

from __future__ import annotations

from src.service.reco_service_v3 import V3RecommenderService, V3ServiceConfig


def main():
    # Default to TEST split for Step 8.6
    cfg = V3ServiceConfig(split="test")
    svc = V3RecommenderService(cfg)

    user_idx = 9764

    out = svc.recommend(
        user_idx=user_idx,
        k=20,
        include_titles=True,
        debug=True,
    )

    print(f"\n[DEMO V3] Recommendations for user_idx={user_idx} | split={out.get('split')}\n")

    for r in out["recommendations"]:
        title = r.get("title")
        score = r.get("score")
        sources = r.get("sources")
        print(f"- {title} | score={score:.4f} | sources={sources}")

    if out.get("debug"):
        print("\n[DEMO V3] Debug (top-5 raw rows)\n")
        for d in out["debug"].get("top5_raw", []):
            print(d)


if __name__ == "__main__":
    main()