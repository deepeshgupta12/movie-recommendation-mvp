# scripts/demo_v4_service.py

from src.service.reco_service_v4 import V4RecommenderService, V4ServiceConfig


def main():
    user_idx = 9764

    svc = V4RecommenderService(V4ServiceConfig(split="val"))
    out = svc.recommend(user_idx=user_idx, k=20, include_titles=True, debug=True)

    print("\n[V4 SERVICE DEMO]\n")
    print("user_idx:", out["user_idx"])
    print("k:", out["k"])

    for i, it in enumerate(out["items"], start=1):
        title = it.get("title") or f"item_idx={it['item_idx']}"
        print(
            f"{i:02d}. {title} | score={it['score']:.4f} | reason={it.get('reason')}"
        )

    dbg = out.get("debug") or {}
    if dbg:
        print("\n[DEBUG]")
        for k, v in dbg.items():
            print(f"- {k}: {v}")


if __name__ == "__main__":
    main()