import json
import urllib.request


def _get(url: str):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    base = "http://127.0.0.1:8004"

    print("[V4 API DEMO]")
    print(_get(f"{base}/health"))

    user_idx = 9764
    out = _get(
        f"{base}/recommend?user_idx={user_idx}&k=20&include_titles=true&debug=true&split=val"
    )

    recs = out.get("recommendations") or out.get("items") or []

    print("\nTop-5:")
    for i, rec in enumerate(recs[:5], 1):
        title = rec.get("title", rec.get("item_idx"))
        score = rec.get("score")
        reason = rec.get("reason")
        print(f"{i:02d}. {title} | score={score} | reason={reason}")

    dbg = out.get("debug") or {}
    print("\n[DEBUG]")
    for k in [
        "split",
        "candidates_path",
        "session_features_path",
        "ranker_path",
        "model_auc",
    ]:
        if k in dbg:
            print(f"- {k}: {dbg[k]}")


if __name__ == "__main__":
    main()