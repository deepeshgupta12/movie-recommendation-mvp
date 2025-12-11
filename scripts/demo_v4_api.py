# scripts/demo_v4_api.py

import json
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8004"


def _get(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        raw = r.read().decode("utf-8")
        try:
            return json.loads(raw)
        except Exception:
            return raw


def main():
    print("[V4 API DEMO]")
    print("[BASE]", BASE)

    health_url = f"{BASE}/health"
    print("[CALL]", health_url)
    try:
        print(json.dumps(_get(health_url), indent=2))
    except Exception as e:
        print("\n[HEALTH FAILED]")
        print(repr(e))
        return

    rec_url = (
        f"{BASE}/recommend?"
        "user_idx=9764&k=20&include_titles=True&debug=True&split=val&apply_diversity=True"
    )
    print("[CALL]", rec_url)

    try:
        out = _get(rec_url)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        print("\n[HTTP ERROR]")
        print(json.dumps({"status": e.code, "body": body, "url": rec_url}, indent=2))
        return
    except Exception as e:
        print("\n[ERROR]")
        print(repr(e))
        return

    recs = out.get("recommendations", [])
    print("\nTop-5:")
    for i, it in enumerate(recs[:5], 1):
        title = it.get("title")
        score = it.get("score")
        reason = it.get("reason")
        bucket = it.get("bucket")
        print(f"{i:02d}. {title} | score={score} | {bucket} | reason={reason}")

    if out.get("debug"):
        print("\n[DEBUG]")
        for k, v in out["debug"].items():
            if k == "feature_order":
                print(f"- {k}: {v}")
            else:
                print(f"- {k}: {v}")


if __name__ == "__main__":
    main()