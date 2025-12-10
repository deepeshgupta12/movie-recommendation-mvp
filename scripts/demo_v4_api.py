# scripts/demo_v4_api.py

from __future__ import annotations

import json
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8004"


def _get(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    print("[V4 API DEMO]")
    print("[BASE]", BASE)

    health_url = f"{BASE}/health"
    print("[CALL]", health_url)
    try:
        print(json.dumps(_get(health_url), indent=2))
    except Exception as e:
        print("[HEALTH FAILED]", repr(e))
        return

    rec_url = (
        f"{BASE}/recommend?"
        "user_idx=9764&k=20&include_titles=True&debug=True&split=val&apply_diversity=True"
    )
    print("[CALL]", rec_url)

    try:
        out = _get(rec_url)
        recs = out.get("recommendations", [])
        print("\nTop-5:")
        for i, r in enumerate(recs[:5], 1):
            print(f"{i:02d}. {r.get('title')} | score={r.get('score')} | reason={r.get('reason')}")

        if out.get("debug"):
            print("\n[DEBUG]")
            for k, v in out["debug"].items():
                print("-", k + ":", v)

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        print("\n[HTTP ERROR]")
        print(json.dumps({
            "http_error": True,
            "status": e.code,
            "body": body,
            "url": rec_url
        }, indent=2))
    except Exception as e:
        print("\n[UNEXPECTED ERROR]")
        print(repr(e))


if __name__ == "__main__":
    main()