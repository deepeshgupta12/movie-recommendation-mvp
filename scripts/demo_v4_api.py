# scripts/demo_v4_api.py
from __future__ import annotations

import json
import urllib.error
import urllib.request


def _get(url: str, timeout: int = 30):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            raw = r.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8")
            return {"http_error": True, "status": e.code, "body": body, "url": url}
        except Exception:
            return {"http_error": True, "status": e.code, "url": url}
    except Exception as e:
        return {"remote_disconnected": True, "detail": str(e), "url": url}


def main():
    base = "http://127.0.0.1:8004"
    print("[V4 API DEMO]")
    print("[BASE]", base)

    health_url = f"{base}/health"
    print("[CALL]", health_url)
    print(json.dumps(_get(health_url), indent=2))

    rec_url = (
        f"{base}/recommend?"
        "user_idx=9764&k=20&include_titles=True&debug=True&split=val"
    )
    print("[CALL]", rec_url)
    out = _get(rec_url)

    if out.get("remote_disconnected"):
        print("\n[SERVER DROPPED CONNECTION]")
        print(json.dumps(out, indent=2))
        print(
            "\nLikely causes:\n"
            "1) uvicorn reload subprocess crash\n"
            "2) fatal error during V4 service recommend()\n"
            "3) native lib abort (rare)\n\n"
            "Next check: run API WITHOUT --reload and watch logs:\n"
            "  python -m uvicorn src.service.api_v4:app --port 8004\n"
        )
        return

    if out.get("http_error"):
        print("\n[HTTP ERROR]")
        print(json.dumps(out, indent=2))
        return

    items = out.get("items", [])
    print("\nTop-5:")
    for i, it in enumerate(items[:5], start=1):
        print(f"{i:02d}. {it.get('title','')} | score={it.get('score')} | reason={it.get('reason')}")

    dbg = out.get("debug", {})
    if dbg:
        print("\n[DEBUG]")
        for k, v in dbg.items():
            print(f"- {k}: {v}")


if __name__ == "__main__":
    main()