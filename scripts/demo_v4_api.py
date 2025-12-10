from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8004"


def _pretty(obj):
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)


def _get(url: str):
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            raw = r.read().decode("utf-8")
            try:
                return json.loads(raw)
            except Exception:
                return {"raw": raw}

    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8")
        except Exception:
            pass
        return {
            "http_error": e.code,
            "reason": str(e.reason),
            "body": body,
            "url": url,
        }

    except http.client.RemoteDisconnected as e:
        return {
            "remote_disconnected": True,
            "detail": str(e),
            "url": url,
        }

    except urllib.error.URLError as e:
        return {
            "url_error": True,
            "detail": str(e.reason),
            "url": url,
        }

    except Exception as e:
        return {
            "unexpected_client_error": type(e).__name__,
            "detail": str(e),
            "url": url,
        }


def main():
    print("[V4 API DEMO]")
    print("[BASE]", BASE)

    health_url = f"{BASE}/health"
    print("[CALL]", health_url)
    health = _get(health_url)
    print(_pretty(health))

    rec_url = (
        f"{BASE}/recommend"
        f"?user_idx=9764"
        f"&k=20"
        f"&include_titles=True"
        f"&debug=True"
        f"&split=val"
    )
    print("[CALL]", rec_url)

    out = _get(rec_url)

    if out.get("remote_disconnected"):
        print("\n[SERVER DROPPED CONNECTION]")
        print(_pretty(out))
        print(
            "\nLikely causes:\n"
            "1) uvicorn reload subprocess crash\n"
            "2) fatal error during V4 service recommend()\n"
            "3) native lib abort (rare, but possible with heavy Polars ops)\n\n"
            "Next check: run API WITHOUT --reload and watch logs."
        )
        return

    if "http_error" in out or out.get("url_error") or out.get("unexpected_client_error"):
        print("\n[ERROR PAYLOAD]")
        print(_pretty(out))
        return

    recs = out.get("recommendations") or out.get("items") or []
    print("\nTop-5:")
    for i, r in enumerate(recs[:5], 1):
        print(f"{i:02d}. {r.get('title')} | score={r.get('score')} | reason={r.get('reason')}")

    dbg = out.get("debug") or {}
    if dbg:
        print("\n[DEBUG]")
        for k in sorted(dbg.keys()):
            print("-", k, ":", dbg.get(k))


if __name__ == "__main__":
    main()