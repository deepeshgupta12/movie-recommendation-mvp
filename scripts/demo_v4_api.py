from __future__ import annotations

import json
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8004"


def _get(url: str):
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
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
    except urllib.error.URLError as e:
        return {"url_error": str(e.reason), "url": url}


def main():
    print("[V4 API DEMO]")
    print("[BASE]", BASE)

    health_url = f"{BASE}/health"
    print("[CALL]", health_url)
    print(_get(health_url))

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

    if "http_error" in out or "url_error" in out:
        print("\n[ERROR PAYLOAD]")
        print(out)
        return

    # OK case
    recs = out.get("recommendations") or []
    print("\nTop-5:")
    for i, r in enumerate(recs[:5], 1):
        title = r.get("title")
        score = r.get("score")
        reason = r.get("reason")
        print(f"{i:02d}. {title} | score={score} | reason={reason}")

    dbg = out.get("debug") or {}
    if dbg:
        print("\n[DEBUG KEYS]")
        for k in sorted(dbg.keys()):
            print("-", k, ":", dbg.get(k))


if __name__ == "__main__":
    main()