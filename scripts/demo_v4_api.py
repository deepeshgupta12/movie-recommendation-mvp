from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional  # noqa: UP035

DEFAULT_API_BASE = os.environ.get("V4_API_BASE", "http://127.0.0.1:8004")


def _get(url: str, timeout: int = 30) -> Dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            raw = r.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8")
        except Exception:
            body = ""
        raise RuntimeError(
            f"HTTPError {e.code} for URL:\n{url}\n\nResponse body:\n{body}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"URLError while calling:\n{url}\n\nIs the API running at {DEFAULT_API_BASE}?"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error while calling:\n{url}\n\n{type(e).__name__}: {e}"
        ) from e


def build_url(path: str, params: Optional[Dict[str, Any]] = None) -> str:
    base = DEFAULT_API_BASE.rstrip("/")
    full = f"{base}{path}"
    if not params:
        return full
    qs = urllib.parse.urlencode(params)
    return f"{full}?{qs}"


def main() -> None:
    print("[V4 API DEMO]")
    print(f"[BASE] {DEFAULT_API_BASE}")

    # 1) Health check
    health_url = build_url("/health")
    print(f"[CALL] {health_url}")
    health = _get(health_url)
    print(health)

    # 2) Recommend call (val default)
    user_idx = int(os.environ.get("USER_IDX", "9764"))
    k = int(os.environ.get("K", "20"))
    split = os.environ.get("SPLIT", "val")
    debug = os.environ.get("DEBUG", "0") == "1"

    rec_url = build_url(
        "/recommend",
        {
            "user_idx": user_idx,
            "k": k,
            "include_titles": True,
            "debug": debug,
            "split": split,
        },
    )
    print(f"[CALL] {rec_url}")

    out = _get(rec_url)

    # The API response model expects "recommendations"
    recs = out.get("recommendations", [])
    if not isinstance(recs, list):
        raise RuntimeError(
            f"Unexpected recommend payload shape.\nKeys: {list(out.keys())}"
        )

    print("\nTop-5:")
    for i, r in enumerate(recs[:5], start=1):
        title = r.get("title") or f"item_idx={r.get('item_idx')}"
        score = r.get("score", 0.0)
        reason = r.get("reason", "")
        print(f"{i:02d}. {title} | score={score} | reason={reason}")

    if debug:
        print("\n[DEBUG]")
        dbg = out.get("debug", {})
        if isinstance(dbg, dict):
            for k, v in dbg.items():
                print(f"- {k}: {v}")

    print("\n[DONE] V4 API demo complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERROR]")
        print(str(e))
        sys.exit(1)