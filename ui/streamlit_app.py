import json
from pathlib import Path
from typing import Any, Dict, List, Optional  # noqa: UP035

import requests
import streamlit as st

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
POSTER_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "poster_cache_v4.json"

DEFAULT_API_BASE = "http://127.0.0.1:8004"
DEFAULT_SPLIT = "val"
DEFAULT_USER_IDX = 9764

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_poster_cache() -> Dict[str, str]:
    if POSTER_CACHE_PATH.exists():
        try:
            return json.loads(POSTER_CACHE_PATH.read_text())
        except Exception:
            # If cache is corrupt, just ignore it
            return {}
    return {}


POSTER_CACHE = load_poster_cache()


def _get(url: str, timeout: int = 60) -> Dict[str, Any]:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _post(url: str, payload: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    if not resp.content:
        return {}
    return resp.json()


def resolve_poster_url(rec: Dict[str, Any]) -> Optional[str]:
    """
    Poster resolution strategy:
    1) If API already attached poster_url -> use that.
    2) Else look up by movieId in poster_cache_v4.json.
    3) Else return None (UI will render a text placeholder).
    """
    if rec.get("poster_url"):
        return rec["poster_url"]

    movie_id = rec.get("movieId") or rec.get("movie_id")
    if movie_id is None:
        return None

    # poster_cache_v4.json likely has string keys
    key_str = str(movie_id)
    if key_str in POSTER_CACHE:
        return POSTER_CACHE[key_str]

    # just in case keys are numeric
    return POSTER_CACHE.get(movie_id)


def group_recommendations(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Very simple grouping based on the `reason` field.
    This mirrors the intent of:
      - "Similar to your taste"
      - "Popular among similar users"
      - Future buckets like "Continue watching" can be wired later.
    """
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "Similar to your taste": [],
        "Popular among similar users": [],
        "Other recommendations": [],
    }
    for rec in items:
        reason = (rec.get("reason") or "").lower()
        if "similar to your taste" in reason:
            buckets["Similar to your taste"].append(rec)
        elif "popular among similar users" in reason or "popular" in reason:
            buckets["Popular among similar users"].append(rec)
        else:
            buckets["Other recommendations"].append(rec)
    # Drop empty buckets
    return {k: v for k, v in buckets.items() if v}


def send_feedback(api_base: str, user_idx: int, rec: Dict[str, Any], event: str) -> None:
    """
    Send feedback to /feedback.

    Allowed events (as per API v4 schema):
      - like
      - remove_like
      - watched
      - watch_later
      - remove_watch_later
      - skip
    """
    payload = {
        "user_idx": user_idx,
        "item_idx": rec.get("item_idx"),
        "movieId": rec.get("movieId"),
        "event": event,
        "score": rec.get("score"),
        "reason": rec.get("reason"),
    }
    try:
        out = _post(f"{api_base}/feedback", payload)
        st.toast(f"Feedback '{event}' recorded.", icon="✅")
        if out.get("detail"):
            st.caption(f"Server detail: {out['detail']}")
    except requests.HTTPError as e:
        try:
            detail = e.response.text
        except Exception:
            detail = str(e)
        st.error(f"Feedback failed: {detail}")
    except Exception as e:
        st.error(f"Feedback failed: {e}")


def render_card(rec: Dict[str, Any], user_idx: int, api_base: str) -> None:
    poster_url = resolve_poster_url(rec)
    title = rec.get("title") or f"Item {rec.get('item_idx')}"
    score = rec.get("score")
    reason = rec.get("reason") or ""

    with st.container(border=True):
        # Poster
        if poster_url:
            st.image(poster_url, use_column_width=True)
        else:
            # Simple placeholder so we never show a broken image
            st.markdown(
                f"""
                <div style="
                    height: 280px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    border-radius: 4px;
                    border: 1px solid #444;
                    font-size: 0.85rem;
                    text-align: center;
                    padding: 0.5rem;
                ">
                    Poster unavailable<br/><span style="opacity:0.7;">{title}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Title + score + reason
        st.markdown(f"**{title}**")
        if score is not None:
            st.caption(f"Score: {score:.4f} | {reason}")
        else:
            st.caption(reason)

        # Feedback buttons
        c1, c2, c3, c4 = st.columns(4)
        if c1.button("Like", key=f"like-{user_idx}-{rec.get('item_idx')}"):
            send_feedback(api_base, user_idx, rec, "like")
        if c2.button("Watched", key=f"watched-{user_idx}-{rec.get('item_idx')}"):
            send_feedback(api_base, user_idx, rec, "watched")
        if c3.button("Watch later", key=f"watchlater-{user_idx}-{rec.get('item_idx')}"):
            send_feedback(api_base, user_idx, rec, "watch_later")
        # Interpret "Start" as a lightweight "watched" signal for now
        if c4.button("Start", key=f"start-{user_idx}-{rec.get('item_idx')}"):
            send_feedback(api_base, user_idx, rec, "watched")


# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Movie Recommendation MVP - V4",
        layout="wide",
    )

    st.title("Movie Recommendation MVP - V4")
    st.caption("Session-aware • Diversity • Live Feedback Loop")

    # Sidebar – controls
    st.sidebar.header("Mode")
    mode = st.sidebar.radio(
        "Select version",
        options=["V4 (Session-aware)", "V3 (Feedback-loop)"],
        index=0,
        help="V3 path is still available for comparison; main focus is V4.",
    )

    st.sidebar.header("API")
    api_base = st.sidebar.text_input("Base URL", DEFAULT_API_BASE)
    split = st.sidebar.selectbox("Split", options=["val", "test"], index=0)

    st.sidebar.header("Request")
    user_idx = st.sidebar.number_input(
        "user_idx",
        min_value=0,
        max_value=200000,
        value=DEFAULT_USER_IDX,
        step=1,
    )
    k = st.sidebar.slider("k", min_value=5, max_value=50, value=20, step=1)
    include_titles = st.sidebar.checkbox("include_titles", value=True)
    debug = st.sidebar.checkbox("debug", value=False)
    apply_diversity = st.sidebar.checkbox("apply_diversity (V4)", value=True)

    # Health check
    health_ok = False
    try:
        health = _get(f"{api_base}/health")
        st.success(f"API ok: {health}")
        health_ok = True
    except Exception as e:
        st.error(f"Health check failed: {e}")

    # Do not proceed if health fails
    if not health_ok:
        return

    # Only V4 UI is implemented in this file; V3 keeps old behaviour via its own route
    if mode.startswith("V3"):
        st.info("V3 UI path is not wired here. Use the older V3 app/script for comparison.")
        return

    if st.button("Get Recommendations"):
        # Build URL exactly like demo_v4_api does
        params = {
            "user_idx": int(user_idx),
            "k": int(k),
            "include_titles": bool(include_titles),
            "debug": bool(debug),
            "split": split,
            "apply_diversity": bool(apply_diversity),
        }
        try:
            resp = requests.get(f"{api_base}/recommend", params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.HTTPError as e:
            try:
                body = e.response.text
            except Exception:
                body = str(e)
            st.error(f"Recommend failed: {body}")
            return
        except Exception as e:
            st.error(f"Recommend failed: {e}")
            return

        items = data.get("items") or data.get("recommendations") or []

        if not items:
            st.info("No recommendations returned.")
            return

        # Group into buckets
        buckets = group_recommendations(items)

        # Optional debug block
        if debug and "debug" in data:
            with st.expander("Debug payload from API"):
                st.json(data["debug"])

        # Render each section
        for section_name, recs in buckets.items():
            st.subheader(section_name)
            cols = st.columns(5)
            for i, rec in enumerate(recs[: k ]):
                with cols[i % 5]:
                    render_card(rec, int(user_idx), api_base)


if __name__ == "__main__":
    main()