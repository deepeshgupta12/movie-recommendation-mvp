from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional  # noqa: UP035

import requests
import streamlit as st

DEFAULT_API = "http://127.0.0.1:8004"


def _safe_get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _safe_post(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_recommendations(api_base: str, user_idx: int, k: int, split: str, debug: bool = False):
    return _safe_get(
        f"{api_base}/recommend",
        {
            "user_idx": user_idx,
            "k": k,
            "include_titles": True,
            "debug": debug,
            "split": split,
        },
    )


def send_feedback(api_base: str, user_idx: int, item_idx: int, event: str):
    return _safe_post(
        f"{api_base}/feedback",
        {"user_idx": user_idx, "item_idx": item_idx, "event": event},
    )


def group_by_category(recs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    g = defaultdict(list)
    for r in recs:
        g[r.get("category", "Recommended")].append(r)
    # Stable order preference
    order = [
        "Because you watched",
        "Similar to your taste",
        "Popular among similar users",
        "Recommended",
    ]
    out = {}
    for k in order:
        if k in g:
            out[k] = g[k]
    for k, v in g.items():
        if k not in out:
            out[k] = v
    return out


st.set_page_config(page_title="Movie Recommendation MVP - V4", layout="wide")

st.title("Movie Recommendation MVP - V4")
st.caption("Session-aware + diversity-aware ranking with live feedback loop and cached posters.")

with st.sidebar:
    st.header("Controls")
    api_base = st.text_input("API base URL", value=DEFAULT_API)
    split = st.selectbox("Split", options=["val", "test"], index=0)
    user_idx = st.number_input("User index", min_value=0, value=9764, step=1)
    k = st.slider("Top-K", min_value=5, max_value=50, value=20, step=5)
    debug = st.checkbox("Debug payload", value=False)

    st.divider()
    st.write("Make sure API is running:")
    st.code("python -m uvicorn src.service.api_v4:app --reload --port 8004")

if "last_recs" not in st.session_state:
    st.session_state.last_recs = None

if "last_payload" not in st.session_state:
    st.session_state.last_payload = None


colA, colB = st.columns([1, 1])

with colA:
    if st.button("Get recommendations", type="primary"):
        try:
            payload = fetch_recommendations(api_base, int(user_idx), int(k), split, debug)
            st.session_state.last_payload = payload
            st.session_state.last_recs = payload.get("recommendations", [])
        except Exception as e:
            st.error(f"Failed to fetch recommendations: {e}")

with colB:
    if st.button("Refresh after feedback"):
        try:
            payload = fetch_recommendations(api_base, int(user_idx), int(k), split, debug)
            st.session_state.last_payload = payload
            st.session_state.last_recs = payload.get("recommendations", [])
        except Exception as e:
            st.error(f"Failed to refresh: {e}")


recs = st.session_state.last_recs or []

if not recs:
    st.info("Click 'Get recommendations' to load results.")
    st.stop()


grouped = group_by_category(recs)

for category, items in grouped.items():
    st.subheader(category)

    # Horizontal grid
    rows = (len(items) + 4) // 5
    idx = 0

    for _ in range(rows):
        cols = st.columns(5)
        for c in cols:
            if idx >= len(items):
                break

            it = items[idx]
            idx += 1

            title = it.get("title") or "Untitled"
            poster = it.get("poster_url")
            score = it.get("score", 0.0)
            item_idx = int(it.get("item_idx"))

            with c:
                # Poster always shows: service guarantees placeholder if missing
                st.image(poster, use_container_width=True)

                st.markdown(f"**{title}**")
                st.caption(f"score: {score:.4f}")
                st.caption(it.get("reason", ""))

                # Interaction row
                b1, b2 = st.columns(2)
                b3, b4 = st.columns(2)

                with b1:
                    if st.button("Like", key=f"like_{category}_{item_idx}"):
                        try:
                            send_feedback(api_base, int(user_idx), item_idx, "like")
                            st.success("Liked")
                        except Exception as e:
                            st.error(str(e))

                with b2:
                    if st.button("Watched", key=f"watched_{category}_{item_idx}"):
                        try:
                            send_feedback(api_base, int(user_idx), item_idx, "watched")
                            st.success("Marked watched")
                        except Exception as e:
                            st.error(str(e))

                with b3:
                    if st.button("Watch later", key=f"later_{category}_{item_idx}"):
                        try:
                            send_feedback(api_base, int(user_idx), item_idx, "watch_later")
                            st.success("Saved for later")
                        except Exception as e:
                            st.error(str(e))

                with b4:
                    if st.button("Skip", key=f"skip_{category}_{item_idx}"):
                        try:
                            send_feedback(api_base, int(user_idx), item_idx, "skip")
                            st.success("Skipped")
                        except Exception as e:
                            st.error(str(e))


if debug and st.session_state.last_payload:
    st.divider()
    st.subheader("Debug")
    st.json(st.session_state.last_payload.get("debug", {}))