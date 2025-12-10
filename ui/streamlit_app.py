# ui/streamlit_app.py
from __future__ import annotations

import json
from typing import Any, Dict, List  # noqa: UP035

import requests
import streamlit as st

DEFAULT_BASE = "http://127.0.0.1:8004"
PLACEHOLDER_POSTER = "https://via.placeholder.com/300x450?text=No+Poster"


def api_get(url: str) -> Dict[str, Any]:
    try:
        r = requests.get(url, timeout=20)
        return r.json()
    except Exception as e:
        return {"error": str(e), "url": url}


def api_post(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        r = requests.post(url, json=payload, timeout=20)
        return r.json()
    except Exception as e:
        return {"error": str(e), "url": url, "payload": payload}


def group_by_reason(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "Because you watched": [],
        "Similar to your taste": [],
        "Popular among similar users": [],
        "Recommended for you": [],
    }

    for it in items:
        r = str(it.get("reason") or "Recommended for you")
        key = r
        if r.startswith("Because you watched"):
            key = "Because you watched"
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(it)

    # remove empty
    return {k: v for k, v in buckets.items() if v}


def render_card(base: str, split: str, user_idx: int, it: Dict[str, Any]):
    title = it.get("title", "")
    item_idx = it.get("item_idx")
    poster = it.get("poster_url") or PLACEHOLDER_POSTER
    score = it.get("score")
    reason = it.get("reason", "")

    with st.container(border=True):
        cols = st.columns([1, 3])
        with cols[0]:
            st.image(poster, use_container_width=True)

        with cols[1]:
            st.markdown(f"**{title}**")
            st.caption(f"Score: {score:.4f}" if isinstance(score, (int, float)) else f"Score: {score}")
            st.caption(reason)

            bcols = st.columns(4)
            if bcols[0].button("Like", key=f"like-{user_idx}-{item_idx}-{reason}"):
                api_post(f"{base}/feedback?split={split}", {"user_idx": user_idx, "item_idx": item_idx, "event": "like"})
                st.toast("Liked")

            if bcols[1].button("Watched", key=f"watched-{user_idx}-{item_idx}-{reason}"):
                api_post(f"{base}/feedback?split={split}", {"user_idx": user_idx, "item_idx": item_idx, "event": "watched"})
                st.toast("Marked watched")

            if bcols[2].button("Watch Later", key=f"later-{user_idx}-{item_idx}-{reason}"):
                api_post(f"{base}/feedback?split={split}", {"user_idx": user_idx, "item_idx": item_idx, "event": "watch_later"})
                st.toast("Added to watch later")

            if bcols[3].button("Dislike", key=f"dislike-{user_idx}-{item_idx}-{reason}"):
                api_post(f"{base}/feedback?split={split}", {"user_idx": user_idx, "item_idx": item_idx, "event": "dislike"})
                st.toast("Disliked")


def main():
    st.set_page_config(page_title="Movie Reco MVP - V4", layout="wide")

    st.title("Movie Recommendation MVP - V4")
    st.caption("Session-aware ranking + live feedback + poster cache-first rendering")

    with st.sidebar:
        base = st.text_input("API Base", value=DEFAULT_BASE)
        split = st.selectbox("Split", ["val", "test"], index=0)
        user_idx = st.number_input("User Index", min_value=0, value=9764, step=1)
        k = st.slider("K", min_value=5, max_value=50, value=20, step=1)
        include_titles = st.checkbox("Include titles", value=True)
        debug = st.checkbox("Debug", value=False)

        st.divider()
        if st.button("Refresh recommendations"):
            st.session_state["refresh_token"] = st.session_state.get("refresh_token", 0) + 1

    # Load user state (so you can see clicks are actually registered)
    st_state = api_get(f"{base}/user_state?user_idx={user_idx}&split={split}")
    with st.expander("Your interaction state (live)"):
        st.json(st_state)

    # Get recommendations
    rec_url = (
        f"{base}/recommend?"
        f"user_idx={user_idx}&k={k}&include_titles={str(include_titles)}&debug={str(debug)}&split={split}"
    )
    out = api_get(rec_url)

    if out.get("error") or out.get("recommend_failed"):
        st.error("Failed to fetch recommendations.")
        st.json(out)
        return

    items = out.get("items", [])

    if not items:
        st.warning("No recommendations returned for this user.")
        st.json(out)
        return

    # Grouped sections
    buckets = group_by_reason(items)

    for heading, group in buckets.items():
        st.subheader(heading)
        for it in group:
            render_card(base, split, int(user_idx), it)

    if debug and out.get("debug"):
        st.divider()
        st.subheader("Debug")
        st.json(out["debug"])


if __name__ == "__main__":
    main()