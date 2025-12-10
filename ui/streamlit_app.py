from __future__ import annotations

import json
from typing import Dict, List, Optional  # noqa: UP035

import requests
import streamlit as st

# ---------- Config ----------
st.set_page_config(
    page_title="Movie Recommendation MVP",
    layout="wide",
)

DEFAULT_API_V4 = "http://127.0.0.1:8004"
DEFAULT_API_V3 = "http://127.0.0.1:8003"


# ---------- Helpers ----------
def _api_get(url: str, params: dict):
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _api_post(url: str, payload: dict):
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def _render_card(item: dict, api_base: str, split: str):
    title = item.get("title") or f"item_idx:{item.get('item_idx')}"
    poster_url = item.get("poster_url")

    with st.container(border=True):
        if poster_url:
            st.image(poster_url, use_container_width=True)
        else:
            # Light placeholder block
            st.markdown(
                "<div style='height:240px; background:#111827; border-radius:8px; "
                "display:flex; align-items:center; justify-content:center; color:#9CA3AF;'>"
                "Poster unavailable"
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown(f"**{title}**")
        st.caption(f"item_idx: {item.get('item_idx')} | movieId: {item.get('movieId')}")
        st.write(item.get("reason", ""))
        st.caption(f"score: {round(float(item.get('score', 0.0)), 4)}")

        cols = st.columns(4)
        user_idx = st.session_state.get("user_idx")

        def send(event: str):
            if user_idx is None:
                return
            payload = {
                "user_idx": int(user_idx),
                "item_idx": int(item["item_idx"]),
                "event": event,
                "split": split,
            }
            _api_post(f"{api_base}/feedback", payload)

        if cols[0].button("Start", key=f"start_{split}_{item['item_idx']}"):
            send("start")
            st.session_state["refresh"] = True

        if cols[1].button("Watched", key=f"watched_{split}_{item['item_idx']}"):
            send("watched")
            st.session_state["refresh"] = True

        if cols[2].button("Like", key=f"like_{split}_{item['item_idx']}"):
            send("like")
            st.session_state["refresh"] = True

        if cols[3].button("Watch Later", key=f"wl_{split}_{item['item_idx']}"):
            send("watch_later")
            st.session_state["refresh"] = True


def _render_section(title: str, items: List[dict], api_base: str, split: str):
    if not items:
        return
    st.subheader(title)
    # 5-column grid
    cols = st.columns(5)
    for i, item in enumerate(items):
        with cols[i % 5]:
            _render_card(item, api_base=api_base, split=split)


# ---------- Sidebar ----------
st.sidebar.title("Mode")

version = st.sidebar.radio(
    "Select version",
    ["V4 (Session-aware + Online Feedback Overlay)", "V3 (Feedback-loop)"],
    index=0,
)

st.sidebar.divider()

st.sidebar.title("API")
api_base = st.sidebar.text_input(
    "Base URL",
    value=DEFAULT_API_V4 if version.startswith("V4") else DEFAULT_API_V3,
)

split = st.sidebar.selectbox("Split", ["val", "test"], index=0)

st.sidebar.divider()

st.sidebar.title("Request")
user_idx = st.sidebar.number_input("user_idx", min_value=0, value=9764, step=1)
k = st.sidebar.slider("k", min_value=5, max_value=50, value=20, step=1)
include_titles = st.sidebar.checkbox("include_titles", value=True)
debug = st.sidebar.checkbox("debug", value=False)
apply_feedback = st.sidebar.checkbox("apply_feedback_overlay (V4)", value=True)

if st.sidebar.button("Reset My Feedback"):
    try:
        _api_post(f"{api_base}/feedback", {"user_idx": int(user_idx), "item_idx": -1, "event": "reset", "split": split})
        st.sidebar.success("Feedback reset.")
    except Exception as e:
        st.sidebar.error(str(e))


# ---------- Main ----------
st.title("Movie Recommendation MVP")

st.session_state["user_idx"] = int(user_idx)
st.session_state.setdefault("refresh", False)

# Health check
try:
    health = _api_get(f"{api_base}/health", {})
    st.success(f"API ok: {health}")
except Exception as e:
    st.error(f"API not reachable: {e}")

get_btn = st.button("Get Recommendations")

should_fetch = get_btn or st.session_state.get("refresh", False)

if should_fetch:
    st.session_state["refresh"] = False
    params = {
        "user_idx": int(user_idx),
        "k": int(k),
        "include_titles": bool(include_titles),
        "debug": bool(debug),
        "split": split,
    }
    if version.startswith("V4"):
        params["apply_feedback"] = bool(apply_feedback)

    try:
        rec = _api_get(f"{api_base}/recommend", params)
        st.session_state["last_rec"] = rec
    except Exception as e:
        st.error(f"Recommend failed: {e}")

rec = st.session_state.get("last_rec")

if rec:
    sections = rec.get("sections") or {}
    items = rec.get("items") or rec.get("recommendations") or []

    # Continue Watching row first (Netflix feel)
    _render_section("Continue Watching", sections.get("continue_watching", []), api_base, split)

    # Session-aware row
    _render_section("Because you watched", sections.get("because_you_watched", []), api_base, split)

    # Taste row
    _render_section("Similar to your taste", sections.get("similar_to_your_taste", []), api_base, split)

    # Popular row
    _render_section("Popular among similar users", sections.get("popular_among_similar_users", []), api_base, split)

    # Fallback row
    _render_section("More for you", sections.get("more_for_you", []), api_base, split)

    # If API didn't provide sections, fallback to flat render
    if not sections and items:
        st.subheader("Top Picks For You")
        cols = st.columns(5)
        for i, item in enumerate(items):
            with cols[i % 5]:
                _render_card(item, api_base=api_base, split=split)

    if debug and rec.get("debug"):
        st.divider()
        st.subheader("Debug")
        st.code(json.dumps(rec["debug"], indent=2))