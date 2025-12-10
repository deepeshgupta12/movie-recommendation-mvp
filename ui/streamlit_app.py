# ui/streamlit_app.py

from __future__ import annotations

import time
from typing import Any, Dict, List  # noqa: UP035

import requests
import streamlit as st

API_BASE_DEFAULT = "http://127.0.0.1:8004"
PLACEHOLDER_POSTER = "https://via.placeholder.com/300x450?text=No+Poster"


def _get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def _post(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def _group_by_reason(recs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in recs:
        key = r.get("reason") or "Recommended for you"
        groups.setdefault(key, []).append(r)
    return groups


def _render_card(rec: Dict[str, Any], user_idx: int, api_base: str):
    poster = rec.get("poster_url") or PLACEHOLDER_POSTER
    title = rec.get("title") or f"Item {rec.get('item_idx')}"
    score = rec.get("score")

    with st.container(border=True):
        st.image(poster, use_container_width=True)
        st.markdown(f"**{title}**")
        st.caption(f"Score: {score:.4f} | {rec.get('reason')}")

        c1, c2, c3, c4 = st.columns(4)
        item_idx = rec["item_idx"]

        def send(ev_type: str):
            _post(f"{api_base}/feedback", {
                "user_idx": int(user_idx),
                "item_idx": int(item_idx),
                "event_type": ev_type,
            })

        if c1.button("👍 Like", key=f"like_{user_idx}_{item_idx}"):
            send("liked")
            st.toast("Liked")
        if c2.button("✅ Watched", key=f"watched_{user_idx}_{item_idx}"):
            send("watched")
            st.toast("Marked as watched")
        if c3.button("🕒 Watch later", key=f"wl_{user_idx}_{item_idx}"):
            send("watch_later")
            st.toast("Added to watch later")
        if c4.button("▶️ Start", key=f"start_{user_idx}_{item_idx}"):
            send("started")
            st.toast("Started watching")


def main():
    st.set_page_config(page_title="Movie Reco MVP - V4", layout="wide")
    st.title("Movie Recommendation MVP - V4")
    st.caption("Session-aware + Diversity + Live Feedback Loop")

    with st.sidebar:
        st.subheader("API")
        api_base = st.text_input("Base URL", value=API_BASE_DEFAULT)
        split = st.selectbox("Split", ["val", "test"], index=0)
        apply_diversity = st.checkbox("Apply diversity", value=True)
        debug = st.checkbox("Debug payload", value=False)

        st.divider()
        st.subheader("User")
        user_idx = st.number_input("user_idx", min_value=0, value=9764, step=1)
        k = st.slider("Top-K", 5, 50, 20)

        fetch = st.button("Get recommendations")

    if "last_payload" not in st.session_state:
        st.session_state.last_payload = None
    if "last_fetch_ts" not in st.session_state:
        st.session_state.last_fetch_ts = 0.0

    def do_fetch():
        payload = _get(
            f"{api_base}/recommend",
            {
                "user_idx": int(user_idx),
                "k": int(k),
                "include_titles": True,
                "debug": bool(debug),
                "split": split,
                "apply_diversity": bool(apply_diversity),
            },
        )
        st.session_state.last_payload = payload
        st.session_state.last_fetch_ts = time.time()

    if fetch:
        do_fetch()

    # Auto-refresh after feedback button clicks
    # Streamlit reruns script automatically; we can refresh if we already have payload.
    if st.session_state.last_payload is not None and (time.time() - st.session_state.last_fetch_ts) > 0.2:
        # light refresh hook button
        if st.button("Refresh recommendations"):
            do_fetch()

    payload = st.session_state.last_payload
    if not payload:
        st.info("Select user and click **Get recommendations**.")
        return

    recs = payload.get("recommendations", [])
    if not recs:
        st.warning("No recommendations returned for this user.")
        if debug and payload.get("debug"):
            st.json(payload["debug"])
        return

    # Group display with headings
    groups = _group_by_reason(recs)

    # Keep a predictable order of key buckets
    reason_order = [
        "Because you watched",
        "Similar to your taste",
        "Popular among similar users",
        "Recommended for you",
    ]

    ordered_keys = []
    for ro in reason_order:
        for k_reason in groups.keys():
            if k_reason.startswith(ro):
                if k_reason not in ordered_keys:
                    ordered_keys.append(k_reason)

    for k_reason in groups.keys():
        if k_reason not in ordered_keys:
            ordered_keys.append(k_reason)

    for reason in ordered_keys:
        items = groups.get(reason, [])
        if not items:
            continue

        st.subheader(reason)

        cols = st.columns(5)
        for i, rec in enumerate(items):
            with cols[i % 5]:
                _render_card(rec, user_idx=int(user_idx), api_base=api_base)

    if debug and payload.get("debug"):
        st.divider()
        st.subheader("Debug")
        st.json(payload["debug"])


if __name__ == "__main__":
    main()