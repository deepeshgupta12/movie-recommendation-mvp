# ui/streamlit_app.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional  # noqa: UP035

import requests
import streamlit as st

# ------------------------------
# Config
# ------------------------------

DEFAULT_API_BASE = os.getenv("V3_API_BASE", "http://127.0.0.1:8003")

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()
OMDB_API_KEY = os.getenv("OMDB_API_KEY", "").strip()

POSTER_PLACEHOLDER = "https://via.placeholder.com/342x513.png?text=Poster+Unavailable"

CARDS_PER_ROW = 5
DEFAULT_USER_IDX = 9764
DEFAULT_K = 20


# ------------------------------
# Data model
# ------------------------------

@dataclass
class RecoItem:
    item_idx: int
    title: str
    score: float
    reason: Optional[str] = None
    has_tt: int = 0
    has_seq: int = 0
    has_v2: int = 0
    poster_url: Optional[str] = None


# ------------------------------
# API helpers
# ------------------------------

@st.cache_data(show_spinner=False, ttl=60)
def _api_get_recommendations(
    api_base: str,
    user_idx: int,
    k: int,
    include_titles: bool,
    debug: bool,
    split: str,
) -> Dict[str, Any]:
    url = f"{api_base}/recommend"
    params = {
        "user_idx": int(user_idx),
        "k": int(k),
        "include_titles": bool(include_titles),
        "debug": bool(debug),
        "split": split,
    }
    r = requests.get(url, params=params, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"API error {r.status_code}: {r.text}")
    return r.json()


def _api_post_feedback(api_base: str, user_idx: int, item_idx: int, action: str) -> Dict[str, Any]:
    url = f"{api_base}/feedback"
    payload = {"user_idx": int(user_idx), "item_idx": int(item_idx), "action": action}
    r = requests.post(url, json=payload, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"API error {r.status_code}: {r.text}")
    return r.json()


@st.cache_data(show_spinner=False, ttl=10)
def _api_get_user_state(api_base: str, user_idx: int) -> Dict[str, Any]:
    url = f"{api_base}/user_state"
    params = {"user_idx": int(user_idx)}
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"API error {r.status_code}: {r.text}")
    return r.json()


# ------------------------------
# Poster resolvers
# ------------------------------

@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def _tmdb_search_poster(title: str) -> Optional[str]:
    if not TMDB_API_KEY:
        return None
    try:
        url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": TMDB_API_KEY, "query": title}
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        results = data.get("results") or []
        if not results:
            return None
        poster_path = results[0].get("poster_path")
        if not poster_path:
            return None
        return f"https://image.tmdb.org/t/p/w342{poster_path}"
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def _omdb_search_poster(title: str) -> Optional[str]:
    if not OMDB_API_KEY:
        return None
    try:
        url = "https://www.omdbapi.com/"
        params = {"apikey": OMDB_API_KEY, "t": title}
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        poster = data.get("Poster")
        if poster and poster != "N/A":
            return poster
        return None
    except Exception:
        return None


def _resolve_poster(item: RecoItem) -> str:
    if item.poster_url:
        return item.poster_url
    p = _tmdb_search_poster(item.title)
    if p:
        return p
    p = _omdb_search_poster(item.title)
    if p:
        return p
    return POSTER_PLACEHOLDER


# ------------------------------
# Normalization & bucketing
# ------------------------------

def _normalize_items(payload: Dict[str, Any]) -> List[RecoItem]:
    raw = payload.get("items") or payload.get("recommendations") or []
    out: List[RecoItem] = []
    for r in raw:
        out.append(
            RecoItem(
                item_idx=int(r.get("item_idx")),
                title=str(r.get("title") or "Unknown Title"),
                score=float(r.get("score", 0.0)),
                reason=r.get("reason"),
                has_tt=int(r.get("has_tt", 0)),
                has_seq=int(r.get("has_seq", 0)),
                has_v2=int(r.get("has_v2", 0)),
                poster_url=r.get("poster_url"),
            )
        )
    return out


def _bucket_label(item: RecoItem) -> str:
    if item.has_seq:
        return "Because you watched recently"
    if item.has_tt:
        return "Similar to your taste"
    if item.has_v2:
        return "Popular among similar users"
    if item.reason:
        if "Because you watched" in item.reason:
            return "Because you watched recently"
        if "Similar to your taste" in item.reason:
            return "Similar to your taste"
        if "Popular among similar users" in item.reason:
            return "Popular among similar users"
    return "More picks for you"


def _group_by_bucket(items: List[RecoItem]) -> Dict[str, List[RecoItem]]:
    buckets: Dict[str, List[RecoItem]] = {}
    for it in items:
        key = _bucket_label(it)
        buckets.setdefault(key, []).append(it)
    return buckets


# ------------------------------
# UI render
# ------------------------------

def _render_card(
    api_base: str,
    user_idx: int,
    item: RecoItem,
    state: Dict[str, List[int]],
):
    poster = _resolve_poster(item)
    st.image(poster, use_container_width=True)
    st.markdown(
        f"""
        <div style="font-weight:600; font-size: 0.95rem; line-height:1.2;">
            {item.title}
        </div>
        """,
        unsafe_allow_html=True,
    )
    if item.reason:
        st.caption(item.reason)

    st.caption(f"Score {item.score:.4f}")

    liked = set(state.get("liked", []))
    watched = set(state.get("watched", []))
    started = set(state.get("started", []))

    c1, c2, c3 = st.columns(3)

    with c1:
        if item.item_idx in started:
            if st.button("Started", key=f"unstart_{item.item_idx}"):
                _api_post_feedback(api_base, user_idx, item.item_idx, "unstart")
                _api_get_user_state.clear()
                _api_get_recommendations.clear()
                st.rerun()
        else:
            if st.button("Start", key=f"start_{item.item_idx}"):
                _api_post_feedback(api_base, user_idx, item.item_idx, "start")
                _api_get_user_state.clear()
                st.rerun()

    with c2:
        if item.item_idx in watched:
            if st.button("Watched ✓", key=f"unwatch_{item.item_idx}"):
                _api_post_feedback(api_base, user_idx, item.item_idx, "unwatch")
                _api_get_user_state.clear()
                _api_get_recommendations.clear()
                st.rerun()
        else:
            if st.button("Watched", key=f"watched_{item.item_idx}"):
                _api_post_feedback(api_base, user_idx, item.item_idx, "watched")
                _api_get_user_state.clear()
                _api_get_recommendations.clear()
                st.rerun()

    with c3:
        if item.item_idx in liked:
            if st.button("Liked ✓", key=f"unlike_{item.item_idx}"):
                _api_post_feedback(api_base, user_idx, item.item_idx, "unlike")
                _api_get_user_state.clear()
                _api_get_recommendations.clear()
                st.rerun()
        else:
            if st.button("Like", key=f"like_{item.item_idx}"):
                _api_post_feedback(api_base, user_idx, item.item_idx, "like")
                _api_get_user_state.clear()
                _api_get_recommendations.clear()
                st.rerun()


def _render_row(
    api_base: str,
    user_idx: int,
    items: List[RecoItem],
    state: Dict[str, List[int]],
):
    cols = st.columns(CARDS_PER_ROW)
    for i, it in enumerate(items):
        with cols[i % CARDS_PER_ROW]:
            _render_card(api_base, user_idx, it, state)


def _render_bucket(
    api_base: str,
    user_idx: int,
    title: str,
    items: List[RecoItem],
    state: Dict[str, List[int]],
):
    if not items:
        return
    st.subheader(title)
    for i in range(0, len(items), CARDS_PER_ROW):
        _render_row(api_base, user_idx, items[i:i + CARDS_PER_ROW], state)


# ------------------------------
# Page
# ------------------------------

def main():
    st.set_page_config(page_title="Movie Recommendation MVP V3", layout="wide")
    st.title("Movie Recommendation MVP — V3")

    with st.sidebar:
        st.markdown("### V3 Controls")
        api_base = st.text_input("API Base", value=DEFAULT_API_BASE)
        split = st.selectbox("Split", options=["test", "val"], index=0)
        user_idx = st.number_input("user_idx", min_value=0, value=DEFAULT_USER_IDX, step=1)
        k = st.slider("Top-K", min_value=5, max_value=200, value=DEFAULT_K)
        include_titles = st.checkbox("Include titles", value=True)
        debug = st.checkbox("Debug payload", value=False)

        st.markdown("---")
        st.markdown("### Poster Providers (optional)")
        st.code(
            "export TMDB_API_KEY='...'\nexport OMDB_API_KEY='...'\nexport V3_API_BASE='http://127.0.0.1:8003'",
            language="bash",
        )

        if st.button("Clear UI caches"):
            _api_get_user_state.clear()
            _api_get_recommendations.clear()
            _tmdb_search_poster.clear()
            _omdb_search_poster.clear()
            st.success("Cleared caches.")

    # Server state
    try:
        state = _api_get_user_state(api_base, int(user_idx))
    except Exception:
        state = {"liked": [], "watched": [], "started": []}

    # Continue Watching
    st.subheader("Continue Watching")
    started_ids = state.get("started", [])
    if not started_ids:
        st.info("No items here yet. Click Start on a movie to build your row.")
    else:
        # Build placeholder items for display; titles will render if later enriched by service
        cw_items = [RecoItem(item_idx=i, title="(from activity)", score=0.0) for i in started_ids[:CARDS_PER_ROW]]
        _render_row(api_base, int(user_idx), cw_items, state)

    # Activity sidebar on main page
    with st.expander("My Activity", expanded=False):
        st.write("Liked:", state.get("liked", []))
        st.write("Watched:", state.get("watched", []))
        st.write("Started:", state.get("started", []))

    # Fetch recos
    if st.button("Get Recommendations"):
        try:
            payload = _api_get_recommendations(
                api_base=api_base,
                user_idx=int(user_idx),
                k=int(k),
                include_titles=include_titles,
                debug=debug,
                split=split,
            )
            items = _normalize_items(payload)
            st.session_state["last_payload"] = payload
            st.session_state["last_items"] = items
        except Exception as e:
            st.error(str(e))
            st.session_state["last_payload"] = None
            st.session_state["last_items"] = []

    items: List[RecoItem] = st.session_state.get("last_items", [])

    if items:
        buckets = _group_by_bucket(items)
        ordered = [
            "Because you watched recently",
            "Similar to your taste",
            "Popular among similar users",
            "More picks for you",
        ]

        for key in ordered:
            if key in buckets:
                _render_bucket(api_base, int(user_idx), key, buckets[key], state)

        if debug and st.session_state.get("last_payload"):
            st.markdown("### Debug payload")
            st.json(st.session_state["last_payload"])
    else:
        st.caption("Click 'Get Recommendations' to load V3 results.")


if __name__ == "__main__":
    main()