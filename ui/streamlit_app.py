from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional  # noqa: UP035

import requests
import streamlit as st

DEFAULT_API = "http://127.0.0.1:8000"


def _api_get(path: str, base_url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30):
    url = f"{base_url.rstrip('/')}{path}"
    return requests.get(url, params=params or {}, timeout=timeout)


def fetch_health(base_url: str) -> bool:
    try:
        r = _api_get("/health", base_url, timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def fetch_recommendations(base_url: str, user_idx: int, k: int) -> List[Dict[str, Any]]:
    r = _api_get(f"/recommend/user/{user_idx}", base_url, params={"k": k})
    if r.status_code != 200:
        raise RuntimeError(f"API error ({r.status_code}): {r.text}")
    return r.json()


def fetch_debug(base_url: str, user_idx: int, k: int) -> Dict[str, Any]:
    r = _api_get(f"/recommend/user/{user_idx}/debug", base_url, params={"k": k})
    if r.status_code != 200:
        raise RuntimeError(f"API debug error ({r.status_code}): {r.text}")
    return r.json()


def style():
    st.set_page_config(
        page_title="Movie Recommendation MVP",
        page_icon="🎥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Minimal custom CSS for a cleaner look
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
        .stMetric { background: rgba(0,0,0,0.03); padding: 0.6rem 0.8rem; border-radius: 8px; }
        .movie-card {
            border: 1px solid rgba(0,0,0,0.06);
            border-radius: 12px;
            padding: 14px 16px;
            margin-bottom: 10px;
            background: white;
        }
        .muted { color: rgba(0,0,0,0.55); font-size: 0.9rem; }
        .pill {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 999px;
            margin-right: 6px;
            margin-bottom: 6px;
            background: rgba(0,0,0,0.06);
            font-size: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_movie_card(rank: int, item: Dict[str, Any]):
    title = item.get("title", "")
    genres = item.get("genres", "")
    score = item.get("score", 0.0)
    reasons = item.get("reasons", []) or []

    st.markdown(
        f"""
        <div class="movie-card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div style="max-width:80%;">
                    <div class="muted">Rank #{rank}</div>
                    <div style="font-size:1.15rem; font-weight:600; margin-top:2px;">{title}</div>
                    <div class="muted" style="margin-top:4px;">{genres}</div>
                </div>
                <div>
                    <div class="muted">Rank score</div>
                    <div style="font-size:1.05rem; font-weight:600;">{score:.4f}</div>
                </div>
            </div>
            <div style="margin-top:10px;">
                {"".join([f'<span class="pill">{r}</span>' for r in reasons])}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_debug_panel(debug: Dict[str, Any]):
    st.subheader("Debug view")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Candidates by source (top 10 shown)**")
        cbs = debug.get("candidates_by_source", {})
        small = {k: (v[:10] if isinstance(v, list) else v) for k, v in cbs.items()}
        st.json(small)

    with col2:
        st.markdown("**Feature columns used by ranker**")
        st.code(", ".join(debug.get("feature_cols", [])))

    st.markdown("**Ranked top-K with features**")
    ranked_top = debug.get("ranked_top", [])
    for row in ranked_top:
        with st.expander(f"{row.get('title', '')} | score={row.get('score', 0):.4f}", expanded=False):
            st.markdown(f"**Genres:** {row.get('genres', '')}")
            st.markdown(f"**Sources:** {row.get('sources', [])}")
            st.markdown(f"**Reasons:** {row.get('reasons', [])}")
            st.markdown("**Features:**")
            st.json(row.get("features", {}))


def main():
    style()

    st.title("Movie Recommendation MVP — V2 UI")

    with st.sidebar:
        st.header("Configuration")
        base_url = st.text_input("API base URL", value=DEFAULT_API)
        k = st.slider("Top-K", min_value=5, max_value=50, value=10, step=5)
        user_idx = st.number_input("user_idx", min_value=0, value=9764, step=1)

        show_debug = st.checkbox("Show debug panel", value=False)
        auto_health = st.checkbox("Check API health", value=True)

        st.divider()
        st.markdown("Run API:")
        st.code("uvicorn app.main:app --reload")

    # Health
    api_ok = True
    if auto_health:
        api_ok = fetch_health(base_url)
        if api_ok:
            st.success("API is reachable.")
        else:
            st.warning("API not reachable. Start the FastAPI server to use this UI.")

    # Main action
    cols = st.columns([1, 1, 2])
    with cols[0]:
        go = st.button("Get recommendations", type="primary", use_container_width=True)
    with cols[1]:
        clear = st.button("Clear", use_container_width=True)

    if clear:
        st.session_state.pop("recs", None)
        st.session_state.pop("debug", None)
        st.rerun()

    if go:
        if not api_ok:
            # Try anyway in case health check was disabled or transient
            pass

        with st.spinner("Fetching recommendations..."):
            t0 = time.perf_counter()
            recs = fetch_recommendations(base_url, int(user_idx), int(k))
            t1 = time.perf_counter()

        st.session_state["recs"] = recs
        st.session_state["lat_ms"] = (t1 - t0) * 1000.0

        if show_debug:
            with st.spinner("Fetching debug payload..."):
                dbg = fetch_debug(base_url, int(user_idx), min(int(k), 10))
            st.session_state["debug"] = dbg

    # Render state
    recs = st.session_state.get("recs", [])

    if recs:
        lat_ms = st.session_state.get("lat_ms", None)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("User", str(int(user_idx)))
        with m2:
            st.metric("Top-K", str(int(k)))
        with m3:
            st.metric("API latency (ms)", f"{lat_ms:.2f}" if lat_ms is not None else "—")

        st.subheader("Ranked recommendations")

        for idx, item in enumerate(recs, start=1):
            render_movie_card(idx, item)
    else:
        st.info("Enter a user_idx and click 'Get recommendations'.")

    if show_debug and "debug" in st.session_state:
        st.divider()
        render_debug_panel(st.session_state["debug"])


if __name__ == "__main__":
    main()