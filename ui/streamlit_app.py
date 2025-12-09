from __future__ import annotations

import time
from typing import Any, Dict, List, Optional  # noqa: UP035

import requests
import streamlit as st

from ui.feedback_store import log_feedback
from ui.realtime_rerank import apply_feedback_rerank, get_user_liked_genres

DEFAULT_API = "http://127.0.0.1:8000"
POSTER_CACHE_PATH = "data/processed/item_posters.json"
FEEDBACK_PATH = "data/processed/ui_feedback.jsonl"


def _api_get(path: str, base_url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30):
    url = f"{base_url.rstrip('/')}{path}"
    return requests.get(url, params=params or {}, timeout=timeout)


def fetch_health(base_url: str) -> bool:
    try:
        r = _api_get("/health", base_url, timeout=5)
        return r.status_code == 200
    except Exception:
        return False


@st.cache_data(show_spinner=False, ttl=60)
def fetch_recommendations_cached(base_url: str, user_idx: int, k: int) -> List[Dict[str, Any]]:
    r = _api_get(f"/recommend/user/{user_idx}", base_url, params={"k": k})
    if r.status_code != 200:
        raise RuntimeError(f"API error ({r.status_code}): {r.text}")
    return r.json()


@st.cache_data(show_spinner=False, ttl=60)
def fetch_debug_cached(base_url: str, user_idx: int, k: int) -> Dict[str, Any]:
    r = _api_get(f"/recommend/user/{user_idx}/debug", base_url, params={"k": k})
    if r.status_code != 200:
        raise RuntimeError(f"API debug error ({r.status_code}): {r.text}")
    return r.json()


@st.cache_data(show_spinner=False)
def load_poster_cache() -> Dict[int, str]:
    import json
    from pathlib import Path

    p = Path(POSTER_CACHE_PATH)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        out = {}
        for k, v in raw.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
        return out
    except Exception:
        return {}


def style():
    st.set_page_config(
        page_title="Movie Recommendation MVP",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
        .muted { color: rgba(0,0,0,0.55); font-size: 0.9rem; }

        .movie-row {
            border: 1px solid rgba(0,0,0,0.06);
            border-radius: 14px;
            padding: 14px 16px;
            margin-bottom: 12px;
            background: white;
        }

        .pill {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 999px;
            margin-right: 6px;
            margin-bottom: 6px;
            background: rgba(0,0,0,0.06);
            font-size: 0.8rem;
        }

        .score-box {
            background: rgba(0,0,0,0.03);
            padding: 6px 10px;
            border-radius: 10px;
            display: inline-block;
            font-weight: 600;
            font-size: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_movie_row(
    rank: int,
    item: Dict[str, Any],
    user_idx: int,
    poster_map: Dict[int, str],
):
    item_idx = int(item.get("item_idx", -1))
    title = item.get("title", "")
    genres = item.get("genres", "")
    score = float(item.get("score", 0.0))
    reasons = item.get("reasons", []) or []

    poster_url = poster_map.get(item_idx)

    c1, c2 = st.columns([1, 4])

    with c1:
        if poster_url:
            st.image(poster_url, use_container_width=True)
        else:
            st.caption("No poster cache")

    with c2:
        st.markdown(f"<div class='movie-row'>", unsafe_allow_html=True)
        st.markdown(f"<div class='muted'>Rank #{rank}</div>", unsafe_allow_html=True)
        st.markdown(f"### {title}")
        st.markdown(f"<div class='muted'>{genres}</div>", unsafe_allow_html=True)
        st.markdown(f"<span class='score-box'>Score: {score:.4f}</span>", unsafe_allow_html=True)

        if reasons:
            st.markdown(
                " ".join([f"<span class='pill'>{r}</span>" for r in reasons]),
                unsafe_allow_html=True,
            )

        btn1, btn2, btn3 = st.columns([1, 1, 2])

        context = {"rank": rank, "title": title, "genres": genres}

        with btn1:
            if st.button("Like", key=f"like_{user_idx}_{item_idx}_{rank}"):
                log_feedback(user_idx, item_idx, "like", context=context)
                st.toast("Liked")

        with btn2:
            if st.button("Dislike", key=f"dislike_{user_idx}_{item_idx}_{rank}"):
                log_feedback(user_idx, item_idx, "dislike", context=context)
                st.toast("Disliked")

        with btn3:
            if st.button("Save for later", key=f"save_{user_idx}_{item_idx}_{rank}"):
                log_feedback(user_idx, item_idx, "save", context=context)
                st.toast("Saved")

        st.markdown(f"</div>", unsafe_allow_html=True)


def render_debug_panel(debug: Dict[str, Any]):
    st.subheader("Debug view")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("Candidates by source (top 10 shown)")
        cbs = debug.get("candidates_by_source", {})
        small = {k: (v[:10] if isinstance(v, list) else v) for k, v in cbs.items()}
        st.json(small)

    with col2:
        st.markdown("Feature columns used by ranker")
        st.code(", ".join(debug.get("feature_cols", [])))

    st.markdown("Ranked top-K with features")
    ranked_top = debug.get("ranked_top", [])
    for row in ranked_top:
        with st.expander(f"{row.get('title', '')} | score={row.get('score', 0):.4f}", expanded=False):
            st.markdown(f"Genres: {row.get('genres', '')}")
            st.markdown(f"Sources: {row.get('sources', [])}")
            st.markdown(f"Reasons: {row.get('reasons', [])}")
            st.markdown("Features:")
            st.json(row.get("features", {}))


def render_taste_panel(user_idx: int):
    st.subheader("Your recent taste (from likes)")
    liked = get_user_liked_genres(user_idx, top_n=5)

    if not liked:
        st.info("No liked-genre signals yet for this user.")
        return

    cols = st.columns(len(liked))
    for i, (g, c) in enumerate(liked):
        with cols[i]:
            st.metric(g, c)


def main():
    style()

    st.title("Movie Recommendation MVP — V2 UI")

    poster_map = load_poster_cache()

    with st.sidebar:
        st.header("Configuration")
        base_url = st.text_input("API base URL", value=DEFAULT_API)
        k = st.slider("Top-K", min_value=5, max_value=50, value=10, step=5)
        user_idx = st.number_input("user_idx", min_value=0, value=9764, step=1)

        show_debug = st.checkbox("Show debug panel", value=False)
        auto_health = st.checkbox("Check API health", value=True)
        show_posters = st.checkbox("Show posters", value=True)

        st.divider()
        st.header("Real-time personalization")
        apply_live_boost = st.checkbox("Apply feedback boost", value=True)
        boost_value = st.slider("Boost strength", 0.02, 0.15, 0.08, 0.01)

        st.divider()
        st.caption("Poster cache")
        st.caption(f"Cached posters loaded: {len(poster_map)}")

        st.divider()
        st.caption("Run API")
        st.code("uvicorn app.main:app --reload")

    api_ok = True
    if auto_health:
        api_ok = fetch_health(base_url)
        if api_ok:
            st.success("API is reachable.")
        else:
            st.warning("API not reachable. Start FastAPI to use this UI.")

    cta1, cta2, _ = st.columns([1, 1, 2])
    with cta1:
        go = st.button("Get recommendations", type="primary", use_container_width=True)
    with cta2:
        clear = st.button("Clear", use_container_width=True)

    if clear:
        st.session_state.pop("recs", None)
        st.session_state.pop("debug", None)
        st.session_state.pop("lat_ms", None)
        st.rerun()

    if go:
        with st.spinner("Fetching recommendations..."):
            t0 = time.perf_counter()
            recs = fetch_recommendations_cached(base_url, int(user_idx), int(k))
            t1 = time.perf_counter()

        if apply_live_boost:
            recs = apply_feedback_rerank(
                recs=recs,
                user_idx=int(user_idx),
                boost=float(boost_value),
            )

        st.session_state["recs"] = recs
        st.session_state["lat_ms"] = (t1 - t0) * 1000.0

        if show_debug:
            with st.spinner("Fetching debug payload..."):
                dbg = fetch_debug_cached(base_url, int(user_idx), min(int(k), 10))
            st.session_state["debug"] = dbg

    recs = st.session_state.get("recs", [])

    if recs:
        lat_ms = st.session_state.get("lat_ms", None)

        top = st.container()
        with top:
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("User", str(int(user_idx)))
            with m2:
                st.metric("Top-K", str(int(k)))
            with m3:
                st.metric("API latency (ms)", f"{lat_ms:.2f}" if lat_ms is not None else "—")

        render_taste_panel(int(user_idx))

        st.subheader("Ranked recommendations")

        for idx, item in enumerate(recs, start=1):
            if show_posters:
                render_movie_row(idx, item, int(user_idx), poster_map)
            else:
                st.write(
                    f"{idx}. {item.get('title', '')} | {item.get('genres', '')} | {item.get('reasons', [])}"
                )

        st.caption(f"Feedback is stored locally in {FEEDBACK_PATH}")

    else:
        st.info("Enter a user_idx and click 'Get recommendations'.")

    if show_debug and "debug" in st.session_state:
        st.divider()
        render_debug_panel(st.session_state["debug"])


if __name__ == "__main__":
    main()