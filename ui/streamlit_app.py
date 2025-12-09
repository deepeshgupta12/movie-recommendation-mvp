from __future__ import annotations

import time
from typing import Any, Dict, List, Optional  # noqa: UP035

import requests
import streamlit as st

from ui.diversity_rerank import apply_diversity_pipeline
from ui.feedback_store import log_feedback
from ui.home_rows import (
    filter_recs_by_genres,
    get_title_genres,
    get_trending_items,
)
from ui.realtime_rerank import apply_feedback_rerank, get_user_liked_genres
from ui.session_store import (
    get_continue_watching_items,
    get_last_watched_item,
    log_watch_event,
)

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
        .block-container { padding-top: 1.0rem; padding-bottom: 2rem; }
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


def render_row_title(title: str, subtitle: str = ""):
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(f"## {title}")
        if subtitle:
            st.caption(subtitle)
    with c2:
        st.write("")


def render_horizontal_cards(
    items: List[Dict[str, Any]],
    user_idx: int,
    poster_map: Dict[int, str],
    row_key: str,
):
    if not items:
        st.info("No items available for this row yet.")
        return

    cols = st.columns(min(len(items), 6))
    for i, item in enumerate(items[:6]):
        item_idx = int(item.get("item_idx", -1))
        title = item.get("title", "")
        genres = item.get("genres", "")
        reasons = item.get("reasons", []) or []
        score = float(item.get("score", 0.0))

        with cols[i]:
            poster = poster_map.get(item_idx)
            if poster:
                st.image(poster, use_container_width=True)
            else:
                st.caption("No poster")

            st.markdown(f"**{title}**")
            st.caption(genres)

            if reasons:
                st.markdown(
                    " ".join([f"<span class='pill'>{r}</span>" for r in reasons[:2]]),
                    unsafe_allow_html=True,
                )

            st.markdown(f"<span class='score-box'>Score {score:.3f}</span>", unsafe_allow_html=True)

            context = {"row": row_key, "title": title, "genres": genres}

            if st.button("Start", key=f"{row_key}_start_{user_idx}_{item_idx}_{i}"):
                log_watch_event(user_idx, item_idx, "watch_start", context=context)
                st.toast("Added to Continue Watching")

            if st.button("Watched", key=f"{row_key}_done_{user_idx}_{item_idx}_{i}"):
                log_watch_event(user_idx, item_idx, "watch_complete", context=context)
                st.toast("Marked as watched")

            if st.button("Like", key=f"{row_key}_like_{user_idx}_{item_idx}_{i}"):
                log_feedback(user_idx, item_idx, "like", context={"title": title, "genres": genres})
                st.toast("Liked")


def render_taste_panel(user_idx: int):
    st.subheader("Your recent taste (from likes)")
    liked = get_user_liked_genres(user_idx, top_n=5)

    if not liked:
        st.caption("No liked-genre signals yet.")
        return

    cols = st.columns(len(liked))
    for i, (g, c) in enumerate(liked):
        with cols[i]:
            st.metric(g, c)


def build_continue_watching_row(user_idx: int) -> List[Dict[str, Any]]:
    ids = get_continue_watching_items(user_idx)
    out = []
    for item_idx in ids[:20]:
        title, genres = get_title_genres(item_idx)
        out.append(
            {
                "item_idx": int(item_idx),
                "title": title,
                "genres": genres,
                "score": 1.0,
                "reasons": ["Continue watching"],
            }
        )
    return out


def build_because_you_watched_row(
    user_idx: int,
    recs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    last = get_last_watched_item(user_idx)
    if last is None or last < 0:
        return []

    title, genres = get_title_genres(last)
    filtered = filter_recs_by_genres(recs, genres, limit=12)

    out = []
    for it in filtered:
        new_it = dict(it)
        rs = list(new_it.get("reasons", []) or [])
        label = f"Because you watched {title}" if title else "Because you watched recently"
        if label not in rs:
            rs = [label] + rs
        new_it["reasons"] = rs[:3]
        out.append(new_it)

    return out


def main():
    style()
    st.title("Movie Recommendation MVP — Home")

    poster_map = load_poster_cache()

    with st.sidebar:
        st.header("Configuration")
        base_url = st.text_input("API base URL", value=DEFAULT_API)
        user_idx = st.number_input("user_idx", min_value=0, value=9764, step=1)
        k = st.slider("Top-K (Top Picks)", min_value=5, max_value=50, value=10, step=5)

        auto_health = st.checkbox("Check API health", value=True)
        show_posters = st.checkbox("Show posters", value=True)

        st.divider()
        st.header("Real-time personalization")
        apply_live_boost = st.checkbox("Apply feedback boost", value=True)
        boost_value = st.slider("Boost strength", 0.02, 0.15, 0.08, 0.01)

        st.divider()
        st.header("Slate diversity controls")
        enable_diversity = st.checkbox("Enable diversity re-rank", value=True)
        enable_genre_cap = st.checkbox("Genre cap", value=True)
        max_per_genre = st.slider("Max items per primary genre", 1, 5, 3, 1)

        enable_mmr = st.checkbox("MMR soft diversification", value=True)
        lambda_relevance = st.slider(
            "Relevance vs diversity (higher = more relevance)",
            0.50, 0.95, 0.75, 0.05
        )

        st.divider()
        st.caption("Feedback path")
        st.code(FEEDBACK_PATH)

        st.caption("Run API")
        st.code("uvicorn app.main:app --reload")

    if auto_health:
        api_ok = fetch_health(base_url)
        if api_ok:
            st.success("API is reachable.")
        else:
            st.warning("API not reachable. Start FastAPI.")

    cta1, cta2, _ = st.columns([1, 1, 2])
    with cta1:
        go = st.button("Load Home", type="primary", use_container_width=True)
    with cta2:
        clear = st.button("Clear", use_container_width=True)

    if clear:
        st.session_state.pop("recs", None)
        st.session_state.pop("lat_ms", None)
        st.rerun()

    if go:
        # IMPORTANT:
        # API enforces k <= 50. We fetch a slightly larger pool than display-k
        # but never exceed 50 to stay compliant with validation.
        fetch_k = min(50, max(int(k), 30))

        with st.spinner("Building Home rows..."):
            t0 = time.perf_counter()
            recs = fetch_recommendations_cached(base_url, int(user_idx), fetch_k)
            t1 = time.perf_counter()

        if apply_live_boost:
            recs = apply_feedback_rerank(
                recs=recs,
                user_idx=int(user_idx),
                boost=float(boost_value),
            )

        st.session_state["recs"] = recs
        st.session_state["lat_ms"] = (t1 - t0) * 1000.0

    recs = st.session_state.get("recs", [])

    if not recs:
        st.info("Click 'Load Home' to render rows.")
        return

    lat_ms = st.session_state.get("lat_ms", None)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("User", str(int(user_idx)))
    with m2:
        st.metric("Top Picks K", str(int(k)))
    with m3:
        st.metric("API latency (ms)", f"{lat_ms:.2f}" if lat_ms is not None else "—")

    render_taste_panel(int(user_idx))
    st.divider()

    # Row 1: Continue Watching
    continue_row = build_continue_watching_row(int(user_idx))
    render_row_title("Continue Watching", "Local watch-state simulation")
    render_horizontal_cards(continue_row, int(user_idx), poster_map, row_key="continue")

    st.divider()

    # Row 2: Top Picks For You
    top_pool = recs[: max(int(k) * 3, 30)]
    if enable_diversity:
        top_picks = apply_diversity_pipeline(
            items=top_pool,
            k=int(k),
            enable_genre_cap=enable_genre_cap,
            max_per_genre=int(max_per_genre),
            enable_mmr=enable_mmr,
            lambda_relevance=float(lambda_relevance),
        )
    else:
        top_picks = top_pool[: int(k)]

    render_row_title("Top Picks For You", "V2 ranked hybrid output plus optional diversity re-rank")
    render_horizontal_cards(top_picks, int(user_idx), poster_map, row_key="top_picks")

    st.divider()

    # Row 3: Because You Watched
    because_pool = build_because_you_watched_row(int(user_idx), recs)
    if enable_diversity and because_pool:
        because_row = apply_diversity_pipeline(
            items=because_pool,
            k=min(10, len(because_pool)),
            enable_genre_cap=enable_genre_cap,
            max_per_genre=int(max_per_genre),
            enable_mmr=enable_mmr,
            lambda_relevance=float(lambda_relevance),
        )
    else:
        because_row = because_pool[:10]

    render_row_title("Because You Watched", "Derived from your last watch event")
    render_horizontal_cards(because_row, int(user_idx), poster_map, row_key="because")

    st.divider()

    # Row 4: Trending Now
    trending_pool = get_trending_items(limit=24)
    if enable_diversity and trending_pool:
        trending = apply_diversity_pipeline(
            items=trending_pool,
            k=10,
            enable_genre_cap=enable_genre_cap,
            max_per_genre=int(max_per_genre),
            enable_mmr=enable_mmr,
            lambda_relevance=float(lambda_relevance),
        )
    else:
        trending = trending_pool[:10]

    render_row_title("Trending Now", "Derived locally from item interaction features")
    render_horizontal_cards(trending, int(user_idx), poster_map, row_key="trending")

    st.caption("Step 7.5 adds UI-level slate diversity without changing the model layer.")


if __name__ == "__main__":
    main()