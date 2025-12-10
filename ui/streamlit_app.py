"""
Unified Streamlit UI for Movie Recommendation MVP (V3 + V4)

Fixes included:
1) Restores Netflix-like feedback controls:
   - Start, Watched, Like, Watch Later
   - Sends POST /feedback to the selected API.
2) Buckets rows using flags (has_seq/has_tt/has_v2).
3) Continue Watching row is driven by Streamlit session_state.
4) Robust poster resolution:
   - API poster_url
   - auto-scan data/processed for any *poster*.parquet
   - supports flexible column names for item + poster URL.
5) Correct path usage for app inside /ui.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple  # noqa: UP035

import polars as pl
import requests
import streamlit as st

# -----------------------------
# Project root resolution
# -----------------------------

APP_FILE = Path(__file__).resolve()
PROJECT_ROOT = APP_FILE.parents[1]  # repo root assuming ui/streamlit_app.py
DATA_DIR = PROJECT_ROOT / "data" / "processed"


# -----------------------------
# Streamlit config
# -----------------------------

st.set_page_config(
    page_title="Movie Recommendation MVP",
    layout="wide",
)


# -----------------------------
# Poster map loader (robust)
# -----------------------------

ITEM_COL_CANDIDATES = ["item_idx", "item_id", "itemIndex", "item"]
POSTER_COL_CANDIDATES = ["poster_url", "poster", "posterUrl", "image_url", "img_url"]


def _extract_poster_map(df: pl.DataFrame) -> Dict[int, str]:
    cols = set(df.columns)

    item_col = next((c for c in ITEM_COL_CANDIDATES if c in cols), None)
    poster_col = next((c for c in POSTER_COL_CANDIDATES if c in cols), None)

    if not item_col or not poster_col:
        return {}

    out: Dict[int, str] = {}
    for idx, url in zip(df[item_col].to_list(), df[poster_col].to_list()):
        if idx is None or url is None:
            continue
        try:
            out[int(idx)] = str(url)
        except Exception:
            continue
    return out


@st.cache_data(show_spinner=False)
def load_local_poster_map() -> Dict[int, str]:
    """
    Attempts to load poster mapping from multiple likely sources.

    Strategy:
    1) Check commonly named files.
    2) Scan data/processed for any parquet with 'poster' in filename.
    """
    common_files = [
        DATA_DIR / "item_metadata.parquet",
        DATA_DIR / "item_posters.parquet",
        DATA_DIR / "posters.parquet",
        DATA_DIR / "items.parquet",
        DATA_DIR / "item_features.parquet",
    ]

    # 1) Priority files first
    for p in common_files:
        if p.exists():
            try:
                df = pl.read_parquet(p)
                m = _extract_poster_map(df)
                if m:
                    return m
            except Exception:
                pass

    # 2) Scan any *poster*.parquet
    if DATA_DIR.exists():
        for p in sorted(DATA_DIR.glob("*poster*.parquet")):
            try:
                df = pl.read_parquet(p)
                m = _extract_poster_map(df)
                if m:
                    return m
            except Exception:
                continue

    return {}


def poster_for_item(item: Dict[str, Any], poster_map: Dict[int, str]) -> Optional[str]:
    """
    Priority:
    1) API-provided poster_url
    2) Local poster map lookup
    """
    p = item.get("poster_url")
    if p:
        return p

    idx = item.get("item_idx")
    if idx is not None:
        return poster_map.get(int(idx))

    return None


# -----------------------------
# API helpers
# -----------------------------

def api_get(base: str, path: str, params: Optional[dict] = None) -> dict:
    url = f"{base.rstrip('/')}{path}"
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def api_post(base: str, path: str, payload: dict) -> dict:
    url = f"{base.rstrip('/')}{path}"
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_recommendations(
    base: str,
    user_idx: int,
    k: int,
    include_titles: bool,
    debug: bool,
    split: str,
) -> List[Dict[str, Any]]:
    payload = api_get(
        base,
        "/recommend",
        params={
            "user_idx": user_idx,
            "k": k,
            "include_titles": include_titles,
            "debug": debug,
            "split": split,
        },
    )

    recs = payload.get("recommendations")
    if recs is None:
        recs = payload.get("items", [])

    if not isinstance(recs, list):
        return []

    return recs


def send_feedback(
    base: str,
    user_idx: int,
    item_idx: int,
    event: str,
    split: str,
) -> Tuple[bool, str]:
    """
    Sends feedback to POST /feedback.
    Returns (ok, message).
    """
    try:
        out = api_post(
            base,
            "/feedback",
            {
                "user_idx": user_idx,
                "item_idx": item_idx,
                "event": event,
                "split": split,
            },
        )
        return True, str(out)
    except requests.HTTPError as e:
        # Most common: 404 if endpoint not yet added in v4 API
        try:
            return False, e.response.text
        except Exception:
            return False, str(e)
    except Exception as e:
        return False, str(e)


# -----------------------------
# Category logic (flag-first)
# -----------------------------

def bucket_items(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets = {
        "Because you watched recently": [],
        "Similar to your taste": [],
        "Popular among similar users": [],
        "More picks for you": [],
    }

    for it in items:
        has_seq = int(it.get("has_seq", 0) or 0)
        has_tt = int(it.get("has_tt", 0) or 0)
        has_v2 = int(it.get("has_v2", 0) or 0)

        if has_seq == 1:
            buckets["Because you watched recently"].append(it)
        elif has_tt == 1:
            buckets["Similar to your taste"].append(it)
        elif has_v2 == 1:
            buckets["Popular among similar users"].append(it)
        else:
            buckets["More picks for you"].append(it)

    return {k: v for k, v in buckets.items() if v}


# -----------------------------
# Session-state helpers
# -----------------------------

def ensure_state():
    st.session_state.setdefault("continue_watching", [])  # list of item dicts
    st.session_state.setdefault("liked_items", set())
    st.session_state.setdefault("watched_items", set())
    st.session_state.setdefault("watch_later_items", set())
    st.session_state.setdefault("last_items", [])  # last fetched list


def add_continue_item(item: Dict[str, Any]):
    """
    Add/move item to Continue Watching row.
    Dedup by item_idx.
    """
    cw = st.session_state["continue_watching"]
    idx = item.get("item_idx")
    if idx is None:
        return

    cw = [x for x in cw if x.get("item_idx") != idx]
    cw.insert(0, item)
    st.session_state["continue_watching"] = cw[:20]


# -----------------------------
# Render helpers
# -----------------------------

def render_continue_watching(poster_map: Dict[int, str]):
    st.subheader("Continue Watching")

    cw = st.session_state.get("continue_watching", [])
    if not cw:
        st.info("No items available for this row yet.")
        return

    cols = st.columns(5)
    for i, it in enumerate(cw[:10]):
        with cols[i % 5]:
            title = it.get("title") or f"Item {it.get('item_idx')}"
            poster = poster_for_item(it, poster_map)

            if poster:
                st.image(poster, use_container_width=True)
            else:
                st.markdown(
                    f"""
                    <div style="
                        height:200px;
                        border:1px solid #2a2a2a;
                        border-radius:10px;
                        padding:10px;
                        background:#0f0f0f;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        text-align:center;">
                        <div style="font-size:12px; line-height:1.3;">
                            {title}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.caption(title)


def render_card(
    it: Dict[str, Any],
    poster_map: Dict[int, str],
    base_url: str,
    user_idx: int,
    split: str,
    mode: str,
):
    title = it.get("title") or f"Item {it.get('item_idx')}"
    score = it.get("score")
    idx = it.get("item_idx")

    poster = poster_for_item(it, poster_map)

    if poster:
        st.image(poster, use_container_width=True)
    else:
        st.markdown(
            f"""
            <div style="
                height:240px;
                border:1px solid #2a2a2a;
                border-radius:10px;
                padding:12px;
                background:#0f0f0f;
                display:flex;
                align-items:center;
                justify-content:center;
                text-align:center;">
                <div style="font-size:13px; line-height:1.3;">
                    {title}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(title)
    if idx is not None:
        st.caption(f"item_idx: {idx}")

    if score is not None:
        try:
            st.caption(f"score: {float(score):.4f}")
        except Exception:
            st.caption(f"score: {score}")

    # reason line
    reason = it.get("reason")
    if reason:
        st.caption(reason)

    # Feedback controls
    if idx is not None:
        c1, c2, c3, c4 = st.columns(4)

        # Unique keys include mode so V3/V4 toggles don't collide
        key_base = f"{mode}_{user_idx}_{idx}_{split}"

        with c1:
            if st.button("Start", key=f"start_{key_base}", use_container_width=True):
                ok, msg = send_feedback(base_url, user_idx, int(idx), "start", split)
                add_continue_item(it)
                if ok:
                    st.toast("Start recorded")
                else:
                    st.error(f"/feedback failed: {msg}")

                st.rerun()

        with c2:
            if st.button("Watched", key=f"watched_{key_base}", use_container_width=True):
                ok, msg = send_feedback(base_url, user_idx, int(idx), "watched", split)
                st.session_state["watched_items"].add(int(idx))
                if ok:
                    st.toast("Watched recorded")
                else:
                    st.error(f"/feedback failed: {msg}")

                st.rerun()

        with c3:
            if st.button("Like", key=f"like_{key_base}", use_container_width=True):
                ok, msg = send_feedback(base_url, user_idx, int(idx), "like", split)
                st.session_state["liked_items"].add(int(idx))
                if ok:
                    st.toast("Like recorded")
                else:
                    st.error(f"/feedback failed: {msg}")

                st.rerun()

        with c4:
            if st.button("Watch later", key=f"wl_{key_base}", use_container_width=True):
                ok, msg = send_feedback(base_url, user_idx, int(idx), "watch_later", split)
                st.session_state["watch_later_items"].add(int(idx))
                if ok:
                    st.toast("Watch Later recorded")
                else:
                    st.error(f"/feedback failed: {msg}")

                st.rerun()


def render_card_grid(
    items: List[Dict[str, Any]],
    poster_map: Dict[int, str],
    base_url: str,
    user_idx: int,
    split: str,
    mode: str,
    cols_per_row: int = 5,
):
    cols = st.columns(cols_per_row)

    for i, it in enumerate(items):
        with cols[i % cols_per_row]:
            render_card(it, poster_map, base_url, user_idx, split, mode)


def render_section(
    heading: str,
    items: List[Dict[str, Any]],
    poster_map: Dict[int, str],
    base_url: str,
    user_idx: int,
    split: str,
    mode: str,
):
    st.subheader(heading)
    render_card_grid(items, poster_map, base_url, user_idx, split, mode)
    st.write("")


# -----------------------------
# UI
# -----------------------------

ensure_state()

st.title("Movie Recommendation MVP")

with st.sidebar:
    st.header("Mode")
    mode = st.radio(
        "Select version",
        ["V4 (Session-aware)", "V3 (Feedback-loop)"],
        index=0,
    )

    st.divider()

    st.header("API")
    default_v4 = "http://127.0.0.1:8004"
    default_v3 = "http://127.0.0.1:8003"

    base_url = st.text_input(
        "Base URL",
        value=default_v4 if mode.startswith("V4") else default_v3
    )

    split = st.selectbox("Split", ["val", "test"], index=0)

    st.divider()

    st.header("Request")
    user_idx = st.number_input("user_idx", min_value=0, max_value=200000, value=9764, step=1)
    k = st.slider("k", min_value=5, max_value=200, value=20, step=5)
    include_titles = st.checkbox("include_titles", value=True)
    debug = st.checkbox("debug", value=False)

    fetch_btn = st.button("Get Recommendations", use_container_width=True)


poster_map = load_local_poster_map()

# Top row: Continue Watching
render_continue_watching(poster_map)

st.write("")

if fetch_btn:
    try:
        health = api_get(base_url, "/health")
        st.success(f"API ok: {health}")

        items = fetch_recommendations(
            base=base_url,
            user_idx=int(user_idx),
            k=int(k),
            include_titles=include_titles,
            debug=debug,
            split=split,
        )

        st.session_state["last_items"] = items

        if not items:
            st.warning("No recommendations returned.")
        else:
            buckets = bucket_items(items)

            order = [
                "Because you watched recently",
                "Similar to your taste",
                "Popular among similar users",
                "More picks for you",
            ]

            for key in order:
                if key in buckets:
                    render_section(
                        key,
                        buckets[key],
                        poster_map,
                        base_url,
                        int(user_idx),
                        split,
                        mode,
                    )

        if debug:
            st.caption("Debug is enabled.")

    except requests.HTTPError as e:
        st.error(f"HTTP error: {e}")
        try:
            st.code(e.response.text)
        except Exception:
            pass
    except Exception as e:
        st.error(f"Error: {e}")


st.divider()
st.caption(
    f"App file: {APP_FILE} | Project root: {PROJECT_ROOT}. "
    "Posters render from API poster_url or any local *poster*.parquet under data/processed."
)