"""
V4 Streamlit UI (Session-aware + categorized rows)

Key fixes vs earlier UI:
1) Category separation is driven by flags (has_seq/has_tt/has_v2),
   not only by the reason string.
2) Robust project-root path resolution even when the app lives in /ui.
3) Poster fallback logic supports:
   - API poster_url
   - local metadata parquet with item_idx + poster_url
   - clean placeholder tile if neither exists
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional  # noqa: UP035

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
# Local metadata loaders
# -----------------------------

@st.cache_data(show_spinner=False)
def load_local_poster_map() -> Dict[int, str]:
    """
    Load item_idx -> poster_url from whichever metadata file exists.
    """
    candidates = [
        DATA_DIR / "item_metadata.parquet",
        DATA_DIR / "items.parquet",
        DATA_DIR / "item_features.parquet",
    ]

    for p in candidates:
        if p.exists():
            try:
                df = pl.read_parquet(p)
                cols = set(df.columns)
                if "item_idx" in cols and "poster_url" in cols:
                    out = {}
                    for idx, url in zip(df["item_idx"].to_list(), df["poster_url"].to_list()):
                        if idx is None or url is None:
                            continue
                        out[int(idx)] = str(url)
                    return out
            except Exception:
                continue

    return {}


def poster_for_item(item: Dict[str, Any], poster_map: Dict[int, str]) -> Optional[str]:
    """
    Priority:
    1) API-provided poster_url
    2) Local poster map
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

    # V4 standard response
    recs = payload.get("recommendations")

    # allow fallback for older shapes
    if recs is None:
        recs = payload.get("items", [])

    if not isinstance(recs, list):
        return []

    return recs


# -----------------------------
# Category logic (flag-first)
# -----------------------------

def bucket_items(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Ensures stable Netflix-like rows even if reason strings are repetitive.
    """
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

    # remove empty buckets
    return {k: v for k, v in buckets.items() if len(v) > 0}


# -----------------------------
# Render helpers
# -----------------------------

def render_card_grid(
    items: List[Dict[str, Any]],
    poster_map: Dict[int, str],
    cols_per_row: int = 5,
):
    cols = st.columns(cols_per_row)

    for i, it in enumerate(items):
        col = cols[i % cols_per_row]
        with col:
            title = it.get("title") or f"Item {it.get('item_idx')}"
            score = it.get("score")
            idx = it.get("item_idx")

            poster = poster_for_item(it, poster_map)

            if poster:
                st.image(poster, use_container_width=True)
            else:
                # clean placeholder card
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


def render_section(
    heading: str,
    items: List[Dict[str, Any]],
    poster_map: Dict[int, str],
):
    st.subheader(heading)
    render_card_grid(items, poster_map)
    st.write("")


# -----------------------------
# UI Layout
# -----------------------------

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

        if len(items) == 0:
            st.warning("No recommendations returned.")
        else:
            # Netflix-like row separation
            buckets = bucket_items(items)

            # stable display order
            order = [
                "Because you watched recently",
                "Similar to your taste",
                "Popular among similar users",
                "More picks for you",
            ]

            for key in order:
                if key in buckets:
                    render_section(key, buckets[key], poster_map)

        if debug:
            st.caption("Debug is enabled. Check API debug payload in terminal logs if needed.")

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
    f"App file: {APP_FILE} | Project root resolved as: {PROJECT_ROOT}. "
    "Poster tiles will appear if API or local metadata provides poster_url."
)