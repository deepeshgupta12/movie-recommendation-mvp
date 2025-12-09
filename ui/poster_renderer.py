from __future__ import annotations

from typing import Optional

import streamlit as st


def render_poster_or_placeholder(
    poster_url: Optional[str],
    title: str,
    genres: str,
    height: int = 220,
):
    """
    Renders a real poster if available; otherwise a sleek placeholder tile.
    Keeps row layout stable and OTT-like.
    """
    if poster_url:
        st.image(poster_url, use_container_width=True)
        return

    safe_title = (title or "Unknown Title").strip()
    safe_genres = (genres or "").strip()

    # Simple genre chips (first 2)
    chips = ""
    if safe_genres:
        parts = [g.strip() for g in safe_genres.split("|") if g.strip()]
        parts = parts[:2]
        chips = " ".join([f"<span class='ph-chip'>{p}</span>" for p in parts])

    st.markdown(
        f"""
        <div class="ph-card" style="height:{height}px;">
            <div class="ph-badge">Poster unavailable</div>
            <div class="ph-title">{safe_title}</div>
            <div class="ph-genres">{chips}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )