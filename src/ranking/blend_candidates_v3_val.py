from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple  # noqa: UP035

import numpy as np
import polars as pl

from src.config.settings import settings


def _processed_dir() -> Path:
    p = Path(settings.PROCESSED_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _p(name: str) -> Path:
    return _processed_dir() / name


def _load(name: str) -> pl.DataFrame:
    path = _p(name)
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pl.read_parquet(path)


def _list_to_map(items: List[int] | None, scores: List[float] | None) -> Dict[int, float]:
    m: Dict[int, float] = {}
    if not items or not scores:
        return m
    for i, s in zip(items, scores):
        ii = int(i)
        if ii not in m:
            m[ii] = float(s)
    return m


def _minmax_norm(vals: Dict[int, float]) -> Dict[int, float]:
    if not vals:
        return {}
    arr = np.array(list(vals.values()), dtype=np.float32)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if (vmax - vmin) < 1e-12:
        return {k: 0.0 for k in vals.keys()}
    return {k: (float(v) - vmin) / (vmax - vmin) for k, v in vals.items()}


def main(
    out_k: int = 200,
    w_tt: float = 0.65,
    w_seq: float = 0.25,
    w_v2: float = 0.10,
):
    print("\n[START] V3 val candidate blending (ANN base + optional SEQ/V2)...")

    # Load
    print("[START] Loading input candidate files...")
    ann = _load("ann_candidates_v3_val.parquet").select(["user_idx", "candidates", "tt_scores"])
    seq = _load("seq_candidates_v3_val.parquet").select(["user_idx", "candidates", "seq_scores"])
    v2 = _load("v2_candidates_val.parquet").select(["user_idx", "candidates"])

    print(f"[OK] ann users: {ann.height}")
    print(f"[OK] seq users: {seq.height}")
    print(f"[OK] v2 users:  {v2.height}")

    # IMPORTANT CHANGE:
    # Use ANN as base and LEFT JOIN everything onto it
    print("[START] Joining ANN + SEQ (LEFT)...")
    df = ann.join(seq, on="user_idx", how="left", suffix="_seq")
    print(f"[OK] after ANN+SEQ join rows: {df.height}")

    print("[START] Joining V2 prior (LEFT)...")
    df = df.join(v2, on="user_idx", how="left", suffix="_v2")
    print(f"[OK] after +V2 join rows: {df.height}")

    # Fill missing lists safely
    df = df.with_columns(
        [
            pl.col("candidates_seq").fill_null(pl.lit([])),
            pl.col("seq_scores").fill_null(pl.lit([])),
            pl.col("candidates_v2").fill_null(pl.lit([])),
        ]
    )

    print("[OK] columns in blend frame:")
    print(df.columns)

    print("[START] Blending per-user candidate pools...")

    out_users: List[int] = []
    out_items: List[List[int]] = []
    out_scores: List[List[float]] = []
    out_sources: List[List[str]] = []

    for row in df.iter_rows(named=True):
        u = int(row["user_idx"])

        ann_items = row.get("candidates") or []
        ann_scores = row.get("tt_scores") or []

        seq_items = row.get("candidates_seq") or []
        seq_scores = row.get("seq_scores") or []

        v2_items = row.get("candidates_v2") or []
        v2_set = set(int(x) for x in v2_items)

        ann_map = _list_to_map(ann_items, ann_scores)
        seq_map = _list_to_map(seq_items, seq_scores)

        ann_n = _minmax_norm(ann_map)
        seq_n = _minmax_norm(seq_map)

        pool = set(ann_map.keys()) | set(seq_map.keys()) | v2_set

        scored: List[Tuple[int, float, List[str]]] = []

        for iid in pool:
            sources: List[str] = []

            s_tt = ann_n.get(iid, 0.0)
            s_seq = seq_n.get(iid, 0.0)
            s_v2 = 1.0 if iid in v2_set else 0.0

            if iid in ann_map:
                sources.append("two_tower_ann")
            if iid in seq_map:
                sources.append("sequence_gru")
            if s_v2 > 0:
                sources.append("v2_prior")

            final = (w_tt * s_tt) + (w_seq * s_seq) + (w_v2 * s_v2)
            scored.append((int(iid), float(final), sources))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:out_k]

        out_users.append(u)
        out_items.append([i for i, _, _ in top])
        out_scores.append([s for _, s, _ in top])
        out_sources.append([",".join(srcs) for _, _, srcs in top])

    out = pl.DataFrame(
        {
            "user_idx": out_users,
            "candidates": out_items,
            "blend_scores": out_scores,
            "blend_sources": out_sources,
        }
    )

    out_path = _p("v3_candidates_val.parquet")
    out.write_parquet(out_path)

    print("[DONE] V3 blended candidates saved.")
    print(f"[PATH] {out_path}")
    print(f"[OK] users blended: {out.height}")
    print(f"[OK] weights: w_tt={w_tt}, w_seq={w_seq}, w_v2={w_v2}")


if __name__ == "__main__":
    main()