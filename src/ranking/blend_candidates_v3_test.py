from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple  # noqa: UP035

import polars as pl

from src.config.settings import settings


def _processed_dir() -> Path:
    return Path(settings.PROCESSED_DIR)


def _load_parquet_safe(path: Path) -> Optional[pl.DataFrame]:
    if not path.exists():
        return None
    return pl.read_parquet(path)


def _safe_list(x) -> List:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return []


def _normalize_seq_schema(seq: pl.DataFrame) -> pl.DataFrame:
    """
    Accept both possible shapes:
      A) user_idx, candidates, seq_scores
      B) user_idx, candidates_seq, seq_scores
    Normalize to:
      user_idx, candidates_seq, seq_scores
    """
    cols = set(seq.columns)

    if "candidates_seq" in cols and "seq_scores" in cols:
        # already normalized
        return seq.select(["user_idx", "candidates_seq", "seq_scores"])

    if "candidates" in cols and "seq_scores" in cols:
        return (
            seq.select(["user_idx", "candidates", "seq_scores"])
            .rename({"candidates": "candidates_seq"})
        )

    raise ValueError(
        "Sequence candidates file has unexpected schema. "
        f"Found columns={seq.columns}. "
        "Expected candidates+seq_scores OR candidates_seq+seq_scores."
    )


def _normalize_v2_schema(v2: pl.DataFrame) -> pl.DataFrame:
    """
    Accept both:
      A) user_idx, candidates, ...
      B) user_idx, candidates_v2, ...
    Normalize to:
      user_idx, candidates_v2
    """
    cols = set(v2.columns)

    if "candidates_v2" in cols:
        return v2.select(["user_idx", "candidates_v2"])

    if "candidates" in cols:
        return v2.select(["user_idx", "candidates"]).rename({"candidates": "candidates_v2"})

    raise ValueError(
        "V2 candidates file has unexpected schema. "
        f"Found columns={v2.columns}. Expected candidates OR candidates_v2."
    )


def _blend_one_user(
    tt_items: List[int],
    tt_scores: List[float],
    seq_items: List[int],
    seq_scores: List[float],
    v2_items: List[int],
    w_tt: float,
    w_seq: float,
    w_v2: float,
    max_k: int,
) -> Tuple[List[int], List[float], List[str]]:
    score_map = {}
    src_map = {}

    # Two-tower ANN base
    for i, s in zip(tt_items, tt_scores):
        score_map[i] = score_map.get(i, 0.0) + w_tt * float(s)
        src_map.setdefault(i, set()).add("two_tower_ann")

    # Sequence
    for i, s in zip(seq_items, seq_scores):
        score_map[i] = score_map.get(i, 0.0) + w_seq * float(s)
        src_map.setdefault(i, set()).add("sequence_gru")

    # V2 priors
    for i in v2_items:
        score_map[i] = score_map.get(i, 0.0) + w_v2 * 1.0
        src_map.setdefault(i, set()).add("v2_prior")

    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:max_k]

    out_items = [int(i) for i, _ in ranked]
    out_scores = [float(s) for _, s in ranked]
    out_sources = [",".join(sorted(src_map[i])) for i in out_items]

    return out_items, out_scores, out_sources


def main(
    max_k: int = 200,
    user_cap: int = 50000,
    w_tt: float = 0.65,
    w_seq: float = 0.25,
    w_v2: float = 0.10,
):
    print("\n[START] V3 test candidate blending (ANN base + optional SEQ/V2)...")

    ann_p = _processed_dir() / "ann_candidates_v3_test.parquet"
    seq_p = _processed_dir() / "seq_candidates_v3_test.parquet"
    v2_p = _processed_dir() / "v2_candidates_test.parquet"

    print("[START] Loading input candidate files...")
    ann = _load_parquet_safe(ann_p)
    seq = _load_parquet_safe(seq_p)
    v2 = _load_parquet_safe(v2_p)

    if ann is None:
        raise FileNotFoundError(
            "Missing ann_candidates_v3_test.parquet. "
            "Run src.neural.ann_generate_candidates_test_v3 first."
        )

    print(f"[OK] ann users: {ann.select('user_idx').n_unique()}")

    if seq is None:
        print(
            "[WARN] seq_candidates_v3_test.parquet not found. "
            "Proceeding with ANN-only blending."
        )
    else:
        print(f"[OK] seq users: {seq.select('user_idx').n_unique()}")

    if v2 is None:
        print(
            "[WARN] v2_candidates_test.parquet not found. "
            "Proceeding without V2 priors."
        )
    else:
        print(f"[OK] v2 users:  {v2.select('user_idx').n_unique()}")

    # Normalize ANN schema
    ann = ann.select(["user_idx", "candidates", "tt_scores"])

    blend = ann

    # Normalize + join SEQ
    if seq is not None:
        seq_norm = _normalize_seq_schema(seq)
        print("[START] Joining ANN + SEQ (LEFT)...")
        blend = blend.join(seq_norm, on="user_idx", how="left")
        print(f"[OK] after ANN+SEQ join rows: {blend.height}")
    else:
        blend = blend.with_columns(
            [
                pl.lit(None).alias("candidates_seq"),
                pl.lit(None).alias("seq_scores"),
            ]
        )

    # Normalize + join V2
    if v2 is not None:
        v2_norm = _normalize_v2_schema(v2)
        print("[START] Joining V2 prior (LEFT)...")
        blend = blend.join(v2_norm, on="user_idx", how="left")
        print(f"[OK] after +V2 join rows: {blend.height}")
    else:
        blend = blend.with_columns([pl.lit(None).alias("candidates_v2")])

    print("[OK] columns in blend frame:")
    print(blend.columns)

    # Cap users deterministically
    blend = blend.sort("user_idx").head(user_cap)

    out_user: List[int] = []
    out_candidates: List[List[int]] = []
    out_scores: List[List[float]] = []
    out_sources: List[List[str]] = []

    print("[START] Blending per-user candidate pools...")

    rows = blend.to_dicts()
    for r in rows:
        u = int(r["user_idx"])

        tt_items = _safe_list(r.get("candidates"))
        tt_scores = _safe_list(r.get("tt_scores"))
        seq_items = _safe_list(r.get("candidates_seq"))
        seq_scores = _safe_list(r.get("seq_scores"))
        v2_items = _safe_list(r.get("candidates_v2"))

        # Defensive types
        tt_items = [int(x) for x in tt_items]
        seq_items = [int(x) for x in seq_items]
        v2_items = [int(x) for x in v2_items]
        tt_scores = [float(x) for x in tt_scores]
        seq_scores = [float(x) for x in seq_scores]

        items, scores, sources = _blend_one_user(
            tt_items=tt_items,
            tt_scores=tt_scores,
            seq_items=seq_items,
            seq_scores=seq_scores,
            v2_items=v2_items,
            w_tt=w_tt,
            w_seq=w_seq,
            w_v2=w_v2,
            max_k=max_k,
        )

        out_user.append(u)
        out_candidates.append(items)
        out_scores.append(scores)
        out_sources.append(sources)

    out = pl.DataFrame(
        {
            "user_idx": out_user,
            "candidates": out_candidates,
            "blend_scores": out_scores,
            "blend_sources": out_sources,
        }
    )

    out_path = _processed_dir() / "v3_candidates_test.parquet"
    out.write_parquet(out_path)

    print("[DONE] V3 blended candidates saved.")
    print(f"[PATH] {out_path}")
    print(f"[OK] users blended: {out.select('user_idx').n_unique()}")
    print(f"[OK] weights: w_tt={w_tt}, w_seq={w_seq}, w_v2={w_v2}")


if __name__ == "__main__":
    main()