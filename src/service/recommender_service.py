from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple  # noqa: UP035

import joblib
import polars as pl

from src.config.settings import settings
from src.models.als_cf_conf import ALSConfidenceRecommender
from src.models.item_item import ItemItemSimilarityRecommender
from src.models.popularity import PopularityRecommender
from src.retrieval.genre_neighbors import GenreNeighborsRecommender
from src.retrieval.hybrid import HybridCandidateBlender, SourceWeight


@dataclass
class RecommendItem:
    item_idx: int
    title: str
    genres: str
    score: float
    reasons: List[str]


class V2RecommenderService:
    """
    Production-style local inference service for V2 hybrid + ranker.

    - Loads models once
    - Loads feature store once
    - Computes lightweight user-item genre affinity cross feature
    - Ranks blended candidates with the trained HGB ranker
    """

    def __init__(
        self,
        max_k: int = 50,
        weights: List[SourceWeight] | None = None,
    ) -> None:
        self.max_k = max_k

        self.weights = weights or [
            SourceWeight("pop", 0.4),
            SourceWeight("item_item", 1.0),
            SourceWeight("als", 1.0),
            SourceWeight("genre_nb", 0.8),
        ]

        # Candidate models
        self.pop = PopularityRecommender()
        self.item_item = ItemItemSimilarityRecommender()
        self.als = ALSConfidenceRecommender()
        self.genre_nb = GenreNeighborsRecommender()

        self.blender = HybridCandidateBlender(self.weights)

        # Ranker
        self.ranker = None
        self.feature_cols: List[str] = []
        self.pos_idx: int = 1

        # Feature store maps
        self.user_feat_map: Dict[int, Dict[str, float]] = {}
        self.item_feat_map: Dict[int, Dict[str, float]] = {}
        self.item_genre_count_map: Dict[int, float] = {}
        self.item_genres_map: Dict[int, List[str]] = {}
        self.user_genre_aff_map: Dict[int, Dict[str, Tuple[float, float]]] = {}

        # Titles mapping
        self.item_title_map: Dict[int, Tuple[str, str]] = {}  # item_idx -> (title, genres)

        self._loaded = False

    # --------------------------
    # Public API
    # --------------------------

    def load(self) -> V2RecommenderService:
        if self._loaded:
            return self

        # Fit candidate models
        self.pop.fit()
        self.item_item.fit()
        self.als.fit()
        self.genre_nb.fit()

        # Load ranker + meta
        self._load_ranker()

        # Load feature store artifacts
        self._load_feature_store_maps()

        # Load item title/genre mapping
        self._load_item_title_map()

        self._loaded = True
        return self

    def recommend(self, user_idx: int, k: int = 10) -> List[RecommendItem]:
        self._ensure_loaded()
        k = min(k, self.max_k)

        blended, blended_sources = self._get_blended_candidates_with_sources(user_idx, k=self.max_k)
        if not blended:
            return []

        scores = self._score_candidates(user_idx, blended)
        ranked = [i for i, _ in sorted(zip(blended, scores), key=lambda x: x[1], reverse=True)]
        ranked = ranked[:k]

        out: List[RecommendItem] = []
        for i in ranked:
            title, genres = self.item_title_map.get(i, ("", ""))
            reasons = self._build_reason_tags(user_idx, i, blended_sources.get(i, []))
            out.append(
                RecommendItem(
                    item_idx=i,
                    title=title,
                    genres=genres,
                    score=float(scores[blended.index(i)]),
                    reasons=reasons,
                )
            )
        return out

    def recommend_debug(self, user_idx: int, k: int = 10) -> Dict[str, Any]:
        """
        Returns debug payload:
        - candidates per source
        - blended list
        - feature vectors for top candidates
        - ranked output with titles
        """
        self._ensure_loaded()
        k = min(k, self.max_k)

        per_source = {
            "pop": self.pop.recommend(user_idx, k=self.max_k),
            "item_item": self.item_item.recommend(user_idx, k=self.max_k),
            "als": self.als.recommend(user_idx, k=self.max_k),
            "genre_nb": self.genre_nb.recommend(user_idx, k=self.max_k),
        }

        blended, blended_sources = self._blend_with_sources(per_source, k=self.max_k)
        scores = self._score_candidates(user_idx, blended) if blended else []

        ranked_pairs = sorted(zip(blended, scores), key=lambda x: x[1], reverse=True)[:k]

        top_debug = []
        for item_idx, score in ranked_pairs:
            fd = self._feature_dict(user_idx, item_idx)
            title, genres = self.item_title_map.get(item_idx, ("", ""))
            top_debug.append(
                {
                    "item_idx": item_idx,
                    "title": title,
                    "genres": genres,
                    "score": float(score),
                    "sources": blended_sources.get(item_idx, []),
                    "features": {c: fd.get(c, 0.0) for c in self.feature_cols},
                    "reasons": self._build_reason_tags(user_idx, item_idx, blended_sources.get(item_idx, [])),
                }
            )

        return {
            "user_idx": user_idx,
            "feature_cols": self.feature_cols,
            "candidates_by_source": per_source,
            "blended": blended[:k],
            "ranked_top": top_debug,
        }

    # --------------------------
    # Internals
    # --------------------------

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Service not loaded. Call service.load() first.")

    def _load_ranker(self) -> None:
        models_dir = settings.PROJECT_ROOT / "reports" / "models"
        ranker_path = models_dir / "ranker_hgb_v2.pkl"
        meta_path = models_dir / "ranker_hgb_v2.meta.json"

        if not ranker_path.exists():
            raise FileNotFoundError("ranker_hgb_v2.pkl not found. Train V2 ranker first.")
        if not meta_path.exists():
            raise FileNotFoundError("ranker_hgb_v2.meta.json not found. Train V2 ranker first.")

        self.ranker = joblib.load(ranker_path)

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.feature_cols = meta["feature_cols"]
        positive_class = meta.get("positive_class", 1)

        classes_list = list(self.ranker.classes_)
        if positive_class not in classes_list:
            raise ValueError(f"Positive class {positive_class} not in model.classes_: {classes_list}")

        self.pos_idx = classes_list.index(positive_class)

    def _load_feature_store_maps(self) -> None:
        # Core features
        user_feat = pl.read_parquet(settings.PROCESSED_DIR / "user_features.parquet")
        item_feat = pl.read_parquet(settings.PROCESSED_DIR / "item_features.parquet")

        # Genres
        item_genres = pl.read_parquet(settings.PROCESSED_DIR / "item_genres_expanded.parquet")
        user_genre = pl.read_parquet(settings.PROCESSED_DIR / "user_genre_affinity.parquet")

        # Build user feature map
        self.user_feat_map = {
            int(r[0]): {
                "user_interactions": float(r[1]),
                "user_conf_sum": float(r[2]),
                "user_conf_decay_sum": float(r[3]),
                "user_days_since_last": float(r[4]),
            }
            for r in user_feat.select(
                "user_idx",
                "user_interactions",
                "user_conf_sum",
                "user_conf_decay_sum",
                "user_days_since_last",
            ).iter_rows()
        }

        # Build item feature map
        self.item_feat_map = {
            int(r[0]): {
                "item_interactions": float(r[1]),
                "item_conf_sum": float(r[2]),
                "item_conf_decay_sum": float(r[3]),
                "item_days_since_last": float(r[4]),
            }
            for r in item_feat.select(
                "item_idx",
                "item_interactions",
                "item_conf_sum",
                "item_conf_decay_sum",
                "item_days_since_last",
            ).iter_rows()
        }

        # Item genre count map
        item_genre_count = (
            item_genres.group_by("item_idx")
            .agg(pl.len().alias("item_genre_count"))
        )
        self.item_genre_count_map = {
            int(i): float(c)
            for i, c in item_genre_count.select("item_idx", "item_genre_count").iter_rows()
        }

        # Item -> genres list
        self.item_genres_map = {}
        for i, g in item_genres.select("item_idx", "genre").iter_rows():
            self.item_genres_map.setdefault(int(i), []).append(str(g))

        # User -> genre -> (aff, aff_decay)
        self.user_genre_aff_map = {}
        for u, g, aff, aff_d in user_genre.select(
            "user_idx", "genre", "user_genre_aff", "user_genre_aff_decay"
        ).iter_rows():
            ud = self.user_genre_aff_map.setdefault(int(u), {})
            ud[str(g)] = (float(aff), float(aff_d))

    def _load_item_title_map(self) -> None:
        items_map = pl.read_parquet(settings.PROCESSED_DIR / "items.parquet").select("item_idx", "movieId")
        movies = pl.read_parquet(settings.PROCESSED_DIR / "movies.parquet").select("movieId", "title", "genres")

        joined = (
            items_map.join(movies, on="movieId", how="left")
            .with_columns(
                pl.col("title").fill_null(""),
                pl.col("genres").fill_null(""),
            )
        )

        self.item_title_map = {
            int(r[0]): (str(r[1]), str(r[2]))
            for r in joined.select("item_idx", "title", "genres").iter_rows()
        }

    def _get_blended_candidates_with_sources(self, user_idx: int, k: int) -> Tuple[List[int], Dict[int, List[str]]]:
        per_source = {
            "pop": self.pop.recommend(user_idx, k=k),
            "item_item": self.item_item.recommend(user_idx, k=k),
            "als": self.als.recommend(user_idx, k=k),
            "genre_nb": self.genre_nb.recommend(user_idx, k=k),
        }
        return self._blend_with_sources(per_source, k=k)

    def _blend_with_sources(
        self,
        per_source: Dict[str, List[int]],
        k: int,
    ) -> Tuple[List[int], Dict[int, List[str]]]:
        """
        Uses the existing blender for ordering,
        but also returns a reverse map of item -> sources contributing it.
        """
        src_map: Dict[int, List[str]] = {}

        for s, items in per_source.items():
            for it in items:
                src_map.setdefault(int(it), []).append(s)

        blended = self.blender.blend(per_source, k=k)
        return blended, src_map

    def _feature_dict(self, user_idx: int, item_idx: int) -> Dict[str, float]:
        defaults = {
            "user_interactions": 0.0,
            "user_conf_sum": 0.0,
            "user_conf_decay_sum": 0.0,
            "user_days_since_last": 0.0,
            "item_interactions": 0.0,
            "item_conf_sum": 0.0,
            "item_conf_decay_sum": 0.0,
            "item_days_since_last": 0.0,
            "item_genre_count": 0.0,
            "user_item_genre_aff": 0.0,
            "user_item_genre_aff_decay": 0.0,
        }

        out = dict(defaults)
        out.update(self.user_feat_map.get(user_idx, {}))
        out.update(self.item_feat_map.get(item_idx, {}))

        out["item_genre_count"] = self.item_genre_count_map.get(item_idx, 0.0)

        genres = self.item_genres_map.get(item_idx, [])
        if genres:
            ug = self.user_genre_aff_map.get(user_idx, {})
            if ug:
                affs = []
                affds = []
                for g in genres:
                    a, ad = ug.get(g, (0.0, 0.0))
                    affs.append(a)
                    affds.append(ad)

                if affs:
                    out["user_item_genre_aff"] = sum(affs) / len(affs)
                    out["user_item_genre_aff_decay"] = sum(affds) / len(affds)

        return out

    def _score_candidates(self, user_idx: int, item_list: List[int]) -> List[float]:
        if not item_list:
            return []

        X = []
        for i in item_list:
            fd = self._feature_dict(user_idx, i)
            X.append([fd.get(c, 0.0) for c in self.feature_cols])

        probs = self.ranker.predict_proba(X)[:, self.pos_idx]
        return [float(p) for p in probs]

    def _build_reason_tags(self, user_idx: int, item_idx: int, sources: List[str]) -> List[str]:
        reasons: List[str] = []

        # Source-based reasons
        src_to_label = {
            "pop": "Trending now",
            "item_item": "Similar to your taste",
            "als": "People like you watched this",
            "genre_nb": "Matches your genres",
        }
        for s in sources:
            if s in src_to_label and src_to_label[s] not in reasons:
                reasons.append(src_to_label[s])

        # Affinity-based reason
        genres = self.item_genres_map.get(item_idx, [])
        ug = self.user_genre_aff_map.get(user_idx, {})
        if genres and ug:
            # if any genre affinity is strong, add a personalized tag
            max_aff = 0.0
            best_g = None
            for g in genres:
                a, _ = ug.get(g, (0.0, 0.0))
                if a > max_aff:
                    max_aff = a
                    best_g = g
            if best_g and max_aff >= 0.10:
                reasons.append(f"Because you like {best_g}")

        return reasons[:3]