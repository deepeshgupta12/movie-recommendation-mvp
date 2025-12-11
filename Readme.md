# 🎬 Movie Recommendation MVP — V1 ➜ V4 (through Step 9.x)

A hands-on, locally runnable, **end-to-end recommender systems playground** that evolves from classic baselines to a **Netflix-inspired hybrid stack** with neural retrieval, sequence modeling, candidate blending, learning-to-rank, a lightweight **online feedback loop**, and now a **session-aware, diversity‑aware V4 stack with posters + live interactions**.

This repository is intentionally engineered as a **learning-grade but production-shaped** system:
- clear module boundaries  
- reproducible offline evaluation  
- service + UI wiring  
- explicit trade-offs vs. Netflix/Amazon scale

---

## ✅ What this README includes (final scope checklist)

This final README version covers everything we finalized from **V1 to V4**, plus supporting artifacts:

### Product story & positioning
- 🎯 Problem framing and what “good” looks like
- 🧭 Why this architecture is different from generic toy recommenders
- 🆚 Explicit comparison to **Netflix** and **Amazon Prime Video**

### System architecture (V1 → V2 → V3 → V4)
- 🧱 Component-by-component evolution
- 🔁 Candidate → Ranker → Explain → Feedback loop
- 🧩 How each model contributes to the final slate
- 🧠 What V4 adds: **session awareness + diversity + live feedback effects**

### Evaluation & results
- 📈 Offline ranking metrics you produced in terminal
- 🔍 Interpretation of lifts and what they mean
- ⚠️ Known evaluation limitations in a local MVP setup

### Service + UI (through V4)
- 🚀 API contracts: `/recommend`, `/feedback`, `/health`
- 🖥️ Streamlit UX behavior and Netflix-like sections
- 🧠 “Why this” lightweight explanation rules
- 📸 Poster caching and fallback behavior

### Gaps & future roadmap
- 🧨 The “serious gaps” list
- 🧪 How we bridge them in future work (V5+ style)

---

## 🌟 High-level vision

Most small recommendation demos stop at:
- one algorithm
- no candidate/ranker split
- no explainability
- no online loop

This project does **not**.

Instead, it deliberately mimics the *shape* of a modern large-scale system:

1. **Multiple candidate generators**
2. **Blended retrieval priors**
3. **Re-ranking with consistent feature contracts**
4. **Reason generation**
5. **A feedback loop that changes future results**
6. **Session-aware boosts & diversity re-ordering (V4)**

Even in local constraints, the architecture is designed to be an honest blueprint for core Netflix/Amazon patterns.

---

## 🧠 Architecture at a glance (V3 core)

```text
User Events (ratings/implicit) 
        │
        ▼
Confidence Builder (train/val/test splits)
        │
        ├───────────────┬───────────────────────────┬──────────────────────┐
        ▼               ▼                           ▼                      ▼
V2 CF Priors      Two-Tower Retrieval         GRU Sequence Model     Popular/Genre priors
(item-item/user)  (user/item embeddings)      (next-item intent)     (fallback + cold-ish)
        │               │                           │                      │
        └───────────────┴──────────────┬────────────┴──────────────────────┘
                                       ▼
                           V3 Candidate Blending
                        (ANN base + optional SEQ/V2)
                                       │
                                       ▼
                         V3 Pointwise Ranker (HGB)
                 (blend_score + source flags + user/item features)
                                       │
                                       ▼
                         Recommendation Service + API
                                       │
                                       ▼
                            Streamlit “Netflix-like UI”
                                       │
                                       ▼
                            Feedback events (/feedback)
                           influencing future refresh
```

This is the **V3 backbone**. V4 rides on top of this, adding **session features**, **diversity-aware reordering**, and a more explicit **live feedback behavior** (without changing the fundamental candidate/feature contracts).

---

## 🧠 V4 extras on top of V3

V4 keeps the full V3 stack and adds:

1. **Session features & state**  
   - Session-level interaction tracking (per user) with groups like:
     - `sess_hot` – items the user has interacted with in this session
     - `sess_warm` – closely related items to the recent interactions
     - `sess_cold` – long-tail / exploratory items
   - A `short_term_boost` scalar capturing recent engagement intensity.
   - A `last_item_idx` + `last_title` anchor for “because you watched” style reasoning.

2. **V4 ranker with session-aware features**  
   - New pointwise feature tables for **VAL** and **TEST**:
     - `rank_v4_val_pointwise.parquet`
     - `rank_v4_test_pointwise.parquet`
   - Ranker features (locked order):
     - `blend_score`, `has_tt`, `has_seq`, `has_v2`
     - `short_term_boost`, `sess_hot`, `sess_warm`, `sess_cold`
     - `user_interactions`, `user_conf_sum`, `user_conf_decay_sum`, `user_days_since_last`
     - `item_interactions`, `item_conf_sum`, `item_conf_decay_sum`, `item_days_since_last`

3. **Diversity-aware slate re-ordering (V4 service)**  
   - Optional `apply_diversity` flag in `/recommend`:
     - when `False`, you get the pure HGB ranker ordering
     - when `True`, the top‑K are re-ordered to:
       - avoid over-concentrating on a single franchise/decade
       - mix “similar to your taste” with “popular among similar users”
       - expose a bit more variety while respecting model scores

4. **Stronger feedback wiring (V4 events)**  
   - `/feedback` now supports richer event types:
     - `"like"`, `"remove_like"`
     - `"watched"`
     - `"watch_later"`, `"remove_watch_later"`
     - `"skip"`
   - Events are persisted in `data/processed/feedback_state_v4.json`.
   - Future recommendations (for the same `user_idx`) are adjusted via `short_term_boost` and session group flags.

5. **Poster cache v4 (TMDB-backed, keyed by movieId)**  
   - Script: `scripts.build_poster_cache_v4.py`
   - Data sources:
     - primary: `data/processed/item_meta.parquet`  
       (must contain at least `item_idx`, `movieId`, `title`)
   - Output: `data/processed/poster_cache_v4.json`
     - JSON keyed by **`movieId`**, not `item_idx`, to avoid mismatch.
   - Runtime lookup order in the service:
     1. `poster_cache_v4.json` (movieId → poster path / full URL)
     2. legacy `item_posters.json` (if still present)
     3. gracefully render a **placeholder card** (“Poster unavailable”) instead of blank images.

6. **Improved UI wiring (V4 mode)**  
   - Same Streamlit app, with **V3 / V4 toggle** in the sidebar.
   - V4 mode uses:
     - API base: `http://127.0.0.1:8004`
     - Extra controls: `split` (val/test), `apply_diversity`
   - Each card now exposes:
     - poster (from cache)
     - title
     - score + explanation text
     - inline action buttons: “Like”, “Watched”, “Watch later”, “Start” (mapped to V4 feedback events).

---

## 🧪 Model evolution (V1 → V4)

### V1 — Foundations

**What we aimed to prove:**
- A clean, reproducible offline pipeline
- Baselines that are easy to reason about
- A simple service + UI

**Core building blocks:**
- Popularity priors  
- Genre-based heuristics  
- Item-item collaborative signals  

This stage was about *correctness, data hygiene, and confidence logic*, not sheer metric wins.

---

### V2 — Hybrid candidate ranking

**What changed:**
- Unified V2 feature table  
- Hybrid candidate pool  
- A learned ranker with locked feature order  

**Why it mattered:**
This introduced the **candidate → ranker split** that almost all serious systems rely on.

**Best V2 ranked snapshot (after fixes):**
- Recall@10 ≈ **0.0253**
- Recall@20 ≈ **0.0454**
- Recall@50 ≈ **0.0592**

This was the first moment the system started behaving like a layered recommender rather than a single-model script.

---

### V3 — Neural + Sequence + Proper blended retrieval

V3 adds three major ideas that are table-stakes in modern platforms:

1. **Two-Tower / ANN retrieval**
2. **Sequence intent modeling (GRU)**
3. **Blended candidate priors** feeding a **V3 ranker**

**Why this makes V3 feel “Netflix-like”:**
- retrieval is no longer purely co-occurrence  
- **embedding space** lets you generalize beyond exact overlaps  
- sequence intent gives a “what you watched recently” flavor

V3 is the backbone that V4 builds on.

---

### V4 — Session-aware ranking, diversity & live feedback

V4’s goal is **not** to change the retrieval stack, but to make the
experience feel **more like a product** and less like a static model demo.

**Key objectives:**
- Make the system **aware of the session** (what you just did).
- Allow **quick experiments** with diversity-aware reordering.
- Tighten the **feedback → next recommendation** loop.
- Fix obvious UX issues like **poster/title mismatches** via a better cache.

**What V4 delivers:**
- New session feature tables for VAL/TEST used by the V4 ranker.
- A V4 HGB ranker trained on 10M+ rows for each split, with AUC ≈ **0.80 (VAL)** and ≈ **0.69 (TEST)**.
- A `/recommend` endpoint that can run in **V4 session-aware mode** with `apply_diversity=True`.
- A `/feedback` endpoint that accepts multiple event types and updates a local JSON state.
- A V4 UI mode that:
  - shows posters when available,
  - groups results as “Similar to your taste / Popular among similar users / etc.”,
  - exposes inline actions whose effects you can inspect via `demo_v4_feedback_flow` and repeated `/recommend` calls.

---

## 📈 Results you produced (V3 vs V4 rankers)

### V3 ranked (for reference)

Your V3 ranking metrics (VAL) showed clear lift over V2, especially at higher K, due to richer candidate pools and better features.

### V4 ranked (VAL + TEST)

From your terminal runs:

- **V4 VAL ranker**
  - Train positive rate ≈ **0.000808**
  - AUC ≈ **0.8033**

- **V4 TEST ranker**
  - Train positive rate ≈ **0.001316**
  - AUC ≈ **0.6903**

Evaluation with `scripts.eval_ranked_candidates_v4_val` and
`scripts.eval_ranked_candidates_v4_test` showed reasonable recall/NDCG values across K, demonstrating that the V4 ranker behaves sensibly on **both** validation and test splits.

(Exact numbers depend on your last run; see terminal for the latest snapshot.)

---

## 🧩 Candidate sources and how the ranker sees them

In both V3 and V4 pointwise features, we explicitly encode:

- `blend_score`
- `has_tt` (Two-Tower ANN)
- `has_seq` (GRU)
- `has_v2` (V2 CF priors)
- user-level stats
- item-level stats
- **V4-only:** `short_term_boost`, `sess_hot`, `sess_warm`, `sess_cold`

This makes the ranker **source-aware** and now also **session-aware**, enabling it to learn:
- when sequence is a better signal,
- when embeddings are strong,
- when popularity priors rescue sparse users,
- and when recent interactions should give a near-term boost.

---

## 🧠 “Why this” explanation (V3 & V4 service)

We maintain lightweight explanations for UI clarity:

- If `has_seq == 1` and the item is close to `last_item_idx`  
  → **“Because you watched X recently”**
- Else if `has_tt == 1`  
  → **“Similar to your taste”**
- Else if `has_v2 == 1`  
  → **“Popular among similar users”**
- Otherwise  
  → **“Trending now”** (or similar fallback)

V4 keeps this rule-based layer, but now has better session anchors (e.g., `last_title`) and diversity‑aware reordering, which makes the final slate *feel* more intentional.

---

## 🔁 V3 + V4 feedback loop

### `/feedback` contract (V4)

```jsonc
POST /feedback

{
  "user_idx": 9764,
  "item_idx": 19758,
  "event": "like" // or "remove_like", "watched", "watch_later", "remove_watch_later", "skip"
}
```

- Input validation is strict; an invalid `event` will trigger **422**.
- Successful events are persisted to `data/processed/feedback_state_v4.json`.

### How V4 uses feedback

- Likes & watched events increase `short_term_boost` and contribute to `sess_hot` / `sess_warm` tagging.
- Watch-later flags can be surfaced in separate “Watch later” sections in future UI iterations.
- Skips can be used to slightly penalize items for that user within the same session.

The **demo script** `scripts.demo_v4_feedback_flow` walks through:
- a “before” recommendation slate,
- a couple of feedback calls,
- and an “after” slate so you can see real changes.

---

## 🌈 Streamlit UI — V3 & V4 modes

The Streamlit app (`ui/streamlit_app.py`) has:

- **Mode toggle**
  - **V3 (Feedback-loop)** — uses the V3 API on port `8003`.
  - **V4 (Session-aware)** — uses the V4 API on port `8004`.

- **API controls**
  - Base URL (overridable if needed)
  - Split selector (`val` / `test` in V4)
  - `k` slider
  - `include_titles`
  - `debug`
  - `apply_diversity (V4)`

- **Presentation**
  - Netflix-style rows with cards
  - Cards show poster, title, score, and one-line explanation
  - Inline feedback buttons mapped to `/feedback`

In V4 mode, the expectation is that:
- **posters** resolve via `poster_cache_v4.json` wherever possible,
- **re-ranking** changes when you toggle `apply_diversity`,
- **feedback** calls change subsequent `/recommend` results for that `user_idx`.

Known local issues (depending on TMDB/network):
- some posters will remain missing (network resets or TMDB misses),
- the cache builder logs those errors and continues,
- UI falls back to “Poster unavailable” placeholder cards.

---

## 📸 Poster cache v4 — details

### Builder

```bash
export TMDB_API_KEY="your_tmdb_key_here"

python -m scripts.build_poster_cache_v4
```

This will:
- read `data/processed/item_meta.parquet`,
- iterate over `movieId + title` pairs,
- call TMDB’s search APIs,
- write `data/processed/poster_cache_v4.json` with a compact mapping.

Transient errors (connection reset, rate limits) are logged and skipped; you can re-run to gradually improve coverage.

### Runtime lookup

In the V4 service (`reco_service_v4.py`):

1. For each recommendation, we find its `movieId` from `item_meta`.
2. We look it up in `poster_cache_v4.json`.
3. If found, we attach a full poster URL; if not, we try legacy caches and finally a placeholder.

This design avoids the **title/poster mismatches** that happen when you mix `item_idx`-keyed and `movieId`-keyed caches.

---

## 🆚 How this matches Netflix/Amazon (and where it doesn’t)

### Where we match the *shape*

- ✅ Multi-stage architecture  
- ✅ Candidate blending  
- ✅ Neural retrieval  
- ✅ Sequence intent  
- ✅ Ranker with source-aware + session-aware features  
- ✅ UI sections + explanations  
- ✅ Feedback loop that changes results  
- ✅ Diversity-aware slate reshaping (V4 flag)  
- ✅ Poster caching + graceful fallbacks  

### Where we don’t match the *scale*

- ❌ No real-time streaming features  
- ❌ No massive online feature store  
- ❌ No multi-objective / slate-aware ranker with strict constraints  
- ❌ No contractual/licensing constraints  
- ❌ No household maturity controls  
- ❌ No originals-boosting business rules  

The MVP is intentionally **honest** about these limits while still reflecting the correct design patterns.

---

## 🧪 How to run — quick paths

These assume you have already prepared the raw data and earlier steps as per the project notebooks / scripts.

### 1) V3 core pipeline (summary)

Same as in the previous README — the core sequence + ANN + V3 ranker steps:

```bash
# Neural + sequence prep
python -m src.neural.data_prep
python -m src.sequence.data_prep
python -m src.neural.train_two_tower_v3
python -m src.neural.export_embeddings_v3
python -m src.neural.ann_build_hnsw_v3
python -m src.sequence.train_gru_v3

# VAL pipeline
python -m src.ranking.export_v2_candidates_val
python -m src.neural.ann_generate_candidates_val_v3
python -m src.sequence.generate_candidates_v3
python -m src.ranking.blend_candidates_v3_val
python -m src.ranking.train_ranker_v3_val
python -m scripts.eval_ranked_candidates_v3
```

### 2) V3 TEST pipeline

```bash
python -m scripts.make_v3_test_pipeline
```

This produces the V3 TEST analogs (candidates, features, ranker).

### 3) V4 session features, rankers & eval

Assuming:
- you have V3 candidate files such as `data/processed/v3_candidates_val.parquet` and `v3_candidates_test.parquet`,
- you’ve already run the V4 session feature builders that produce:
  - `data/processed/session_features_v4_val.parquet`
  - `data/processed/session_features_v4_test.parquet`

You can then run:

```bash
# Build V4 VAL pointwise table (10M rows)
python -m src.ranking.build_v4_val_pointwise

# Train V4 VAL ranker
python -m src.ranking.train_ranker_v4_val

# Build V4 TEST pointwise table (10M rows)
python -m src.ranking.build_v4_test_pointwise

# Train V4 TEST ranker
python -m src.ranking.train_ranker_v4_test

# Evaluate V4 on VAL + TEST
python -m scripts.eval_ranked_candidates_v4_val
python -m scripts.eval_ranked_candidates_v4_test
```

You should see AUC / recall / NDCG metrics printed in the terminal for both splits.

### 4) Build poster cache (V4)

```bash
export TMDB_API_KEY="your_tmdb_key_here"
python -m scripts.build_poster_cache_v4
```

This generates `data/processed/poster_cache_v4.json` and logs any TMDB/network errors without failing the run.

### 5) Run V4 API + demo scripts

```bash
# API (no reload, port 8004)
python -m uvicorn src.service.api_v4:app --port 8004

# In another terminal, sanity checks:
python -m scripts.demo_v4_service
python -m scripts.demo_v4_api
python -m scripts.demo_v4_feedback_flow
```

- `demo_v4_service` calls the service in-process and prints out a slate.
- `demo_v4_api` talks to the running FastAPI server on `http://127.0.0.1:8004`.
- `demo_v4_feedback_flow` simulates a before/after experience using feedback events.

### 6) Run Streamlit UI (V3+V4)

```bash
cd ui
streamlit run streamlit_app.py
```

In the UI:
- Select **V4 (Session-aware)** mode.
- Set API base to `http://127.0.0.1:8004`.
- Pick `split = val` (or `test` if you want to experiment with the test pipeline).
- Use the slider + checkboxes to drive `/recommend` calls.
- Interact with “Like / Watched / Watch later / Start” buttons and re-request recommendations to see how the slate changes.

---

## 🔌 API contracts (V3 & V4)

### GET `/health`

```json
{
  "status": "ok",
  "version": "v4"
}
```

### GET `/recommend` (V4)

**Query params**

- `user_idx: int`
- `k: int`
- `include_titles: bool`
- `debug: bool`
- `split: "val" | "test"`
- `apply_diversity: bool` (optional, default `false`)

**Response (conceptual)**

```jsonc
{
  "user_idx": 9764,
  "k": 20,
  "version": "v4",
  "split": "val",
  "apply_diversity": true,
  "items": [
    {
      "item_idx": 22123,
      "movieId": 106916,
      "title": "American Hustle (2013)",
      "poster_url": "https://image.tmdb.org/t/p/w342/....jpg",
      "score": 0.58,
      "reason": "Similar to your taste",
      "has_tt": 1,
      "has_seq": 0,
      "has_v2": 0,
      "short_term_boost": 0.0,
      "sess_hot": 0,
      "sess_warm": 0,
      "sess_cold": 0
    }
  ],
  "debug": {
    "split": "val",
    "candidates_path": ".../data/processed/v3_candidates_val.parquet",
    "session_features_path": ".../data/processed/session_features_v4_val.parquet",
    "ranker_path": ".../reports/models/ranker_hgb_v4_val.pkl",
    "model_auc": 0.8033,
    "feature_order": ["blend_score", "...", "item_days_since_last"],
    "candidate_count": 200,
    "session": {
      "short_term_boost": 0.0,
      "sess_hot": 0,
      "sess_warm": 0,
      "sess_cold": 0,
      "last_item_idx": null,
      "last_title": null
    }
  }
}
```

### POST `/feedback` (V4)

```jsonc
{
  "user_idx": 9764,
  "item_idx": 19758,
  "event": "like"
}
```

On success:

```json
{
  "status": "ok",
  "detail": null
}
```

On invalid `event`:

```jsonc
{
  "detail": [
    {
      "type": "literal_error",
      "loc": ["body", "event"],
      "msg": "Input should be 'like', 'remove_like', 'watched', 'watch_later', 'remove_watch_later' or 'skip'",
      "input": "start"
    }
  ]
}
```

Use the names **exactly** as defined; otherwise FastAPI will return 422.

---

## 🧯 Known limitations (honest MVP notes)

- The two-tower model is trained with **capped users/items** for local feasibility.  
- The sequence model only covers users with sufficient history.  
- The V3/V4 blend weights are **hand-tuned** for now.  
- Poster coverage depends on TMDB and local network reliability.  
- V4 is still **pointwise**, not listwise, and diversity is applied with simple heuristics.  
- Feedback state is stored in a **local JSON file**, not a real feature store.

These are *acceptable* constraints for a learning-grade, local MVP whose aim is to teach architecture and product thinking, not to match FAANG infra.

---

## 🗺️ What’s next (beyond V4)

Logical next steps after V4:

1. **Richer diversity & exploration**
   - slate-aware objective
   - novelty / serendipity constraints

2. **Multi-objective ranking**
   - engagement + completion + freshness + catalog health

3. **Streaming-first architecture**
   - Kafka/Kinesis-based event ingestion
   - near-real-time embedding refresh and session features

4. **Content-aware representations**
   - textual + visual embeddings combined with behavior data

5. **Production-hardening**
   - metrics dashboards
   - A/B experimentation harness
   - canary deploys and model rollback

---

## 🙌 Closing note

The repo now demonstrates a **credible mini-version of a modern recommender stack**:
not because it matches Netflix scale, but because it respects the **correct architecture, data discipline, evaluation rigor, and product UX patterns** that make Netflix/Amazon-like systems feel intelligent.

---

### 📎 Included assets

- `recall_v2_v3.png`
- `ndcg_v2_v3.png`

---

**Maintainer:** Deepesh Kumar Gupta  
**Focus:** Product-grade learning systems, not just model demos 🚀
