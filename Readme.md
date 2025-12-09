# 🎬 Movie Recommendation MVP (V2 + UI)

A **local-first, production-style Movie Recommendation System** built on the **MovieLens 20M** dataset.  
This repo is designed as a serious, stepwise MVP that mirrors **real-world recommender architecture** used by platforms like **Netflix** and **Amazon Prime Video**, but deliberately scoped to remain:

- ✅ **reproducible**
- ✅ **offline-evaluable**
- ✅ **explainable**
- ✅ **fast on Apple Silicon (M1)**
- ✅ **productized locally** with API + UI

If you want to demonstrate an end-to-end recommender—from data to a user-facing experience—without cloud infrastructure, this is the right blueprint.

---

## 🧭 What This Project Is (and Isn’t)

### ✅ This is
A complete **multi-stage recommender MVP** with:

- 📥 Robust ingestion and analytics layer  
- 🔍 Hybrid candidate retrieval  
- 🧮 Time-aware ranking  
- 🧠 Genre affinity cross-features  
- 🏗️ Feature store  
- 🌐 FastAPI service  
- 🖥️ Streamlit UI with posters + feedback loop  
- ⚡ Real-time UI-only personalization  

### ❌ This is not
- A full-scale deep learning recommender platform  
- A true online-learning system  
- A direct replica of Netflix/Prime infrastructure  
- A cloud-scale retrieval/ranking system with petabyte telemetry  

This is intentional. The goal is to be **production-faithful**, not production-identical.

---

## 🧠 Why This Approach Is Different

Most hobby or tutorial recommenders stop at:

- popularity  
- matrix factorization  
- basic item-item similarity  

They often lack:
- robust offline evaluation  
- time-based splitting  
- multi-source retrieval  
- ranking layers  
- explainability  
- product surfaces  

This MVP goes beyond that by adopting **modern multi-stage design**:

### 🔥 Key differentiators
1) **Hybrid retrieval, not single-model recommendations**  
   We generate candidates from multiple “experts”:
   - ⭐ Popularity  
   - 🔁 Item-Item similarity  
   - 🤝 ALS-based collaborative filtering  
   - 🎭 Genre neighbors  

   This is closer to how large platforms build resilient candidate sets.

2) **Time-aware scoring with decay signals**  
   Most public projects ignore recency dynamics.  
   This MVP explicitly models:
   - user recency  
   - item recency  
   - confidence decay  

3) **Feature store-first mindset**  
   We precompute:
   - user-level aggregates  
   - item-level aggregates  
   - genre priors and affinity  
   so the ranker and service work efficiently.

4) **Ranker with cross-feature logic**  
   Instead of naive blending or fixed rules, we train a ranker using:
   - user activity + confidence sums  
   - item popularity + confidence sums  
   - days since last interaction  
   - ✅ user–item genre affinity cross-features  

5) **Explainability baked into service output**  
   Each recommendation can surface:
   - “Trending now”  
   - “Matches your genres”  
   - “Similar to your taste”  
   - “Boosted by your likes” (UI layer)

6) **Full product loop**
   Not just models:
   - API  
   - UI  
   - posters  
   - feedback logging  
   - real-time personalization  

---

## 📺 How This Matches Netflix / Amazon Prime (Conceptually)

Netflix and Amazon Prime operate with:

- massive behavioral telemetry  
- deep personalization stacks  
- multi-model ensembles  
- sophisticated experimentation systems  
- large-scale ranking infrastructure  
- online learning and real-time feature pipelines  

We obviously won’t replicate that scale locally.  
But the **shape of the architecture** is similar.

### ✅ Where we match
- ✅ **Multi-stage architecture** (retrieve → rank)  
- ✅ **Hybrid candidate sources**  
- ✅ **Recency-aware modeling**  
- ✅ **Feature store mindset**  
- ✅ **Explainability + product surfaces**  
- ✅ **Feedback-driven personalization (simulated)**  

### ⚠️ Where we do not match (by design)
- ❌ No neural retrieval models (two-tower)  
- ❌ No real user session tracking  
- ❌ No real-time streaming pipelines (Kafka/Flink)  
- ❌ No A/B experimentation engine  
- ❌ No multi-device context modeling  
- ❌ No large-scale knowledge graph enrichment  
- ❌ No cloud-scale ANN infrastructure  

This repo is a **faithful MVP** of the *architecture pattern*, not the *enterprise system*.

---

## 🧱 System Architecture (MVP)

### 1) 📥 Data Layer
- Downloads MovieLens 20M
- Builds Parquet datasets
- Creates DuckDB views for fast analytics

**Why it matters:**
- Produces a clean offline data foundation  
- Enables faster iteration  
- Mimics real-world warehouse → feature store flows  

---

### 2) 🔍 Candidate Generation (V1.5)

Candidate sources:
- ⭐ Popularity  
- 🔁 Item-Item similarity  
- 🤝 ALS CF (implicit confidence)  
- 🎭 Genre neighbors  

They are combined into a **hybrid retrieval pool**.

**Why it matters:**
- Single-model retrieval is fragile  
- Hybrid retrieval improves recall coverage  
- Mirrors real-world ensemble strategy  

---

### 3) 🏗️ Feature Store (V2)

Generated files:
- 👤 `user_features.parquet`
- 🎞️ `item_features.parquet`
- 🎭 `genre_item_priors.parquet`
- 🧩 Expanded genre + affinity lookups

Feature themes:
- interaction volume  
- confidence sums  
- decay-weighted confidence  
- recency tracking  
- genre priors  

**Why it matters:**
- Separates compute-heavy aggregation from real-time ranking  
- Enables fast and stable inference  
- Reflects production best practice  

---

### 4) 🧮 Ranking (V1 → V2)

Ranker:
- **HistGradientBoostingClassifier**

V2 features include locked ordering:
- user_interactions  
- user_conf_sum  
- user_conf_decay_sum  
- user_days_since_last  
- item_interactions  
- item_conf_sum  
- item_conf_decay_sum  
- item_days_since_last  

Plus:
- ✅ **genre-level cross intelligence**

**Why it matters:**
- Moves beyond heuristic blending  
- Improves top-K precision  
- Creates an interpretable, scalable local ranker  

---

### 5) 🌐 Serving Layer (Step 6)

- **V2RecommenderService**
- **FastAPI**
- Debug endpoints and reason tags

Endpoints:
- `/health`
- `/recommend/user/{user_idx}?k=`
- `/recommend/user/{user_idx}/debug?k=`

**Why it matters:**
- Converts a model into a product capability  
- Makes your MVP callable and testable  
- Enables UI integration  

---

### 6) 🖥️ UI Layer (Step 7)

Built with **Streamlit** to avoid frontend overhead.

#### ✅ Step 7.1
- UI that calls FastAPI
- Displays ranked recommendations

#### ✅ Step 7.2
- 🎞️ Poster support via TMDB (optional)
- Local multi-threaded poster cache
- 👍/👎 “Like/Dislike/Save” feedback stored locally

#### ✅ Step 7.3
- ⚡ Real-time UI-only re-ranking based on likes
- “Your recent taste” panel
- Cached API calls for smoother UX

**Why it matters:**
- Demonstrates a full user-facing loop  
- Enables realistic product demos  
- Simulates personalization before online learning  

---

## 📊 Offline Evaluation Philosophy

We use:
- **time-based splits**
- hybrid recall evaluation  
- ranked hybrid evaluation  

Metrics:
- Recall@10/20/50  
- NDCG@10/20/50  

**Why it matters:**
- Random splits often inflate results  
- Time-based evaluation is closer to real consumption  
- Helps prevent false confidence in model quality  

---

## 📈 Latest Offline Results Snapshot (V2 Ranked Hybrid)

Your recent successful run showed healthy top-K improvements, approximately:

- ✅ recall@10 ≈ 0.0253  
- ✅ recall@20 ≈ 0.0454  
- ✅ recall@50 ≈ 0.0592  
- ✅ NDCG improved across top-K  

Interpretation:
- Candidate layer provides breadth  
- Ranker improves relevance at the top  

---

## 🧰 Tech Stack

- 🐍 Python 3.11
- ⚡ Polars
- 🦆 DuckDB
- 🤝 implicit
- 🌲 scikit-learn
- 📈 tqdm
- 🌐 FastAPI + Uvicorn
- 🧱 Joblib
- 🖥️ Streamlit
- 🌍 Requests

---

## 📁 High-Level Project Layout

```text
movie-recommendation-mvp/
├── app/                        # FastAPI app
├── ui/                         # Streamlit UI + feedback + rerank
├── scripts/                    # CLI runners / demos / eval flows
├── src/
│   ├── config/                 # settings
│   ├── data/                   # download, ingest, duckdb
│   ├── eval/                   # splits + metrics
│   ├── models/                 # popularity, item-item, ALS
│   ├── retrieval/              # hybrid blending + genre neighbors
│   ├── ranking/                # training data + rankers + feature store
│   ├── metadata/               # poster cache (TMDB optional)
│   └── service/                # V2 inference layer
├── data/
│   ├── raw/
│   └── processed/
└── reports/models/             # trained rankers + meta
```

---

## ⚙️ Setup

```bash
cd movie-recommendation-mvp
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

Install dependencies:

```bash
pip install polars duckdb implicit scikit-learn tqdm fastapi uvicorn joblib streamlit requests
```

---

## 🧪 Step-wise Execution

### ✅ Step 1: Data Build

```bash
source .venv/bin/activate
python -m scripts.download_movielens
python -m src.data.ingest_movielens
python -m src.data.create_duckdb
```

---

### ✅ Step 2: V1 Baselines

```bash
python -m scripts.eval_v1_baselines
```

---

### ✅ Step 3: V1.5 Hybrid Candidates

```bash
python -m src.data.prepare_implicit_confidence
python -m src.eval.split_confidence
python -m scripts.eval_v1_5_candidates
```

---

### ✅ Step 4: Ranking (V1)

```bash
python -m src.ranking.build_training_data
python -m src.ranking.train_ranker
python -m scripts.eval_ranked_candidates
```

---

### ✅ Step 5: V2 MVP Upgrades

```bash
python -m src.ranking.feature_store
python -m scripts.eval_v2_candidates
python -m src.ranking.train_ranker_v2
python -m scripts.eval_ranked_candidates_v2
```

---

### ✅ Step 6: Serving

```bash
uvicorn app.main:app --reload
```

Test:

```bash
curl http://127.0.0.1:8000/health
curl "http://127.0.0.1:8000/recommend/user/9764?k=10"
curl "http://127.0.0.1:8000/recommend/user/9764/debug?k=5"
```

---

### ✅ Step 7: UI

Run UI:

```bash
python -m scripts.run_ui
```

---

## 🎞️ Optional Posters (TMDB)

Set key:

```bash
export TMDB_API_KEY="YOUR_KEY"
```

Build cache:

```bash
python -m scripts.build_posters_cache
```

Output:

- `data/processed/item_posters.json`

---

## 👍 Feedback Loop (Local)

UI logs feedback to:

- `data/processed/ui_feedback.jsonl`

Actions:
- 👍 Like
- 👎 Dislike
- ⭐ Save

---

## ⚡ Real-time Personalization (UI-only)

When enabled in UI:
- reads liked genres from feedback logs  
- applies a small boost to matching items  
- reorders the visible top-K  

This simulates an online personalization feel  
without retraining or streaming infra.

---

## 🔍 Why This MVP Is Useful in the Real World

Even without deep learning:

- This repo is a strong template for:
  - enterprise prototyping  
  - internal stakeholder demos  
  - offline evaluation pipelines  
  - recommender learning projects  
  - local-first experimentation  

It teaches the *product shape* of recommender systems:
not just the models.

---

## 🛣️ Roadmap Beyond This MVP

Potential next expansions aligned with production-grade systems:

### 🧠 Retrieval upgrades
- Two-tower embedding retrieval  
- ANN with FAISS  

### 🎯 Ranking upgrades
- Pairwise or listwise ranking  
- LightGBM ranking objectives  

### 🔁 Session intelligence
- “Because you watched X” logic  
- short-term intent modeling  
- time-of-day personalization  

### 🧪 Experimentation
- offline compare harness  
- synthetic A/B simulation  
- evaluation dashboards  

---

## 🙌 Credits & Data

- Dataset: **MovieLens 20M** by GroupLens Research  
- Used here for education, research, and MVP prototyping.

---

## 📌 Status

✅ **V2 multi-stage recommender + API + UI + feedback loop complete**  
Next optional milestone: **Step 7.4** (session-style UI modules and “Because you watched” experiences).
