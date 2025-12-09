# 🎬 Movie Recommendation MVP (V2)

A serious, production-style **Movie Recommendation System MVP** built on the **MovieLens 20M** dataset.  
This project implements a modern **multi-stage recommender** inspired by real-world patterns used by platforms like Netflix and Amazon Prime—adapted for **local-first execution on Apple Silicon (M1)**.

---

## ✨ What You Get

### ✅ A complete end-to-end recommender pipeline
- **Data ingestion → Feature store → Candidate retrieval → Ranking → API**
- Offline evaluation with **Recall@K** and **NDCG@K**
- **Explainable recommendation reasons** (lightweight but meaningful)
- **FastAPI service** with debug endpoints

### ✅ V2 MVP highlights
- Hybrid candidate generation (**multi-source**)
- Time-aware signals with **decay**
- User–item **genre affinity cross-features**
- Improved top-K ranking quality
- Built to run efficiently on M1

---

## 🧠 System Architecture (MVP)

### 1) 📥 Data Layer
- Download MovieLens 20M
- Convert to Parquet
- Create a DuckDB layer with enriched views

### 2) 🔍 Candidate Generation
Multiple retrieval signals:
- ⭐ **Popularity**
- 🔁 **Item–Item similarity**
- 🤝 **ALS confidence collaborative filtering**
- 🎭 **Genre neighbors**

These are blended into a **hybrid candidate set**.

### 3) 🏗️ Feature Store (V2)
Precomputed features stored as Parquet:
- 👤 `user_features`
- 🎞️ `item_features`
- 🎭 `genre_item_priors`
- 🧩 `item_genres_expanded`
- ❤️ `user_genre_affinity`

### 4) 🧮 Ranking (V2)
A lightweight ranker:
- **HistGradientBoostingClassifier**
- Uses:
  - user activity + recency
  - item popularity + recency
  - ✅ **user–item genre affinity cross-features**
- Produces strong **top-K** improvements

### 5) 🚀 Serving Layer
- Local inference service
- FastAPI endpoints
- Debug + explainability

---

## 📊 Recent Offline Results (V2 Ranked Hybrid)

The V2 ranker with genre-cross signals achieved:

- **recall@10 ≈ 0.0253**
- **recall@20 ≈ 0.0454**
- **recall@50 ≈ 0.0592**
- Improved NDCG across top-K

This confirms healthy multi-stage behavior:
- ✅ candidates provide coverage  
- ✅ ranker improves precision at the top  

---

## 🧰 Tech Stack

- 🐍 Python 3.11
- ⚡ Polars
- 🦆 DuckDB
- 🤝 implicit (ALS + nearest neighbor models)
- 🌲 scikit-learn (ranking model)
- 📈 tqdm (progress bars)
- 🌐 FastAPI + Uvicorn
- 🧱 Joblib

---

## 📁 Project Structure (high level)

```text
movie-recommendation-mvp/
├── app/                        # FastAPI app
├── scripts/                    # CLI demos + eval runners
├── src/
│   ├── data/                   # download, ingest, duckdb
│   ├── eval/                   # splits + metrics
│   ├── models/                 # popularity, item-item, ALS
│   ├── retrieval/              # hybrid blending + genre neighbors
│   ├── ranking/                # training data + rankers + feature store
│   └── service/                # V2 recommender inference layer
├── data/
│   ├── raw/
│   └── processed/
├── reports/models/             # trained rankers
└── README.md
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
pip install polars duckdb implicit scikit-learn tqdm fastapi uvicorn joblib
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

### ✅ Step 3: V1.5 Candidates

```bash
python -m src.data.prepare_implicit_confidence
python -m src.eval.split_confidence
python -m scripts.eval_v1_5_candidates
```

---

### ✅ Step 4: V1 Ranking

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

## 🎯 Local Demo (Service)

```bash
python -m scripts.demo_v2_service
```

Outputs:
- Top movies with genres
- Scores
- 1–3 reason tags per recommendation

---

## 🌐 Run API

```bash
uvicorn app.main:app --reload
```

### ✅ Test endpoints

```bash
curl http://127.0.0.1:8000/health
```

```bash
curl "http://127.0.0.1:8000/recommend/user/9764?k=10"
```

```bash
curl "http://127.0.0.1:8000/recommend/user/9764/debug?k=5"
```

---

## ⚡ Performance Smoke (Optional)

```bash
python -m scripts.smoke_perf_v2
```

Gives:
- mean / median / p90 / p95 / p99 latency metrics

---

## 🧩 Design Principles (MVP)

- ✅ **Multi-stage architecture**
- ✅ **Time-awareness via decay**
- ✅ **Hybrid retrieval**
- ✅ **Cross-features for ranking**
- ✅ **Local-first execution**
- ✅ **Explainable outputs**
- ✅ **Clean git checkpoints**

---

## 🛣️ Next Possible Upgrades (Post MVP)

### 🎯 Ranking Depth
- Pairwise ranking loss
- LambdaRank-style objective
- LightGBM ranker (optional)

### 🧠 Representation Learning
- Two-tower retrieval
- Embedding-based ANN search (FAISS)

### 🔁 Session + Context
- Short-term session modeling
- Time-of-day preference
- Device/context signals (synthetic for offline)

### 🖥️ UI Layer
- Streamlit demo
- Minimal React dashboard

---

## 🙌 Credits & Data

- Dataset: **MovieLens 20M** by GroupLens Research  
- For research/learning and local MVP experimentation.

---

## ✅ Quick One-Liner

If you want a single command flow later, we can formalize:
- `make data`
- `make v2`
- `make api`

---

## 📌 Status

**V2 MVP complete and productized locally with serving + debug.**  
Next milestone: lightweight UI or advanced ranking experiments.
