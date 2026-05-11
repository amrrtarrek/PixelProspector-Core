# PixelProspector V4.0 — System Architecture & Logic

> **An Autonomous Agentic Flywheel for Indie Game Publishing Discovery**

---

## Table of Contents

1. [What Is PixelProspector?](#1-what-is-pixelprospector)
2. [Big Picture — The Agentic Flywheel](#2-big-picture--the-agentic-flywheel)
3. [The V4.0 Data Contract](#3-the-v40-data-contract)
4. [The Inner Loop — Per-Review Pipeline](#4-the-inner-loop--per-review-pipeline)
   - [Stage 1 — Data Ingestion](#stage-1--data-ingestion)
   - [Stage 2 — Unsupervised Clustering](#stage-2--unsupervised-clustering)
   - [Stage 3 — Supervised ML & Explainability](#stage-3--supervised-ml--explainability)
   - [Stage 4 — Forecasting](#stage-4--forecasting)
   - [Stage 5 — FastAPI ReAct Agent](#stage-5--fastapi-react-agent)
5. [The 5 Intelligent Score Signals](#5-the-5-intelligent-score-signals)
6. [The 7-Path ReAct Router](#6-the-7-path-react-router)
7. [The Outer Loop — Autonomous Self-Healing](#7-the-outer-loop--autonomous-self-healing)
8. [The Dashboard — 6 Monitoring Zones](#8-the-dashboard--6-monitoring-zones)
9. [Complete System Flow Diagram](#9-complete-system-flow-diagram)
10. [Technology Stack](#10-technology-stack)

---

## 1. What Is PixelProspector?

PixelProspector is a **5-stage autonomous ML pipeline** that reads raw Steam game reviews and determines whether an indie game has genuine publishing potential — without human supervision.

It solves a core problem in the indie gaming industry: **thousands of games are published on Steam every year, but genuine opportunities are buried under noise** — fake reviews, spam, low-effort feedback. Traditional scoring systems are static and go stale as review patterns shift.

PixelProspector addresses this with an **Agentic Flywheel** — a system that not only scores games and reviewers, but also **detects when its own models are drifting** and **autonomously retrains itself** to stay accurate.

---

## 2. Big Picture — The Agentic Flywheel

The system has two concentric loops:

```
┌─────────────────────────────────────────────────────────┐
│                    AGENTIC FLYWHEEL                     │
│                                                         │
│   ┌──────────────────────────────────────────────────┐  │
│   │               INNER LOOP                         │  │
│   │  (runs on every review — real-time scoring)      │  │
│   │                                                  │  │
│   │  [1] Ingest → [2] Cluster → [3] SVM+SHAP        │  │
│   │       → [4] ARIMA → [5] ReAct Decision           │  │
│   └──────────────────────────────────────────────────┘  │
│                          ↕                              │
│   ┌──────────────────────────────────────────────────┐  │
│   │               OUTER LOOP                         │  │
│   │  (runs every 5 minutes — system health monitor)  │  │
│   │                                                  │  │
│   │  Drift Check → Auto-Retrain → Cooldown           │  │
│   └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

- The **Inner Loop** processes each review and produces a scored, actionable output.
- The **Outer Loop** watches for model degradation and heals the system automatically.
- Both loops write to **PostgreSQL** and are visible on the **Streamlit dashboard**.

---

## 3. The V4.0 Data Contract

Every record flowing through the system must conform to the **V4.0 JSON Contract** — a strict schema validated by Pydantic V2. This is the single source of truth that all 5 stages agree on.

```json
{
  "interaction_metadata": {
    "user_id":         "string",
    "game_id":         "string (prefix: st_)",
    "timestamp":       "ISO-8601",
    "developer_email": "string",
    "primary_genre":   "string",
    "triage_status":   "Pass | Rejected"
  },
  "game_ml_features": {
    "gameplay_addictiveness": 0.0–1.0,
    "technical_polish":       0.0–1.0,
    "aesthetic_appeal":       0.0–1.0,
    "narrative_depth":        0.0–1.0,
    "replayability":          0.0–1.0,
    "viral_momentum":         0.0–1.0
  },
  "user_review_features": {
    "insight_depth":          0.0–1.0,
    "toxicity_level":         0.0–1.0,
    "genre_expertise":        0.0–1.0,
    "sentiment_consistency":  0.0–1.0
  },
  "intelligent_score_signals": {
    "S_class_severity":        float,
    "Gap_SVM_confidence":      0.0–1.0,
    "mu_geometric_membership": 0.0–1.0,
    "ARIMA_trend_multiplier":  0.5–2.0,
    "SHAP_cosine_similarity":  -1.0–1.0
  },
  "llm_audit_log": "string"
}
```

> **Key rule:** Signals are `0.0` at ingest time. They are progressively filled by Stages 3, 4, and 5.

---

## 4. The Inner Loop — Per-Review Pipeline

### Stage 1 — Data Ingestion

**Owner:** Member 1 | **File:** `01_data_ingestion/ingest.py`

This stage converts raw, unstructured Steam review text into a validated V4.0 JSON record.

#### How it works:

```
Raw Review Text (CSV or live submission)
          ↓
   Build User Prompt
          ↓
  Gemini 2.5-flash LLM
  (Structured JSON Output)
          ↓
   Pydantic V2 Validation
   (strict schema gate)
          ↓
  PostgreSQL (via SQLAlchemy)
```

#### LLM Role:
The LLM reads the raw review and infers all 10 numeric features (6 game + 4 user). It does not invent — it extracts signals from the review text using a precise system prompt with feature definitions.

#### Fallback:
If `GEMINI_API_KEY` is not set or LLM fails, a **deterministic heuristic parser** runs instead — using keyword counting and play-hour statistics to estimate features. Output is valid V4.0 but less accurate.

#### Triage Status:
All records enter as `"triage_status": "Pass"`. The Pre-scoring Triage Gate in Stage 5 may flip this to `"Rejected"`.

#### Two ingestion flows:
| Flow | Trigger | Description |
|------|---------|-------------|
| **Flow A (Batch)** | CSV file | Kaggle Steam reviews, batch processed |
| **Flow B (Live)** | Dashboard | Free-text user submission, analysed in real-time |

---

### Stage 2 — Unsupervised Clustering

**Owner:** Member 2 | **File:** `02_unsupervised_ml/train_clusters.py`

This stage builds the system's **memory** — the ground truth clusters that everything else references.

#### How it works:

```
Raw Feature Matrix (from PostgreSQL)
          ↓
   DBSCAN Outlier Filter
   (removes noise points before clustering)
          ↓
   Fuzzy C-Means (FCM)
   (soft clustering — every point has membership in ALL clusters)
          ↓
   ┌─────────────────┐    ┌─────────────────┐
   │  Game Pipeline  │    │  User Pipeline  │
   │  3 clusters:    │    │  3 clusters:    │
   │  Flop           │    │  Spam           │
   │  Niche          │    │  Casual         │
   │  Breakout       │    │  Expert         │
   └────────┬────────┘    └────────┬────────┘
            └──────────┬───────────┘
                       ↓
        4 Output Artifacts:
        • fcm_centroids.pkl      → used by Stages 3 & 4
        • cluster_mean_shap.pkl  → used by Stage 3 (Implicit RAG)
        • training_data.pkl      → used by Stage 3 (SVM training)
        • pixel_prospector.index → FAISS index, used by Stage 5 (Explicit RAG)
```

#### Why FCM over k-means?
Hard clustering (k-means) forces every game into exactly one bucket. FCM gives each game a **probability of belonging to each cluster** — a game might be 60% Niche, 30% Breakout, 10% Flop. This soft membership is more realistic for borderline cases and is used directly in Signal 3.

#### Two RAG Mechanisms built here:
| RAG Type | Artifact | Used by |
|----------|----------|---------|
| **Implicit RAG** | `cluster_mean_shap.pkl` — average SHAP vector per cluster | Stage 3 (Signal 5) |
| **Explicit RAG** | `pixel_prospector.index` — FAISS vector index | Stage 5 (borderline voting) |

---

### Stage 3 — Supervised ML & Explainability

**Owner:** Member 3 | **File:** `03_supervised_ml/live_inference.py`

This stage classifies every live record and computes 4 of the 5 intelligent signals.

#### How it works:

```
V4.0 JSON Payload
          ↓
  Extract game_ml_features (6D vector)
  Extract user_review_features (4D vector)
          ↓
  ┌──────────────────┐  ┌──────────────────┐
  │   Game SVM       │  │   User SVM       │
  │   RBF kernel     │  │   RBF kernel     │
  │   probability=T  │  │   probability=T  │
  │   → class label  │  │   → class label  │
  │   → proba array  │  │   → proba array  │
  └────────┬─────────┘  └────────┬─────────┘
           └──────────┬──────────┘
                      ↓
        Compute 4 Signals (see Section 5)
                      ↓
        Signal 4 slot left as None (→ filled by Stage 4)
```

#### SHAP Explainability (Signal 5):
```
Game Feature Vector
          ↓
  SHAP KernelExplainer
  (background = FCM centroids)
          ↓
  Live SHAP Vector (6D)
          ↓
  Cosine Similarity vs Cluster Mean SHAP
  (from cluster_mean_shap.pkl — Stage 2's Implicit RAG)
          ↓
  Signal 5: SHAP_cosine_similarity
```

This answers: *"Is the model's reasoning for this record consistent with how it normally reasons about games in this cluster?"* A high score means the explanation is trustworthy.

---

### Stage 4 — Forecasting

**Owner:** Member 4 | **File:** `04_forecasting/drift_monitor.py`

This stage provides the temporal dimension — how is the cluster's momentum trending over time?

#### ARIMA Signal (Signal 4):

```
PostgreSQL: viral_momentum column
(ordered by timestamp, all historical records)
          ↓
  Minimum 10 data points required
          ↓
  pmdarima.auto_arima()
  (Seasonal ARIMA, auto-selects best p,d,q,P,D,Q)
          ↓
  forecast = predict(n_periods=1)
  baseline = mean of last 4 observations
          ↓
  raw_multiplier = forecast / baseline
          ↓
  clamp to [0.50, 2.00]
          ↓
  Signal 4: ARIMA_trend_multiplier
```

| Multiplier | Meaning | Effect on Score |
|------------|---------|-----------------|
| `> 1.0` | Rising trend — cluster is growing | Score amplified |
| `= 1.0` | Neutral / insufficient data | No effect |
| `< 1.0` | Declining trend | Score dampened |

---

### Stage 5 — FastAPI ReAct Agent

**Owner:** Member 5 | **File:** `05_fastapi_agent/core/agent_router.py`

This is the final integration hub — it receives all signals, computes the Intelligent Score, and routes to one of 7 action paths.

#### Request flow through the FastAPI `/v1/predict` endpoint:

```
POST /v1/predict  (V4.0 JSON payload)
          ↓
  ① Pre-scoring Triage Gate
     if toxicity_level > threshold OR insight_depth too low:
         → return "Rejected" immediately
          ↓
  ② Receive 5 Intelligent Score Signals
          ↓
  ③ Compute Intelligent Score
     Score = S×0.4 + Gap×0.2 + μ×0.1 + ARIMA×0.2 + SHAP×0.1
          ↓
  ④ 7-Path ReAct Router
     (see Section 6)
          ↓
  ⑤ Generative Explainability
     SHAP values → LLM → human-readable narrative
          ↓
  ⑥ Multi-Agent Actions
     Community Agent → community profile
     Investor Agent  → investor pitch
          ↓
  ⑦ Write to PostgreSQL + append to /recent_actions buffer
          ↓
  Return JSON response
```

---

## 5. The 5 Intelligent Score Signals

All 5 signals combine into a single **Intelligent Score** weighted as:

```
Score = S×0.4 + Gap×0.2 + μ×0.1 + ARIMA×0.2 + SHAP×0.1
```

| # | Signal | Formula | Owner | Meaning |
|---|--------|---------|-------|---------|
| **1** | `S_class_severity` | `P_game(top) × Trust_user` | Stage 3 | Dynamic severity — expert reviewers amplify the game's score |
| **2** | `Gap_SVM_confidence` | `P_top - P_second` | Stage 3 | How decisive the SVM is — high gap = high confidence |
| **3** | `μ_geometric_membership` | `1 / (1 + d_min)` | Stage 3 | How close the game sits to its nearest FCM centroid |
| **4** | `ARIMA_trend_multiplier` | `forecast / baseline` clamped [0.5, 2.0] | Stage 4 | Is the cluster's viral momentum rising or falling? |
| **5** | `SHAP_cosine_similarity` | `1 - cosine_distance(live_shap, cluster_mean_shap)` | Stage 3 | Is the model's reasoning consistent with the cluster's pattern? |

#### Trust Multiplier (used in Signal 1):
```
User Class → Spam:    0.2× (heavily dampened)
             Casual:  1.0× (neutral)
             Expert:  1.5× (amplified)
```

---

## 6. The 7-Path ReAct Router

The router examines the Intelligent Score and SHAP cosine together to choose exactly one of 7 paths:

```
                     Score > 0.8
                    AND SHAP > 0.8?
                        │
              ┌─────── YES ────────┐
              │                   │
      Path 1: Direct Dispatch     │
      (publish immediately)       │
                                  │
              ┌──────── NO ───────┘
              │
        SHAP < 0.5?
              │
    ┌──── YES ────┐
    │             │
  Path 2:         │
  SHAP Re-check   │
  (uncertain model)
                  │
           ┌──── NO ────┐
           │
     Score very low?
           │
  ┌─── YES ───┐
  │           │
Path 3:       │
Flop          │
Rejection     │
              │
       ┌──── NO ────┐
       │
  Score > 0.8
  but SHAP moderate?
       │
  ┌── YES ──┐
  │         │
Path 4:     │
Human       │
Review      │
            │
     ┌──── NO ────┐
     │
  Borderline score?
     │
  ┌── YES ──────────────┐
  │                     │
  FAISS vote positive?  FAISS vote negative?
  │                     │
Path 5:              Path 6:
RAG High             RAG Low
(promote)            (demote)
                     │
              ┌──── else ────┐
              │
           Path 7:
           Neutral Hold
```

---

## 7. The Outer Loop — Autonomous Self-Healing

**File:** `04_forecasting/drift_monitor.py`

The Outer Loop runs **every 5 minutes** and answers the question: *"Are our models still accurate, or has the data distribution shifted?"*

### Dual-Signal Drift Detection

Drift is only confirmed when **BOTH** signals fire simultaneously — an AND gate that prevents false positives.

#### Signal A — Spatial Centroid Drift
```
Database: most recent 50 records (game_ml_features)
          ↓
  Compute live centroid = mean(50 records)
          ↓
  Euclidean distance to each saved FCM centroid
          ↓
  min_distance > 0.35 threshold?
          ↓
  Signal A FIRED (spatial drift detected)
```

*Meaning: the typical game review coming in today looks significantly different from the clusters we trained on.*

#### Signal B — Confidence Decline
```
Database: most recent 30 Gap_SVM_confidence values
          ↓
  Fit least-squares linear trend line
          ↓
  slope < -0.10 threshold?
          ↓
  Signal B FIRED (SVM confidence is declining)
```

*Meaning: the SVM is becoming less decisive — it's increasingly uncertain which cluster a game belongs to.*

### The AND Gate
```
Signal A fired AND Signal B fired
             ↓
      Drift Confirmed
```

If only one fires, the system logs a warning but does **not** retrain. This prevents a single anomalous batch from triggering an unnecessary retrain.

### Autonomous Retrain Sequence

```
Drift Confirmed
      ↓
  Cooldown check (30-minute lock active?)
      │
  ┌── YES ──┐         ┌── NO ───────────────────────────┐
  │         │         │
  Log       │    Step 1: Invoke Member 2 (subprocess)
  "on       │    python train_clusters.py
  cooldown" │    → Re-clusters all PostgreSQL data
            │    → Generates new fcm_centroids.pkl
            │    → Regenerates FAISS index
            │         ↓
            │    Step 2: Invoke Member 3 (subprocess)
            │    python live_inference.py --train
            │    → Retrains Game SVM on new clusters
            │    → Retrains User SVM on new clusters
            │         ↓
            │    Reload fresh centroids into memory
            │         ↓
            │    Reset ARIMA baseline (Signal 4)
            │         ↓
            │    Start 30-minute cooldown timer
            │         ↓
            │    Write DriftEvent to PostgreSQL
            │    (visible in Dashboard Zone 6)
            └──────────────────────────────────────────┘
```

### Outer Loop Timing Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Poll interval | 300 seconds | Checks drift every 5 minutes |
| Cooldown | 1800 seconds | 30-minute lock after retrain |
| Spatial window | 50 records | Records used for live centroid |
| Confidence window | 30 records | Records used for trend fitting |
| Centroid threshold | 0.35 | Max Euclidean drift before Signal A fires |
| Confidence threshold | -0.10 | Max slope before Signal B fires |
| ARIMA min history | 10 points | Minimum data before ARIMA activates |

---

## 8. The Dashboard — 6 Monitoring Zones

**File:** `01_data_ingestion/dashboard.py` | **Run:** `streamlit run dashboard.py`

The Streamlit dashboard is **read-only** — it displays system state but contains no ML logic. It queries PostgreSQL and the FastAPI `/health` and `/recent_actions` endpoints.

| Zone | Name | What It Shows |
|------|------|---------------|
| **Zone 1** | Cluster Health | Bar charts of average game/user feature values — shows how cluster characteristics look right now |
| **Zone 2** | Live Scoring Feed | Table of recent scored records + 5-signal KPI metrics + live review submission tab (Gemini-powered) |
| **Zone 3** | Action Dispatch | ReAct router decisions from FastAPI, API online/offline status |
| **Zone 4** | SHAP Reliability | Line chart of SHAP cosine similarity over time + High/Medium/Low reliability breakdown |
| **Zone 5** | Outcome Tracking | Pass rate progress bar, triage counts, genre distribution |
| **Zone 6** | System Alerts | Active drift events (unhealed) and auto-healed events from the Outer Loop |

---

## 9. Complete System Flow Diagram

```
╔══════════════════════════════════════════════════════════════════════════╗
║                    PIXELPROSPECTOR V4.0 SYSTEM FLOW                     ║
╚══════════════════════════════════════════════════════════════════════════╝

  Raw Steam Review (CSV or Live Dashboard Submission)
                          │
                          ▼
  ┌─────────────────────────────────────────────────────┐
  │  STAGE 1 — DATA INGESTION                           │
  │  Gemini 2.5-flash → V4.0 JSON Contract              │
  │  Pydantic V2 validation → PostgreSQL                │
  └─────────────────────────┬───────────────────────────┘
                            │  (Signals = 0.0 at this point)
                            ▼
  ┌─────────────────────────────────────────────────────┐
  │  STAGE 2 — UNSUPERVISED CLUSTERING (offline/batch)  │
  │  DBSCAN outlier filter → FCM (3 game + 3 user       │
  │  clusters) → FAISS index + SHAP means               │
  └─────────────────────────┬───────────────────────────┘
                            │  fcm_centroids.pkl
                            │  cluster_mean_shap.pkl
                            │  training_data.pkl
                            │  pixel_prospector.index
                            ▼
  ┌─────────────────────────────────────────────────────┐
  │  STAGE 3 — SUPERVISED ML & EXPLAINABILITY           │
  │  Game SVM → Flop / Niche / Breakout                 │
  │  User SVM → Spam / Casual / Expert                  │
  │  → Signal 1 (S_class_severity)                      │
  │  → Signal 2 (Gap_SVM_confidence)                    │
  │  → Signal 3 (μ_geometric_membership)                │
  │  → Signal 5 (SHAP_cosine_similarity)                │
  └─────────────────────────┬───────────────────────────┘
                            │
                            ▼
  ┌─────────────────────────────────────────────────────┐
  │  STAGE 4 — FORECASTING                              │
  │  S-ARIMA on viral_momentum history                  │
  │  → Signal 4 (ARIMA_trend_multiplier)                │
  └─────────────────────────┬───────────────────────────┘
                            │  All 5 signals ready
                            ▼
  ┌─────────────────────────────────────────────────────┐
  │  STAGE 5 — FASTAPI REACT AGENT                      │
  │                                                     │
  │  ① Triage Gate (toxicity/insight check)             │
  │  ② Intelligent Score = weighted sum of 5 signals    │
  │  ③ 7-Path ReAct Router                              │
  │  ④ Generative Explainability (SHAP → narrative)     │
  │  ⑤ Multi-Agent: Community + Investor agents         │
  │  ⑥ Write to PostgreSQL                              │
  └─────────────────────────┬───────────────────────────┘
                            │
                            ▼
  ┌─────────────────────────────────────────────────────┐
  │  STREAMLIT DASHBOARD (6 Zones)                      │
  │  Real-time monitoring of the full pipeline          │
  └─────────────────────────────────────────────────────┘

  ════════════════════════════════════════════════════════

  OUTER LOOP (runs in parallel, every 5 minutes)

  ┌─────────────────────────────────────────────────────┐
  │  DUAL-SIGNAL DRIFT DETECTOR                         │
  │                                                     │
  │  Signal A: live centroid vs FCM centroids           │
  │            (Euclidean distance > 0.35?)             │
  │  Signal B: Gap_SVM trend slope < -0.10?             │
  │                                                     │
  │  A AND B both fired?                                │
  └──────────────┬──────────────────────────────────────┘
                 │ YES — drift confirmed
                 ▼
  ┌─────────────────────────────────────────────────────┐
  │  AUTONOMOUS RETRAIN                                 │
  │  subprocess: train_clusters.py   (Member 2)         │
  │  subprocess: live_inference.py --train (Member 3)   │
  │  → Reload centroids → Reset ARIMA baseline          │
  │  → 30-minute cooldown → Log DriftEvent              │
  └─────────────────────────────────────────────────────┘
                 │
                 └──────────────────────────────────────┐
                                                        │
                                                        ▼
                              Back to INNER LOOP with fresh models
```

---

## 10. Technology Stack

| Stage | Component | Technology |
|-------|-----------|------------|
| **Stage 1** | LLM Analysis | Google Gemini 2.5-flash (`google-generativeai`) |
| **Stage 1** | Data Validation | Pydantic V2 |
| **Stage 1** | Database ORM | SQLAlchemy + PostgreSQL |
| **Stage 1** | Dashboard | Streamlit |
| **Stage 2** | Outlier Filtering | DBSCAN (`scikit-learn`) |
| **Stage 2** | Clustering | Fuzzy C-Means (`scikit-fuzzy`) |
| **Stage 2** | Vector Index | FAISS (`faiss-cpu`) |
| **Stage 3** | Classification | SVM with RBF kernel (`scikit-learn`) |
| **Stage 3** | Model Persistence | joblib |
| **Stage 3** | Explainability | SHAP KernelExplainer |
| **Stage 4** | Forecasting | pmdarima (`auto_arima`) |
| **Stage 4** | Drift Detection | NumPy (least-squares linear fit) |
| **Stage 5** | API Layer | FastAPI + Uvicorn |
| **Stage 5** | Agent Framework | LangChain |
| **All** | Async Runtime | Python `asyncio` |
| **All** | Testing | pytest |

---

*Last updated: May 2026 — PixelProspector V4.0*
