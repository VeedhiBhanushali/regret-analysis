# Regret Analysis

Analyzing regret patterns across career, immigration, and relationship decisions using Reddit discourse. This project collects regret-related posts from nine subreddits, engineers structured NLP features (sentiment, emotion, temporal markers, psycholinguistic cues), and applies statistical and machine-learning methods to understand what predicts decision reversal.

**Live Website:** [https://regret-analysis.uc.r.appspot.com](https://regret-analysis.uc.r.appspot.com)

---

## Setup

**Requirements:** Python 3.10+

```bash
git clone https://github.com/VeedhiBhanushali/regret-analysis.git
cd regret-analysis
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Environment variables** (only needed for specific pipeline stages):

| Variable | Used by | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | `src/llm_annotate.py` | GPT-4o-mini label validation |
| `GOOGLE_APPLICATION_CREDENTIALS` | `src/upload_to_firestore.py` | Firestore upload |

---

## Pipeline

The project follows a five-stage pipeline from raw data collection to deployed website:

```
Reddit API ──► Raw CSVs ──► Filtered/Structured ──► Feature Engineering ──► Modeling & Stats ──► Website
 (collect)     (data_raw)     (data_clean)           (sentiment, NLP)      (results, plots)    (index.html)
```

**Stage 1 — Collection:** `src/collect_reddit_buckets.py` scrapes posts from Reddit's public JSON API across nine subreddits (r/cscareerquestions, r/careerguidance, r/jobs, r/careeradvice, r/USCIS, r/IWantOut, r/immigration, r/relationship_advice, r/relationships). Raw data lands in `data_raw/`.

**Stage 2 — Filtering & Structuring:** `src/filter_regret_posts.py` applies regex-based regret extraction with exclusion of hypothetical language. `src/build_structured_*.py` scripts assign reversal labels (from action verbs), urgency scores, and time-to-regret estimates. `src/merge_all_domains.py` and `src/merge_all_structured.py` unify all domains into `data_clean/all_domains_final.csv` (3,604 posts).

**Stage 3 — Feature Engineering:** VADER sentiment (`src/feature_sentiment.py`), improved temporal extraction (`src/feature_temporal_improved.py`), 7-class emotion classification via distilRoBERTa (`src/emotion_classifier.py`), sentence embeddings with all-MiniLM-L6-v2 + PCA (`src/embed_features.py`), NMF topic modeling (`src/topic_model_crossdomain.py`), event taxonomy (`src/event_taxonomy.py`), and psycholinguistic features (`src/psycholinguistic_features.py`).

**Stage 4 — Modeling & Statistics:** Cross-domain reversal classification with LR/RF/GBM and bootstrap CIs (`src/model_reversal_crossdomain.py`), leave-one-domain-out generalization (`src/cross_domain_generalization.py`), Cox PH regression (`src/cox_regression_v2.py`), propensity score matching (`src/propensity_matching.py`), cohort/temporal analysis (`src/cohort_temporal.py`), subreddit confounding analysis (`src/subreddit_mixed_effects.py`), label validation via comment scraping (`src/scrape_comments.py`, `src/finalize_comments.py`), and LLM-based annotation (`src/llm_annotate.py`). Outputs go to `results/` (JSON) and `plots/` (PNG).

**Stage 5 — Deployment:** Static website (`index.html`) served via Google App Engine. Interactive Regret Analyzer component calls the ML inference service on Cloud Run. Dataset is stored in Google Cloud Firestore.

---

## Repository Structure

```
RegretAnalysis/
├── index.html              # Main website (served via App Engine)
├── app.yaml                # App Engine deployment config
├── requirements.txt        # Python dependencies
├── src/                    # All analysis scripts (43 modules)
│   ├── collect_reddit_buckets.py     # Reddit data collection
│   ├── filter_regret_posts.py        # Regret post filtering
│   ├── build_structured_*.py         # Structured feature extraction (per domain)
│   ├── merge_all_domains.py          # Cross-domain merge
│   ├── feature_sentiment.py          # VADER sentiment scoring
│   ├── feature_temporal_improved.py  # Time-to-regret extraction
│   ├── emotion_classifier.py         # 7-class emotion (distilRoBERTa)
│   ├── embed_features.py             # Sentence embeddings (MiniLM + PCA)
│   ├── topic_model_crossdomain.py    # NMF topic modeling
│   ├── psycholinguistic_features.py  # Agency, hedging, future orientation
│   ├── model_reversal_crossdomain.py # Main reversal classifier (LR/RF/GBM)
│   ├── cross_domain_generalization.py# Leave-one-domain-out evaluation
│   ├── cox_regression_v2.py          # Cox PH survival analysis
│   ├── propensity_matching.py        # PSM (career vs immigration)
│   ├── cohort_temporal.py            # Temporal/era analysis
│   ├── upload_to_firestore.py        # Firestore data upload
│   └── ...                           # Additional analysis scripts
├── inference/              # ML inference service (Cloud Run)
│   ├── app.py              # FastAPI application
│   ├── Dockerfile          # Container definition
│   ├── requirements.txt    # Inference-specific dependencies
│   └── models/             # Serialized model artifacts (joblib)
├── data_raw/               # Raw Reddit CSV dumps (~21.8k posts)
├── data_clean/             # Processed datasets, embeddings, checkpoints
├── plots/                  # Generated figures (31 PNGs)
└── results/                # JSON outputs from models and statistics
```

---

## System Design

```
┌─────────────────────────────────────────────────────────────┐
│                        End Users                            │
│                    (Web Browsers)                            │
└──────────────┬──────────────────────┬───────────────────────┘
               │                      │
               │ Static pages         │ POST /analyze
               │ (HTML, plots)        │ (user text input)
               ▼                      ▼
┌──────────────────────┐   ┌──────────────────────────────┐
│   Google App Engine   │   │   Google Cloud Run            │
│                       │   │                               │
│  index.html           │   │  FastAPI inference service     │
│  plots/*.png          │   │  - VADER sentiment            │
│  Static file serving  │   │  - Emotion classifier         │
│                       │   │    (distilRoBERTa)            │
│  Auto-scales to zero  │   │  - TF-IDF + LR reversal      │
│  when idle            │   │    prediction                 │
│                       │   │  - Domain classification      │
│                       │   │                               │
│                       │   │  Scales 0-2 instances         │
│                       │   │  512MB RAM, 1 vCPU            │
└──────────────────────┘   └──────────────┬───────────────┘
                                          │
                                          │ Read/Write
                                          ▼
                            ┌──────────────────────────┐
                            │  Google Cloud Firestore    │
                            │                           │
                            │  regret_posts collection  │
                            │   (3,604 documents)       │
                            │                           │
                            │  query_logs collection    │
                            │   (analyzer usage logs)   │
                            └──────────────────────────┘
```

**Scaling:** App Engine serves static content with automatic scaling (scales to zero when idle, no minimum instances). Cloud Run hosts the inference service as a Docker container with concurrency-based autoscaling (0-2 instances, cold-start ~10s due to model loading). Firestore provides serverless document storage with automatic scaling.

---

## Inference Service

Located in `inference/`, the ML inference service runs as a Docker container on Google Cloud Run.

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `POST` | `/analyze` | Analyze user-provided text for regret patterns |
| `GET` | `/health` | Health check |

**Request (`POST /analyze`):**
```json
{
  "text": "I regret moving to the US for my PhD...",
  "domain_hint": "auto"
}
```

**Response:**
```json
{
  "emotion": "sadness",
  "emotion_confidence": 0.84,
  "domain": "immigration",
  "reversal_probability": 0.41,
  "urgency_score": 0.72,
  "interpretation": "High urgency sadness post in the immigration domain..."
}
```

**Models used:**
- **VADER** — urgency and sentiment scoring (no GPU required)
- **distilRoBERTa** (`j-hartmann/emotion-english-distilroberta-base`) — 7-class emotion classification
- **TF-IDF + Logistic Regression** — domain classification and reversal probability (serialized via joblib from `src/model_reversal_crossdomain.py`)

**Docker build:**
```bash
cd inference
docker build -t regret-inference .
docker run -p 8080:8080 regret-inference
```

---

## Cloud Data Storage

**Google Cloud Firestore** stores the structured dataset and usage logs:

| Collection | Documents | Fields | Purpose |
|---|---|---|---|
| `regret_posts` | 3,604 | domain, subreddit, title, regret_sentence, vader_compound, urgency_score, reversal, emotion, time_to_regret_days, topic, event_type | Full structured dataset consumed by inference service for baseline statistics |
| `query_logs` | variable | text, domain, emotion, reversal_probability, timestamp | Anonymous logs of Regret Analyzer usage |

Data is uploaded via `src/upload_to_firestore.py`, which reads `data_clean/all_domains_final.csv` and batch-writes to Firestore. The inference service reads from `regret_posts` to compute domain-specific baselines for the interpretation text.

---

## Key Results

- **Domain differences are significant** (p < 0.001): relationships reverse at 39.5%, career at 38.5%, immigration at only 27.6%
- **Structural constraints suppress action**: immigration posters have 36% lower hazard of reversal (Cox HR = 0.64) even with strong regret language
- **Emotion predicts reversal**: sadness-labeled posts reverse at 39.9% vs 26.3% for neutral (chi-square p = 0.002)
- **Temporal patterns diverge**: career reversal increases over time while immigration declines (year x domain interaction p = 0.009)
- **Community engagement is a null predictor**: upvotes and comments add negligible signal (+0.005 AUC)
- **Best model**: Random Forest achieves AUC 0.652 [0.610, 0.691] for reversal prediction using TF-IDF + sentiment + domain features
