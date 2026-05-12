import os
import re
import json
import joblib
import numpy as np
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline as hf_pipeline

app = FastAPI(title="Regret Analysis Inference")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = Path(__file__).parent / "models"
FIRESTORE_DB = os.getenv("FIRESTORE_DB", "regret-data")

vader = SentimentIntensityAnalyzer()
emotion_clf = None
reversal_model = None
tfidf_vectorizer = None
domain_model = None
domain_tfidf = None

URGENCY_WORDS = [
    "rushed", "pressure", "deadline", "last minute", "forced",
    "urgent", "quick decision", "panicked", "impulsive",
]

DOMAIN_BASELINES = {
    "career": {"reversal_rate": 0.385, "median_time_days": 210},
    "immigration": {"reversal_rate": 0.276, "median_time_days": 365},
    "relationships": {"reversal_rate": 0.395, "median_time_days": 60},
}


class AnalyzeRequest(BaseModel):
    text: str
    domain_hint: str = "auto"


class AnalyzeResponse(BaseModel):
    emotion: str
    emotion_confidence: float
    domain: str
    reversal_probability: float
    urgency_score: float
    interpretation: str


@app.on_event("startup")
def load_models():
    global emotion_clf, reversal_model, tfidf_vectorizer, domain_model, domain_tfidf

    print("Loading emotion classifier...")
    emotion_clf = hf_pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1,
        truncation=True,
        max_length=512,
        device=-1,
    )

    reversal_path = MODEL_DIR / "reversal_model.joblib"
    tfidf_path = MODEL_DIR / "tfidf_vectorizer.joblib"
    domain_model_path = MODEL_DIR / "domain_model.joblib"
    domain_tfidf_path = MODEL_DIR / "domain_tfidf.joblib"

    if reversal_path.exists():
        print("Loading reversal model...")
        reversal_model = joblib.load(reversal_path)
    if tfidf_path.exists():
        print("Loading TF-IDF vectorizer...")
        tfidf_vectorizer = joblib.load(tfidf_path)
    if domain_model_path.exists():
        print("Loading domain classifier...")
        domain_model = joblib.load(domain_model_path)
    if domain_tfidf_path.exists():
        domain_tfidf = joblib.load(domain_tfidf_path)

    print("All models loaded.")


def compute_urgency(text: str) -> float:
    text_lower = text.lower()
    count = sum(1 for w in URGENCY_WORDS if w in text_lower)
    return min(count / 3.0, 1.0)


def classify_domain(text: str, hint: str) -> str:
    if hint != "auto" and hint in DOMAIN_BASELINES:
        return hint
    if domain_model is not None and domain_tfidf is not None:
        X = domain_tfidf.transform([text])
        return domain_model.predict(X)[0]
    text_lower = text.lower()
    career_kw = ["job", "career", "salary", "company", "quit", "fired", "hired",
                 "promotion", "boss", "coworker", "work", "resign"]
    imm_kw = ["visa", "h1b", "immigration", "green card", "uscis", "move to",
              "moved to", "country", "citizen", "immigrant", "asylum"]
    rel_kw = ["boyfriend", "girlfriend", "partner", "marriage", "divorced",
              "breakup", "broke up", "relationship", "wife", "husband", "dating"]
    scores = {
        "career": sum(1 for k in career_kw if k in text_lower),
        "immigration": sum(1 for k in imm_kw if k in text_lower),
        "relationships": sum(1 for k in rel_kw if k in text_lower),
    }
    return max(scores, key=scores.get) if max(scores.values()) > 0 else "career"


def predict_reversal(text: str, vader_scores: dict, urgency: float, has_time: int) -> float:
    if reversal_model is not None and tfidf_vectorizer is not None:
        from scipy.sparse import hstack, csr_matrix
        X_text = tfidf_vectorizer.transform([text])
        X_num = csr_matrix([[vader_scores["compound"], vader_scores["neg"], urgency, has_time]])
        X = hstack([X_text, X_num])
        proba = reversal_model.predict_proba(X)[0][1]
        return round(float(proba), 3)
    reversal_words = ["quit", "left", "resigned", "divorced", "moved back",
                      "withdrew", "dropped out", "changed careers", "switched"]
    text_lower = text.lower()
    hits = sum(1 for w in reversal_words if w in text_lower)
    base = 0.37
    signal = min(hits * 0.08, 0.25)
    sentiment_adj = -vader_scores["compound"] * 0.05
    return round(min(max(base + signal + sentiment_adj, 0.05), 0.95), 3)


def has_temporal_marker(text: str) -> int:
    patterns = [
        r"\d+\s+(day|week|month|year)s?\s+(ago|later)",
        r"after\s+\d+\s+(day|week|month|year)s?",
        r"(in|back in|since)\s+20\d{2}",
        r"(last|past)\s+(week|month|year|few\s+(months|years))",
    ]
    text_lower = text.lower()
    return 1 if any(re.search(p, text_lower) for p in patterns) else 0


def build_interpretation(emotion: str, domain: str, reversal_prob: float, urgency: float) -> str:
    baseline = DOMAIN_BASELINES.get(domain, {})
    baseline_rate = baseline.get("reversal_rate", 0.37)
    median_days = baseline.get("median_time_days", 180)

    urg_label = "Low" if urgency < 0.3 else ("Medium" if urgency < 0.6 else "High")

    comp = "above" if reversal_prob > baseline_rate else "below"
    parts = [
        f"{urg_label}-urgency {emotion} post classified in the {domain} domain.",
        f"The estimated reversal probability ({reversal_prob:.0%}) is {comp} the {domain} baseline ({baseline_rate:.0%}).",
        f"In our dataset, {domain} regret typically surfaces around {median_days} days after the decision.",
    ]

    if emotion == "sadness":
        parts.append("Sadness is the most common emotion in regret posts and is associated with higher reversal rates (39.9%).")
    elif emotion == "anger":
        parts.append("Anger-driven regret often reflects interpersonal conflict and tends toward action.")
    elif emotion == "neutral":
        parts.append("Neutral-toned posts tend to have lower reversal rates (26.3%), suggesting ongoing deliberation.")

    return " ".join(parts)


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    text = req.text.strip()

    vader_scores = vader.polarity_scores(text)

    emo_result = emotion_clf(text[:512])[0]
    emotion = emo_result["label"]
    emotion_conf = round(emo_result["score"], 3)

    domain = classify_domain(text, req.domain_hint)
    urgency = compute_urgency(text)
    has_time = has_temporal_marker(text)
    reversal_prob = predict_reversal(text, vader_scores, urgency, has_time)
    interpretation = build_interpretation(emotion, domain, reversal_prob, urgency)

    return AnalyzeResponse(
        emotion=emotion,
        emotion_confidence=emotion_conf,
        domain=domain,
        reversal_probability=reversal_prob,
        urgency_score=round(urgency, 3),
        interpretation=interpretation,
    )


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": reversal_model is not None}
