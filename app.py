from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from textblob import TextBlob

# ======================
# Load model and vectorizer
# ======================
model = joblib.load("best_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

app = FastAPI(title="Fake News Detection API")

# ======================
# Request / Response models
# ======================
class NewsRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    sentiment: float

# ======================
# Prediction endpoint
# ======================

import re
import numpy as np

CLICKBAIT_PHRASES = [
    "breaking",
    "shocking",
    "you won't believe",
    "secret",
    "unbelievable",
    "revealed",
    "this will change",
    "what happens next"
]

def detect_clickbait(text: str):
    text_lower = text.lower()
    return [phrase for phrase in CLICKBAIT_PHRASES if phrase in text_lower]

def get_top_tfidf_words(text_vectorized, vectorizer, top_n=5):
    feature_names = vectorizer.get_feature_names_out()
    scores = text_vectorized.toarray()[0]
    top_indices = np.argsort(scores)[-top_n:][::-1]
    return [feature_names[i] for i in top_indices if scores[i] > 0]


@app.post("/predict")
def predict_news(data: NewsRequest):
    text = data.text.strip()

    # ===== Vectorize =====
    text_vectorized = vectorizer.transform([text])
    proba = model.predict_proba(text_vectorized)[0]

    true_prob = float(proba[0])
    fake_prob = float(proba[1])

    raw_confidence = max(true_prob, fake_prob)
    confidence = min(raw_confidence, 0.85)

    # ===== Label logic =====
    if raw_confidence < 0.6:
        label = "UNCERTAIN"
    elif fake_prob > true_prob:
        label = "LIKELY FAKE"
    else:
        label = "LIKELY TRUE"

    # ===== Extra analysis =====
    sentiment = float(TextBlob(text).sentiment.polarity)
    word_count = len(text.split())
    text_length = len(text)

    clickbait = detect_clickbait(text)
    top_words = get_top_tfidf_words(text_vectorized, vectorizer)

    # ===== Explanation =====
    reasons = []
    if clickbait:
        reasons.append("clickbait phrases detected")
    if abs(sentiment) > 0.3:
        reasons.append("emotionally charged language")
    if word_count < 50:
        reasons.append("very short news-style text")

    explanation = (
        "The model classified this text based on writing style patterns. "
        + ("Key reasons: " + ", ".join(reasons) if reasons else "No strong stylistic signals detected.")
    )

    return {
        "label": label,
        "true_probability": round(true_prob, 3),
        "fake_probability": round(fake_prob, 3),
        "confidence": round(confidence, 2),
        "sentiment": round(sentiment, 3),
        "text_length": text_length,
        "word_count": word_count,
        "clickbait_phrases": clickbait,
        "top_tfidf_words": top_words,
        "explanation": explanation
    }
