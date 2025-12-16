from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from textblob import TextBlob

# ======================
# Load model and vectorizer
# ======================
model = joblib.load("best_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

app = FastAPI(
    title="Fake News Detection API",
    description="Style-based political fake news detection",
    version="1.0.1",
)

# ======================
# Request / Response Models
# ======================
class NewsRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    true_probability: float
    fake_probability: float
    confidence: float
    sentiment: float
    text_length: int
    word_count: int
    clickbait_phrases: list
    top_tfidf_words: list
    explanation: str

# ======================
# Helper Functions
# ======================
CLICKBAIT_PHRASES = [
    "breaking",
    "shocking",
    "you won't believe",
    "secret",
    "unbelievable",
    "revealed",
    "this will change",
    "what happens next",
]

def detect_clickbait(text: str):
    text_lower = text.lower()
    return [p for p in CLICKBAIT_PHRASES if p in text_lower]

def get_top_tfidf_words(text_vectorized, vectorizer, top_n=5):
    feature_names = vectorizer.get_feature_names_out()
    scores = text_vectorized.toarray()[0]
    top_indices = np.argsort(scores)[-top_n:][::-1]
    return [feature_names[i] for i in top_indices if scores[i] > 0]

# ======================
# Prediction Endpoint
# ======================
@app.post("/predict", response_model=PredictionResponse)
def predict_news(data: NewsRequest):
    text = data.text.strip()

    # ===== Empty input =====
    if not text:
        return {
            "label": "LIKELY TRUE",
            "true_probability": 0.0,
            "fake_probability": 0.0,
            "confidence": 0.0,
            "sentiment": 0.0,
            "text_length": 0,
            "word_count": 0,
            "clickbait_phrases": [],
            "top_tfidf_words": [],
            "explanation": (
                "The input text is empty. "
                "No indicators of misinformation were detected."
            ),
        }

    # ===== Vectorization & prediction =====
    text_vectorized = vectorizer.transform([text])
    proba = model.predict_proba(text_vectorized)[0]

    # ===== Safe class mapping =====
    class_to_index = {c: i for i, c in enumerate(model.classes_)}

    true_prob = float(proba[class_to_index.get("true", 0)])
    fake_prob = float(proba[class_to_index.get("fake", 1)])

    # ===== Majority decision (NO UNCERTAIN) =====
# ======================
# LABEL DECISION LOGIC
# true >= 40% is GOOD
# ======================
    if true_prob >= 0.40:
        label = "LIKELY TRUE"
    else:
        label = "LIKELY FAKE"


    # ===== Confidence (UI-compatible) =====
    raw_confidence = max(true_prob, fake_prob)
    confidence = round(min(raw_confidence, 0.85), 2)

    # ===== Article stats =====
    sentiment = float(TextBlob(text).sentiment.polarity)
    word_count = len(text.split())
    text_length = len(text)

    clickbait = detect_clickbait(text)
    top_words = get_top_tfidf_words(text_vectorized, vectorizer)

    # ======================
    # Explanation logic
    # ======================
    if label == "LIKELY FAKE":
        explanation = (
            "The model detected more linguistic patterns associated with "
            "misleading or manipulative news content than with neutral reporting."
        )

        if clickbait:
            explanation += " Clickbait-style language was detected."

        if abs(sentiment) > 0.3:
            explanation += " The article uses emotionally charged language."

    else:
        explanation = (
            "The article shows more characteristics of neutral, factual reporting "
            "than of misleading or manipulative content."
        )

    # ======================
    # Response (unchanged schema)
    # ======================
    return {
        "label": label,
        "true_probability": round(true_prob, 3),
        "fake_probability": round(fake_prob, 3),
        "confidence": confidence,
        "sentiment": round(sentiment, 3),
        "text_length": text_length,
        "word_count": word_count,
        "clickbait_phrases": clickbait,
        "top_tfidf_words": top_words,
        "explanation": explanation,
    }
