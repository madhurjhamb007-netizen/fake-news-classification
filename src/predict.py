import re
import string
import joblib
import numpy as np
import nltk
from nltk.corpus import stopwords

# Load stopwords
try:
    stop_words = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")
    stop_words = stopwords.words("english")

# Preserve negation words
for word in ["no", "not", "nor"]:
    if word in stop_words:
        stop_words.remove(word)

# Terms removed during shortcut-reduced training
words_to_remove = ["reuters", "nytimes", "via", "getty"]

bias_pattern = re.compile(
    r"\b(" + "|".join(words_to_remove) + r")\b",
    re.IGNORECASE
)

punctuation_pattern = re.compile(
    f"[{re.escape(string.punctuation)}]"
)

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = punctuation_pattern.sub(" ", text)

    words = text.split()

    filtered_words = [
        word for word in words
        if word not in stop_words
        and not bias_pattern.fullmatch(word)
    ]

    return " ".join(filtered_words)


# Load trained model and vectorizer
MODEL_PATH = "xgb_unbiased_model.pkl"
VECTORIZER_PATH = "cv_unbiased_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def predict_news(raw_text):
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("Please provide a valid news article.")

    cleaned_text = clean_text(raw_text)

    if not cleaned_text.strip():
        raise ValueError("No usable text remains after preprocessing.")

    features = vectorizer.transform([cleaned_text])
    probabilities = model.predict_proba(features)[0]

    predicted_class = model.classes_[np.argmax(probabilities)]

    label_map = {
        0: "Real",
        1: "Fake"
    }

    return {
        "prediction": label_map[predicted_class],
        "confidence_real": float(probabilities[0]),
        "confidence_fake": float(probabilities[1])
    }
