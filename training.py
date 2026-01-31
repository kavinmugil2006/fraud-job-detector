# training.py

import pandas as pd
import re
import nltk
import joblib

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# =============================
# Download NLTK
# =============================
nltk.download("stopwords")


# =============================
# Load Dataset
# =============================
DATA_PATH = r"C:\Users\namik\OneDrive\Desktop\fakejobdetector\fake_job_postings.csv"

print("Loading dataset...")

data = pd.read_csv(DATA_PATH)

print("Dataset Loaded!")
print("Rows:", len(data))
print("\nColumns Found:")
print(data.columns)


# =============================
# Detect Text & Label Columns
# =============================

TEXT_COL = None
LABEL_COL = None

for col in data.columns:
    name = col.lower()

    if "text" in name or "desc" in name or "job" in name:
        TEXT_COL = col

    if "label" in name or "fraud" in name or "fake" in name:
        LABEL_COL = col


if TEXT_COL is None or LABEL_COL is None:
    print("\n❌ Could not auto-detect columns!")
    print("Please rename your columns manually.")
    exit()


print("\nUsing Columns:")
print("Text  :", TEXT_COL)
print("Label :", LABEL_COL)


# =============================
# Keep Required Data
# =============================

data = data[[TEXT_COL, LABEL_COL]]

data.dropna(inplace=True)

data.columns = ["text", "label"]   # Rename safely


# =============================
# Stopwords
# =============================
stop_words = set(stopwords.words("english"))


# =============================
# Clean Text
# =============================
def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"[^a-z\s]", "", text)

    words = text.split()

    words = [w for w in words if w not in stop_words]

    return " ".join(words)


# =============================
# Preprocess
# =============================
print("\nCleaning text...")

data["clean_text"] = data["text"].apply(clean_text)

print("Cleaning Done!")


# =============================
# Split
# =============================
X = data["clean_text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =============================
# Vectorize
# =============================
print("Vectorizing...")

vectorizer = TfidfVectorizer(
    max_features=6000,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# =============================
# Train
# =============================
print("Training...")

model = MultinomialNB()

model.fit(X_train_vec, y_train)

print("Training Done!")


# =============================
# Evaluate
# =============================
y_pred = model.predict(X_test_vec)

acc = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(acc * 100, 2), "%\n")

print(classification_report(y_test, y_pred))


# =============================
# Save
# =============================
joblib.dump(model, "fake_job_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel Saved ❤️🔥")
print("Pipeline Completed Successfully 😘💙")
