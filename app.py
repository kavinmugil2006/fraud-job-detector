import streamlit as st
import re
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download("stopwords")
nltk.download("punkt")


# =============================
# LOAD MODEL
# =============================

@st.cache_resource
def load_model():

    model = joblib.load("model.pkl")
    tfidf = joblib.load("vectorizer.pkl")

    return model, tfidf


model, tfidf = load_model()

stop_words = set(stopwords.words("english"))


# =============================
# CLEAN TEXT
# =============================

def clean_text(text):

    text = text.lower()
    text = re.sub(r"[^a-z ]", " ", text)

    tokens = word_tokenize(text)

    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]

    return " ".join(tokens)


# =============================
# UI
# =============================

st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🕵️"
)

st.title("🕵️ Fake Job Scam Detector")

st.write("Paste job description to analyze.")


sample = """
Work from home.
Earn 50000 per month.
No interview.
Pay registration fee.
WhatsApp now.
"""

if st.button("Use Sample"):
    st.session_state.text = sample


text = st.text_area(
    "Job Description",
    height=200,
    key="text"
)


# =============================
# ANALYZE
# =============================

if st.button("Analyze"):

    if text.strip() == "":
        st.warning("Enter job description first")

    else:

        clean = clean_text(text)

        vec = tfidf.transform([clean])

        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]


        st.subheader("Result")

        st.write("Genuine:", round(proba[0]*100,2), "%")
        st.write("Fraud  :", round(proba[1]*100,2), "%")


        if pred == 1:
            st.error("❌ Likely Scam")
        else:
            st.success("✅ Likely Genuine")


st.caption("Developed by Bumblebee ❤️")
