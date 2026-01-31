# test_model.py

import joblib


# Load model & vectorizer
model = joblib.load("fake_job_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


def predict_job(text):

    text = text.lower()

    text_vec = vectorizer.transform([text])

    prediction = model.predict(text_vec)[0]
    prob = model.predict_proba(text_vec).max()

    return prediction, prob


# =============================
# Manual Testing
# =============================
print("Fake Job Detection Test Mode ❤️\n")

while True:

    text = input("Enter Job Description (or type exit):\n> ")

    if text.lower() == "exit":
        break

    pred, confidence = predict_job(text)

    if pred == 1:
        print("\n⚠️ FAKE JOB DETECTED!")
    else:
        print("\n✅ REAL JOB")

    print("Confidence:", round(confidence * 100, 2), "%\n")

print("Testing Finished 💙")