from flask import Flask, render_template, request, jsonify
import joblib
import re
import html
import numpy as np
from scipy.sparse import hstack, csr_matrix

app = Flask(__name__)

# Load saved model and transformers
model = joblib.load("saved_models/logistic_regression.joblib")
tfidf = joblib.load("saved_models/tfidf_vectorizer.joblib")
scaler = joblib.load("saved_models/scaler.joblib")
label_encoders = joblib.load("saved_models/label_encoders.joblib")

def clean_text(text):
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'#URL_\w+#', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^\w\s.,!?;:\'-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def extract_fraud_signals(text):
    t = text.lower()
    return {
        'has_fee': int(bool(re.search(r'fee|registration cost|processing fee|verification fee', t))),
        'has_pay_to_work': int(bool(re.search(r'pay.*to (start|begin|access|join)|payment.*required', t))),
        'has_gmail_contact': int(bool(re.search(r'@gmail\.com|@yahoo\.com|@hotmail\.com', t))),
        'has_messaging_app': int(bool(re.search(r'whatsapp|telegram|wechat', t))),
        'has_unrealistic_pay': int(bool(re.search(r'earn.*\$\d{4,}.*per\s*(week|day)', t))),
        'has_easy_money': int(bool(re.search(r'be your own boss|financial freedom|unlimited income', t))),
        'has_no_experience': int(bool(re.search(r'no experience|no skills|anyone can', t))),
        'has_urgent': int(bool(re.search(r'start (today|now|immediately)|urgent hiring', t))),
        'has_crypto': int(bool(re.search(r'crypto|bitcoin|blockchain|web3|nft', t))),
    }

def extract_features(text):
    cleaned = clean_text(text)
    words = cleaned.split()

    # TF-IDF features
    tfidf_features = tfidf.transform([cleaned])

    # Fraud signals
    fraud = extract_fraud_signals(text)

    # All 22 numerical features in exact training order:
    # NUM_COLS (3) + ENGINEERED_COLS (10) + FRAUD_SIGNAL_COLS (9)
    num_features = np.array([[
        # NUM_COLS
        0,  # telecommuting
        0,  # has_company_logo
        0,  # has_questions
        # ENGINEERED_COLS
        len(cleaned),                                                      # text_length
        len(words),                                                        # word_count
        np.mean([len(w) for w in words]) if words else 0,                 # avg_word_length
        int(bool(re.search(r'\$|salary|pay|compensation|wage', cleaned, re.IGNORECASE))),  # has_salary_mention
        int(bool(re.search(r'@\w+\.\w+', cleaned))),                     # has_email
        cleaned.count('!'),                                                # exclamation_count
        sum(1 for c in text if c.isupper()) / max(len(text), 1),          # upper_ratio
        int(len(cleaned) < 50),                                            # missing_profile (approximate)
        0,                                                                 # missing_benefits
        0,                                                                 # missing_requirements
        # FRAUD_SIGNAL_COLS
        fraud['has_fee'],
        fraud['has_pay_to_work'],
        fraud['has_gmail_contact'],
        fraud['has_messaging_app'],
        fraud['has_unrealistic_pay'],
        fraud['has_easy_money'],
        fraud['has_no_experience'],
        fraud['has_urgent'],
        fraud['has_crypto'],
    ]])
    num_scaled = scaler.transform(num_features)

    # Categorical features (use "Unknown" for all)
    cat_arrays = []
    for col, le in label_encoders.items():
        if "Unknown" in le.classes_:
            encoded = le.transform(["Unknown"])
        else:
            encoded = np.array([0])
        cat_arrays.append(encoded.reshape(-1, 1))
    cat_matrix = np.hstack(cat_arrays)

    # Combine all
    X = hstack([tfidf_features, csr_matrix(num_scaled), csr_matrix(cat_matrix)])
    return X

def detect_red_flags(text):
    flags = []
    t = text.lower()
    if re.search(r'no experience|no skills required|anyone can apply', t):
        flags.append("No experience or skills required")
    if re.search(r'work from home|work remotely|wfh', t):
        flags.append("Work from home emphasis")
    if re.search(r'fee|payment.*required|pay.*to (start|begin|access)', t):
        flags.append("Upfront fee or payment required")
    if re.search(r'earn.*\$\d{3,}.*per\s*(day|week|hour)|make.*\$\d{4,}', t):
        flags.append("Unrealistic earning claims")
    if re.search(r'be your own boss|unlimited income|financial freedom|passive income', t):
        flags.append("Get-rich-quick language")
    if re.search(r'whatsapp|telegram|signal|wechat', t):
        flags.append("Contact via messaging apps")
    if re.search(r'crypto|bitcoin|blockchain|web3|nft|defi', t):
        flags.append("Cryptocurrency or Web3 buzzwords")
    if re.search(r'!!!|!!\s*!', t):
        flags.append("Excessive exclamation marks")
    if len(text.split()) < 50:
        flags.append("Very short job description")
    if re.search(r'@(gmail|yahoo|hotmail|outlook)\.com', t):
        flags.append("Personal email instead of corporate")
    if re.search(r'refundable|refunded.*paycheck', t):
        flags.append("Refundable fee claim (common scam tactic)")
    if re.search(r'start (today|now|immediately)|urgent|asap|limited spots', t):
        flags.append("Urgency or pressure tactics")
    return flags

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text.strip():
            return jsonify({'error': 'Please enter a job description'})

        X = extract_features(text)
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]

        fraud_prob = round(probability[1] * 100, 2)
        legit_prob = round(probability[0] * 100, 2)

        red_flags = detect_red_flags(text)

        result = {
            'prediction': 'FRAUDULENT' if prediction == 1 else 'LEGITIMATE',
            'fraud_probability': fraud_prob,
            'legit_probability': legit_prob,
            'risk_level': 'HIGH' if fraud_prob > 70 else 'MEDIUM' if fraud_prob > 30 else 'LOW',
            'red_flags': red_flags,
            'total_red_flags': len(red_flags)
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  FRAUD JOB DETECTOR")
    print("  Model: Logistic Regression (EMSCAD trained)")
    print("  Open http://127.0.0.1:5000 in your browser")
    print("=" * 50 + "\n")
    app.run(debug=True, port=5000)