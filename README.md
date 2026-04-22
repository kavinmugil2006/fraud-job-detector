# 🔍 Fraudulent Job Posting Detection

> **A Comparative Study of Machine Learning and Deep Learning Models for Detecting Online Recruitment Fraud**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-red.svg)](https://pytorch.org)
[![BERT](https://img.shields.io/badge/BERT-base--uncased-yellow.svg)](https://huggingface.co/bert-base-uncased)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Overview

This project builds and compares **7 machine learning and deep learning models** for detecting fraudulent job postings on the [EMSCAD benchmark dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction). Our key finding: **TF-IDF + SVM outperforms BERT** on this imbalanced dataset (F1=0.850 vs 0.808), challenging the assumption that transformers always win.

A **Flask web application** is included for real-time fraud detection with red flag analysis.

## 🏆 Key Results

| Model | F1 Score | ROC-AUC | Framework |
|-------|----------|---------|-----------|
| **SVM** | **0.850 ± 0.034** | **0.991** | sklearn |
| XGBoost | 0.847 ± 0.032 | 0.990 | sklearn |
| BERT | 0.808 ± 0.035 | 0.980 | PyTorch + HuggingFace |
| Random Forest | 0.736 ± 0.040 | 0.981 | sklearn |
| Logistic Regression | 0.709 ± 0.024 | 0.991 | sklearn |
| BiLSTM | 0.552 ± 0.095 | 0.976 | PyTorch (GPU) |
| Transformer | 0.536 ± 0.098 | 0.970 | PyTorch (GPU) |

After hyperparameter tuning: **SVM F1 = 0.913** | **AUC = 0.995**

## 🔬 Research Highlights

- **Stratified 5-fold cross-validation** with all feature engineering inside each fold (no data leakage)
- **McNemar's statistical significance test** — SVM significantly outperforms BERT (p < 0.001)
- **Hyperparameter optimization** using Optuna Bayesian search (50 trials)
- **SMOTE analysis** — class weighting outperforms synthetic oversampling
- **Threshold optimization** — F1=0.912 at optimal threshold of 0.10
- **Feature explainability** — coefficient analysis identifying top fraud/legitimacy indicators
- **Ablation study** — measuring contribution of TF-IDF, engineered features, and fraud signals

## 📊 Feature Engineering

**50,027 total features** across three categories:

- **TF-IDF (50,000)**: Unigram to trigram features with sublinear TF scaling
- **Engineered (13)**: Text length, word count, missing profile/benefits/requirements, salary mentions, email detection, exclamation count, uppercase ratio
- **Fraud Signals (9)**: Fee detection, messaging app references, unrealistic pay, no-experience claims, urgency language, crypto mentions

### Top Fraud Indicators
| Feature | Coefficient |
|---------|------------|
| "data entry" | +2.37 |
| "earn" | +2.33 |
| "money" | +1.76 |
| missing_profile | +1.13 |
| "administrative assistant" | +1.70 |

### Top Legitimacy Indicators
| Feature | Coefficient |
|---------|------------|
| word_count | -2.35 |
| "software" | -1.52 |
| "experience" | -1.26 |
| "marketing" | -1.26 |

## 🌐 Web Application

A Flask-based web app for real-time fraud detection:

- Paste any job description for instant analysis
- Three-tier risk classification: LOW / MEDIUM / HIGH
- Automated red flag detection (upfront fees, messaging apps, unrealistic earnings, etc.)
- Sample fraud and legitimate job descriptions included

### Live Testing Results

| Test Case | Prediction | Confidence |
|-----------|-----------|------------|
| Obvious fraud (WFH scam) | FRAUDULENT | 100% |
| Sneaky fraud (professional-looking) | FRAUDULENT | 87.8% |
| Subtle fraud (borderline) | SUSPICIOUS | 33.0% |
| Legitimate job (nurse posting) | LEGITIMATE | 81.4% |

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/fraud-job-detector.git
cd fraud-job-detector
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download [EMSCAD dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) and place `fake_job_postings.csv` in the project root.

### 5. Train the models
Open `ieee_fraud_detector.ipynb` in Jupyter/VS Code and run all cells.

### 6. Run the web app
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser.

## 📁 Project Structure

```
fraud-job-detector/
├── ieee_fraud_detector.ipynb  # Complete research notebook (5-fold CV, 7 models)
├── app.py                     # Flask web application
├── templates/
│   └── index.html             # Web UI
├── saved_models/              # Trained models (generated after training)
│   ├── logistic_regression.joblib
│   ├── tfidf_vectorizer.joblib
│   ├── scaler.joblib
│   └── label_encoders.joblib
├── requirements.txt
├── ieee_results.png           # Model comparison charts
├── feature_importance.png     # SHAP/coefficient analysis
├── roc_pr_curves.png          # ROC and PR curves
├── eda_analysis.png           # Exploratory data analysis
└── README.md
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| ML Frameworks | Scikit-learn 1.8, XGBoost 3.2 |
| Deep Learning | PyTorch 2.11 (CUDA 12.8) |
| NLP | TF-IDF, BERT (HuggingFace Transformers) |
| Web Framework | Flask |
| Tuning | Optuna (Bayesian optimization) |
| GPU | NVIDIA RTX 4050 (6 GB VRAM) |

## 📄 Citation

If you use this work, please cite:

```bibtex
@inproceedings{fraud_job_detector_2026,
  title={Beyond BERT: Why TF-IDF with SVM Outperforms Transformer Models for Fraudulent Job Posting Detection on Imbalanced Datasets},
  author={Kavinmugil A and Prasanna K},
  year={2026}
}
```

## 📧 Contact

- **Kavinmugil A** — kavinmugilarulmaran@gmail.com
- **Prasanna K** — prasannakp2005@gmail.com

Department of Artificial Intelligence and Data Science, St. Joseph's College of Engineering, Chennai, India

## 📝 License

This project is licensed under the MIT License.
