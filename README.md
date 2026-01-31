🏆 Fake Job Scam Detection System
📌 Overview

This project is an AI-based Fake Job and Internship Scam Detection System developed using Natural Language Processing (NLP) and Machine Learning techniques. It helps users identify fraudulent job postings and avoid online scams by analyzing job descriptions.

The system uses both traditional machine learning and deep learning models to provide accurate predictions and is deployed as a web application using Streamlit.

🚀 Features

Detects fake and genuine job postings

Uses NLP for text preprocessing

Supports TF-IDF + Logistic Regression

Supports DistilBERT for deep learning

Provides fraud probability

User-friendly web interface

Real-time prediction system

🛠️ Technologies Used

Python

Pandas, NumPy

Scikit-learn

NLTK

Transformers (DistilBERT)

Streamlit

Joblib

📂 Project Structure
Fake_job_detector/
│
├── training.py
├── bert_train.py / distilbert_train.py
├── test_model.py
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── fake_job_postings.csv
└── README.md

📊 Dataset

Large-scale job posting dataset

Contains genuine and fraudulent samples

Labeled with binary classes

0 → Genuine

1 → Fraud

⚙️ Installation

Clone the repository

git clone https://github.com/your-username/Fake-Job-Detector.git
cd Fake-Job-Detector


Create virtual environment

python -m venv venv


Activate virtual environment

venv\Scripts\activate


Install dependencies

pip install -r requirements.txt

🧠 Model Training
Train ML Model
python training.py

Train DistilBERT Model
python distilbert_train.py

🧪 Test the Model
python test_model.py

🌐 Run Web Application
streamlit run app.py


Open browser and access:

http://localhost:8501

📈 Results

High accuracy on large dataset

Effective scam detection

Reliable performance on real-world examples

🔮 Future Enhancements

Use advanced transformer models (BERT, RoBERTa)

Live job scraping

Company verification system

Mobile application

Cloud deployment

👨‍💻 Author

Kavinmugil A
B.Tech AI & Data Science Student

📜 License

This project is for educational purposes only.

🙏 Acknowledgements

Kaggle for dataset

Hugging Face for transformer models

Open-source community
