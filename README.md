# 🧠 AI Fake News Detection System

An AI-powered web application that classifies news articles as **Fake or Real** using Natural Language Processing (NLP) and Machine Learning.

---

## 🚀 Features

* 🔍 Real-time fake news detection
* 📊 Confidence score with visual progress bar
* 📈 Probability breakdown (Fake vs Real)
* 🧠 NLP preprocessing (cleaning, stopword removal)
* 🔎 Keyword-based explanation of predictions
* 🎨 Premium Streamlit UI (Dark theme, dashboard layout)

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Machine Learning:** Scikit-learn (Logistic Regression)
* **NLP:** NLTK, TF-IDF Vectorization
* **Frontend/UI:** Streamlit

---

## 📂 Project Structure

```
fake-news-detector/
│
├── app.py                 # Streamlit UI
├── model.py               # Model training script
├── utils.py               # Text preprocessing
├── model.pkl              # Trained ML model
├── vectorizer.pkl         # TF-IDF vectorizer
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Jeevakumar415/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
```

---

## ▶️ Run the Application

### Option 1: Direct Run (Using Pre-trained Model)

```bash
python -m streamlit run app.py
```

---

### Option 2: Train Model First (If needed)

```bash
python model.py
python -m streamlit run app.py
```

---

## 📊 Model Details

* **Algorithm:** Logistic Regression
* **Feature Extraction:** TF-IDF (Unigrams + Bigrams)
* **Dataset:** Combined multiple labeled text datasets (Fake & Legit news)
* **Accuracy:** ~65% – 85% (after optimization)

---

## 🧠 How It Works

1. User inputs news text
2. Text is preprocessed (cleaning, stopword removal)
3. TF-IDF converts text into numerical features
4. Machine Learning model predicts Fake or Real
5. UI displays:

   * Prediction result
   * Confidence score
   * Probability breakdown
   * Key influencing words

---

## ⚠️ Note

* Dataset is not included in this repository to keep it lightweight
* Pre-trained model files (`model.pkl`, `vectorizer.pkl`) are included for direct usage
* If needed, you can retrain the model using your own dataset

---

## 🚀 Future Enhancements

* Upgrade to Transformer models (BERT)
* Real-time news API integration
* Explainable AI (SHAP/LIME)
* Deployment on cloud platforms

---

## 👨‍💻 Author

**Jeevakumar**
MCA Student | AI Enthusiast

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share your feedback!
