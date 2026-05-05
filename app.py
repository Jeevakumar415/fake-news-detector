import streamlit as st
import pickle
import numpy as np
from utils import clean_text

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page config
st.set_page_config(page_title="Fake News AI", layout="wide")

# 🔥 CUSTOM CSS (DARK + GLASS UI)
st.markdown("""
<style>
body {
    background-color: #0e1117;
}

.stApp {
    background: linear-gradient(135deg, #0e1117, #111827);
    color: white;
}

.card {
    background: rgba(255, 255, 255, 0.05);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
}

.title {
    font-size: 32px;
    font-weight: bold;
}

.subtitle {
    color: #9ca3af;
}
</style>
""", unsafe_allow_html=True)

# 🔥 HEADER
st.markdown("<div class='title'>🧠 Fake News Detection AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Real-time NLP-based News Verification System</div>", unsafe_allow_html=True)

st.write("")

# 🔥 LAYOUT
col1, col2 = st.columns([1, 1])

# ---------------- LEFT ----------------
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("✍️ Enter News Content")

    input_text = st.text_area("", height=250)

    analyze_btn = st.button("🚀 Analyze")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- RIGHT ----------------
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 AI Analysis")

    if analyze_btn:
        if input_text.strip() == "":
            st.warning("Please enter text.")
        else:
            cleaned = clean_text(input_text)
            vectorized = vectorizer.transform([cleaned])

            prediction = model.predict(vectorized)[0]
            probability = model.predict_proba(vectorized)[0]

            confidence = max(probability) * 100

            # 🔥 RESULT BADGE
            if prediction == 0:
                st.markdown(f"### ❌ Fake News")
            else:
                st.markdown(f"### ✅ Real News")

            # 🔥 CONFIDENCE
            st.write(f"Confidence Score: {confidence:.2f}%")
            st.progress(int(confidence))

            # 🔥 PROBABILITY CHART
            fake_prob = probability[0] * 100
            real_prob = probability[1] * 100

            st.write("### 📊 Prediction Breakdown")
            st.bar_chart({
                "Fake": [fake_prob],
                "Real": [real_prob]
            })

            # 🔥 KEYWORDS
            st.write("### 🔍 Key Influencing Words")

            feature_names = vectorizer.get_feature_names_out()
            vector_data = vectorized.toarray()[0]

            top_indices = np.argsort(vector_data)[-8:]
            words = [feature_names[i] for i in top_indices if vector_data[i] > 0]

            if words:
                st.write(" • " + " • ".join(reversed(words)))
            else:
                st.write("No strong keywords found.")

    else:
        st.info("Enter text and click Analyze")

    st.markdown("</div>", unsafe_allow_html=True)