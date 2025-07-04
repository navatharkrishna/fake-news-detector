import streamlit as st
import joblib
import pandas as pd
from deep_translator import GoogleTranslator
import os
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from pathlib import Path

# -----------------------------
# ✅ Page Configuration
# -----------------------------
st.set_page_config(page_title="📰 Fake News Detector", layout="wide", page_icon="🕵️")

# ✅ OpenAI API Key from secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# -----------------------------
# ✅ Paths & Data Setup
# -----------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = DATA_DIR / "feedback_dataset.csv"
MODEL_PATH = DATA_DIR / "fake_news_model.pkl"
VEC_PATH = DATA_DIR / "tfidf_vectorizer.pkl"

# Load or create dataset
if DATA_PATH.exists():
    feedback_df = pd.read_csv(DATA_PATH)
else:
    feedback_df = pd.DataFrame(columns=["text", "label"])

# -----------------------------
# ✅ Function: Train Model
# -----------------------------
def train_model_if_needed():
    global vec, clf
    if len(feedback_df) >= 500:
        st.toast("⚙️ Training ML model...", icon="🧠")
        vec = TfidfVectorizer(stop_words='english', max_df=0.7)
        X_train = vec.fit_transform(feedback_df['text'])
        y_train = feedback_df['label']
        clf = PassiveAggressiveClassifier(max_iter=100)
        clf.fit(X_train, y_train)
        joblib.dump(clf, MODEL_PATH)
        joblib.dump(vec, VEC_PATH)
        st.success("✅ Model trained and saved!")

vec, clf = None, None
if MODEL_PATH.exists() and VEC_PATH.exists():
    vec = joblib.load(VEC_PATH)
    clf = joblib.load(MODEL_PATH)
else:
    train_model_if_needed()

# -----------------------------
# ✅ CSS Styling
# -----------------------------
st.markdown("""
    <style>
    .highlight-fake {
        background-color: #ffe6e6;
        padding: 1.8rem;
        border-left: 8px solid #ff1a1a;
        font-size: 1.4rem;
        color: #990000;
        border-radius: 10px;
        font-weight: bold;
    }
    .highlight-real {
        background-color: #e6ffe6;
        padding: 1.8rem;
        border-left: 8px solid #33cc33;
        font-size: 1.4rem;
        color: #006600;
        border-radius: 10px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# ✅ App Layout
# -----------------------------
st.title("📰 Fake News Detector")
st.caption("🔍 Powered by Machine Learning & GPT Intelligence")

tab1, tab2 = st.tabs(["🔎 Analyze News", "🌐 Trusted Sources"])

with tab1:
    st.subheader("📜 Paste News Article Below")
    news_input = st.text_area("News Content", height=180, placeholder="Paste your news text here...")

    if st.button("🚀 Analyze News"):
        if not news_input.strip():
            st.warning("⚠️ Please enter some news text.")
        else:
            translated = GoogleTranslator(source='auto', target='en').translate(news_input)
            result = None
            confidence = ""

            if vec and clf:
                try:
                    X_input = vec.transform([translated])
                    pred = clf.predict(X_input)[0]
                    prob = clf.decision_function(X_input)[0]
                    confidence = f"Confidence Score: {abs(prob):.2f}"
                    result = pred.upper()
                    st.toast("✅ Checked with local ML model", icon="📊")
                except Exception:
                    result = None
                    st.warning("⚠️ Local model failed. Switching to GPT...")

            if result not in ["REAL", "FAKE"]:
                st.info("🤖 GPT is analyzing the content...")
                try:
                    prompt = f"""
                    You are a fact-checking assistant.
                    Classify the following news article as either REAL or FAKE.
                    Only respond with one word: REAL or FAKE.

                    News: {translated}
                    """
                    response = openai.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    result = response.choices[0].message.content.strip().upper()

                    if result in ["REAL", "FAKE"]:
                        feedback_df = pd.concat([
                            feedback_df,
                            pd.DataFrame({"text": [translated], "label": [result]})
                        ], ignore_index=True)
                        feedback_df.to_csv(DATA_PATH, index=False)
                        st.toast("✅ Saved to feedback dataset", icon="💾")
                        train_model_if_needed()
                except Exception as e:
                    st.error(f"OpenAI GPT Error: {e}")

            if result == "FAKE":
                st.markdown("<div class='highlight-fake'>🚨 FAKE NEWS DETECTED</div>", unsafe_allow_html=True)
            elif result == "REAL":
                st.markdown("<div class='highlight-real'>✅ THIS NEWS APPEARS TO BE REAL</div>", unsafe_allow_html=True)
            else:
                st.error("❓ Could not determine if the news is real or fake.")

            if confidence:
                st.markdown(f"**{confidence}**")

            st.markdown("#### 🔎 Translated Input:")
            st.code(translated, language="text")

with tab2:
    st.subheader("🗞️ Trusted News Sources")
    st.markdown("#### 🔸 English News")
    st.markdown("- [NDTV](https://www.ndtv.com)\n- [India Today](https://www.indiatoday.in)\n- [The Hindu](https://www.thehindu.com)")
    st.markdown("#### 🔸 Hindi News")
    st.markdown("- [Aaj Tak](https://www.aajtak.in)\n- [Dainik Bhaskar](https://www.bhaskar.com)")
    st.markdown("#### 🔸 Marathi News")
    st.markdown("- [Lokmat](https://www.lokmat.com)\n- [Sakal](https://www.esakal.com)")
