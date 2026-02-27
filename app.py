import streamlit as st
import pandas as pd
import string
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detection App")
st.write("Enter a news headline below to check if it is Fake or Real.")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("fake_news.csv")
    df = df.dropna()
    return df

df = load_data()

# -----------------------------
# TEXT CLEANING
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df["text"] = df["text"].apply(clean_text)

# -----------------------------
# TRAIN MODEL
# -----------------------------
@st.cache_resource
def train_model():
    X = df["text"]
    y = df["label"]

    vectorizer = TfidfVectorizer(stop_words="english")
    X_vectorized = vectorizer.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_vectorized, y)

    return model, vectorizer

model, vectorizer = train_model()

# -----------------------------
# USER INPUT
# -----------------------------
user_input = st.text_area("Enter News Text", height=150)

if st.button("🔍 Predict", use_container_width=True):

    if user_input.strip() == "":
        st.warning("⚠ Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vectorized_input = vectorizer.transform([cleaned])

        prediction = model.predict(vectorized_input)[0]
        probabilities = model.predict_proba(vectorized_input)[0]

        # Take only confidence of predicted class
        confidence = max(probabilities) * 100

        st.markdown("---")

        if prediction.lower() == "fake":
            st.error("🚨 Prediction: FAKE NEWS")
        else:
            st.success("✅ Prediction: REAL NEWS")

        st.write(f"### Confidence Score: {confidence:.2f}%")
        st.progress(int(confidence))
