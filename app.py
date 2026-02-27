import streamlit as st
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
data = pd.read_csv("fake_news.csv")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

data["clean_text"] = data["text"].apply(preprocess)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["clean_text"])
y = data["label"]

model = LogisticRegression()
model.fit(X, y)

st.title("Fake News Detection App")

user_input = st.text_area("Enter News Text")

if st.button("Predict"):
    clean_input = preprocess(user_input)
    vector_input = vectorizer.transform([clean_input])
    prediction = model.predict(vector_input)
    st.success(f"Prediction: {prediction[0]}")