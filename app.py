import streamlit as st
import pandas as pd
import joblib
import nltk
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from summarization_utils import clean_text, extractive_reduce, extract_topics, plot_wordcloud

# --- Page Config ---
st.set_page_config(page_title="Dynamic Text Analysis Platform", layout="wide")

st.title("🧠 Dynamic Text Analysis & Insights Platform")
st.write("Analyze, summarize, and extract actionable insights from your Kaggle dataset.")

# --- Load Dataset ---
@st.cache_data(show_spinner=True)
def load_kaggle_data(csv_path: str):
    df = pd.read_csv(csv_path)
    return df

uploaded_file = st.file_uploader("Upload your Kaggle sentiment dataset (CSV)", type=["csv"])
if uploaded_file:
    df = load_kaggle_data(uploaded_file)
    st.success(f"✅ Loaded {len(df)} records.")
else:
    st.warning("Please upload your Kaggle dataset to continue.")
    st.stop()

# --- Sentiment Model Training ---
@st.cache_resource(show_spinner=True)
def prepare_or_load_sentiment_model(_df):
    _df["cleaned"] = _df["text"].apply(clean_text)
    X_train, X_test, y_train, y_test = train_test_split(_df["cleaned"], _df["sentiment"], test_size=0.2, random_state=42)
    vec = TfidfVectorizer(max_features=5000)
    X_train_vec = vec.fit_transform(X_train)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_vec, y_train)
    report = classification_report(y_test, model.predict(vec.transform(X_test)), output_dict=True)
    return model, vec, report

if "text" not in df.columns or "sentiment" not in df.columns:
    st.error("Dataset must contain columns named `text` and `sentiment`.")
    st.stop()

model, vec, report = prepare_or_load_sentiment_model(df)
st.success("✅ Sentiment model trained successfully using your Kaggle dataset!")

# --- Frontend Layout ---
col1, col2, col3, col4 = st.columns(4)
col5, col6, col7 = st.columns(3)

with col1:
    if st.button("🔍 Show Dataset"):
        st.dataframe(df.head(10))

with col2:
    if st.button("📊 Sentiment Report"):
        st.json(report)

with col3:
    if st.button("🧾 Extract Topics"):
        topics = extract_topics(df["text"])
        for i, t in enumerate(topics):
            st.write(f"**Topic {i+1}:** {', '.join(t)}")

with col4:
    if st.button("☁️ Word Cloud"):
        plt = plot_wordcloud(" ".join(df["text"].astype(str)))
        if plt:
            st.pyplot(plt)

# Second row
with col5:
    text_input = st.text_area("Enter text for summarization or analysis", height=150)

with col6:
    if st.button("🧠 Extractive Summary"):
        summary = extractive_reduce(text_input)
        st.subheader("Extractive Summary")
        st.write(summary)

with col7:
    if st.button("❤️ Predict Sentiment"):
        if text_input.strip():
            X_vec = vec.transform([clean_text(text_input)])
            pred = model.predict(X_vec)[0]
            st.write(f"Predicted Sentiment: **{pred}**")
        else:
            st.warning("Please enter text first.")
