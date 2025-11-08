# app.py
import os
import io
import pandas as pd
import string
import nltk
import joblib
import warnings
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import LatentDirichletAllocation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

import kagglehub
from summarization_utils import (
    clean_text,
    extractive_reduce,
    abstractive_summarize_text,
    extract_keywords,
    extract_topics,
    generate_recommendations
)

warnings.filterwarnings("ignore")

MODEL_DIR = 'models'
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
reverse_sentiment_mapping = {v: k for k, v in sentiment_mapping.items()}

st.set_page_config(page_title="TalkTective Studio", layout="wide", page_icon="💬")

# -------------------------
# Load Kaggle dataset
# -------------------------
@st.cache_data(show_spinner="📦 Loading dataset...")
def load_dataset():
    path = kagglehub.dataset_download("abhi8923shriv/sentiment-analysis-dataset")
    df = pd.read_csv(os.path.join(path, 'train.csv'), encoding='latin-1')
    df.dropna(subset=['text', 'selected_text'], inplace=True)
    df['cleaned_text'] = df['text'].apply(clean_text)
    return df

df = load_dataset()

# -------------------------
# Train models (cached)
# -------------------------
@st.cache_resource
def train_models(_X_train, _y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(_X_train.toarray(), _y_train)

    lda_model = LatentDirichletAllocation(n_components=5, random_state=42,
                                          learning_method='batch', max_iter=10)
    lda_model.fit(_X_train)
    return clf, lda_model

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(df['cleaned_text'])
y = df['sentiment'].map(sentiment_mapping).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf, lda_model = train_models(X_train, y_train)

# -------------------------
# Streamlit UI
# -------------------------
st.title("💬 TalkTective")
st.caption("Developed by **Lahari Reddy** - the AI detective that investigates your text✨")

text_input = st.text_area("📝 Enter Text:", height=160)
uploaded = st.file_uploader("📄 Or upload a text file (.txt):", type=["txt"])
if uploaded:
    text_input = uploaded.read().decode("utf-8", errors="ignore")

st.markdown("---")

# Buttons in 2 rows: 4 + 3
row1 = st.columns(4)
row2 = st.columns(3)

buttons_row1 = [
    ("🧠 Sentiment Analysis", "sentiment"),
    ("✂️ Extractive Summary", "extractive"),
    ("🪶 Abstractive Summary", "abstractive"),
    ("☁️ Word Cloud", "wordcloud")
]
buttons_row2 = [
    ("🧩 Keywords", "keywords"),
    ("📊 Topics", "topics"),
    ("🎯 Insights", "insights")
]

choice = None
for (label, val), col in zip(buttons_row1, row1):
    with col:
        if st.button(label):
            choice = val
for (label, val), col in zip(buttons_row2, row2):
    with col:
        if st.button(label):
            choice = val

# -------------------------
# Helper functions
# -------------------------
def analyze_sentiment(text):
    clean_t = clean_text(text)
    X_vec = tfidf_vectorizer.transform([clean_t]).toarray()
    probs = clf.predict_proba(X_vec)[0]
    results = {reverse_sentiment_mapping[c]: float(p) for c, p in zip(clf.classes_, probs)}
    top_sent = reverse_sentiment_mapping[clf.classes_[np.argmax(probs)]]
    return results, top_sent

def plot_bar_chart(values_dict, dark_mode=False, title=""):
    labels = list(values_dict.keys())
    vals = [values_dict[k] for k in labels]

    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
    colors = ['#F87171', '#FACC15', '#34D399'] if not dark_mode else ['#FF6B6B', '#FFD166', '#06D6A0']
    ax.bar(labels, vals, color=colors[:len(labels)], width=0.35)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    return fig

# -------------------------
# Main logic
# -------------------------
if text_input and text_input.strip():
    if choice == "sentiment":
        st.subheader("🧠 Sentiment Analysis")
        sentiment_probs, top_sent = analyze_sentiment(text_input)
        st.success(f"Predicted Sentiment: **{top_sent.upper()}**")
        fig = plot_bar_chart(sentiment_probs, title="Sentiment Confidence")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.pyplot(fig)

    elif choice == "extractive":
        st.subheader("✂️ Extractive Summary")
        st.info(extractive_reduce(text_input))

    elif choice == "abstractive":
        st.subheader("🪶 Abstractive Summary")
        st.info(abstractive_summarize_text(text_input))

    elif choice == "wordcloud":
        st.subheader("☁️ Word Cloud")
        from wordcloud import WordCloud
        wc_img = WordCloud(width=500, height=300, background_color="white").generate(clean_text(text_input)).to_image()
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(wc_img, width=500)

    elif choice == "keywords":
        st.subheader("🧩 Key Keywords")
        kw = extract_keywords(text_input)
        st.success(', '.join(kw) if kw else 'No keywords')

    elif choice == "topics":
        st.subheader("📊 Extracted Topics")
        topics = extract_topics([text_input], vectorizer=tfidf_vectorizer, lda_model=lda_model, top_n=7)
        for k, v in topics.items():
            st.markdown(f"**{k}:** {', '.join(v)}")
        # Plot bar chart for first topic probabilities
        topic_vec = tfidf_vectorizer.transform([clean_text(text_input)])
        topic_probs = lda_model.transform(topic_vec)[0]
        topic_dict = {f"Topic {i+1}": float(p) for i, p in enumerate(topic_probs)}
        fig = plot_bar_chart(topic_dict, title="Topic Probabilities")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.pyplot(fig)

    elif choice == "insights":
        st.subheader("🎯 Insights")
        sentiment_probs, top_sent = analyze_sentiment(text_input)
        kw = extract_keywords(text_input)
        topics = extract_topics([text_input], vectorizer=tfidf_vectorizer, lda_model=lda_model)
        recs = generate_recommendations(text_input, top_sent, kw, topics)
        for r in recs:
            st.markdown(f"- {r}")

else:
    st.info("💡 Enter text above or upload a file to start analysis.")
