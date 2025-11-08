# app.py — Lahari Reddy | TalkTective Frontend

import os
import io
import warnings
import pandas as pd
import string
import numpy as np
import joblib
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import LatentDirichletAllocation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
import kagglehub

# Import custom utils
from summarization_utils import (
    clean_text,
    extractive_reduce,
    abstractive_summarize_text,
    extract_keywords,
    extract_topics,
    generate_recommendations
)

warnings.filterwarnings("ignore")

MODEL_DIR = "models"
RANDOM_STATE = 42
MAX_FEATURES = 5000

# --- Streamlit Page ---
st.set_page_config(
    page_title="TalkTective Studio | Lahari Reddy",
    layout="wide",
    page_icon="💬"
)

st.markdown("""
<style>
.stButton>button {background: linear-gradient(90deg, #6C63FF, #00BFA6); color: white; border-radius: 8px; padding: 0.5em 1em; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load & preprocess Kaggle dataset
# -----------------------------
@st.cache_data(show_spinner="📦 Loading dataset...")
def load_data():
    path = kagglehub.dataset_download("abhi8923shriv/sentiment-analysis-dataset")
    file_path = os.path.join(path, "train.csv")
    df = pd.read_csv(file_path, encoding='latin-1')
    df.dropna(subset=['text', 'selected_text'], inplace=True)
    df['cleaned_text'] = df['text'].apply(clean_text)
    return df

df = load_data()

# -----------------------------
# Train sentiment model
# -----------------------------
@st.cache_resource(show_spinner="🔧 Training Sentiment Model...")
def train_sentiment_model(_df):
    vec = TfidfVectorizer(max_features=MAX_FEATURES)
    X = vec.fit_transform(_df['cleaned_text'])
    y = _df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train.toarray(), y_train)
    return clf, vec

clf, vec = train_sentiment_model(df)

# -----------------------------
# Train LDA model
# -----------------------------
@st.cache_resource(show_spinner="🔧 Training Topic Model...")
def train_lda_model(_df):
    lda_vec = TfidfVectorizer(max_features=1000)
    X = lda_vec.fit_transform(_df['cleaned_text'])
    lda_model = LatentDirichletAllocation(n_components=5, random_state=RANDOM_STATE)
    lda_model.fit(X)
    return lda_model, lda_vec

lda_model, lda_vec = train_lda_model(df)

# -----------------------------
# UI Input
# -----------------------------
st.title("💬 TalkTective Studio")
text_input = st.text_area("📝 Enter Text or paste here:", height=160)
uploaded = st.file_uploader("📄 Upload text file (.txt)", type=["txt"])
if uploaded:
    text_input = uploaded.read().decode("utf-8", errors="ignore")

# Buttons in 2 rows: 4 + 3
choice = None
row1, row2 = st.columns(4), st.columns(3)
buttons_row1 = [("🧠 Sentiment Analysis", "sentiment"),
                ("✂️ Extractive Summary", "extractive"),
                ("🪶 Abstractive Summary", "abstractive"),
                ("☁️ Word Cloud", "wordcloud")]

buttons_row2 = [("🧩 Keywords", "keywords"),
                ("📊 Topics", "topics"),
                ("🎯 Insights", "insights")]

for (label, val), col in zip(buttons_row1, row1):
    with col:
        if st.button(label):
            choice = val

for (label, val), col in zip(buttons_row2, row2):
    with col:
        if st.button(label):
            choice = val

st.markdown("---")
dark_mode = st.get_option("theme.base") == "dark"

# -----------------------------
# Helper functions
# -----------------------------
def analyze_sentiment(text, vec, clf):
    clean_t = clean_text(text)
    X = vec.transform([clean_t]).toarray()
    probs = clf.predict_proba(X)[0]
    classes = clf.classes_
    results = {str(c): float(p) for c, p in zip(classes, probs)}
    top = str(classes[np.argmax(probs)])
    return results, top

def plot_sentiment_bar(sentiment_dict):
    labels = list(sentiment_dict.keys())
    vals = [sentiment_dict[k] for k in labels]
    fig, ax = plt.subplots(figsize=(5,3))
    bars = ax.bar(labels, vals, color=['#F87171', '#FACC15', '#34D399'])
    ax.set_ylim(0,1.05)
    ax.set_ylabel("Probability")
    ax.set_title("Sentiment Confidence")
    plt.tight_layout()
    return fig

def generate_wordcloud_img(text):
    from wordcloud import WordCloud
    if not text.strip(): return Image.new("RGB", (500,300), "white")
    wc = WordCloud(width=500, height=300, max_words=150, background_color="white", colormap="viridis").generate(clean_text(text))
    return wc.to_image()

# -----------------------------
# Main Logic
# -----------------------------
if text_input and text_input.strip():
    if choice == "sentiment":
        st.subheader("🧠 Sentiment Analysis")
        sentiment_probs, top_sent = analyze_sentiment(text_input, vec, clf)
        st.success(f"Predicted Sentiment: **{top_sent.upper()}**")
        fig = plot_sentiment_bar(sentiment_probs)
        st.pyplot(fig, use_container_width=False)

    elif choice == "extractive":
        st.subheader("✂️ Extractive Summary")
        st.info(extractive_reduce(text_input))

    elif choice == "abstractive":
        st.subheader("🪶 Abstractive Summary")
        st.info(abstractive_summarize_text(text_input))

    elif choice == "wordcloud":
        st.subheader("☁️ Word Cloud")
        st.image(generate_wordcloud_img(text_input), width=500)

    elif choice == "keywords":
        st.subheader("🧩 Keywords")
        kws = extract_keywords(text_input)
        st.success(", ".join(kws))

    elif choice == "topics":
        st.subheader("📊 Top Topic Words")
        topic_words = extract_topics(text_input, lda_model, lda_vec)
        st.info(topic_words)

    elif choice == "insights":
        st.subheader("🎯 Actionable Insights")
        sentiment_probs, top_sent = analyze_sentiment(text_input, vec, clf)
        kws = extract_keywords(text_input)
        topic_words = extract_topics(text_input, lda_model, lda_vec)
        recs = generate_recommendations(text_input, top_sent, kws, topic_words.split(", "))
        for r in recs:
            st.markdown(f"- {r}")

    # PDF Download
    if st.button("📥 Download Full Report (PDF)"):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = [Paragraph("💬 TalkTective Report", styles["Title"]), Spacer(1,8)]
        elements.append(Paragraph("Original Text:", styles["Heading2"]))
        elements.append(Paragraph(text_input[:1200]+("..." if len(text_input)>1200 else ""), styles["Normal"]))
        elements.append(Spacer(1,6))

        sentiment_probs, top_sent = analyze_sentiment(text_input, vec, clf)
        elements.append(Paragraph("Sentiment:", styles["Heading2"]))
        elements.append(Paragraph(top_sent.upper(), styles["Normal"]))
        elements.append(Spacer(1,6))

        elements.append(Paragraph("Extractive Summary:", styles["Heading2"]))
        elements.append(Paragraph(extractive_reduce(text_input), styles["Normal"]))
        elements.append(Spacer(1,6))

        elements.append(Paragraph("Abstractive Summary:", styles["Heading2"]))
        elements.append(Paragraph(abstractive_summarize_text(text_input), styles["Normal"]))
        elements.append(Spacer(1,6))

        elements.append(Paragraph("Keywords:", styles["Heading2"]))
        elements.append(Paragraph(", ".join(extract_keywords(text_input)), styles["Normal"]))
        elements.append(Spacer(1,6))

        elements.append(Paragraph("Top Topic Words:", styles["Heading2"]))
        elements.append(Paragraph(extract_topics(text_input, lda_model, lda_vec), styles["Normal"]))
        elements.append(Spacer(1,6))

        elements.append(Paragraph("Recommendations:", styles["Heading2"]))
        recs = generate_recommendations(text_input, top_sent, extract_keywords(text_input), extract_topics(text_input, lda_model, lda_vec).split(", "))
        for r in recs:
            elements.append(Paragraph(r, styles["Normal"]))

        doc.build(elements)
        st.download_button("⬇️ Save PDF", data=buffer.getvalue(), file_name="TalkTective_Report.pdf", mime="application/pdf")

else:
    st.info("💡 Enter text above or upload a text file to start analysis.")
