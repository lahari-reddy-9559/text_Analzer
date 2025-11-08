# app.py
import os
import streamlit as st
import pandas as pd
import string
import nltk
import joblib
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import matplotlib.pyplot as plt
from summarization_utils import (
    clean_text, extractive_reduce, abstractive_summarize_text,
    extract_keywords, extract_topics, generate_recommendations, load_or_train_lda
)

# --- Streamlit setup ---
st.set_page_config(page_title="TalkTective Studio", layout="wide", page_icon="💬")
st.title("💬 TalkTective")
st.caption("Developed by **Lahari Reddy** - AI that analyzes text and topics.")

# --- Load Kaggle dataset for sentiment + LDA ---
@st.cache_data(show_spinner="Loading dataset...")
def load_dataset_and_train_models():
    import kagglehub
    path = kagglehub.dataset_download("abhi8923shriv/sentiment-analysis-dataset")
    df = pd.read_csv(os.path.join(path, "train.csv"), encoding="latin-1")
    df.dropna(subset=["text", "selected_text"], inplace=True)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    df["cleaned_text"] = df["text"].apply(lambda t: clean_text(t))
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df["cleaned_text"])
    y = df["sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    # Train LDA on Kaggle data
    lda_model, lda_vectorizer = load_or_train_lda(df["cleaned_text"].tolist(), n_topics=5)
    return clf, tfidf, lda_model, lda_vectorizer

clf, tfidf_vec, lda_model, lda_vectorizer = load_dataset_and_train_models()

# --- User input ---
text_input = st.text_area("📝 Enter Text:", placeholder="Type or paste text here...", height=160)
uploaded = st.file_uploader("📄 Or upload a .txt file:", type=["txt"])
if uploaded:
    text_input = uploaded.read().decode("utf-8", errors="ignore")

st.markdown("---")

# --- Buttons layout (2 rows: 4 + 3) ---
row1_cols = st.columns(4)
row2_cols = st.columns(3)

button_list_row1 = [
    ("🧠 Sentiment Analysis", "sentiment"),
    ("✂️ Extractive Summary", "extractive"),
    ("🪶 Abstractive Summary", "abstractive"),
    ("☁️ Word Cloud", "wordcloud")
]
button_list_row2 = [
    ("🧩 Keywords", "keywords"),
    ("📊 Topics", "topics"),
    ("🎯 Insights", "insights")
]

choice = None
for (label, val), col in zip(button_list_row1, row1_cols):
    if col.button(label):
        choice = val
for (label, val), col in zip(button_list_row2, row2_cols):
    if col.button(label):
        choice = val

# --- Dark mode check ---
def is_dark_mode():
    try:
        return st.get_option("theme.base") == "dark"
    except Exception:
        return False
dark_mode = is_dark_mode()

# --- Utility: sentiment analysis plot ---
def plot_sentiment_bar(sent_dict):
    labels = list(sent_dict.keys())
    vals = [sent_dict[k] for k in labels]
    colors = ['#F87171', '#FACC15', '#34D399'] if not dark_mode else ['#FF6B6B', '#FFD166', '#06D6A0']
    fig, ax = plt.subplots(figsize=(5,3))
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylim(0,1)
    ax.set_ylabel("Probability")
    ax.set_title("Sentiment Confidence")
    return fig

# --- Main logic ---
if text_input and text_input.strip():
    clean_input = clean_text(text_input)

    if choice == "sentiment":
        st.subheader("🧠 Sentiment Analysis")
        X_in = tfidf_vec.transform([clean_input])
        probs = clf.predict_proba(X_in)[0]
        labels = clf.classes_
        sent_dict = {labels[i]: float(probs[i]) for i in range(len(labels))}
        top_sent = max(sent_dict, key=sent_dict.get)
        st.success(f"Predicted Sentiment: **{top_sent.upper()}**")
        fig = plot_sentiment_bar(sent_dict)
        c1,c2,c3 = st.columns([1,2,1])
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
        wc = WordCloud(width=500, height=300, background_color="white").generate(clean_input)
        c1,c2,c3 = st.columns([1,2,1])
        with c2:
            st.image(wc.to_image(), use_column_width=False, width=500)

    elif choice == "keywords":
        st.subheader("🧩 Keywords")
        kw = extract_keywords(text_input)
        st.success(", ".join(kw))

    elif choice == "topics":
        st.subheader("📊 Topics")
        top_words = extract_topics(text_input, lda_model=lda_model, lda_vectorizer=lda_vectorizer, n_words=6)
        c1,c2,c3 = st.columns([1,2,1])
        with c2:
            st.bar_chart({w:1 for w in top_words})
            st.info(f"Top Topic Words: {', '.join(top_words)}")

    elif choice == "insights":
        st.subheader("🎯 Actionable Insights")
        X_in = tfidf_vec.transform([clean_input])
        probs = clf.predict_proba(X_in)[0]
        labels = clf.classes_
        top_sent = labels[probs.argmax()]
        kw = extract_keywords(text_input)
        topics_words = extract_topics(text_input, lda_model=lda_model, lda_vectorizer=lda_vectorizer, n_words=6)
        recs = generate_recommendations(text_input, top_sent, kw, topics_words)
        for r in recs:
            st.markdown(f"- {r}")

else:
    st.info("💡 Enter text above or upload a file to start analysis.")
