# app.py — TalkTective Frontend (integrated with Kaggle dataset)
import os
import io
import string
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from wordcloud import WordCloud
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# Import backend utilities (must be in same folder)
from summarization_utils import (
    clean_text,
    extractive_reduce,
    abstractive_summarize_text,
    extract_keywords_corpus,
    extract_topics_for_input,
    generate_recommendations,
)

warnings.filterwarnings("ignore")

# Basic settings
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
RANDOM_STATE = 42
MAX_FEATURES = 5000

# Sentiment label mapping (dataset uses strings 'negative','neutral','positive')
sentiment_mapping = {"negative": 0, "neutral": 1, "positive": 2}
reverse_sentiment_mapping = {v: k for k, v in sentiment_mapping.items()}

# Streamlit config
st.set_page_config(page_title="TalkTective Studio | Lahari Reddy", layout="wide", page_icon="💬")

# Visual CSS (keeps your original look)
st.markdown(
    """
<style>
body {
    background: linear-gradient(135deg, #FDEFF9 0%, #ECF4FF 50%, #E8F9F0 100%);
    font-family: 'Poppins', sans-serif;
}
div.block-container {
    padding-top: 1.6rem;
    background-color: rgba(255, 255, 255, 0.94);
    border-radius: 14px;
    padding: 20px 24px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.06);
}
h1, h2, h3 {
    color: #4B0082;
    font-weight: 600;
}
.stButton>button {
    background: linear-gradient(90deg, #6C63FF, #00BFA6);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5em 1.0em;
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# Load Kaggle dataset & prepare vectorizer and optionally models
# -------------------------
@st.cache_data(show_spinner="📦 Loading Kaggle dataset and vectorizer...")
def load_kaggle_dataset_and_vect():
    """
    Download dataset (kagglehub) and prepare TfidfVectorizer over cleaned text.
    Returns df, vectorizer, and cleaned corpus (list of cleaned_text).
    """
    try:
        import kagglehub  # local import to avoid import error earlier if not installed
        path = kagglehub.dataset_download("abhi8923shriv/sentiment-analysis-dataset")
        df = pd.read_csv(os.path.join(path, "train.csv"), encoding="latin-1")
    except Exception:
        # If kagglehub or dataset not available, return None to let UI show error
        return None, None, None

    # Basic cleaning prerequisites (ensure NLTK packages)
    try:
        import nltk

        for pkg in ["punkt", "stopwords", "wordnet"]:
            try:
                nltk.data.find(f"tokenizers/{pkg}" if pkg == "punkt" else f"corpora/{pkg}")
            except LookupError:
                nltk.download(pkg, quiet=True)
    except Exception:
        pass

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    def clean_local(t):
        t = str(t).lower().translate(str.maketrans("", "", string.punctuation))
        w = [lemmatizer.lemmatize(x) for x in t.split() if x not in stop_words]
        return " ".join(w)

    # Keep only rows with text & sentiment
    if "text" not in df.columns or "sentiment" not in df.columns:
        return None, None, None

    df = df.dropna(subset=["text", "sentiment"]).copy()
    df["cleaned_text"] = df["text"].apply(clean_local)

    # Build TF-IDF vectorizer for training sentiment model (if needed)
    vec = TfidfVectorizer(max_features=MAX_FEATURES)
    vec.fit(df["cleaned_text"].tolist())
    corpus = df["cleaned_text"].tolist()
    return df, vec, corpus


# Training and saving sentiment model (RandomForest as per your original code)
@st.cache_resource(show_spinner="🔧 Training or loading sentiment model...")
def prepare_or_load_sentiment_model(df, vec):
    """
    If saved model exists in models/, load it. Otherwise train on the Kaggle dataset and save.
    Returns trained classifier and vectorizer.
    """
    clf_path = os.path.join(MODEL_DIR, "rf_sentiment.pkl")
    vec_path = os.path.join(MODEL_DIR, "tfidf.pkl")

    # If files exist, load
    if os.path.exists(clf_path) and os.path.exists(vec_path):
        try:
            clf = joblib.load(clf_path)
            vec_loaded = joblib.load(vec_path)
            return clf, vec_loaded
        except Exception:
            # proceed to retrain if loading fails
            pass

    # Train model
    try:
        # prepare X and y from df
        X = vec.transform(df["cleaned_text"].tolist())
        y_series = df["sentiment"]
        # map labels to numeric
        y_num = pd.Series(y_series).map(sentiment_mapping).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y_num, test_size=0.2, random_state=RANDOM_STATE, stratify=y_num)
        clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        clf.fit(X_train.toarray(), y_train)
        # save
        joblib.dump(clf, clf_path)
        joblib.dump(vec, vec_path)
        return clf, vec
    except Exception as e:
        # If training fails, bubble up None so UI can show error
        st.error(f"Model training/loading error: {e}")
        return None, None


# -------------------------
# Helper visual & util functions
# -------------------------
def is_streamlit_dark():
    """Return True if Streamlit theme is dark; fallback False."""
    try:
        base = st.get_option("theme.base")
        return base == "dark"
    except Exception:
        return False


def analyze_sentiment(text, vec, clf):
    """Return dict of probabilities per label (strings) and top label."""
    clean_t = clean_text(text)
    if not clean_t:
        return {"negative": 0.0, "neutral": 0.0, "positive": 0.0}, "neutral"
    X = vec.transform([clean_t]).toarray()
    probs = clf.predict_proba(X)[0]
    # clf.classes_ are numeric labels (0,1,2)
    results = {}
    for c, p in zip(clf.classes_, probs):
        label = reverse_sentiment_mapping.get(int(c), str(c))
        results[label] = float(p)
    top = reverse_sentiment_mapping[int(clf.classes_[np.argmax(probs)])]
    return results, top


def generate_wc_image(text, dark_mode=False):
    clean_t = clean_text(text)
    if not clean_t:
        bg = "black" if dark_mode else "white"
        im = Image.new("RGB", (500, 300), color=bg)
        return im
    wc = WordCloud(
        width=500,
        height=300,
        background_color="black" if dark_mode else "white",
        colormap="plasma" if dark_mode else "viridis",
        max_words=150,
    ).generate(clean_t)
    return wc.to_image()


def plot_compact_bar(sentiment_dict, dark_mode=False):
    labels = list(sentiment_dict.keys())
    vals = [sentiment_dict[k] for k in labels]
    if dark_mode:
        bg = "#0b0f14"
        text_color = "white"
        bar_colors = ["#FF6B6B", "#FFD166", "#06D6A0"]
    else:
        bg = "white"
        text_color = "#222222"
        bar_colors = ["#F87171", "#FACC15", "#34D399"]
    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
    ax.bar(labels, vals, color=bar_colors[: len(labels)], width=0.35, edgecolor="gray")
    ax.set_ylim(0, 1.05)
    ax.set_title("Sentiment Confidence", fontsize=10, color=text_color, pad=6)
    ax.set_ylabel("Probability", color=text_color, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.tick_params(colors=text_color, which="both")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.setp(ax.get_xticklabels(), fontsize=9, color=text_color)
    plt.setp(ax.get_yticklabels(), fontsize=8, color=text_color)
    plt.tight_layout()
    return fig


# -------------------------
# App UI
# -------------------------
st.title("💬 TalkTective")
st.caption("Developed by **Lahari Reddy** - the AI detective that investigates your text✨")

# Load dataset and vectorizer
df, vec, corpus = load_kaggle_dataset_and_vect()
if df is None or vec is None or corpus is None:
    st.error("Could not load Kaggle dataset. Ensure Kaggle credentials and kagglehub are configured and dataset exists.")
    st.stop()

# Prepare or load sentiment model
clf, vec = prepare_or_load_sentiment_model(df, vec)
if clf is None or vec is None:
    st.error("Sentiment model could not be trained or loaded. Check logs.")
    st.stop()

# Sidebar options
with st.sidebar:
    st.header("Options")
    use_abstractive = st.checkbox("Enable abstractive summarization (requires transformers)", value=False)
    abstr_model = st.text_input("Abstractive model name", value="t5-small")
    extract_ratio = st.slider("Extractive summary ratio", 0.05, 0.8, 0.25)
    n_topics = st.slider("Number of topics (LDA)", 2, 8, 4)
    top_k_keywords = st.slider("Top keywords to show (global)", 3, 20, 8)

# Input area
text_input = st.text_area("📝 Enter Text:", placeholder="Paste or type text to analyze...", height=180)
uploaded = st.file_uploader("📄 Or upload a text file (.txt):", type=["txt"])
if uploaded:
    try:
        text_input = uploaded.read().decode("utf-8", errors="ignore")
    except Exception:
        text_input = str(uploaded.read())

st.markdown("---")

# ---------- TWO ROWS: Row1 (4 buttons) Row2 (3 buttons) ----------
st.markdown("<br>", unsafe_allow_html=True)
row1 = st.columns(4)
with row1[0]:
    sentiment_btn = st.button("🧠 Sentiment Analysis", key="btn_sentiment")
with row1[1]:
    extractive_btn = st.button("✂️ Extractive Summary", key="btn_extractive")
with row1[2]:
    abstractive_btn = st.button("🪶 Abstractive Summary", key="btn_abstractive")
with row1[3]:
    wordcloud_btn = st.button("☁️ Word Cloud", key="btn_wordcloud")

st.markdown("<br>", unsafe_allow_html=True)
row2 = st.columns(3)
with row2[0]:
    keywords_btn = st.button("🧩 Keywords", key="btn_keywords")
with row2[1]:
    topics_btn = st.button("📊 Topics", key="btn_topics")
with row2[2]:
    insights_btn = st.button("🎯 Insights", key="btn_insights")
st.markdown("---")

dark_mode = is_streamlit_dark()

# Main logic
if text_input and text_input.strip():

    # Precompute some shared things (cleaned input)
    cleaned_input = clean_text(text_input)

    # Sentiment
    if sentiment_btn:
        st.subheader("🧠 Sentiment Analysis")
        sentiment_probs, top_sent = analyze_sentiment(text_input, vec, clf)
        st.success(f"Predicted Sentiment: **{top_sent.upper()}**")
        fig = plot_compact_bar(sentiment_probs, dark_mode=dark_mode)
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.pyplot(fig, use_container_width=False)

    # Extractive
    if extractive_btn:
        st.subheader("✂️ Extractive Summary")
        st.info(extractive_reduce(text_input, ratio=extract_ratio, min_sentences=1, max_sentences=6))

    # Abstractive
    if abstractive_btn:
        st.subheader("🪶 Abstractive Summary")
        if use_abstractive:
            try:
                st.info(abstractive_summarize_text(text_input, model_name=abstr_model))
            except Exception as e:
                st.error(f"Abstractive summarization error: {e}")
        else:
            st.info("Abstractive summarization disabled. Enable it in the sidebar.")

    # Wordcloud
    if wordcloud_btn:
        st.subheader("☁️ Word Cloud Visualization")
        wc_img = generate_wc_image(text_input, dark_mode=dark_mode)
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(wc_img, use_column_width=False, width=500)

    # Keywords (global from corpus + local)
    if keywords_btn:
        st.subheader("🧩 Keywords (global corpus + input highlights)")
        global_keywords = extract_keywords_corpus(corpus, top_n=top_k_keywords)
        # show global top keywords
        st.markdown("**Global dataset top keywords:**")
        for term, score in global_keywords:
            st.write(f"- **{term}** — score {score:.3f}")
        # local highlights (terms from input)
        st.markdown("**Top terms in your input (extracted locally):**")
        local_vect = TfidfVectorizer(stop_words="english", max_features=1000)
        try:
            X_local = local_vect.fit_transform([cleaned_input])
            scores = zip(local_vect.get_feature_names_out(), X_local.toarray()[0])
            top_local = sorted(scores, key=lambda x: x[1], reverse=True)[:10]
            st.write(", ".join([w for w, s in top_local if s > 0]))
        except Exception:
            st.info("No local keywords could be extracted from the input.")

    # Topics - use Kaggle corpus to build topics and then show which topics input maps to
    if topics_btn:
        st.subheader("📊 Extracted Topics")
        st.info("Building topic model from Kaggle corpus (this uses the dataset to get robust topics).")
        topic_words, input_topic_dist = extract_topics_for_input(cleaned_input, corpus, n_topics=n_topics, n_top_words=8)
        if not topic_words:
            st.info("Not enough data to build topics.")
        else:
            for i, words in enumerate(topic_words, 1):
                st.markdown(f"**Topic {i}:** {', '.join(words)}")
            st.markdown("---")
            st.markdown("**Input document topic distribution:**")
            for i, p in enumerate(input_topic_dist, 1):
                st.write(f"Topic {i}: {p:.3f}")

    # Insights
    if insights_btn:
        st.subheader("🎯 Actionable Insights")
        sentiment_probs, top_sent = analyze_sentiment(text_input, vec, clf)
        # global keywords (top 8) + local
        global_kws = [t for t, _ in extract_keywords_corpus(corpus, top_n=8)]
        # topics (just the word lists)
        topic_words, _ = extract_topics_for_input(cleaned_input, corpus, n_topics=n_topics, n_top_words=6)
        recs = generate_recommendations(text_input, top_sent, global_kws, topic_words)
        for r in recs:
            st.info(r)

    # PDF generation
    if st.button("📥 Download Full Report (PDF)", key="btn_pdf"):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = [
            Paragraph("<b>Text Insight Studio - Compact Report</b>", styles["Title"]),
            Spacer(1, 8),
            Paragraph("Original Text:", styles["Heading2"]),
            Paragraph((text_input[:1200] + ("..." if len(text_input) > 1200 else "")), styles["Normal"]),
            Spacer(1, 8),
        ]
        # sentiment
        sentiment_probs, top_sent = analyze_sentiment(text_input, vec, clf)
        elements.append(Paragraph("Predicted Sentiment:", styles["Heading2"]))
        elements.append(Paragraph(str(top_sent).upper(), styles["Normal"]))
        elements.append(Spacer(1, 6))
        # extractive summary
        elements.append(Paragraph("Extractive Summary:", styles["Heading2"]))
        elements.append(Paragraph(extractive_reduce(text_input, ratio=extract_ratio), styles["Normal"]))
        elements.append(Spacer(1, 6))
        # abstractive (if available)
        try:
            elements.append(Paragraph("Abstractive Summary:", styles["Heading2"]))
            if use_abstractive:
                elements.append(Paragraph(abstractive_summarize_text(text_input, model_name=abstr_model), styles["Normal"]))
            else:
                elements.append(Paragraph("Abstractive not enabled in sidebar.", styles["Normal"]))
            elements.append(Spacer(1, 6))
        except Exception:
            pass
        # wordcloud image
        wc_img = generate_wc_image(text_input, dark_mode=dark_mode)
        img_path = "wordcloud_500x300.png"
        wc_img.save(img_path)
        elements.append(RLImage(img_path, width=5.0 * inch, height=3.0 * inch))
        elements.append(Spacer(1, 8))
        # keywords and topics and recs
        kws = [t for t, _ in extract_keywords_corpus(corpus, top_n=8)]
        topic_words, _ = extract_topics_for_input(clean_text(text_input), corpus, n_topics=n_topics, n_top_words=6)
        recs = generate_recommendations(text_input, top_sent, kws, topic_words)
        elements.append(Paragraph("Top Keywords:", styles["Heading2"]))
        elements.append(Paragraph(", ".join(kws), styles["Normal"]))
        elements.append(Spacer(1, 6))
        elements.append(Paragraph("Topics:", styles["Heading2"]))
        for t in topic_words:
            elements.append(Paragraph(", ".join(t), styles["Normal"]) if isinstance(t, list) else Paragraph(str(t), styles["Normal"]))
        elements.append(Spacer(1, 6))
        elements.append(Paragraph("Recommendations:", styles["Heading2"]))
        for r in recs:
            elements.append(Paragraph(r, styles["Normal"]))
        doc.build(elements)
        st.download_button(
            "⬇️ Save Compact PDF Report",
            data=buffer.getvalue(),
            file_name="Text_Insight_Compact_Report.pdf",
            mime="application/pdf",
        )
else:
    st.info("💡 Enter text above or upload a file to start analysis.")

# Footer
st.markdown("---")
st.caption("Built with your Kaggle dataset and auditable NLP building blocks. Customize rules and models for your domain.")
