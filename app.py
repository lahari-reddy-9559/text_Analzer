# app.py — TalkTective Complete
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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from wordcloud import WordCloud
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# Import backend utilities
from summarization_utils import (
    clean_text,
    extractive_reduce,
    abstractive_summarize_text,
    extract_keywords_corpus,
    generate_recommendations,
    topic_modeling_corpus
)

warnings.filterwarnings("ignore")

# Basic settings
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
RANDOM_STATE = 42
MAX_FEATURES = 5000

sentiment_mapping = {"negative": 0, "neutral": 1, "positive": 2}
reverse_sentiment_mapping = {v: k for k, v in sentiment_mapping.items()}

st.set_page_config(page_title="TalkTective Studio | Lahari Reddy", layout="wide", page_icon="💬")

# Visual CSS
st.markdown("""
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
""", unsafe_allow_html=True)

# ---------------- Load Kaggle dataset and prepare vectorizer ----------------
@st.cache_data(show_spinner="📦 Loading Kaggle dataset...")
def load_kaggle_dataset_and_vect():
    try:
        import kagglehub
        path = kagglehub.dataset_download("abhi8923shriv/sentiment-analysis-dataset")
        df = pd.read_csv(os.path.join(path, "train.csv"), encoding="latin-1")
    except Exception:
        return None, None, None

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    def clean_local(t):
        t = str(t).lower().translate(str.maketrans("", "", string.punctuation))
        w = [lemmatizer.lemmatize(x) for x in t.split() if x not in stop_words]
        return " ".join(w)

    if "text" not in df.columns or "sentiment" not in df.columns:
        return None, None, None

    df = df.dropna(subset=["text","sentiment"]).copy()
    df["cleaned_text"] = df["text"].apply(clean_local)

    vec = TfidfVectorizer(max_features=MAX_FEATURES)
    vec.fit(df["cleaned_text"].tolist())
    corpus = df["cleaned_text"].tolist()
    return df, vec, corpus

# ---------------- Train / Load Sentiment Model ----------------
@st.cache_resource(show_spinner="🔧 Training/loading sentiment model...")
def prepare_or_load_sentiment_model(df, vec):
    clf_path = os.path.join(MODEL_DIR, "rf_sentiment.pkl")
    vec_path = os.path.join(MODEL_DIR, "tfidf.pkl")
    if os.path.exists(clf_path) and os.path.exists(vec_path):
        try:
            clf = joblib.load(clf_path)
            vec_loaded = joblib.load(vec_path)
            return clf, vec_loaded
        except:
            pass

    X = vec.transform(df["cleaned_text"].tolist())
    y_num = pd.Series(df["sentiment"]).map(sentiment_mapping).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y_num, test_size=0.2, random_state=RANDOM_STATE, stratify=y_num)
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train.toarray(), y_train)
    joblib.dump(clf, clf_path)
    joblib.dump(vec, vec_path)
    return clf, vec

# ---------------- Helper functions ----------------
def analyze_sentiment(text, vec, clf):
    clean_t = clean_text(text)
    X = vec.transform([clean_t]).toarray()
    probs = clf.predict_proba(X)[0]
    results = {reverse_sentiment_mapping[int(c)]: float(p) for c,p in zip(clf.classes_, probs)}
    top = reverse_sentiment_mapping[int(clf.classes_[np.argmax(probs)])]
    return results, top

def generate_wc_image(text, dark_mode=False):
    clean_t = clean_text(text)
    if not clean_t:
        bg = "black" if dark_mode else "white"
        return Image.new("RGB", (500,300), color=bg)
    wc = WordCloud(width=500,height=300,background_color="black" if dark_mode else "white",
                   colormap="plasma" if dark_mode else "viridis", max_words=150).generate(clean_t)
    return wc.to_image()

def plot_compact_bar(sentiment_dict, dark_mode=False):
    labels = list(sentiment_dict.keys())
    vals = [sentiment_dict[k] for k in labels]
    bg, text_color, bar_colors = ("#0b0f14", "white", ["#FF6B6B","#FFD166","#06D6A0"]) if dark_mode else ("white","#222222",["#F87171","#FACC15","#34D399"])
    fig, ax = plt.subplots(figsize=(5,3), dpi=100)
    ax.bar(labels, vals, color=bar_colors[:len(labels)], width=0.35, edgecolor="gray")
    ax.set_ylim(0,1.05)
    ax.set_title("Sentiment Confidence", fontsize=10, color=text_color)
    ax.set_ylabel("Probability", color=text_color, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.tick_params(colors=text_color, which="both")
    for spine in ["top","right"]:
        ax.spines[spine].set_visible(False)
    plt.setp(ax.get_xticklabels(), fontsize=9, color=text_color)
    plt.setp(ax.get_yticklabels(), fontsize=8, color=text_color)
    plt.tight_layout()
    return fig

# ---------------- Single top LDA topic ----------------
def extract_top_topic(input_text, corpus, n_top_words=8):
    lda, feature_names, topic_words_list, _ = topic_modeling_corpus(corpus, n_topics=1, n_top_words=n_top_words)
    if not topic_words_list:
        return [], []
    return topic_words_list[0], []

# ---------------- Load Data & Models ----------------
df, vec, corpus = load_kaggle_dataset_and_vect()
if df is None or vec is None or corpus is None:
    st.error("Dataset not loaded. Check Kaggle credentials.")
    st.stop()
clf, vec = prepare_or_load_sentiment_model(df, vec)
if clf is None or vec is None:
    st.error("Sentiment model could not be trained/loaded.")
    st.stop()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Options")
    use_abstractive = st.checkbox("Enable Abstractive Summarization", value=False)
    abstr_model = st.text_input("Abstractive model", value="t5-small")
    extract_ratio = st.slider("Extractive summary ratio", 0.05, 0.8, 0.25)
    top_k_keywords = st.slider("Top Keywords", 3, 20, 8)

# ---------------- Text input ----------------
text_input = st.text_area("📝 Enter Text", placeholder="Paste text...", height=180)
uploaded = st.file_uploader("📄 Or upload text file (.txt)", type=["txt"])
if uploaded:
    text_input = uploaded.read().decode("utf-8", errors="ignore")

st.markdown("---")

row1 = st.columns(4)
with row1[0]:
    sentiment_btn = st.button("🧠 Sentiment Analysis")
with row1[1]:
    extractive_btn = st.button("✂️ Extractive Summary")
with row1[2]:
    abstractive_btn = st.button("🪶 Abstractive Summary")
with row1[3]:
    wordcloud_btn = st.button("☁️ Word Cloud")

row2 = st.columns(3)
with row2[0]:
    keywords_btn = st.button("🧩 Keywords")
with row2[1]:
    topics_btn = st.button("📊 Topics")
with row2[2]:
    insights_btn = st.button("🎯 Insights")

dark_mode = st.get_option("theme.base") == "dark"

# ---------------- Main Logic ----------------
if text_input and text_input.strip():
    cleaned_input = clean_text(text_input)

    if sentiment_btn:
        st.subheader("🧠 Sentiment Analysis")
        sentiment_probs, top_sent = analyze_sentiment(text_input, vec, clf)
        st.success(f"Predicted Sentiment: **{top_sent.upper()}**")
        st.pyplot(plot_compact_bar(sentiment_probs, dark_mode=dark_mode), use_container_width=False)

    if extractive_btn:
        st.subheader("✂️ Extractive Summary")
        st.info(extractive_reduce(text_input, ratio=extract_ratio, min_sentences=1, max_sentences=6))

    if abstractive_btn:
        st.subheader("🪶 Abstractive Summary")
        if use_abstractive:
            try:
                st.info(abstractive_summarize_text(text_input, model_name=abstr_model))
            except:
                st.error("Abstractive summarization failed.")
        else:
            st.info("Abstractive disabled. Enable in sidebar.")

    if wordcloud_btn:
        st.subheader("☁️ Word Cloud")
        st.image(generate_wc_image(text_input, dark_mode=dark_mode), width=500)

    if keywords_btn:
        st.subheader("🧩 Keywords")
        global_keywords = extract_keywords_corpus(corpus, top_n=top_k_keywords)
        st.markdown("**Global Keywords:**")
        st.write(", ".join([t for t,_ in global_keywords]))

    if topics_btn:
        st.subheader("📊 Topic Words (Single Top Topic)")
        topic_words, _ = extract_top_topic(text_input, corpus)
        st.info(", ".join(topic_words))

    if insights_btn:
        st.subheader("🎯 Actionable Insights")
        sentiment_probs, top_sent = analyze_sentiment(text_input, vec, clf)
        kws = [t for t,_ in extract_keywords_corpus(corpus, top_n=8)]
        topic_words, _ = extract_top_topic(text_input, corpus)
        recs = generate_recommendations(text_input, top_sent, kws, [topic_words])
        for r in recs:
            st.info(r)

    # ---------------- PDF Download ----------------
    pdf_content = {
        "Predicted Sentiment": analyze_sentiment(text_input, vec, clf)[1].upper(),
        "Extractive Summary": extractive_reduce(text_input, ratio=extract_ratio),
        "Abstractive Summary": abstractive_summarize_text(text_input, model_name=abstr_model) if use_abstractive else "Disabled",
        "Keywords": ", ".join([t for t,_ in extract_keywords_corpus(corpus, top_n=top_k_keywords)]),
        "Topic Words": ", ".join(extract_top_topic(text_input, corpus)[0])
    }

    def generate_pdf(content_dict):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = [Paragraph("💬 TalkTective Report", styles['Title']), Spacer(1,12)]
        for key,value in content_dict.items():
            elements.append(Paragraph(f"<b>{key}:</b>", styles['Heading3']))
            elements.append(Paragraph(str(value), styles['Normal']))
            elements.append(Spacer(1,12))
        doc.build(elements)
        buffer.seek(0)
        return buffer

    st.download_button(
        "📥 Download PDF Report",
        data=generate_pdf(pdf_content),
        file_name="TalkTective_Report.pdf",
        mime="application/pdf"
    )
else:
    st.info("💡 Enter text or upload a file to start analysis.")
