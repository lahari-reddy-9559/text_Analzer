# app.py — TalkTective | Lahari Reddy

import kagglehub
import os
import pandas as pd
import string
import nltk
import joblib
import warnings
import numpy as np
import io
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
import tensorflow as tf

try:
    from summarization_utils import (
        clean_text as clean_text_util,
        extractive_reduce,
        abstractive_summarize_text,
        extract_keywords,
        extract_topics,
        generate_recommendations,
    )
except Exception:
    st.error("summarization_utils.py missing or not found.")
    st.stop()

warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

MODEL_DIR = 'models'
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
reverse_sentiment_mapping = {v: k for k, v in sentiment_mapping.items()}
MAX_FEATURES = 5000
RANDOM_STATE = 42

st.set_page_config(page_title="TalkTective | Lahari Reddy", layout="wide", page_icon="💬")

# Theme auto-detect and adaptive style
theme_bg = st.get_option("theme.backgroundColor")
is_dark = theme_bg and theme_bg.lower() not in ["#ffffff", "#fafafa", "white"]

bg_color = "#0E1117" if is_dark else "#FAFAFA"
fg_color = "#E0E0E0" if is_dark else "#1E1E1E"
card_bg = "#1E1E1E" if is_dark else "white"
shadow = "rgba(255,255,255,0.06)" if is_dark else "rgba(0,0,0,0.06)"

st.markdown(f"""
<style>
body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stSidebar"] {{
    background-color: {bg_color} !important;
    color: {fg_color} !important;
}}
div.block-container {{
    background-color: {card_bg} !important;
    border-radius: 12px;
    box-shadow: 0px 4px 20px {shadow};
    padding: 25px 30px;
}}
h1, h2, h3, label, p {{
    color: {fg_color} !important;
}}
.stButton>button {{
    width: 100%;
    height: 42px;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 15px;
}}
[data-testid="stHorizontalBlock"] {{
    gap: 8px !important;
}}
[data-testid="stButton"]:nth-child(1) button {{ background: linear-gradient(90deg, #6C63FF, #00BFA6); }}
[data-testid="stButton"]:nth-child(2) button {{ background: linear-gradient(90deg, #F87171, #F97316); }}
[data-testid="stButton"]:nth-child(3) button {{ background: linear-gradient(90deg, #60A5FA, #2563EB); }}
[data-testid="stButton"]:nth-child(4) button {{ background: linear-gradient(90deg, #FACC15, #16A34A); }}
[data-testid="stButton"]:nth-child(5) button {{ background: linear-gradient(90deg, #EC4899, #8B5CF6); }}
[data-testid="stButton"]:nth-child(6) button {{ background: linear-gradient(90deg, #14B8A6, #0D9488); }}
[data-testid="stButton"]:nth-child(7) button {{ background: linear-gradient(90deg, #22C55E, #15803D); }}
[data-testid="stButton"]:nth-child(8) button {{ background: linear-gradient(90deg, #6366F1, #8B5CF6); }}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading dataset...")
def load_and_preprocess_data():
    path = kagglehub.dataset_download("abhi8923shriv/sentiment-analysis-dataset")
    df = pd.read_csv(os.path.join(path, 'train.csv'), encoding='latin-1')
    df.dropna(subset=['text', 'selected_text'], inplace=True)

    for pkg in ['punkt', 'stopwords', 'wordnet']:
        try:
            nltk.data.find(f'tokenizers/{pkg}' if pkg == 'punkt' else f'corpora/{pkg}')
        except LookupError:
            nltk.download(pkg, quiet=True)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def clean_local(t):
        t = str(t).lower().translate(str.maketrans('', '', string.punctuation))
        w = [lemmatizer.lemmatize(x) for x in t.split() if x not in stop_words]
        return ' '.join(w)

    df['cleaned_text'] = df['text'].apply(clean_local)
    vec = TfidfVectorizer(max_features=MAX_FEATURES)
    vec.fit(df['cleaned_text'])
    X = vec.transform(df['cleaned_text'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, df['sentiment'], test_size=0.2, random_state=RANDOM_STATE, stratify=df['sentiment']
    )
    y_train_num = pd.Series(y_train).map(sentiment_mapping).astype(int)
    return df, vec, X_train, y_train_num, X_test, y_test


@st.cache_resource
def train_and_save_models(_X_train, _y_train_num, _vec):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(_X_train, _y_train_num)
    joblib.dump(clf, os.path.join(MODEL_DIR, 'logreg_sentiment.pkl'))
    joblib.dump(_vec, os.path.join(MODEL_DIR, 'tfidf.pkl'))
    return clf, _vec


def analyze_sentiment(text, vec, clf):
    clean_t = clean_text_util(text)
    X = vec.transform([clean_t]).toarray()
    probs = clf.predict_proba(X)[0]
    results = {reverse_sentiment_mapping[int(c)]: float(p) for c, p in zip(clf.classes_, probs)}
    top = reverse_sentiment_mapping[int(clf.classes_[np.argmax(probs)])]
    return results, top


def generate_wc_image(text, dark_mode=False):
    bg = "black" if dark_mode else "white"
    colormap = "plasma" if dark_mode else "viridis"
    clean_t = clean_text_util(text)
    wc = WordCloud(width=500, height=300, background_color=bg,
                   colormap=colormap, max_words=150).generate(clean_t)
    return wc.to_image()


def plot_compact_bar(sentiment_dict, dark_mode=False):
    labels = list(sentiment_dict.keys())
    vals = [sentiment_dict[k] for k in labels]
    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
    bar_colors = ['#F87171', '#FACC15', '#34D399']
    bg_color = "#0E1117" if dark_mode else "#FFFFFF"
    fg_color = "#E0E0E0" if dark_mode else "#222222"
    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)
    ax.bar(labels, vals, color=bar_colors[:len(labels)], width=0.35, edgecolor='gray')
    ax.set_ylim(0, 1.05)
    ax.set_title("Sentiment Confidence", fontsize=10, color=fg_color, pad=6)
    ax.set_ylabel("Probability", color=fg_color, fontsize=9)
    ax.tick_params(colors=fg_color)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return fig


if 'clf' not in st.session_state:
    df, vec, X_train, y_train_num, _, _ = load_and_preprocess_data()
    clf, tfidf = train_and_save_models(X_train, y_train_num, vec)
    st.session_state.clf = clf
    st.session_state.vec = tfidf

clf = st.session_state.clf
vec = st.session_state.vec

st.title("TalkTective")
st.caption("Developed by Lahari Reddy — Dynamic Text Intelligence Platform")

text_input = st.text_area("Enter text:", placeholder="Type or paste text here...", height=160)
uploaded = st.file_uploader("Upload a text file (.txt):", type=["txt"])
if uploaded:
    text_input = uploaded.read().decode("utf-8", errors="ignore")

st.markdown("---")

row1 = st.columns(4)
with row1[0]:
    sentiment_btn = st.button("Sentiment")
with row1[1]:
    extractive_btn = st.button("Extractive")
with row1[2]:
    abstractive_btn = st.button("Abstractive")
with row1[3]:
    wordcloud_btn = st.button("Word Cloud")

row2 = st.columns(4)
with row2[0]:
    keywords_btn = st.button("Keywords")
with row2[1]:
    topics_btn = st.button("Topics")
with row2[2]:
    insights_btn = st.button("Insights")
with row2[3]:
    pdf_btn = st.button("Download PDF")

st.markdown("<hr>", unsafe_allow_html=True)

if text_input and text_input.strip():
    if sentiment_btn:
        sentiment_probs, top_sent = analyze_sentiment(text_input, vec, clf)
        st.subheader("Sentiment Analysis")
        st.success(f"Predicted Sentiment: {top_sent.upper()}")
        fig = plot_compact_bar(sentiment_probs, dark_mode=is_dark)
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.pyplot(fig, use_container_width=False)

    if extractive_btn:
        st.subheader("Extractive Summary")
        st.info(extractive_reduce(text_input))

    if abstractive_btn:
        st.subheader("Abstractive Summary")
        st.info(abstractive_summarize_text(text_input))

    if wordcloud_btn:
        st.subheader("Word Cloud")
        wc_img = generate_wc_image(text_input, dark_mode=is_dark)
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(wc_img, use_column_width=False, width=500)

    if keywords_btn:
        st.subheader("Key Keywords")
        st.success(', '.join(extract_keywords(text_input)))

    if topics_btn:
        st.subheader("Extracted Topics")
        for t in extract_topics(text_input):
            st.info(t)

    if insights_btn:
        st.subheader("Insights")
        sentiment_probs, top_sent = analyze_sentiment(text_input, vec, clf)
        kw = extract_keywords(text_input)
        topics = extract_topics(text_input)
        recs = generate_recommendations(text_input, top_sent, kw, topics)
        for r in recs:
            st.markdown(f"- {r}")

    if pdf_btn:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = [
            Paragraph("TalkTective - Report", styles["Title"]),
            Spacer(1, 8),
            Paragraph("Original Text:", styles["Heading2"]),
            Paragraph(text_input[:1200] + ("..." if len(text_input) > 1200 else ""), styles["Normal"]),
            Spacer(1, 8)
        ]
        sentiment_probs, top_sent = analyze_sentiment(text_input, vec, clf)
        elements.append(Paragraph("Predicted Sentiment:", styles["Heading2"]))
        elements.append(Paragraph(str(top_sent).upper(), styles["Normal"]))
        elements.append(Spacer(1, 6))
        elements.append(Paragraph("Extractive Summary:", styles["Heading2"]))
        elements.append(Paragraph(extractive_reduce(text_input), styles["Normal"]))
        elements.append(Spacer(1, 6))
        try:
            elements.append(Paragraph("Abstractive Summary:", styles["Heading2"]))
            elements.append(Paragraph(abstractive_summarize_text(text_input), styles["Normal"]))
        except Exception:
            pass
        wc_img = generate_wc_image(text_input, dark_mode=is_dark)
        img_path = "wordcloud_500x300.png"
        wc_img.save(img_path)
        elements.append(RLImage(img_path, width=5.0 * inch, height=3.0 * inch))
        elements.append(Spacer(1, 8))
        kws = extract_keywords(text_input)
        topics = extract_topics(text_input)
        recs = generate_recommendations(text_input, top_sent, kws, topics)
        elements.append(Paragraph("Keywords:", styles["Heading2"]))
        elements.append(Paragraph(', '.join(kws), styles["Normal"]))
        elements.append(Spacer(1, 6))
        elements.append(Paragraph("Topics:", styles["Heading2"]))
        for t in topics:
            elements.append(Paragraph(t, styles["Normal"]))
        elements.append(Spacer(1, 6))
        elements.append(Paragraph("Recommendations:", styles["Heading2"]))
        for r in recs:
            elements.append(Paragraph(r, styles["Normal"]))
        doc.build(elements)
        st.download_button("Save PDF Report", data=buffer.getvalue(), file_name="TalkTective_Report.pdf", mime="application/pdf")
else:
    st.info("Enter text or upload a file to start.")
