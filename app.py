# app.py — TalkTective Studio | Lahari Reddy

import os, io, string, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
import joblib
import kagglehub
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import LatentDirichletAllocation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

from summarization_utils import (
    clean_text as clean_text_util,
    extractive_reduce,
    abstractive_summarize_text,
    extract_keywords,
    extract_topics,
    generate_recommendations,
)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="TalkTective Studio | Lahari Reddy",
    layout="wide",
    page_icon="💬"
)

# Visual theme
st.markdown("""
<style>
body { background: linear-gradient(135deg, #FDEFF9 0%, #ECF4FF 50%, #E8F9F0 100%);
       font-family: 'Poppins', sans-serif; }
div.block-container { padding-top:1.6rem; background-color: rgba(255,255,255,0.94);
       border-radius:14px; padding:20px 24px; box-shadow:0px 4px 20px rgba(0,0,0,0.06); }
h1,h2,h3 { color:#4B0082; font-weight:600; }
.stButton>button { background: linear-gradient(90deg, #6C63FF, #00BFA6);
       color:white; border:none; border-radius:8px; font-weight:600; padding:0.5em 1.0em; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Load & preprocess dataset
# -----------------------
@st.cache_data(show_spinner="📦 Loading dataset & models...")
def load_and_preprocess_data():
    path = kagglehub.dataset_download("abhi8923shriv/sentiment-analysis-dataset")
    df = pd.read_csv(os.path.join(path, 'train.csv'), encoding='latin-1')
    df.dropna(subset=['text', 'selected_text'], inplace=True)

    # NLTK
    for pkg in ['punkt', 'stopwords', 'wordnet']:
        try: nltk.data.find(f'tokenizers/{pkg}' if pkg=='punkt' else f'corpora/{pkg}')
        except LookupError: nltk.download(pkg, quiet=True)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def clean_local(t):
        t = str(t).lower().translate(str.maketrans('', '', string.punctuation))
        w = [lemmatizer.lemmatize(x) for x in t.split() if x not in stop_words]
        return ' '.join(w)

    df['cleaned_text'] = df['text'].apply(clean_local)
    vec = TfidfVectorizer(max_features=5000)
    vec.fit(df['cleaned_text'])

    X = vec.transform(df['cleaned_text'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment']
    )
    return df, vec, X_train, y_train

# -----------------------
# Train models
# -----------------------
@st.cache_resource
def train_models(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train.toarray(), y_train)
    lda = LatentDirichletAllocation(n_components=5, random_state=42, learning_method='batch', max_iter=10)
    lda.fit(X_train)
    return clf, lda

# Load data and models
df, tfidf_vectorizer, X_train, y_train = load_and_preprocess_data()
clf, lda_model = train_models(X_train, y_train)

sentiment_mapping = {'negative':0, 'neutral':1, 'positive':2}
reverse_sentiment_mapping = {v:k for k,v in sentiment_mapping.items()}

# -----------------------
# Helpers
# -----------------------
def analyze_sentiment(text):
    clean_t = clean_text_util(text)
    X = tfidf_vectorizer.transform([clean_t]).toarray()
    probs = clf.predict_proba(X)[0]
    results = {reverse_sentiment_mapping[c]: float(p) for c,p in zip(clf.classes_, probs)}
    top = reverse_sentiment_mapping[clf.classes_[np.argmax(probs)]]
    return results, top

def plot_sentiment_bar(sentiment_dict):
    labels = list(sentiment_dict.keys())
    vals = [sentiment_dict[l] for l in labels]
    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar(labels, vals, color=['#F87171','#FACC15','#34D399'])
    ax.set_ylim(0,1)
    ax.set_title("Sentiment Confidence")
    return fig

def plot_topic_probs(text):
    X = tfidf_vectorizer.transform([clean_text_util(text)])
    topic_probs = lda_model.transform(X)[0]
    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar([f"Topic {i+1}" for i in range(len(topic_probs))], topic_probs, color='skyblue')
    ax.set_ylim(0,1)
    ax.set_title("Topic Probabilities")
    top_topic = np.argmax(topic_probs)+1
    confidence = topic_probs[top_topic-1]
    return fig, top_topic, confidence

# -----------------------
# UI
# -----------------------
st.title("💬 TalkTective")
text_input = st.text_area("📝 Enter text:", placeholder="Type or paste text...", height=160)
uploaded = st.file_uploader("📄 Upload a text file (.txt):", type=["txt"])
if uploaded:
    text_input = uploaded.read().decode("utf-8", errors="ignore")

st.markdown("---")
cols = st.columns(4)
choice = None
buttons = [("🧠 Sentiment Analysis","sentiment"),
           ("✂️ Extractive Summary","extractive"),
           ("🪶 Abstractive Summary","abstractive"),
           ("☁️ Word Cloud","wordcloud"),
           ("🧩 Keywords","keywords"),
           ("📊 Topics","topics"),
           ("🎯 Insights","insights")]

for (label,val),col in zip(buttons, cols + [cols[-1]]*(len(buttons)-len(cols))):
    with col:
        if st.button(label):
            choice = val
st.markdown("---")

if text_input and text_input.strip():
    if choice=="sentiment":
        st.subheader("🧠 Sentiment Analysis")
        sentiment_probs, top_sent = analyze_sentiment(text_input)
        st.success(f"Predicted Sentiment: **{top_sent.upper()}**")
        st.pyplot(plot_sentiment_bar(sentiment_probs))
    elif choice=="extractive":
        st.subheader("✂️ Extractive Summary")
        st.info(extractive_reduce(text_input))
    elif choice=="abstractive":
        st.subheader("🪶 Abstractive Summary")
        try:
            st.info(abstractive_summarize_text(text_input))
        except Exception as e:
            st.error(f"Abstractive error: {e}")
    elif choice=="wordcloud":
        st.subheader("☁️ Word Cloud")
        from wordcloud import WordCloud
        wc_img = WordCloud(width=500, height=300, background_color='white').generate(text_input)
        st.image(wc_img.to_image(), width=500)
    elif choice=="keywords":
        st.subheader("🧩 Keywords")
        st.success(', '.join(extract_keywords(text_input)))
    elif choice=="topics":
        st.subheader("📊 Topic Modeling")
        fig, top_topic, confidence = plot_topic_probs(text_input)
        st.pyplot(fig)
        st.info(f"Top Topic: Topic {top_topic} (Confidence: {confidence:.2f})")
    elif choice=="insights":
        st.subheader("🎯 Insights")
        sentiment_probs, top_sent = analyze_sentiment(text_input)
        kws = extract_keywords(text_input)
        recs = generate_recommendations(text_input, top_sent, kws, extract_topics(text_input))
        for r in recs:
            st.markdown(f"- {r}")
else:
    st.info("💡 Enter text or upload a file to start analysis.")
