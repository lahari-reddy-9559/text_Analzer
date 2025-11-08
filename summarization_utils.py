# summarization_utils.py
import re
import math
import heapq
import nltk
import numpy as np
from typing import List
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import joblib
import os
import streamlit as st

# --- Setup ---
lemmatizer = WordNetLemmatizer()
STOPWORDS = set(nltk.corpus.stopwords.words("english"))

# LDA & TFIDF paths
MODEL_DIR = "models"
LDA_MODEL_PATH = os.path.join(MODEL_DIR, "lda_model.pkl")
LDA_VECTORIZER_PATH = os.path.join(MODEL_DIR, "lda_tfidf.pkl")

# --- Core NLP Utilities ---
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

def split_sentences(text: str) -> List[str]:
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return sents or [text.strip()]

def word_tokens(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r"\w+", text)]

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = t.translate(str.maketrans("", "", r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""))
    toks = [lemmatizer.lemmatize(w) for w in t.split() if w not in STOPWORDS]
    return " ".join(toks)

# --- Extractive Summarization ---
def extractive_reduce(text: str, ratio: float = 0.3, min_sentences: int = 1, max_sentences: int = 6) -> str:
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return text
    word_freq = {}
    for sent in sentences:
        for w in word_tokens(sent):
            if w not in STOPWORDS:
                word_freq[w] = word_freq.get(w, 0) + 1
    if not word_freq:
        return " ".join(sentences[:min_sentences])
    max_freq = max(word_freq.values())
    for w in word_freq:
        word_freq[w] /= max_freq
    scores = []
    for i, sent in enumerate(sentences):
        words = word_tokens(sent)
        score = sum(word_freq.get(w, 0) for w in words) / (len(words) + 1e-6)
        scores.append((score, i, sent))
    keep = max(min_sentences, min(max_sentences, math.ceil(len(sentences) * ratio)))
    top = heapq.nlargest(keep, scores, key=lambda x: (x[0], -x[1]))
    top_sorted = sorted(top, key=lambda x: x[1])
    return " ".join(s for (_score, _i, s) in top_sorted)

# --- Abstractive Summarization ---
@st.cache_resource(show_spinner=False)
def make_abstractive_pipeline(model_name: str = "t5-small"):
    try:
        from transformers import pipeline, AutoTokenizer
        import torch
        device = 0 if torch.cuda.is_available() else -1
        return pipeline("summarization", model=model_name, tokenizer=model_name, device=device)
    except Exception as e:
        raise RuntimeError(f"Transformers unavailable: {e}")

def abstractive_summarize_text(text: str, model_name: str = "t5-small",
                               max_length: int = 120, min_length: int = 20,
                               use_reduced: bool = True) -> str:
    try:
        reduced = extractive_reduce(text, ratio=0.25) if use_reduced else text
        summarizer = make_abstractive_pipeline(model_name)
        out = summarizer(reduced, max_length=max_length, min_length=min_length, do_sample=False)
        if isinstance(out, list) and out:
            return out[0].get("summary_text", "").strip()
        return str(out)
    except Exception:
        return "Abstractive summarization unavailable (check Transformers)."

# --- Keywords Extraction ---
def extract_keywords(text: str, top_n: int = 8) -> List[str]:
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform([text])
    scores = zip(tfidf.get_feature_names_out(), np.asarray(X.sum(axis=0)).ravel())
    sorted_keywords = sorted(scores, key=lambda x: x[1], reverse=True)
    return [k for k, _ in sorted_keywords[:top_n]]

# --- Pretrained LDA Topic Model ---
def load_or_train_lda(corpus_texts: List[str], n_topics: int = 5, n_words: int = 6):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    if os.path.exists(LDA_MODEL_PATH) and os.path.exists(LDA_VECTORIZER_PATH):
        lda_model = joblib.load(LDA_MODEL_PATH)
        lda_vectorizer = joblib.load(LDA_VECTORIZER_PATH)
    else:
        lda_vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
        X = lda_vectorizer.fit_transform(corpus_texts)
        lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=15)
        lda_model.fit(X)
        joblib.dump(lda_model, LDA_MODEL_PATH)
        joblib.dump(lda_vectorizer, LDA_VECTORIZER_PATH)
    return lda_model, lda_vectorizer

def extract_topics(text: str, lda_model=None, lda_vectorizer=None, n_words: int = 6) -> List[str]:
    if lda_model is None or lda_vectorizer is None:
        return ["No LDA model loaded"]
    X = lda_vectorizer.transform([text])
    topic_probs = lda_model.transform(X)[0]
    top_topic_idx = topic_probs.argmax()
    feature_names = lda_vectorizer.get_feature_names_out()
    topic_words_idx = lda_model.components_[top_topic_idx].argsort()[-n_words:][::-1]
    top_words = [feature_names[i] for i in topic_words_idx]
    return top_words

# --- Recommendations ---
def generate_recommendations(text: str, sentiment_label: str, keywords: List[str], topics: List[str]) -> List[str]:
    recs = []
    low_sent = sentiment_label.lower()
    if "negative" in low_sent:
        recs.append("⚠️ Text indicates dissatisfaction — consider deeper root-cause analysis.")
    elif "neutral" in low_sent:
        recs.append("🟡 Neutral tone — possible lack of engagement or clarity.")
    elif "positive" in low_sent:
        recs.append("✅ Positive insights — maintain and amplify these strengths.")
    if any(word in text.lower() for word in ["delay", "slow", "issue", "problem"]):
        recs.append("📊 Operational delays detected — optimize workflow or communication.")
    if "customer" in text.lower():
        recs.append("💬 Customer-focused improvement recommended.")
    if not recs:
        recs.append("ℹ️ No specific action detected — consider contextual review.")
    return recs
