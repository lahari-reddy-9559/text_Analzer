import re
import math
import heapq
import nltk
import warnings
import pandas as pd
from nltk.stem import WordNetLemmatizer
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from wordcloud import WordCloud
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# --- Setup ---
try:
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    STOPWORDS = set(nltk.corpus.stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
except Exception:
    STOPWORDS = set()
    lemmatizer = None

# Regex for sentence splitting
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

def split_sentences(text: str) -> List[str]:
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return sents or [text.strip()]

def word_tokens(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r"\w+", text)]

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = text.split()
    if lemmatizer:
        tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in STOPWORDS]
    return " ".join(tokens)

# --- Extractive Summarization ---
def extractive_reduce(text: str, ratio: float = 0.3, min_sentences: int = 1, max_sentences: int = 6) -> str:
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return text
    freq = {}
    for sent in sentences:
        for w in word_tokens(sent):
            freq[w] = freq.get(w, 0) + 1
    scores = []
    for i, sent in enumerate(sentences):
        s = sum(freq.get(w, 0) for w in word_tokens(sent))
        scores.append((s, i, sent))
    keep = max(min_sentences, min(max_sentences, math.ceil(len(sentences) * ratio)))
    top = heapq.nlargest(keep, scores, key=lambda x: (x[0], -x[1]))
    top_sorted = sorted(top, key=lambda x: x[1])
    return " ".join([s for (_score, _i, s) in top_sorted])

# --- Topic Extraction ---
def extract_topics(texts, n_topics=3, n_words=6):
    cleaned_texts = [clean_text(t) for t in texts if isinstance(t, str) and t.strip()]
    if not cleaned_texts:
        return []
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
    X = vectorizer.fit_transform(cleaned_texts)
    nmf = NMF(n_components=n_topics, random_state=42)
    W = nmf.fit_transform(X)
    H = nmf.components_
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for i, topic_vec in enumerate(H):
        top_words = [feature_names[j] for j in topic_vec.argsort()[:-n_words - 1:-1]]
        topics.append(top_words)
    return topics

# --- Visualization ---
def plot_wordcloud(text):
    if not isinstance(text, str) or not text.strip():
        return None
    wc = WordCloud(width=600, height=400, background_color="white").generate(text)
    plt.figure(figsize=(6, 4))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    return plt
