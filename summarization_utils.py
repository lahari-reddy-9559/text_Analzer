import re
import math
import heapq
import nltk
import numpy as np
from typing import List
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# --- Setup ---
try:
    lemmatizer = WordNetLemmatizer()
    STOPWORDS = set(nltk.corpus.stopwords.words("english"))
except Exception:
    lemmatizer = None
    STOPWORDS = set()

# --- Text Utilities ---
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
    toks = [lemmatizer.lemmatize(w) for w in t.split() if w and w not in STOPWORDS] if lemmatizer else t.split()
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

# --- Abstractive Summarization (requires Transformers) ---
def abstractive_summarize_text(text: str, summarizer=None, max_length: int = 120, min_length: int = 20, use_reduced: bool = True) -> str:
    reduced = extractive_reduce(text, ratio=0.25) if use_reduced else text
    if summarizer is None:
        return reduced
    trimmed = reduced
    out = summarizer(trimmed, max_length=max_length, min_length=min_length, do_sample=False)
    if isinstance(out, list) and out:
        return out[0].get("summary_text", "").strip()
    return str(out)

# --- Keywords ---
def extract_keywords(text: str, top_n: int = 8) -> List[str]:
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform([text])
    scores = zip(tfidf.get_feature_names_out(), np.asarray(X.sum(axis=0)).ravel())
    sorted_keywords = sorted(scores, key=lambda x: x[1], reverse=True)
    return [k for k, _ in sorted_keywords[:top_n]]

# --- Topics using pre-trained LDA ---
def extract_topics(text: str, lda_model, lda_vectorizer, n_words: int = 6) -> str:
    X = lda_vectorizer.transform([text])
    topic_probs = lda_model.transform(X)[0]
    top_topic_idx = topic_probs.argmax()
    feature_names = lda_vectorizer.get_feature_names_out()
    top_words_idx = lda_model.components_[top_topic_idx].argsort()[-n_words:][::-1]
    top_words = [feature_names[i] for i in top_words_idx]
    return ", ".join(top_words)

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
