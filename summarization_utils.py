# summarization_utils.py
import re, string, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from heapq import nlargest

from typing import List

try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("corpora/stopwords")
except:
    nltk.download("stopwords", quiet=True)
try:
    nltk.data.find("corpora/wordnet")
except:
    nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ----------------- Text cleaning -----------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

# ----------------- Extractive summarization -----------------
def extractive_reduce(text: str, ratio: float = 0.25, min_sentences: int = 1, max_sentences: int = 6) -> str:
    from nltk.tokenize import sent_tokenize, word_tokenize
    sentences = sent_tokenize(text)
    if not sentences:
        return ""
    word_freq = {}
    for word in word_tokenize(text.lower()):
        if word.isalpha() and word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    ranked_sentences = sorted(sentences, key=lambda s: sum(word_freq.get(w,0) for w in word_tokenize(s.lower())), reverse=True)
    num_sentences = max(min_sentences, min(max_sentences, int(len(sentences)*ratio)))
    return " ".join(ranked_sentences[:num_sentences])

# ----------------- Abstractive summarization -----------------
def abstractive_summarize_text(text: str, model_name: str = "t5-small") -> str:
    if not text.strip():
        return ""
    from transformers import pipeline
    summarizer = pipeline("summarization", model=model_name, truncation=True)
    summary = summarizer(text, max_length=120, min_length=30, do_sample=False)
    return summary[0]["summary_text"]

# ----------------- Keyword extraction -----------------
def extract_keywords_corpus(corpus: List[str], top_n: int = 10):
    cleaned = [clean_text(t) for t in corpus]
    if not cleaned:
        return []
    vect = TfidfVectorizer(max_features=5000)
    X = vect.fit_transform(cleaned)
    scores = X.sum(axis=0).A1
    features = vect.get_feature_names_out()
    top = sorted(zip(features, scores), key=lambda x: x[1], reverse=True)
    return top[:top_n]

# ----------------- LDA topic modeling (single top topic) -----------------
def topic_modeling_corpus(corpus: List[str], n_top_words: int = 8):
    cleaned = [clean_text(t) for t in corpus if t.strip()]
    if not cleaned:
        return [], None
    vect = CountVectorizer(max_features=5000)
    X = vect.fit_transform(cleaned)
    lda = LatentDirichletAllocation(n_components=1, random_state=42)
    lda.fit(X)
    feature_names = vect.get_feature_names_out()
    comp = lda.components_[0]
    top_idx = comp.argsort()[:-n_top_words-1:-1]
    top_words = [feature_names[i] for i in top_idx]
    return top_words, lda

# ----------------- Recommendations -----------------
def generate_recommendations(text: str, sentiment: str, keywords: List[str], topic_words: List[str]):
    recs = []
    sentiment = sentiment.lower()
    if sentiment=="negative":
        recs.append("⚠️ Negative tone detected — consider addressing top complaints.")
    elif sentiment=="positive":
        recs.append("✅ Positive feedback — maintain strengths.")
    else:
        recs.append("🟡 Neutral tone — clarify key points if needed.")
    if keywords:
        recs.append(f"Focus on these keywords: {', '.join(keywords[:5])}")
    if topic_words:
        recs.append(f"Top topic words: {', '.join(topic_words)}")
    return recs
