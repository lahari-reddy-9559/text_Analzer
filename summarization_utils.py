# summarization_utils.py
import re
import math
import heapq
import nltk
import streamlit as st
import numpy as np
from typing import List, Tuple
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Ensure common NLTK resources are available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

# Setup
try:
    lemmatizer = WordNetLemmatizer()
    STOPWORDS = set(nltk.corpus.stopwords.words("english"))
except Exception:
    lemmatizer = None
    STOPWORDS = set()

_TRANSFORMERS_AVAILABLE = False


def try_enable_transformers():
    """Return (available, error_message_or_none)."""
    global _TRANSFORMERS_AVAILABLE
    if _TRANSFORMERS_AVAILABLE:
        return True, None
    try:
        from transformers import pipeline, AutoTokenizer  # noqa: F401
        import torch  # noqa: F401
        _TRANSFORMERS_AVAILABLE = True
        return True, None
    except Exception as e:
        _TRANSFORMERS_AVAILABLE = False
        return False, f"Transformers unavailable ({str(e)[:200]})"


# --- Core utilities ---
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return sents or [text.strip()]


def word_tokens(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r"\w+", text)]


def clean_text(text: str) -> str:
    """Normalize, remove punctuation, lemmatize and remove stopwords (if available)."""
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = t.translate(str.maketrans("", "", r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""))
    toks = []
    for w in t.split():
        if not w or w in STOPWORDS:
            continue
        if lemmatizer:
            try:
                toks.append(lemmatizer.lemmatize(w))
            except Exception:
                toks.append(w)
        else:
            toks.append(w)
    return " ".join(toks)


# --- Improved extractive summarization ---
def extractive_reduce(text: str, ratio: float = 0.3, min_sentences: int = 1, max_sentences: int = 6) -> str:
    """Selects most-informative sentences using normalized token frequency weighting."""
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return text

    # Build frequency excluding stopwords
    word_freq = {}
    for sent in sentences:
        for w in word_tokens(sent):
            if w in STOPWORDS:
                continue
            word_freq[w] = word_freq.get(w, 0) + 1

    if not word_freq:
        return " ".join(sentences[:min_sentences])

    # Normalize frequency (TF-like)
    max_freq = max(word_freq.values())
    for k in list(word_freq.keys()):
        word_freq[k] = word_freq[k] / float(max_freq)

    # Score sentences by average normalized token score, with light favoring of unique tokens
    scores = []
    for i, sent in enumerate(sentences):
        words = word_tokens(sent)
        if not words:
            score = 0.0
        else:
            score = sum(word_freq.get(w, 0.0) for w in words) / (len(words) + 1e-9)
            score = score * math.sqrt(len(set(words)))  # slight boost for sentence lexical variety
        scores.append((score, i, sent))

    keep = max(min_sentences, min(max_sentences, math.ceil(len(sentences) * ratio)))
    top = heapq.nlargest(keep, scores, key=lambda x: (x[0], -x[1]))
    top_sorted = sorted(top, key=lambda x: x[1])
    return " ".join(s for (_sc, _i, s) in top_sorted)


# --- Abstractive summarization helpers (optional) ---
@st.cache_resource(show_spinner=False)
def make_abstractive_pipeline(model_name: str = "t5-small"):
    avail, err = try_enable_transformers()
    if not avail:
        raise RuntimeError(err or "transformers not available")
    from transformers import pipeline
    import torch as _torch
    device = 0 if _torch.cuda.is_available() else -1
    return pipeline("summarization", model=model_name, tokenizer=model_name, device=device)


def trim_for_model(text: str, model_name: str, fraction_of_model_max: float = 0.9) -> str:
    avail, err = try_enable_transformers()
    if not avail:
        return text
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_max = getattr(tokenizer, "model_max_length", 512) or 1024
    budget = max(64, int(model_max * fraction_of_model_max))
    sentences = split_sentences(text)
    if not sentences:
        return text

    def token_count(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=False, truncation=False))

    joined = " ".join(sentences)
    if token_count(joined) <= budget:
        return joined

    trimmed_sents = []
    cur = 0
    for sent in sentences:
        tcount = token_count(sent)
        if cur + tcount + 2 <= budget:
            trimmed_sents.append(sent)
            cur += tcount
        elif cur == 0:
            ids = tokenizer.encode(sent, add_special_tokens=False)[:budget]
            return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return " ".join(trimmed_sents)


def abstractive_summarize_text(text: str, model_name: str = "t5-small",
                               max_length: int = 120, min_length: int = 20, use_reduced: bool = True) -> str:
    avail, err = try_enable_transformers()
    if not avail:
        raise RuntimeError(err or "transformers not available")
    reduced = extractive_reduce(text, ratio=0.25, min_sentences=1, max_sentences=6) if use_reduced else text
    trimmed = trim_for_model(reduced, model_name)
    summarizer = make_abstractive_pipeline(model_name)
    out = summarizer(trimmed, max_length=max_length, min_length=min_length, do_sample=False)
    if isinstance(out, list) and out:
        return out[0].get("summary_text", "").strip()
    return str(out)


# --- Keywords and topic modeling (improved to avoid duplicates) ---
def extract_keywords_corpus(corpus: List[str], top_n: int = 10) -> List[Tuple[str, float]]:
    """
    Extract top keywords from corpus using TF-IDF. Returns list of (term, score).
    Use corpus (list of documents) for robust global keywords.
    """
    if not corpus:
        return []
    vect = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_df=0.8, min_df=2, max_features=5000)
    X = vect.fit_transform(corpus)
    scores = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(vect.get_feature_names_out())
    idx = np.argsort(scores)[::-1][:top_n]
    return [(terms[i], float(scores[i])) for i in idx]


def topic_modeling_corpus(corpus: List[str], n_topics: int = 4, n_top_words: int = 8):
    """
    Robust topic modeling over a corpus (list of docs).
    Uses CountVectorizer with min_df/max_df to avoid extremely common terms.
    Returns (lda_model, feature_names, topic_word_lists, doc_topic_matrix)
    """
    if not corpus or len(corpus) < 3:
        return None, [], [], np.array([])

    count_vect = CountVectorizer(stop_words="english", max_df=0.75, min_df=2, max_features=5000)
    X = count_vect.fit_transform(corpus)
    lda = LatentDirichletAllocation(n_components=min(n_topics, max(1, X.shape[0] // 2)), random_state=42, learning_method="batch")
    doc_topic = lda.fit_transform(X)
    feature_names = count_vect.get_feature_names_out()
    topic_words = []
    for comp in lda.components_:
        top_idx = comp.argsort()[:-n_top_words - 1:-1]
        topic_words.append([feature_names[i] for i in top_idx])
    return lda, feature_names, topic_words, doc_topic


def extract_topics_for_input(input_text: str, corpus: List[str], n_topics: int = 4, n_top_words: int = 8):
    """
    Build LDA over the main corpus (e.g., Kaggle dataset) and then infer which topics the input_text
    most strongly belongs to (gives topic proportions + top words per topic).
    Returns (topic_words, input_topic_dist)
    """
    if not corpus:
        return [], []

    lda, feature_names, topic_words, doc_topic = topic_modeling_corpus(corpus, n_topics=n_topics, n_top_words=n_top_words)
    if lda is None:
        return [], []

    # Build same count vectorizer to transform input
    count_vect = CountVectorizer(stop_words="english", max_df=0.75, min_df=2, max_features=5000)
    count_vect.fit(corpus)  # fit on corpus
    input_vec = count_vect.transform([input_text])
    input_topic_dist = lda.transform(input_vec)  # (1, n_topics)
    return topic_words, list(input_topic_dist[0])


# --- Actionable recommendations (simple rule-engine) ---
def generate_recommendations(text: str, sentiment_label: str, top_keywords: List[str], topics: List[str]) -> List[str]:
    """
    Produce human-readable recommendations based on signals in text + sentiment.
    top_keywords is a list of strings, topics is list of strings (topic word lists).
    """
    recs = []
    low_sent = (sentiment_label or "").lower()
    if "negative" in low_sent:
        recs.append("⚠️ Negative tone detected — consider a triage of top complaints.")
    elif "neutral" in low_sent:
        recs.append("🟡 Neutral tone — consider clarifying messaging or improving engagement.")
    elif "positive" in low_sent:
        recs.append("✅ Positive feedback — identify strengths to replicate.")

    # keyword-based heuristics
    joined = " ".join([kw.lower() for kw in top_keywords]) + " " + text.lower()
    if any(w in joined for w in ["bug", "error", "fail", "crash"]):
        recs.append("🐞 Technical issues found — reproduce and prioritize the most frequent errors.")
    if any(w in joined for w in ["delay", "slow", "sluggish", "lag"]):
        recs.append("⏱ Operational slowness observed — investigate performance and process bottlenecks.")
    if "price" in joined or "cost" in joined:
        recs.append("💰 Pricing concerns — consider pricing review or clearer communication.")
    if "customer" in joined or "support" in joined:
        recs.append("💬 Customer-facing improvements recommended (support, docs, communication).")

    if not recs:
        recs.append("ℹ️ No strong automated recommendations. Consider manual review or domain-specific rules.")

    # remove duplicates, preserve order
    seen = set()
    final = []
    for r in recs:
        if r not in seen:
            final.append(r)
            seen.add(r)
    return final
