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

# NLTK resources
for pkg in ["punkt", "stopwords", "wordnet"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if pkg == "punkt" else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

lemmatizer = WordNetLemmatizer()
STOPWORDS = set(nltk.corpus.stopwords.words("english"))

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

def split_sentences(text: str) -> List[str]:
    if not text: return []
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return sents or [text.strip()]

def word_tokens(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r"\w+", text)]

def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    t = text.lower().translate(str.maketrans("", "", r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""))
    toks = []
    for w in t.split():
        if not w or w in STOPWORDS: continue
        try:
            toks.append(lemmatizer.lemmatize(w))
        except Exception:
            toks.append(w)
    return " ".join(toks)

def extractive_reduce(text: str, ratio: float = 0.3, min_sentences: int = 1, max_sentences: int = 6) -> str:
    sentences = split_sentences(text)
    if len(sentences) <= 1: return text
    word_freq = {}
    for sent in sentences:
        for w in word_tokens(sent):
            if w in STOPWORDS: continue
            word_freq[w] = word_freq.get(w, 0) + 1
    if not word_freq: return " ".join(sentences[:min_sentences])
    max_freq = max(word_freq.values())
    for k in word_freq: word_freq[k] /= max_freq
    scores = []
    for i, sent in enumerate(sentences):
        words = word_tokens(sent)
        score = sum(word_freq.get(w,0) for w in words)/(len(words)+1e-9) if words else 0
        score *= math.sqrt(len(set(words)))
        scores.append((score,i,sent))
    keep = max(min_sentences, min(max_sentences, math.ceil(len(sentences)*ratio)))
    top = heapq.nlargest(keep, scores, key=lambda x:(x[0],-x[1]))
    top_sorted = sorted(top, key=lambda x:x[1])
    return " ".join(s for (_sc,_i,s) in top_sorted)

# --- Abstractive summarization ---
_TRANSFORMERS_AVAILABLE = False
def try_enable_transformers():
    global _TRANSFORMERS_AVAILABLE
    if _TRANSFORMERS_AVAILABLE: return True, None
    try:
        from transformers import pipeline, AutoTokenizer  # noqa
        import torch
        _TRANSFORMERS_AVAILABLE = True
        return True, None
    except Exception as e:
        return False, f"Transformers unavailable ({str(e)[:200]})"

@st.cache_resource(show_spinner=False)
def make_abstractive_pipeline(model_name: str="t5-small"):
    avail, err = try_enable_transformers()
    if not avail: raise RuntimeError(err)
    from transformers import pipeline
    import torch as _torch
    device = 0 if _torch.cuda.is_available() else -1
    return pipeline("summarization", model=model_name, tokenizer=model_name, device=device)

def trim_for_model(text: str, model_name: str, fraction_of_model_max: float=0.9) -> str:
    avail, err = try_enable_transformers()
    if not avail: return text
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_max = getattr(tokenizer,"model_max_length",512) or 1024
    budget = max(64,int(model_max*fraction_of_model_max))
    sentences = split_sentences(text)
    joined = " ".join(sentences)
    if len(tokenizer.encode(joined, add_special_tokens=False, truncation=False)) <= budget:
        return joined
    trimmed_sents=[]
    cur=0
    for sent in sentences:
        tcount=len(tokenizer.encode(sent, add_special_tokens=False))
        if cur+tcount+2<=budget:
            trimmed_sents.append(sent)
            cur+=tcount
        elif cur==0:
            ids=tokenizer.encode(sent, add_special_tokens=False)[:budget]
            return tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=True)
    return " ".join(trimmed_sents)

def abstractive_summarize_text(text:str, model_name:str="t5-small",
                               max_length:int=120,min_length:int=20,use_reduced:bool=True)->str:
    avail, err = try_enable_transformers()
    if not avail: raise RuntimeError(err)
    reduced = extractive_reduce(text, ratio=0.25, min_sentences=1, max_sentences=6) if use_reduced else text
    trimmed = trim_for_model(reduced, model_name)
    summarizer = make_abstractive_pipeline(model_name)
    out = summarizer(trimmed, max_length=max_length, min_length=min_length, do_sample=False)
    if isinstance(out,list) and out: return out[0].get("summary_text","").strip()
    return str(out)

# --- Keywords ---
def extract_keywords_corpus(corpus:List[str], top_n:int=10)->List[Tuple[str,float]]:
    if not corpus: return []
    vect=TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_df=0.8, min_df=2, max_features=5000)
    X=vect.fit_transform(corpus)
    scores=np.asarray(X.sum(axis=0)).ravel()
    terms=np.array(vect.get_feature_names_out())
    idx=np.argsort(scores)[::-1][:top_n]
    return [(terms[i], float(scores[i])) for i in idx]

# --- Single-topic LDA ---
def extract_top_topic(input_text:str, corpus:List[str], n_top_words:int=8):
    if not corpus or len(corpus)<3: return [],0.0
    count_vect = CountVectorizer(stop_words="english", max_df=0.75, min_df=2, max_features=5000)
    X = count_vect.fit_transform(corpus)
    lda = LatentDirichletAllocation(n_components=min(4,X.shape[0]//2), random_state=42, learning_method="batch")
    lda.fit(X)
    input_vec = count_vect.transform([input_text])
    topic_dist = lda.transform(input_vec)[0]
    top_idx = np.argmax(topic_dist)
    feature_names = count_vect.get_feature_names_out()
    top_words_idx = lda.components_[top_idx].argsort()[:-n_top_words-1:-1]
    topic_words = [feature_names[i] for i in top_words_idx]
    return topic_words, topic_dist[top_idx]

# --- Recommendations ---
def generate_recommendations(text:str, sentiment_label:str, top_keywords:List[str], topics:List[str])->List[str]:
    recs=[]
    low_sent=(sentiment_label or "").lower()
    if "negative" in low_sent: recs.append("⚠️ Negative tone detected — consider triage of complaints.")
    elif "neutral" in low_sent: recs.append("🟡 Neutral tone — consider clarifying messaging.")
    elif "positive" in low_sent: recs.append("✅ Positive feedback — replicate strengths.")
    joined=" ".join([kw.lower() for kw in top_keywords])+ " "+text.lower()
    if any(w in joined for w in ["bug","error","fail","crash"]): recs.append("🐞 Technical issues found — prioritize errors.")
    if any(w in joined for w in ["delay","slow","sluggish","lag"]): recs.append("⏱ Operational slowness observed.")
    if "price" in joined or "cost" in joined: recs.append("💰 Pricing concerns — review communication.")
    if "customer" in joined or "support" in joined: recs.append("💬 Customer-facing improvements recommended.")
    if not recs: recs.append("ℹ️ No strong automated recommendations. Manual review suggested.")
    # remove duplicates
    seen=set(); final=[]
    for r in recs:
        if r not in seen:
            final.append(r); seen.add(r)
    return final
