# summarization_utils.py
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

# Ensure necessary NLTK downloads
for pkg in ['punkt', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(f'tokenizers/{pkg}' if pkg=='punkt' else f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg, quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# -------------------------
# Text cleaning
# -------------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(words)

# -------------------------
# Extractive summarization (simple)
# -------------------------
def extractive_reduce(text: str, max_sentences=3) -> str:
    from nltk.tokenize import sent_tokenize
    from sklearn.feature_extraction.text import TfidfVectorizer
    sentences = sent_tokenize(text)
    if len(sentences) <= max_sentences:
        return text
    vec = TfidfVectorizer(stop_words='english')
    X = vec.fit_transform(sentences)
    scores = X.sum(axis=1)
    top_idx = np.argsort(scores, axis=0)[-max_sentences:].flatten()
    summary = ' '.join([sentences[i] for i in sorted(top_idx)])
    return summary

# -------------------------
# Abstractive summarization placeholder
# -------------------------
def abstractive_summarize_text(text: str) -> str:
    # Dummy placeholder, replace with HuggingFace summarization if desired
    return extractive_reduce(text)

# -------------------------
# Keyword extraction
# -------------------------
def extract_keywords(text: str, top_n=10):
    vec = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vec.fit_transform([text])
    scores = X.toarray().flatten()
    idx = scores.argsort()[-top_n:][::-1]
    return [vec.get_feature_names_out()[i] for i in idx]

# -------------------------
# Topic modeling
# -------------------------
class TopicModeler:
    def __init__(self, tfidf_vectorizer: TfidfVectorizer, lda_model: LatentDirichletAllocation):
        self.vec = tfidf_vectorizer
        self.lda = lda_model

    def predict_topic(self, text: str, top_words=5):
        cleaned = clean_text(text)
        X = self.vec.transform([cleaned])
        topic_probs = self.lda.transform(X)[0]
        topic_idx = topic_probs.argmax()
        feature_names = self.vec.get_feature_names_out()
        top_words_list = [feature_names[i] for i in self.lda.components_[topic_idx].argsort()[:-top_words-1:-1]]
        return topic_idx + 1, float(topic_probs[topic_idx]), top_words_list

def extract_topics(text: str, top_words=5):
    if 'topic_modeler' in globals():
        return topic_modeler.predict_topic(text, top_words)[2]
    else:
        return ["Topic model not initialized"]

# -------------------------
# Recommendations placeholder
# -------------------------
def generate_recommendations(text: str, sentiment:str=None, keywords=None, topics=None):
    recs = []
    if sentiment == 'positive':
        recs.append("Keep the positive tone.")
    elif sentiment == 'negative':
        recs.append("Consider addressing negative aspects.")
    if keywords:
        recs.append(f"Focus on keywords: {', '.join(keywords[:5])}.")
    if topics:
        recs.append(f"Topic relevance: {', '.join(topics)}.")
    return recs
