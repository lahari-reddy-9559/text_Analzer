# app.py
import os, io, string, warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from wordcloud import WordCloud
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

from summarization_utils import (
    clean_text,
    extractive_reduce,
    abstractive_summarize_text,
    extract_keywords_corpus,
    topic_modeling_corpus,
    generate_recommendations,
)

warnings.filterwarnings("ignore")

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
RANDOM_STATE = 42
MAX_FEATURES = 5000
sentiment_mapping = {"negative":0,"neutral":1,"positive":2}
reverse_sentiment_mapping = {v:k for k,v in sentiment_mapping.items()}

st.set_page_config(page_title="TalkTective Studio", layout="wide", page_icon="💬")

# ----------------- Load dataset & vectorizer -----------------
@st.cache_data
def load_kaggle_dataset_and_vect():
    try:
        import kagglehub
        path = kagglehub.dataset_download("abhi8923shriv/sentiment-analysis-dataset")
        df = pd.read_csv(os.path.join(path,"train.csv"), encoding="latin-1")
    except:
        return None, None, None
    df = df.dropna(subset=["text","sentiment"])
    lemmatizer = WordNetLemmatizer()
    stop_words_local = set(stopwords.words("english"))
    def clean_local(t):
        t=str(t).lower().translate(str.maketrans("","",string.punctuation))
        return " ".join([lemmatizer.lemmatize(w) for w in t.split() if w not in stop_words_local])
    df["cleaned_text"]=df["text"].apply(clean_local)
    vec = TfidfVectorizer(max_features=MAX_FEATURES)
    vec.fit(df["cleaned_text"].tolist())
    return df, vec, df["cleaned_text"].tolist()

@st.cache_resource
# Change this function signature
@st.cache_resource
def prepare_or_load_sentiment_model(_vec, df):
    import joblib
    import os
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    MODEL_DIR = "models"
    os.makedirs(MODEL_DIR, exist_ok=True)
    RANDOM_STATE = 42
    clf_path = os.path.join(MODEL_DIR,"rf_sentiment.pkl")
    vec_path = os.path.join(MODEL_DIR,"tfidf.pkl")

    if os.path.exists(clf_path) and os.path.exists(vec_path):
        clf = joblib.load(clf_path)
        vec_loaded = joblib.load(vec_path)
        return clf, vec_loaded

    X = _vec.transform(df["cleaned_text"].tolist())
    y = df["sentiment"].map({"negative":0,"neutral":1,"positive":2}).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train.toarray(), y_train)

    joblib.dump(clf, clf_path)
    joblib.dump(_vec, vec_path)
    return clf, _vec


# ----------------- Utilities -----------------
def analyze_sentiment(text, vec, clf):
    clean_t = clean_text(text)
    if not clean_t:
        return {"negative":0,"neutral":0,"positive":0},"neutral"
    X = vec.transform([clean_t]).toarray()
    probs = clf.predict_proba(X)[0]
    results = {reverse_sentiment_mapping[c]: float(p) for c,p in zip(clf.classes_,probs)}
    top = reverse_sentiment_mapping[int(clf.classes_[np.argmax(probs)])]
    return results, top

def generate_wc_image(text):
    clean_t = clean_text(text)
    if not clean_t:
        return Image.new("RGB",(500,300),"white")
    wc = WordCloud(width=500,height=300,background_color="white",max_words=150).generate(clean_t)
    return wc.to_image()

# ----------------- Load models -----------------
df, vec, corpus = load_kaggle_dataset_and_vect()
if df is None or vec is None:
    st.error("Could not load Kaggle dataset. Check kagglehub.")
    st.stop()
clf, vec = prepare_or_load_sentiment_model(df, vec)

# ----------------- App UI -----------------
st.title("💬 TalkTective")
text_input = st.text_area("Enter text here:",height=180)
uploaded = st.file_uploader("Or upload a text file (.txt):", type=["txt"])
if uploaded:
    try:
        text_input = uploaded.read().decode("utf-8",errors="ignore")
    except:
        text_input = str(uploaded.read())

st.markdown("---")
row1 = st.columns(4)
with row1[0]:
    sentiment_btn = st.button("Sentiment")
with row1[1]:
    extractive_btn = st.button("Extractive Summary")
with row1[2]:
    abstractive_btn = st.button("Abstractive Summary")
with row1[3]:
    wordcloud_btn = st.button("WordCloud")

row2 = st.columns(2)
with row2[0]:
    keywords_btn = st.button("Keywords")
with row2[1]:
    topics_btn = st.button("Top Topic Words")

# ----------------- Main Logic -----------------
if text_input and text_input.strip():
    cleaned_input = clean_text(text_input)

    if sentiment_btn:
        st.subheader("Sentiment Analysis")
        probs, top_sent = analyze_sentiment(text_input, vec, clf)
        st.write(f"Predicted Sentiment: **{top_sent.upper()}**")
    
    if extractive_btn:
        st.subheader("Extractive Summary")
        st.write(extractive_reduce(text_input))

    if abstractive_btn:
        st.subheader("Abstractive Summary")
        try:
            st.write(abstractive_summarize_text(text_input))
        except:
            st.info("Abstractive model not available.")

    if wordcloud_btn:
        st.subheader("WordCloud")
        st.image(generate_wc_image(text_input), width=500)

    if keywords_btn:
        st.subheader("Keywords")
        global_keywords = extract_keywords_corpus(corpus, top_n=8)
        st.write(", ".join([k for k,_ in global_keywords]))

    if topics_btn:
        st.subheader("Top Topic Words")
        top_words, lda_model = topic_modeling_corpus(corpus, n_top_words=8)
        st.write(", ".join(top_words))

    # ----------------- Recommendations -----------------
    if sentiment_btn or keywords_btn or topics_btn:
        st.subheader("Actionable Recommendations")
        top_words, _ = topic_modeling_corpus(corpus, n_top_words=8)
        global_keywords = [k for k,_ in extract_keywords_corpus(corpus, top_n=8)]
        probs, top_sent = analyze_sentiment(text_input, vec, clf)
        recs = generate_recommendations(text_input, top_sent, global_keywords, top_words)
        for r in recs:
            st.info(r)

    # ----------------- PDF Download -----------------
    if st.button("Download Full Report PDF"):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer,pagesize=A4)
        styles = getSampleStyleSheet()
        elems = [Paragraph("TalkTective Report",styles["Title"]),Spacer(1,8)]
        elems.append(Paragraph("Original Text:",styles["Heading2"]))
        elems.append(Paragraph(text_input[:1200]+("..." if len(text_input)>1200 else ""),styles["Normal"]))
        elems.append(Spacer(1,6))
        probs, top_sent = analyze_sentiment(text_input, vec, clf)
        elems.append(Paragraph("Sentiment:",styles["Heading2"]))
        elems.append(Paragraph(top_sent,styles["Normal"]))
        elems.append(Spacer(1,6))
        elems.append(Paragraph("Extractive Summary:",styles["Heading2"]))
        elems.append(Paragraph(extractive_reduce(text_input),styles["Normal"]))
        elems.append(Spacer(1,6))
        try:
            elems.append(Paragraph("Abstractive Summary:",styles["Heading2"]))
            elems.append(Paragraph(abstractive_summarize_text(text_input),styles["Normal"]))
            elems.append(Spacer(1,6))
        except:
            pass
        wc_img = generate_wc_image(text_input)
        img_path = "wc_tmp.png"
        wc_img.save(img_path)
        elems.append(RLImage(img_path,width=5*inch,height=3*inch))
        elems.append(Spacer(1,6))
        elems.append(Paragraph("Keywords:",styles["Heading2"]))
        elems.append(Paragraph(", ".join(global_keywords),styles["Normal"]))
        elems.append(Spacer(1,6))
        elems.append(Paragraph("Top Topic Words:",styles["Heading2"]))
        top_words, _ = topic_modeling_corpus(corpus, n_top_words=8)
        elems.append(Paragraph(", ".join(top_words),styles["Normal"]))
        elems.append(Spacer(1,6))
        recs = generate_recommendations(text_input, top_sent, global_keywords, top_words)
        elems.append(Paragraph("Recommendations:",styles["Heading2"]))
        for r in recs:
            elems.append(Paragraph(r,styles["Normal"]))
        doc.build(elems)
        st.download_button("⬇️ Download PDF", data=buffer.getvalue(), file_name="TalkTective_Report.pdf", mime="application/pdf")
