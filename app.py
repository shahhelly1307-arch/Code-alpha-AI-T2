import streamlit as st
import pandas as pd
import json
import nltk
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer

# --- 1. NLP SETUP ---
@st.cache_resource
def setup_nlp():
    # Downloads necessary components for the lemmatizer and tokenizer
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('punkt_tab')

setup_nlp()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    return " ".join([lemmatizer.lemmatize(token) for token in tokens if token.isalnum()])

# --- 2. DATA LOADING ---
@st.cache_data
def load_data():
    with open('faqs.json', 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

df = load_data()

# --- 3. UI CONFIGURATION ---
st.set_page_config(page_title="Nova AI Interface", layout="wide")

# Custom CSS for the Cyber-Luxe Theme
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #7b2ff7, #f107a3);
        color: white;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0E1117 !important;
        border-right: 2px solid #00FFA3;
    }
    
    .status-box {
        background-color: rgba(0, 255, 163, 0.1);
        border: 1px solid #00FFA3;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-family: monospace;
        color: #00FFA3;
    }

    .q-block {
        background: rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 5px solid #00FFA3;
    }

    .a-block {
        background: rgba(0, 0, 0, 0.3);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 25px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. THE CHAT ENGINE ---
def get_response(user_input):
    processed_input = preprocess_text(user_input)
    corpus = df['question'].apply(preprocess_text).tolist()
    corpus.append(processed_input)
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Calculate similarity between input and FAQ database
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    idx = similarity_scores.argmax()
    confidence = similarity_scores[0][idx]
    
    if confidence > 0.2:
        return df.iloc[idx]['answer'], confidence
    else:
        return "I'm sorry, I couldn't find a high-confidence match for that query in the MASF framework database.", 0

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("📂 Project Details")
    st.markdown("---")
    st.write("**Framework:** MASF AI")
    st.write("**Developer:** Shrutika")
    st.markdown("---")
    st.markdown('<div class="status-box">SYSTEM: ONLINE</div>', unsafe_allow_html=True)
    st.markdown('<div class="status-box" style="margin-top:10px">ENGINE: TF-IDF V2</div>', unsafe_allow_html=True)

# --- 6. MAIN INTERFACE ---
st.markdown('<h1 style="text-align: center;">🤖 <i>Nova AI Chatbot</i></h1>', unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []

# Quick Command Buttons
st.markdown("### ⚡ QUICK COMMANDS")
questions = df['question'].tolist()
cols = st.columns(len(questions))
for i, q in enumerate(questions):
    if cols[i].button(q):
        ans, conf = get_response(q)
        st.session_state.history.append({"q": q, "a": ans, "c": conf})

# Chat History Display
for item in st.session_state.history:
    st.markdown(f'<div class="q-block"><b>User:</b> {item["q"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="a-block"><b>Nova AI:</b> {item["a"]}</div>', unsafe_allow_html=True)

# User Input Field
with st.form(key='chat_form', clear_on_submit=True):
    user_query = st.text_input("Enter your query regarding seismic risk or the MASF framework:")
    submit = st.form_submit_button("Send Signal")
    
    if submit and user_query:
        ans, conf = get_response(user_query)
        st.session_state.history.append({"q": user_query, "a": ans, "c": conf})
        st.rerun()
