import streamlit as st
import pandas as pd
import json
import nltk
import time
import requests
from streamlit_lottie import st_lottie
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer

# --- 1. NLP SETUP ---
@st.cache_resource
def setup_nlp():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('punkt_tab')

setup_nlp()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    return " ".join([lemmatizer.lemmatize(token) for token in tokens if token.isalnum()])

# --- 2. ASSET LOADING ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200: return None
    return r.json()

# Main Robot Animation (Matching the "Next-Gen" vibe)
lottie_main = load_lottieurl("https://lottie.host/8172906e-8360-449e-9988-0320a1630985/B1pU53Y34i.json")

# --- 3. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        with open('faqs.json', 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except FileNotFoundError:
        return pd.DataFrame({"question": ["Hello"], "answer": ["Please upload faqs.json"]})

df = load_data()

# --- 4. UI CONFIGURATION & CUSTOM VOXA THEME ---
st.set_page_config(page_title="Nova Chatterix", layout="wide")

st.markdown("""
    <style>
    /* Background from Image 2: Deep Dark with Cyan Glow */
    .stApp {
        background: radial-gradient(circle at center, #001a1a 0%, #050505 100%);
        color: #ffffff;
    }
    
    /* Header Font: Pixel/Tech style similar to Image 1 */
    @import url('https://fonts.googleapis.com/css2?family=DotGothic16&display=swap');
    
    .hero-text {
        font-family: 'DotGothic16', sans-serif;
        font-size: 5rem !important;
        color: #00FFA3;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 10px;
        margin-bottom: 0px;
        text-shadow: 0 0 20px rgba(0, 255, 163, 0.5);
    }

    /* Circle Line Aesthetic from Image 2 */
    .glow-circle {
        height: 2px;
        background: linear-gradient(90deg, transparent, #00FFA3, transparent);
        margin-bottom: 30px;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #050505 !important;
        border-right: 1px solid rgba(0, 255, 163, 0.3);
    }

    /* Chat Blocks */
    .q-block {
        background: rgba(0, 255, 163, 0.05);
        border: 1px solid rgba(0, 255, 163, 0.2);
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
    }
    .a-block {
        background: rgba(255, 255, 255, 0.02);
        border-left: 3px solid #00FFA3;
        padding: 15px;
        border-radius: 0 10px 10px 0;
        margin-bottom: 15px;
    }

    /* Buttons */
    div.stButton > button {
        background: rgba(0, 255, 163, 0.1) !important;
        color: #00FFA3 !important;
        border: 1px solid #00FFA3 !important;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 5. CHAT ENGINE ---
def get_response(user_input):
    processed_input = preprocess_text(user_input)
    corpus = df['question'].apply(preprocess_text).tolist()
    corpus.append(processed_input)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    idx = similarity_scores.argmax()
    if similarity_scores[0][idx] > 0.2:
        return df.iloc[idx]['answer']
    return "Query not recognized in neural database."

# --- 6. SIDEBAR (Your Name Restored) ---
with st.sidebar:
    st.title("Settings")
    with st.expander("⚙️ System Config"):
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    st.markdown("---")
    st.write("**Developer:** Helly Shah")
    st.write("**Version:** 2.0.1")
    st.markdown('<p style="color:#00FFA3;">● ENGINE: ACTIVE</p>', unsafe_allow_html=True)

# --- 7. MAIN INTERFACE (Landing Scene) ---
# Hero Header based on Image 1
st.markdown('<p class="hero-text">NOVA CHATTERIX</p>', unsafe_allow_html=True)
st.markdown('<div class="glow-circle"></div>', unsafe_allow_html=True)

# Central Robot Animation (Image 2 Style - No Text)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st_lottie(lottie_main, height=300, key="main_bot")

if 'history' not in st.session_state:
    st.session_state.history = []

# Quick Commands
questions = df['question'].tolist()
cols = st.columns(3)
clicked_q = None
for i, q in enumerate(questions[:3]):
    if cols[i].button(q): clicked_q = q

# Input Form
with st.form(key='chat_form', clear_on_submit=True):
    user_query = st.text_input("Signal input:", placeholder="Ask anything...")
    submit = st.form_submit_button("Transmit")

final_query = clicked_q if clicked_q else (user_query if submit else None)

if final_query:
    ans = get_response(final_query)
    st.session_state.history.append({"q": final_query, "a": ans})
    st.rerun()

# History Display
for item in reversed(st.session_state.history):
    st.markdown(f'<div class="q-block">👤 {item["q"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="a-block">🤖 {item["a"]}</div>', unsafe_allow_html=True)
