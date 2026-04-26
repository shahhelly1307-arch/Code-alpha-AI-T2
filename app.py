import streamlit as st
import pandas as pd
import json
import nltk
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

# --- 2. ASSET LOADING (Safety Check Included) ---
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

# Selecting a robot that fits the orbital/next-gen style
lottie_robot = load_lottieurl("https://lottie.host/8172906e-8360-449e-9988-0320a1630985/B1pU53Y34i.json")

# --- 3. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        with open('faqs.json', 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except FileNotFoundError:
        return pd.DataFrame({"question": ["Hello"], "answer": ["Database signal missing. Please upload faqs.json"]})

df = load_data()

# --- 4. UI CONFIGURATION (Exact Aesthetic from Images) ---
st.set_page_config(page_title="Nova Chatterix", layout="wide")

st.markdown("""
    <style>
    /* Background: Deep Obsidian with Center Glow (Image 2) */
    .stApp {
        background: radial-gradient(circle at center, #002b2b 0%, #050505 100%);
        color: #ffffff;
    }
    
    /* Header: Pixel Style (Image 1 - Voxa Style) */
    @import url('https://fonts.googleapis.com/css2?family=DotGothic16&display=swap');
    
    .pixel-header {
        font-family: 'DotGothic16', sans-serif;
        font-size: 5rem;
        color: #00FFA3;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 12px;
        margin-top: 20px;
        margin-bottom: 0px;
        text-shadow: 0 0 25px rgba(0, 255, 163, 0.7);
    }

    /* Half Circle / Orbital Line (Image 2 Style) */
    .orbital-line {
        height: 2px;
        background: linear-gradient(90deg, transparent, #00FFA3, transparent);
        width: 70%;
        margin: 0 auto 30px auto;
        box-shadow: 0 0 15px #00FFA3;
    }

    /* Sidebar Restored */
    [data-testid="stSidebar"] {
        background-color: #050505 !important;
        border-right: 1px solid rgba(0, 255, 163, 0.2);
    }

    /* Chat Elements */
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
        margin-bottom: 20px;
    }
    
    /* Clean Buttons */
    div.stButton > button {
        background: transparent !important;
        color: #00FFA3 !important;
        border: 1px solid #00FFA3 !important;
        width: 100%;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background: #00FFA3 !important;
        color: #000 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 5. LOGIC ENGINE ---
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
    return "Neural buffer empty. Signal not recognized."

# --- 6. SIDEBAR (Your Identity) ---
with st.sidebar:
    st.title("Settings")
    with st.expander("⚙️ System Configuration"):
        if st.button("Clear Neural History"):
            st.session_state.history = []
            st.rerun()
    st.markdown("---")
    st.write("**Developer:** Helly Shah")
    st.write("**Project:** Nova Chatterix")
    st.markdown('<p style="color:#00FFA3;">● SYSTEM: ONLINE</p>', unsafe_allow_html=True)

# --- 7. MAIN INTERFACE (Voxa + Orbital Design) ---
st.markdown('<p class="pixel-header">NOVA CHATTERIX</p>', unsafe_allow_html=True)
st.markdown('<div class="orbital-line"></div>', unsafe_allow_html=True)

# Central Robot (Image 2 Landing Style)
if lottie_robot:
    st_lottie(lottie_robot, height=320, key="hero_bot")

if 'history' not in st.session_state:
    st.session_state.history = []

# Quick Commands
st.markdown("### ⚡ TRANSMIT SIGNALS")
questions = df['question'].tolist()
cols = st.columns(3)
clicked_q = None
for i, q in enumerate(questions[:3]):
    if cols[i].button(q): clicked_q = q

# Form
with st.form(key='chat_form', clear_on_submit=True):
    user_query = st.text_input("Input Signal:", placeholder="Enter command...")
    submit = st.form_submit_button("SEND")

final_query = clicked_q if clicked_q else (user_query if submit else None)

if final_query:
    ans = get_response(final_query)
    st.session_state.history.append({"q": final_query, "a": ans})
    st.rerun()

# History
for item in reversed(st.session_state.history):
    st.markdown(f'<div class="q-block">👤 <b>SIGNAL:</b> {item["q"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="a-block">🤖 <b>NOVA:</b> {item["a"]}</div>', unsafe_allow_html=True)
