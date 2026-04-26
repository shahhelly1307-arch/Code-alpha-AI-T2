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

# --- 2. ASSET LOADING ---
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

lottie_robot = load_lottieurl("https://lottie.host/8172906e-8360-449e-9988-0320a1630985/B1pU53Y34i.json")

# --- 3. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        with open('faqs.json', 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except:
        return pd.DataFrame({"question": ["Who developed this?"], "answer": ["Developed by Helly Shah."]})

df = load_data()

# --- 4. UI CONFIGURATION (IMAGE 1 & 2 ACCURACY) ---
st.set_page_config(page_title="Nova Chatterix", layout="wide")

st.markdown("""
    <style>
    /* VT323 is more uniform and thicker for pixel styles */
    @import url('https://fonts.googleapis.com/css2?family=VT323&display=swap');

    /* Background: Deep Obsidian with a centered Cyan Aura */
    .stApp {
        background: radial-gradient(circle at center, #005a5a 0%, #020b0b 40%, #000000 100%) !important;
        color: #ffffff;
    }
    
    /* THE FONT: Uniform, Extra Thick, Single Line */
    .voxa-style-title {
        font-family: 'VT323', monospace !important;
        font-size: clamp(3rem, 10vw, 7.5rem) !important; 
        font-weight: 900 !important;   /* Maximum thickness */
        color: #00FFA3 !important;      /* VOXA Cyan */
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 10px;
        margin: 0px;
        white-space: nowrap;            /* Single line focus */
        text-shadow: 
            0 0 10px #00FFA3, 
            0 0 30px rgba(0, 255, 163, 0.6),
            0 0 50px rgba(0, 255, 163, 0.3); 
        line-height: 1;
    }

    /* Orbital Line */
    .orbital-line {
        height: 3px;
        background: linear-gradient(90deg, transparent, #00FFA3, transparent);
        width: 80%;
        margin: 5px auto 30px auto;
        box-shadow: 0 0 15px #00FFA3;
    }

    /* UI Consistency */
    div.stButton > button {
        background: rgba(0, 255, 163, 0.05) !important;
        color: #00FFA3 !important;
        border: 2px solid #00FFA3 !important;
        font-family: 'VT323', monospace !important;
        font-size: 1.5rem !important;
    }
    div.stButton > button:hover {
        background: #00FFA3 !important;
        color: #000 !important;
        box-shadow: 0 0 20px #00FFA3;
    }

    .stTextInput input {
        background-color: rgba(0, 0, 0, 0.5) !important;
        border: 1px solid #00FFA3 !important;
        color: #ffffff !important;
        font-family: 'VT323', monospace !important;
        font-size: 1.4rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 5. LOGIC ENGINE ---
def get_response(user_input):
    if "developer" in user_input.lower() or "who developed" in user_input.lower():
        return "This project was developed by Helly Shah as a technical demonstration of NLP and professional UI integration."
    
    processed_input = preprocess_text(user_input)
    corpus = df['question'].apply(preprocess_text).tolist()
    corpus.append(processed_input)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    idx = similarity_scores.argmax()
    if similarity_scores[0][idx] > 0.2:
        return df.iloc[idx]['answer']
    return "Neural signal weak. Command not recognized."

# --- 6. SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='color:#00FFA3; font-family:VT323;'>SYSTEM CONFIG</h2>", unsafe_allow_html=True)
    if st.button("CLEAR LOGS"):
        st.session_state.history = []
        st.rerun()
    st.markdown("---")
    st.write("**Operator:** Helly Shah")
    st.write("**Core:** Nova Chatterix")
    st.markdown('<p style="color:#00FFA3;">● LINK: ACTIVE</p>', unsafe_allow_html=True)

# --- 7. MAIN INTERFACE ---
# UNIFORM THICK HEADER
st.markdown('<p class="voxa-style-title">NOVA CHATTERIX</p>', unsafe_allow_html=True)
st.markdown('<div class="orbital-line"></div>', unsafe_allow_html=True)

# Robot Hero with center aura
if lottie_robot:
    st_lottie(lottie_robot, height=350, key="hero_bot")

if 'history' not in st.session_state:
    st.session_state.history = []

# Action Signals
st.markdown("<h3 style='color:white; font-family:VT323;'>⚡ TRANSMIT SIGNALS</h3>", unsafe_allow_html=True)
questions = df['question'].tolist()
cols = st.columns(3)
clicked_q = None
for i, q in enumerate(questions[:6]):
    if cols[i % 3].button(q, key=f"q_{i}"):
        clicked_q = q

# Input
with st.form(key='chat_form', clear_on_submit=True):
    user_query = st.text_input("Enter Command:", placeholder="Transmitting...")
    submit = st.form_submit_button("SEND")

final_query = clicked_q if clicked_q else (user_query if submit else None)

if final_query:
    ans = get_response(final_query)
    st.session_state.history.append({"q": final_query, "a": ans})
    st.rerun()

# History
for item in reversed(st.session_state.history):
    st.markdown(f'''
    <div style="background:rgba(0, 255, 163, 0.03); border-left:5px solid #00FFA3; padding:20px; margin-bottom:15px; font-family:VT323; font-size:1.3rem;">
        <b style="color:#00FFA3">SIGNAL:</b> {item["q"]}<br><br>
        <b style="color:white">NOVA:</b> {item["a"]}
    </div>
    ''', unsafe_allow_html=True)
