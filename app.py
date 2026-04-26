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
        # Fallback if file is missing
        return pd.DataFrame({"question": ["Developer"], "answer": ["Developed by Helly Shah."]})

df = load_data()

# --- 4. UI CONFIGURATION (VOXA PIXEL MATCH) ---
st.set_page_config(page_title="Nova Chatterix", layout="wide")

st.markdown("""
    <style>
    /* Import Silkscreen: Every letter is a blocky pixel */
    @import url('https://fonts.googleapis.com/css2?family=Silkscreen:wght@700&display=swap');

    /* Background: Deep Center-Cyan Glow */
    .stApp {
        background: radial-gradient(circle at center, #004d4d 0%, #001a1a 40%, #050505 100%) !important;
        color: #ffffff;
    }
    
    /* THE FIX: Force thickness on H, T, E, I */
    .voxa-header {
        font-family: 'Silkscreen', cursive !important;
        font-size: clamp(2.5rem, 8vw, 5rem) !important;
        font-weight: 700 !important;
        color: #00FFA3 !important;      /* EXACT VOXA CYAN */
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 12px;
        margin: 40px 0 10px 0;
        white-space: nowrap;            /* Single line focus */
        
        /* Forces blocky thickness on thin letters */
        -webkit-text-stroke: 2px #00FFA3; 
        text-shadow: 
            3px 3px 0px rgba(0, 0, 0, 0.7),
            0 0 20px rgba(0, 255, 163, 0.8),
            0 0 40px rgba(0, 255, 163, 0.4);
    }

    /* Glow Divider */
    .glow-line {
        height: 4px;
        background: linear-gradient(90deg, transparent, #00FFA3, #00FFA3, transparent);
        width: 80%;
        margin: 0 auto 40px auto;
        box-shadow: 0 0 25px #00FFA3;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #050505 !important;
        border-right: 2px solid #00FFA3;
    }

    /* Pixelated Buttons */
    div.stButton > button {
        background: rgba(0, 255, 163, 0.05) !important;
        color: #00FFA3 !important;
        border: 2px solid #00FFA3 !important;
        font-family: 'Silkscreen', cursive !important;
        text-transform: uppercase;
        border-radius: 0px !important;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background: #00FFA3 !important;
        color: #000 !important;
        box-shadow: 0 0 20px #00FFA3;
    }

    /* Input Field */
    .stTextInput input {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid #00FFA3 !important;
        color: #ffffff !important;
        font-family: 'Silkscreen', cursive !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 5. LOGIC ENGINE ---
def get_response(user_input):
    # Fixed Developer Credit
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
    return "Signal Error. Data Mismatch."

# --- 6. SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='color:#00FFA3; font-family:Silkscreen;'>SYSTEM</h2>", unsafe_allow_html=True)
    if st.button("PURGE HISTORY"):
        st.session_state.history = []
        st.rerun()
    st.markdown("---")
    st.write("**Architect:** Helly Shah")
    st.write("**Core:** Nova Chatterix")
    st.markdown('<p style="color:#00FFA3;">● STATUS: ONLINE</p>', unsafe_allow_html=True)

# --- 7. MAIN INTERFACE ---
# THE UNIFORM THICK HEADER
st.markdown('<p class="voxa-header">NOVA CHATTERIX</p>', unsafe_allow_html=True)
st.markdown('<div class="glow-line"></div>', unsafe_allow_html=True)

# Hero Bot with center aura
if lottie_robot:
    st_lottie(lottie_robot, height=350, key="main_bot")

if 'history' not in st.session_state:
    st.session_state.history = []

# Signal Buttons
st.markdown("### 📡 ACTIVE SIGNALS")
questions = df['question'].tolist()
cols = st.columns(3)
clicked_q = None
for i, q in enumerate(questions[:6]):
    if cols[i % 3].button(q, key=f"q_{i}"):
        clicked_q = q

with st.form(key='chat_form', clear_on_submit=True):
    user_query = st.text_input("Transmit Command:", placeholder="Enter your signal...")
    submit = st.form_submit_button("SEND")

final_query = clicked_q if clicked_q else (user_query if submit else None)

if final_query:
    ans = get_response(final_query)
    st.session_state.history.append({"q": final_query, "a": ans})
    st.rerun()

# History Log
for item in reversed(st.session_state.history):
    st.markdown(f'''
    <div style="background:rgba(0, 255, 163, 0.05); border-left:5px solid #00FFA3; padding:15px; margin-bottom:10px;">
        <b style="color:#00FFA3">INCOMING:</b> {item["q"]}<br><br>
        <b>NOVA:</b> {item["a"]}
    </div>
    ''', unsafe_allow_html=True)
