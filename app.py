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

# --- 2. ASSET LOADING (ROBOT FIX) ---
def load_lottieurl(url: str):
    try:
        # Increased timeout to ensure robot loads
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# High-tech robot asset
lottie_robot = load_lottieurl("https://lottie.host/8172906e-8360-449e-9988-0320a1630985/B1pU53Y34i.json")

# --- 3. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        with open('faqs.json', 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except:
        return pd.DataFrame({"question": ["Developer"], "answer": ["Developed by Helly Shah."]})

df = load_data()

# --- 4. UI CONFIGURATION (VOXA AURA & BLOCK FONT) ---
st.set_page_config(page_title="Nova Chatterix", layout="wide")

st.markdown("""
    <style>
    /* Import Silkscreen for thick pixel consistency */
    @import url('https://fonts.googleapis.com/css2?family=Silkscreen:wght@700&display=swap');

    /* FORCE BACKGROUND: Cyan Radial Aura */
    .stApp {
        background: radial-gradient(circle at center, #005a5a 0%, #020b0b 45%, #000000 100%) !important;
        background-attachment: fixed !important;
    }
    
    /* THE HEADER: Forced Block Thickness for H, T, E, I */
    .voxa-header {
        font-family: 'Silkscreen', cursive !important;
        font-size: clamp(2rem, 8vw, 4.8rem) !important;
        color: #00FFA3 !important;      /* EXACT VOXA CYAN */
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 10px;
        margin-top: 30px;
        margin-bottom: 0px;
        white-space: nowrap;
        
        /* Force blockiness on thin letters */
        -webkit-text-stroke: 2.5px #00FFA3; 
        text-shadow: 
            0 0 15px rgba(0, 255, 163, 0.8),
            0 0 35px rgba(0, 255, 163, 0.4);
    }

    /* Glow Divider */
    .glow-line {
        height: 3px;
        background: linear-gradient(90deg, transparent, #00FFA3, #00FFA3, transparent);
        width: 85%;
        margin: 5px auto 40px auto;
        box-shadow: 0 0 20px #00FFA3;
    }

    /* Sidebar Content */
    [data-testid="stSidebar"] {
        background-color: #050505 !important;
        border-right: 2px solid #00FFA3;
    }

    /* Pixelated Signal Buttons */
    div.stButton > button {
        background: rgba(0, 255, 163, 0.05) !important;
        color: #00FFA3 !important;
        border: 2px solid #00FFA3 !important;
        font-family: 'Silkscreen', cursive !important;
        text-transform: uppercase;
        border-radius: 0px !important;
        font-size: 0.9rem !important;
    }
    div.stButton > button:hover {
        background: #00FFA3 !important;
        color: #000 !important;
        box-shadow: 0 0 15px #00FFA3;
    }

    /* Chat Input */
    .stTextInput input {
        background-color: rgba(0, 0, 0, 0.6) !important;
        border: 1px solid #00FFA3 !important;
        color: #ffffff !important;
        font-family: 'Silkscreen', sans-serif !important;
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
    return "Neural Link Failure. Command not found."

# --- 6. SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='color:#00FFA3; font-family:Silkscreen;'>SYSTEM</h2>", unsafe_allow_html=True)
    # Changed to CLEAR CACHE as requested
    if st.button("CLEAR CACHE"):
        st.session_state.history = []
        st.rerun()
    st.markdown("---")
    st.write("**Operator:** Helly Shah")
    st.write("**System:** NOVA CHATTERIX")
    st.markdown('<p style="color:#00FFA3;">● LINK: ACTIVE</p>', unsafe_allow_html=True)

# --- 7. MAIN INTERFACE ---
# THE FIXED THICK HEADER
st.markdown('<p class="voxa-header">NOVA CHATTERIX</p>', unsafe_allow_html=True)
st.markdown('<div class="glow-line"></div>', unsafe_allow_html=True)

# Robot Animation
if lottie_robot:
    st_lottie(lottie_robot, height=350, key="main_bot")
else:
    st.info("Searching for Robot Signal...")

if 'history' not in st.session_state:
    st.session_state.history = []

# Signal Frequencies
st.markdown("### ⚡ ACTIVE FREQUENCIES")
questions = df['question'].tolist()
cols = st.columns(3)
clicked_q = None
for i, q in enumerate(questions[:6]):
    if cols[i % 3].button(q, key=f"q_{i}"):
        clicked_q = q

with st.form(key='chat_form', clear_on_submit=True):
    user_query = st.text_input("Transmit Command:", placeholder="Input signal...")
    submit = st.form_submit_button("SEND")

final_query = clicked_q if clicked_q else (user_query if submit else None)

if final_query:
    ans = get_response(final_query)
    st.session_state.history.append({"q": final_query, "a": ans})
    st.rerun()

# Display History
for item in reversed(st.session_state.history):
    st.markdown(f'''
    <div style="background:rgba(0, 255, 163, 0.04); border-left:4px solid #00FFA3; padding:15px; margin-bottom:10px;">
        <b style="color:#00FFA3">INCOMING:</b> {item["q"]}<br><br>
        <b>NOVA:</b> {item["a"]}
    </div>
    ''', unsafe_allow_html=True)
