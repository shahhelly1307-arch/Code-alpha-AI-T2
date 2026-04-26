import streamlit as st
import pandas as pd
import json
import nltk
import requests
import streamlit.components.v1 as components
from streamlit_lottie import st_lottie
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer

# --- 1. NLP SETUP ---
@st.cache_resource
def setup_nlp():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

setup_nlp()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    return " ".join([lemmatizer.lemmatize(token) for token in tokens if token.isalnum()])

# --- 2. ASSET LOADING ---
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

lottie_main = load_lottieurl("https://lottie.host/8172906e-8360-449e-9988-0320a1630985/B1pU53Y34i.json")

# --- 3. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        with open('faqs.json', 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame({
            "question": ["System Status", "Identity Check"], 
            "answer": ["Database signal active. Please check faqs.json.", "I am NOVA, your neural interface."]
        })

df = load_data()

# --- 4. THE NOVA CHATTERIX UI ---
st.set_page_config(page_title="Nova Chatterix", layout="wide")

# MERGED: Original HTML/CSS Character Header
hero_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <link href="https://fonts.googleapis.com/css2?family=Silkscreen:wght@700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background-color: transparent;
            display: flex;
            justify-content: center;
            align-items: center;
            font-family: 'Silkscreen', cursive;
            overflow: hidden;
            height: 450px;
        }
        .container {
            display: flex;
            flex-direction: column;
            align-items: center;
            position: relative;
            width: 100%;
        }
        .robot-box {
            margin-bottom: -130px; 
            z-index: 10;
            animation: hover 4s ease-in-out infinite;
        }
        .eye-pupil { animation: eyeMovement 6s ease-in-out infinite; }
        .eye-lid { transform-origin: center; animation: blink 4s ease-in-out infinite; }
        .waving-arm { transform-origin: 145px 135px; animation: bigWave 0.8s ease-in-out 2; }
        .arc-svg {
            width: 950px;
            height: 350px;
            z-index: 5;
            filter: drop-shadow(0 0 20px rgba(92, 225, 255, 0.4));
        }
        .nova-brand {
            margin-top: -80px;
            font-size: 1.3rem;
            letter-spacing: 12px;
            text-transform: uppercase;
            background: linear-gradient(to bottom, #ffffff, #5ce1ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            opacity: 0.9;
        }
        @keyframes hover { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-25px); } }
        @keyframes bigWave { 0%, 100% { transform: rotate(0deg); } 50% { transform: rotate(-25deg) scale(1.1); } }
        @keyframes eyeMovement { 0%, 10%, 100% { transform: translate(0, 0); } 20%, 40% { transform: translate(3px, -1px); } 50%, 70% { transform: translate(-3px, 1px); } 80% { transform: translate(0, -2px); } }
        @keyframes blink { 0%, 45%, 55%, 100% { transform: scaleY(1); } 50% { transform: scaleY(0.1); } }
    </style>
</head>
<body>
<div class="container">
    <div class="robot-box">
        <svg width="280" height="280" viewBox="0 0 200 200">
            <defs>
                <linearGradient id="bodyGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stop-color="#b85cff" />
                    <stop offset="60%" stop-color="#7a7aff" />
                    <stop offset="100%" stop-color="#5ce1ff" />
                </linearGradient>
                <filter id="glow">
                    <feGaussianBlur stdDeviation="6" result="blur" />
                    <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
            </defs>
            <circle cx="100" cy="35" r="4" fill="#b85cff" />
            <rect x="98" y="38" width="4" height="12" fill="#b85cff" />
            <path d="M55 85 C 55 40, 145 40, 145 85 C 145 130, 125 170, 100 170 C 75 170, 55 130, 55 85" fill="url(#bodyGrad)" stroke="white" stroke-width="0.5" />
            <rect x="65" y="75" width="70" height="42" rx="21" fill="#0c121d" stroke="#5ce1ff" stroke-width="1" />
            <g class="eye-lid">
                <circle cx="82" cy="95" r="8" fill="white" />
                <circle class="eye-pupil" cx="82" cy="95" r="3.5" fill="#0c121d" />
                <circle cx="118" cy="95" r="8" fill="white" />
                <circle class="eye-pupil" cx="118" cy="95" r="3.5" fill="#0c121d" />
            </g>
            <path d="M92 122 Q100 128 108 122" stroke="white" stroke-width="2" fill="none" stroke-linecap="round" />
            <path d="M56 130 C 25 135, 25 165, 58 155" fill="#7a7aff" stroke="white" stroke-width="1" />
            <g class="waving-arm">
                <path d="M144 130 C 175 135, 175 165, 142 155" fill="#7a7aff" stroke="white" stroke-width="1" />
            </g>
            <ellipse cx="100" cy="170" rx="35" ry="12" fill="#5ce1ff" filter="url(#glow)" opacity="0.7" />
        </svg>
    </div>
    <svg class="arc-svg" viewBox="0 0 800 300">
        <path d="M40 260 Q400 20 760 260" stroke="#5ce1ff" stroke-width="12" fill="none" opacity="0.2" filter="blur(15px)" />
        <path d="M40 260 Q400 20 760 260" stroke="url(#arcGrad)" stroke-width="5" fill="none" stroke-linecap="round" />
        <defs>
            <linearGradient id="arcGrad">
                <stop offset="0%" stop-color="transparent"/>
                <stop offset="50%" stop-color="#5ce1ff"/>
                <stop offset="100%" stop-color="transparent"/>
            </linearGradient>
        </defs>
    </svg>
    <h1 class="nova-brand">Nova Chatterix</h1>
</div>
</body>
</html>
"""

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Silkscreen:wght@700&display=swap');
    
    html, body, [class*="css"], .stText, .stMarkdown, .stButton, input, label {
        font-family: 'Silkscreen', cursive !important;
    }

    .stApp {
        background-color: #050505 !important; 
        background-image: 
            radial-gradient(circle at 0% 0%, rgba(0, 229, 255, 0.2) 0%, transparent 60%), 
            radial-gradient(circle at 100% 100%, rgba(180, 82, 255, 0.2) 0%, transparent 60%),
            linear-gradient(135deg, #001214 0%, #11001c 100%) !important;
        background-attachment: fixed !important;
        background-size: cover;
        color: #ffffff;
    }
    
    .voxa-header {
        font-family: 'Silkscreen', cursive !important;
        font-size: clamp(2.5rem, 6vw, 8rem) !important; 
        font-weight: 700 !important;
        background: linear-gradient(to right, #00e5ff, #b452ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        text-transform: uppercase;
        white-space: nowrap; 
        letter-spacing: -3px;
        margin-top: 10px;
        margin-bottom: 0px;
        filter: drop-shadow(0 0 15px rgba(0, 229, 255, 0.4));
    }

    .orbital-line {
        height: 3px;
        background: linear-gradient(90deg, transparent, #00e5ff, transparent);
        width: 80%;
        margin: 0 auto 40px auto;
        box-shadow: 0 0 15px #00e5ff;
    }

    [data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.8) !important;
        border-right: 1px solid rgba(0, 229, 255, 0.3);
    }

    .sidebar-label {
        color: #00e5ff;
        font-size: 0.9rem;
        letter-spacing: 2px;
        font-weight: bold;
    }

    div.stButton > button {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #ffffff !important;
        border: 1px solid rgba(0, 229, 255, 0.5) !important;
        border-radius: 50px !important;
        font-size: 0.85rem !important;
        transition: 0.3s;
        padding: 10px 20px;
    }

    div.stButton > button:hover {
        background: rgba(0, 229, 255, 0.2) !important;
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.4);
        border: 1px solid #00e5ff !important;
    }
    
    .stTextInput input {
        background-color: rgba(20, 20, 20, 0.7) !important;
        border: 1px solid rgba(0, 229, 255, 0.5) !important;
        color: #ffffff !important;
        border-radius: 10px !important;
    }

    .chat-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(0, 229, 255, 0.2);
        border-left: 5px solid #00e5ff;
        padding: 20px;
        margin-bottom: 15px;
        backdrop-filter: blur(10px);
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 5. LOGIC ENGINE ---
def get_response(user_input):
    dev_query = user_input.lower()
    if any(x in dev_query for x in ["developed", "creator", "who made", "built by", "developer"]):
        return "This interface was developed by Helly as a professional demonstration of NLP and advanced UI design."
    
    if "nova" in dev_query and "who are you" in dev_query:
        return "I am NOVA, a high-frequency neural interface designed for rapid data retrieval."

    processed_input = preprocess_text(user_input)
    corpus = df['question'].apply(preprocess_text).tolist()
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    user_vec = vectorizer.transform([processed_input])
    similarity_scores = cosine_similarity(user_vec, tfidf_matrix)
    
    idx = similarity_scores.argmax()
    if similarity_scores[0][idx] > 0.2:
        return df.iloc[idx]['answer']
    return "Neural Signal Mismatch. Data not found in current frequency."

# --- 6. SIDEBAR ---
with st.sidebar:
    st.markdown('<p class="sidebar-label">INTERFACE SETTINGS</p>', unsafe_allow_html=True)
    if st.button("CLEAR ACTIVE CACHE"):
        st.session_state.history = []
        st.rerun()
    
    st.markdown("---")
    st.markdown('<p class="sidebar-label">SYSTEM CREDENTIALS</p>', unsafe_allow_html=True)
    st.write("**DEVELOPER:** Helly")
    st.write("**ENGINE:** NOVA-V2")
    
    st.markdown("---")
    st.markdown('<p style="color:#00e5ff; font-weight:bold;">● SYSTEM: ONLINE</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#b452ff; font-weight:bold;">● SIGNAL: ACTIVE</p>', unsafe_allow_html=True)

# --- 7. MAIN INTERFACE ---

# DISPLAY HERO CHARACTER FIRST
components.html(hero_html, height=400)

st.markdown('<div class="orbital-line"></div>', unsafe_allow_html=True)

if lottie_main:
    col_rob, _ = st.columns([1, 4])
    with col_rob:
        st_lottie(lottie_main, height=150, key="main_robot")

if 'history' not in st.session_state:
    st.session_state.history = []

st.markdown("### 📡 ACTIVE FREQUENCIES")
questions_list = df['question'].tolist()
cols = st.columns(3)
clicked_q = None

for i, q in enumerate(questions_list):
    if cols[i % 3].button(q, key=f"q_{i}"):
        clicked_q = q

with st.form(key='chat_form', clear_on_submit=True):
    user_query = st.text_input("Transmit Command:", placeholder="AWAITING SIGNAL...")
    submit = st.form_submit_button("TRANSMIT")

final_query = clicked_q if clicked_q else (user_query if submit else None)

if final_query:
    ans = get_response(final_query)
    st.session_state.history.insert(0, {"q": final_query, "a": ans})
    st.rerun()

for item in st.session_state.history:
    st.markdown(f'''
    <div class="chat-card">
        <b style="color:#00e5ff">SIGNAL:</b> {item["q"]}<br><br>
        <b style="color:#b452ff">NOVA:</b> {item["a"]}
    </div>
    ''', unsafe_allow_html=True)
