import streamlit as st
import pandas as pd
import json
import nltk
import requests
import time
from streamlit_lottie import st_lottie
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer

# --- SETUP ---
@st.cache_resource
def setup_nlp():
    for d in ['punkt', 'wordnet', 'punkt_tab']:
        try:
            nltk.data.find(f'tokenizers/{d}' if 'punkt' in d else f'corpora/{d}')
        except:
            nltk.download(d, quiet=True)

setup_nlp()
lemmatizer = WordNetLemmatizer()

def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

lottie_robot = load_lottieurl("https://lottie.host/8172906e-8360-449e-9988-0320a1630985/B1pU53Y34i.json")

if 'phase' not in st.session_state:
    st.session_state.phase = "intro"

# --- STYLING (Including the Half-Circle from your image) ---
st.set_page_config(page_title="Novo Chatterix", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Silkscreen:wght@700&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Silkscreen', cursive !important;
        background-color: #050505 !important;
        color: white;
    }

    /* THE GLOWING HALF-CIRCLE AT BOTTOM */
    .half-circle-glow {
        position: fixed;
        bottom: -350px;
        left: 50%;
        transform: translateX(-50%);
        width: 1200px;
        height: 700px;
        background: radial-gradient(circle at 50% 0%, #00e5ff 0%, #b452ff 35%, transparent 70%);
        border-radius: 50%;
        z-index: -1;
        opacity: 0.9;
        filter: blur(30px);
    }

    .voxa-header {
        font-size: clamp(3rem, 10vw, 7rem);
        background: linear-gradient(to right, #00e5ff, #b452ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 0;
        filter: drop-shadow(0 0 20px rgba(0, 229, 255, 0.5));
    }

    .chat-card {
        background: rgba(255, 255, 255, 0.05);
        border-left: 5px solid #00e5ff;
        padding: 20px;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIC ---
if st.session_state.phase == "intro":
    # 1. Show the Half-Circle
    st.markdown('<div class="half-circle-glow"></div>', unsafe_allow_html=True)
    
    st.markdown('<div style="height: 15vh;"></div>', unsafe_allow_html=True)
    
    # 2. Show the Robot
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if lottie_robot:
            st_lottie(lottie_robot, height=450, key="intro_robot")
        else:
            st.write("🤖 [Robot Loading...]")
        st.markdown('<p class="voxa-header">NOVA CHATTERIX</p>', unsafe_allow_html=True)
    
    # 3. Wait 2 seconds and switch
    time.sleep(2.5)
    st.session_state.phase = "chatbot"
    st.rerun()

else:
    # --- YOUR ORIGINAL CHATBOT PAGE ---
    with st.sidebar:
        st.title("SYSTEM ONLINE")
        if st.button("RE-RUN INTRO"):
            st.session_state.phase = "intro"
            st.rerun()

    st.markdown('<p class="voxa-header" style="font-size: 4rem;">NOVA CHATTERIX</p>', unsafe_allow_html=True)
    
    # (The rest of your chatbot logic goes here)
    # This part will show only after the robot waves for 2 seconds.
    st.write("### 📡 SYSTEM READY")
    st.info("Neural Link Established. Frequency Active.")
