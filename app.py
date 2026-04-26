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

# --- 1. GLOBAL CONFIG & SHARED STYLES ---
st.set_page_config(page_title="Nova Chatterix", layout="wide")

# CSS applied to BOTH intro and main app to prevent "jumpy" UI
SHARED_STYLES = """
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

    /* Transition Animation */
    .fade-in {
        animation: fadeIn 1.5s ease-in;
    }
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }

    .voxa-header {
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
        filter: drop-shadow(0 0 15px rgba(0, 229, 255, 0.4));
    }

    div.stButton > button {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #ffffff !important;
        border: 1px solid rgba(0, 229, 255, 0.5) !important;
        border-radius: 50px !important;
        transition: 0.3s;
    }
    
    div.stButton > button:hover {
        background: rgba(0, 229, 255, 0.2) !important;
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.4);
        border: 1px solid #00e5ff !important;
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
"""
st.markdown(SHARED_STYLES, unsafe_allow_html=True)

# --- 2. SESSION STATE ---
if 'intro_done' not in st.session_state:
    st.session_state.intro_done = False
if 'history' not in st.session_state:
    st.session_state.history = []

# --- 3. INTRO PAGE ---
if not st.session_state.intro_done:
    intro_html = """
    <div class="fade-in" style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 80vh;">
        <style>
            .robot-box { margin-bottom: -130px; z-index: 10; animation: hover 4s ease-in-out infinite; }
            .eye-lid { transform-origin: center; animation: blink 4s ease-in-out infinite; }
            .waving-arm { transform-origin: 145px 135px; animation: bigWave 0.8s ease-in-out 2; }
            @keyframes hover { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-25px); } }
            @keyframes blink { 0%, 45%, 55%, 100% { transform: scaleY(1); } 50% { transform: scaleY(0.1); } }
            @keyframes bigWave { 0%, 100% { transform: rotate(0deg); } 50% { transform: rotate(-25deg) scale(1.1); } }
        </style>
        <div class="robot-box">
            <svg width="280" height="280" viewBox="0 0 200 200">
                <defs>
                    <linearGradient id="bodyGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                        <stop offset="0%" stop-color="#b85cff" /><stop offset="60%" stop-color="#7a7aff" /><stop offset="100%" stop-color="#5ce1ff" />
                    </linearGradient>
                </defs>
                <circle cx="100" cy="35" r="4" fill="#b85cff" />
                <rect x="98" y="38" width="4" height="12" fill="#b85cff" />
                <path d="M55 85 C 55 40, 145 40, 145 85 C 145 130, 125 170, 100 170 C 75 170, 55 130, 55 85" fill="url(#bodyGrad)" stroke="white" stroke-width="0.5" />
                <rect x="65" y="75" width="70" height="42" rx="21" fill="#0c121d" stroke="#5ce1ff" stroke-width="1" />
                <g class="eye-lid"><circle cx="82" cy="95" r="8" fill="white" /><circle cx="118" cy="95" r="8" fill="white" /></g>
                <g class="waving-arm"><path d="M144 130 C 175 135, 175 165, 142 155" fill="#7a7aff" stroke="white" stroke-width="1" /></g>
            </svg>
        </div>
        <svg width="800" height="300" viewBox="0 0 800 300">
            <path d="M40 260 Q400 20 760 260" stroke="#5ce1ff" stroke-width="5" fill="none" opacity="0.6" />
        </svg>
        <h1 class="voxa-header" style="margin-top:-100px">Nova Chatterix</h1>
    </div>
    """
    st.components.v1.html(intro_html, height=700)
    time.sleep(2.5) 
    st.session_state.intro_done = True
    st.rerun()

# --- 4. NLP & DATA ---
@st.cache_resource
def setup_nlp():
    for pkg in ['punkt', 'wordnet', 'punkt_tab']:
        try: nltk.data.find(f'tokenizers/{pkg}' if 'punkt' in pkg else f'corpora/{pkg}')
        except LookupError: nltk.download(pkg)

setup_nlp()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    return " ".join([lemmatizer.lemmatize(token) for token in tokens if token.isalnum()])

@st.cache_data
def load_data():
    try:
        with open('faqs.json', 'r') as f: return pd.DataFrame(json.load(f))
    except:
        return pd.DataFrame({"question": ["System Status"], "answer": ["Signal Active. Check faqs.json."]})

df = load_data()

# --- 5. LOGIC ---
def get_response(user_input):
    query = user_input.lower()
    if any(x in query for x in ["creator", "who made", "developer"]):
        return "This interface was developed by Helly as a professional demonstration of NLP."
    
    processed_input = preprocess_text(user_input)
    corpus = df['question'].apply(preprocess_text).tolist()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    user_vec = vectorizer.transform([processed_input])
    scores = cosine_similarity(user_vec, tfidf_matrix)
    idx = scores.argmax()
    return df.iloc[idx]['answer'] if scores[0][idx] > 0.2 else "Neural Signal Mismatch."

# --- 6. MAIN APP INTERFACE ---
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;">
        <svg width="100" height="100" viewBox="0 0 200 200">
            <path d="M55 85 C 55 40, 145 40, 145 85 C 145 130, 125 170, 100 170 C 75 170, 55 130, 55 85" fill="url(#bodyGrad)" stroke="white" stroke-width="0.5" />
        </svg>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<p style="color:#00e5ff; letter-spacing:2px; font-size:0.8rem;">SYSTEM CREDENTIALS</p>', unsafe_allow_html=True)
    st.write("**DEVELOPER:** Helly")
    if st.button("RESET INTERFACE"):
        st.session_state.history = []
        st.rerun()

st.markdown('<div class="fade-in">', unsafe_allow_html=True)
st.markdown('<p class="voxa-header">NOVA CHATTERIX</p>', unsafe_allow_html=True)
st.markdown('<div style="height:3px; background:linear-gradient(90deg, transparent, #00e5ff, transparent); width:80%; margin: 0 auto 40px auto;"></div>', unsafe_allow_html=True)

# Frequency Buttons
st.markdown("### 📡 ACTIVE FREQUENCIES")
cols = st.columns(3)
clicked_q = None
for i, q in enumerate(df['question'].tolist()):
    if cols[i % 3].button(q, key=f"btn_{i}"):
        clicked_q = q

# Input Form
with st.form(key='chat_form', clear_on_submit=True):
    user_query = st.text_input("Transmit Command:", placeholder="AWAITING SIGNAL...")
    submit = st.form_submit_button("TRANSMIT")

final_query = clicked_q if clicked_q else (user_query if submit else None)

if final_query:
    ans = get_response(final_query)
    st.session_state.history.insert(0, {"q": final_query, "a": ans})
    st.rerun()

# Chat History
for item in st.session_state.history:
    st.markdown(f'''
    <div class="chat-card fade-in">
        <b style="color:#00e5ff">SIGNAL:</b> {item["q"]}<br><br>
        <b style="color:#b452ff">NOVA:</b> {item["a"]}
    </div>
    ''', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
