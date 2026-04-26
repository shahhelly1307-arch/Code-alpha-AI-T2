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

# --- 1. NLP & DATA SETUP ---
@st.cache_resource
def setup_nlp():
    for d in ['punkt', 'wordnet', 'punkt_tab']:
        try:
            nltk.data.find(f'tokenizers/{d}' if 'punkt' in d else f'corpora/{d}')
        except LookupError:
            nltk.download(d, quiet=True)

setup_nlp()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    return " ".join([lemmatizer.lemmatize(token) for token in tokens if token.isalnum()])

@st.cache_data
def load_data():
    try:
        with open('faqs.json', 'r') as f:
            return pd.DataFrame(json.load(f))
    except:
        return pd.DataFrame({"question": ["System Status"], "answer": ["Active. Check faqs.json"]})

def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=10)
        return r.json() if r.status_code == 200 else None
    except:
        return None

lottie_main = load_lottieurl("https://lottie.host/8172906e-8360-449e-9988-0320a1630985/B1pU53Y34i.json")
df = load_data()

# --- 2. STATE MANAGEMENT ---
if 'flow' not in st.session_state:
    st.session_state.flow = "intro"

# --- 3. PAGE CONFIG & STYLING ---
st.set_page_config(page_title="Novo Chatterix", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Silkscreen:wght@700&display=swap');
    
    html, body, [class*="css"], .stText, .stMarkdown, .stButton, input, label {
        font-family: 'Silkscreen', cursive !important;
    }

    .stApp {
        background-color: #050505 !important; 
        background-image: 
            radial-gradient(circle at 0% 0%, rgba(0, 229, 255, 0.15) 0%, transparent 60%), 
            linear-gradient(135deg, #001214 0%, #11001c 100%) !important;
    }
    
    /* THE HALF-CIRCLE GLOW AT THE BOTTOM (IMAGE STYLE) */
    .intro-half-circle {
        position: fixed;
        bottom: -350px;
        left: 50%;
        transform: translateX(-50%);
        width: 1200px;
        height: 700px;
        background: radial-gradient(circle at 50% 0%, #00e5ff 0%, #b452ff 35%, transparent 70%);
        border-radius: 50%;
        z-index: -1;
        opacity: 0.8;
        filter: blur(20px);
    }

    .voxa-header {
        font-size: clamp(2.5rem, 6vw, 6rem) !important; 
        background: linear-gradient(to right, #00e5ff, #b452ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        text-transform: uppercase;
        margin-bottom: 0px;
        filter: drop-shadow(0 0 15px rgba(0, 229, 255, 0.4));
    }

    .orbital-line {
        height: 3px;
        background: linear-gradient(90deg, transparent, #00e5ff, transparent);
        width: 80%; margin: 20px auto 40px auto;
        box-shadow: 0 0 15px #00e5ff;
    }

    .chat-card {
        background: rgba(255, 255, 255, 0.03);
        border-left: 5px solid #00e5ff;
        padding: 20px; margin-bottom: 15px;
        border-radius: 4px;
        border: 1px solid rgba(0, 229, 255, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. PAGE 1: THE INTRO ---
if st.session_state.flow == "intro":
    # Hide the sidebar during intro
    st.markdown("<style>[data-testid='stSidebar'] {display: none;}</style>", unsafe_allow_html=True)
    
    st.markdown('<div style="height: 15vh;"></div>', unsafe_allow_html=True)
    
    # 1. The Waving/Shaking Robot
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if lottie_main:
            st_lottie(lottie_main, height=450, key="intro_anim")
        # 2. Chatbot Name
        st.markdown('<p class="voxa-header">NOVA CHATTERIX</p>', unsafe_allow_html=True)
    
    # 3. The Half-Circle Glow
    st.markdown('<div class="intro-half-circle"></div>', unsafe_allow_html=True)
    
    # Wait for robot to wave (2.5 seconds)
    time.sleep(2.5)
    st.session_state.flow = "chat"
    st.rerun()

# --- 5. PAGE 2: THE CHATBOT ---
else:
    # Sidebar
    with st.sidebar:
        st.markdown('<p style="color:#00e5ff; font-weight:bold;">● SYSTEM: ACTIVE</p>', unsafe_allow_html=True)
        if st.button("RESET INTERFACE"):
            st.session_state.flow = "intro"
            st.session_state.history = []
            st.rerun()
        st.write("**DEVELOPER:** Helly")

    # Header
    st.markdown('<p class="voxa-header">NOVA CHATTERIX</p>', unsafe_allow_html=True)
    st.markdown('<div class="orbital-line"></div>', unsafe_allow_html=True)

    # Robot Icon for Page 2
    col_r, _ = st.columns([1, 5])
    with col_r:
        if lottie_main:
            st_lottie(lottie_main, height=120, key="chat_robot")

    # Logic Engine
    def get_response(user_input):
        processed_input = preprocess_text(user_input)
        corpus = df['question'].apply(preprocess_text).tolist()
        corpus.append(processed_input)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        idx = similarity_scores.argmax()
        return df.iloc[idx]['answer'] if similarity_scores[0][idx] > 0.2 else "Neural Signal Mismatch."

    if 'history' not in st.session_state:
        st.session_state.history = []

    # Frequency Buttons
    st.markdown("### 📡 ACTIVE FREQUENCIES")
    questions_list = df['question'].tolist()
    cols = st.columns(3)
    clicked_q = None
    for i, q in enumerate(questions_list):
        if cols[i % 3].button(q, key=f"q_{i}"):
            clicked_q = q

    # Input Form
    with st.form(key='chat_form', clear_on_submit=True):
        u_input = st.text_input("Transmit Command:", placeholder="AWAITING SIGNAL...")
        submit = st.form_submit_button("TRANSMIT")

    final_q = clicked_q if clicked_q else (u_input if submit else None)

    if final_q:
        ans = get_response(final_q)
        st.session_state.history.append({"q": final_q, "a": ans})
        st.rerun()

    # Cards
    for item in reversed(st.session_state.history):
        st.markdown(f'''
        <div class="chat-card">
            <b style="color:#00e5ff">SIGNAL:</b> {item["q"]}<br><br>
            <b style="color:#b452ff">NOVO:</b> {item["a"]}
        </div>
        ''', unsafe_allow_html=True)
