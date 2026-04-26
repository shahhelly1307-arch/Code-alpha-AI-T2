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

# This is the animated robot that "tells hi"
lottie_main = load_lottieurl("https://lottie.host/8172906e-8360-449e-9988-0320a1630985/B1pU53Y34i.json")

# --- 3. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        with open('faqs.json', 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame({"question": ["System Status"], "answer": ["Database signal active. Please check faqs.json file."]})\

df = load_data()

# --- 4. SESSION STATE FOR TRANSITION ---
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = 'intro'

# --- 5. THE UI STYLING ---
st.set_page_config(page_title="Novo Chatterix", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Silkscreen:wght@700&display=swap');
    
    html, body, [class*="css"], .stText, .stMarkdown, .stButton, input, label {
        font-family: 'Silkscreen', cursive !important;
    }

    .stApp {
        background-color: #050505 !important; 
        background-image: 
            radial-gradient(circle at 0% 0%, rgba(0, 229, 255, 0.1) 0%, transparent 50%), 
            radial-gradient(circle at 100% 100%, rgba(180, 82, 255, 0.1) 0%, transparent 50%),
            linear-gradient(135deg, #010a0a 0%, #0a010d 100%) !important;
        color: #ffffff;
    }
    
    /* THE HEADER */
    .voxa-header {
        font-size: clamp(2rem, 8vw, 6rem) !important; 
        font-weight: 700 !important;
        background: linear-gradient(to right, #00e5ff, #b452ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        text-transform: uppercase;
        margin: 0;
        filter: drop-shadow(0 0 20px rgba(0, 229, 255, 0.5));
    }

    /* THE HALF CIRCLE GLOW (From your Image) */
    .half-circle-glow {
        position: fixed;
        bottom: -350px;
        left: 50%;
        transform: translateX(-50%);
        width: 1000px;
        height: 600px;
        background: radial-gradient(circle at 50% 0%, #00e5ff 0%, #b452ff 35%, transparent 70%);
        border-radius: 50%;
        z-index: -1;
        opacity: 0.7;
    }

    .orbital-line {
        height: 2px;
        background: linear-gradient(90deg, transparent, #00e5ff, transparent);
        width: 70%;
        margin: 0 auto 30px auto;
        box-shadow: 0 0 10px #00e5ff;
    }

    /* Sidebars and Cards */
    [data-testid="stSidebar"] { background-color: #000000 !important; border-right: 1px solid #00e5ff33; }
    
    .chat-card {
        background: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #00e5ff;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 6. INTRO SCREEN LOGIC ---
if st.session_state.app_mode == 'intro':
    # Hide everything but the intro
    st.markdown("<style>[data-testid='stSidebar'] {display:none;} #MainMenu {visibility:hidden;}</style>", unsafe_allow_html=True)
    
    # Empty space to push content down
    st.markdown('<div style="height: 15vh;"></div>', unsafe_allow_html=True)
    
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        if lottie_main:
            st_lottie(lottie_main, height=350, key="waving_robot")
        st.markdown('<h1 class="voxa-header">NOVA CHATTERIX</h1>', unsafe_allow_html=True)
    
    # Show the half circle at bottom
    st.markdown('<div class="half-circle-glow"></div>', unsafe_allow_html=True)
    
    # Wait for the robot to wave twice (approx 2.5 seconds)
    time.sleep(2.5)
    st.session_state.app_mode = 'chat'
    st.rerun()

# --- 7. CHAT INTERFACE LOGIC ---
else:
    # --- LOGIC ENGINE ---
    def get_response(user_input):
        dev_query = user_input.lower()
        if any(x in dev_query for x in ["developed", "creator", "who made", "built by", "developer"]):
            return "This interface was developed by Helly as a professional demonstration of NLP and advanced UI design."
        processed_input = preprocess_text(user_input)
        corpus = df['question'].apply(preprocess_text).tolist()
        corpus.append(processed_input)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        idx = similarity_scores.argmax()
        if similarity_scores[0][idx] > 0.2:
            return df.iloc[idx]['answer']
        return "Neural Signal Mismatch. Data not found in current frequency."

    # SIDEBAR
    with st.sidebar:
        st.markdown('<p style="color:#00e5ff;">SYSTEM STATUS</p>', unsafe_allow_html=True)
        if st.button("RESTART INTERFACE"):
            st.session_state.app_mode = 'intro'
            st.session_state.history = []
            st.rerun()
        st.write("**DEVELOPER:** Helly")

    # MAIN CHAT UI
    st.markdown('<h1 class="voxa-header">NOVA CHATTERIX</h1>', unsafe_allow_html=True)
    st.markdown('<div class="orbital-line"></div>', unsafe_allow_html=True)

    if 'history' not in st.session_state:
        st.session_state.history = []

    # Small robot icon for chat view
    col_icon, _ = st.columns([1, 5])
    with col_icon:
        if lottie_main:
            st_lottie(lottie_main, height=120, key="chat_robot")

    # Question Buttons
    st.markdown("### 📡 FREQUENCIES")
    questions_list = df['question'].tolist()
    cols = st.columns(3)
    clicked_q = None

    for i, q in enumerate(questions_list):
        if cols[i % 3].button(q, key=f"btn_{i}"):
            clicked_q = q

    # Input Form
    with st.form(key='chat_form', clear_on_submit=True):
        user_query = st.text_input("Transmit Command:", placeholder="AWAITING SIGNAL...")
        submit = st.form_submit_button("TRANSMIT")

    final_query = clicked_q if clicked_q else (user_query if submit else None)

    if final_query:
        ans = get_response(final_query)
        st.session_state.history.append({"q": final_query, "a": ans})
        st.rerun()

    # Display History
    for item in reversed(st.session_state.history):
        st.markdown(f'''
        <div class="chat-card">
            <b style="color:#00e5ff">SIGNAL:</b> {item["q"]}<br>
            <b style="color:#b452ff">NOVO:</b> {item["a"]}
        </div>
        ''', unsafe_allow_html=True)
