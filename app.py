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

# --- 1. SETUP & ASSETS ---
@st.cache_resource
def setup_nlp():
    for d in ['punkt', 'wordnet', 'punkt_tab']:
        try:
            nltk.data.find(f'tokenizers/{d}' if 'punkt' in d else f'corpora/{d}')
        except:
            nltk.download(d, quiet=True)

setup_nlp()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    return " ".join([lemmatizer.lemmatize(token) for token in tokens if token.isalnum()])

def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

# The waving robot for the intro
lottie_robot = load_lottieurl("https://lottie.host/8172906e-8360-449e-9988-0320a1630985/B1pU53Y34i.json")

@st.cache_data
def load_data():
    try:
        with open('faqs.json', 'r') as f:
            return pd.DataFrame(json.load(f))
    except:
        return pd.DataFrame({"question": ["System Status"], "answer": ["Active. Check faqs.json"]})

df = load_data()

# --- 2. STATE CONTROL ---
if 'app_state' not in st.session_state:
    st.session_state.app_state = "intro"

# --- 3. UI STYLING (YOUR ORIGINAL THEME) ---
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
            radial-gradient(circle at 0% 0%, rgba(0, 229, 255, 0.2) 0%, transparent 60%), 
            linear-gradient(135deg, #001214 0%, #11001c 100%) !important;
        background-attachment: fixed !important;
        background-size: cover;
    }

    /* THE HALF-CIRCLE GLOW (MATCHING YOUR IMAGE) */
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
        opacity: 0.8;
        filter: blur(25px);
    }

    .voxa-header {
        font-size: clamp(2.5rem, 6vw, 8rem) !important; 
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

    .orbital-line {
        height: 3px;
        background: linear-gradient(90deg, transparent, #00e5ff, transparent);
        width: 80%; margin: 20px auto 40px auto;
        box-shadow: 0 0 15px #00e5ff;
    }

    .chat-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(0, 229, 255, 0.2);
        border-left: 5px solid #00e5ff;
        padding: 20px; margin-bottom: 15px;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. PAGE 1: INTRO (THE IMAGE LOOK) ---
if st.session_state.app_state == "intro":
    # Hide sidebar for the intro
    st.markdown("<style>[data-testid='stSidebar'] {display: none;}</style>", unsafe_allow_html=True)
    
    st.markdown('<div style="height: 15vh;"></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if lottie_robot:
            st_lottie(lottie_robot, height=450, key="intro_wave")
        st.markdown('<p class="voxa-header">NOVA CHATTERIX</p>', unsafe_allow_html=True)
    
    # Half-Circle at bottom
    st.markdown('<div class="half-circle-glow"></div>', unsafe_allow_html=True)
    
    # Wait 2.5 seconds then switch to your chatbot
    time.sleep(2.5)
    st.session_state.app_state = "chatbot"
    st.rerun()

# --- 5. PAGE 2: YOUR ORIGINAL CHATBOT (NO CHANGES) ---
else:
    with st.sidebar:
        st.markdown('<p style="color:#00e5ff; font-weight:bold;">● SYSTEM: ONLINE</p>', unsafe_allow_html=True)
        if st.button("RE-RUN INTRO"):
            st.session_state.app_state = "intro"
            st.rerun()
        st.write("**DEVELOPER:** Helly")
        st.write("**ENGINE:** NPCL V2.0")

    st.markdown('<p class="voxa-header">NOVO CHATTERIX</p>', unsafe_allow_html=True)
    st.markdown('<div class="orbital-line"></div>', unsafe_allow_html=True)

    if lottie_robot:
        col_rob, _ = st.columns([1, 4])
        with col_rob:
            st_lottie(lottie_robot, height=150, key="chat_robot_small")

    if 'history' not in st.session_state:
        st.session_state.history = []

    def get_response(user_input):
        processed_input = preprocess_text(user_input)
        corpus = df['question'].apply(preprocess_text).tolist()
        corpus.append(processed_input)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        idx = similarity_scores.argmax()
        return df.iloc[idx]['answer'] if similarity_scores[0][idx] > 0.2 else "Neural Signal Mismatch."

    st.markdown("### 📡 ACTIVE FREQUENCIES")
    questions_list = df['question'].tolist()
    cols = st.columns(3)
    clicked_q = None
    for i, q in enumerate(questions_list):
        if cols[i % 3].button(q, key=f"q_{i}"):
            clicked_q = q

    with st.form(key='chat_form', clear_on_submit=True):
        u_input = st.text_input("Transmit Command:", placeholder="AWAITING SIGNAL...")
        submit = st.form_submit_button("TRANSMIT")

    final_q = clicked_q if clicked_q else (u_input if submit else None)

    if final_q:
        ans = get_response(final_q)
        st.session_state.history.append({"q": final_q, "a": ans})
        st.rerun()

    for item in reversed(st.session_state.history):
        st.markdown(f'''
        <div class="chat-card">
            <b style="color:#00e5ff">SIGNAL:</b> {item["q"]}<br><br>
            <b style="color:#b452ff">NOVO:</b> {item["a"]}
        </div>
        ''', unsafe_allow_html=True)
