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

# --- 1. NLP SETUP (Standardizing Indentation to avoid U+00A0 errors) ---
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
        return r.json() if r.status_code == 200 else None
    except: return None

lottie_main = load_lottieurl("https://lottie.host/8172906e-8360-449e-9988-0320a1630985/B1pU53Y34i.json")

@st.cache_data
def load_data():
    try:
        with open('faqs.json', 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except:
        return pd.DataFrame({"question": ["System Status"], "answer": ["Database active. Check faqs.json."]})

df = load_data()

# --- 3. UI CONFIG & YOUR PREVIOUS STYLING ---
st.set_page_config(page_title="Novo Chatterix", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Silkscreen:wght@700&display=swap');
    
    /* UNIVERSAL FONT LOCK */
    * {
        font-family: 'Silkscreen', cursive !important;
    }

    html, body, [class*="css"], .stText, .stMarkdown, .stButton, input, label {
        font-family: 'Silkscreen', cursive !important;
    }

    /* PREVIOUS BACKGROUND: Symmetrical Sides (Blue - Purple - Blue) */
    .stApp {
        background-color: #000000 !important;
        background-image: 
            linear-gradient(90deg, 
                rgba(0, 229, 255, 0.45) 0%, 
                rgba(180, 82, 255, 0.45) 50%, 
                rgba(0, 229, 255, 0.45) 100%) !important;
        background-attachment: fixed !important;
        background-size: cover;
        color: #ffffff;
    }
    
    /* FIRST PAGE: Splash Text */
    .splash-text {
        font-size: 8rem;
        text-align: center;
        margin-top: 15%;
        background: linear-gradient(to right, #00e5ff, #b452ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 0 30px rgba(0, 229, 255, 0.8));
    }

    /* SECOND PAGE: Your Original Large Header */
    .voxa-header {
        font-size: clamp(2.5rem, 6vw, 8rem) !important; 
        font-weight: 700 !important;
        background: linear-gradient(to right, #00e5ff, #b452ff, #00e5ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: -3px;
        margin-top: 10px;
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
        background-color: rgba(0, 0, 0, 0.9) !important;
        border-right: 2px solid #00e5ff;
    }

    /* Your Original Chat Card Style */
    .chat-card {
        background: rgba(0, 229, 255, 0.03);
        border: 1px solid #00e5ff;
        border-left: 5px solid #00e5ff;
        padding: 20px;
        margin-bottom: 15px;
        backdrop-filter: blur(10px);
    }

    div.stButton > button {
        background: rgba(0, 229, 255, 0.05) !important;
        color: #00e5ff !important;
        border: 2px solid #00e5ff !important;
        border-radius: 0px !important;
        transition: 0.3s;
    }
    
    .stTextInput input {
        background-color: rgba(20, 20, 20, 0.9) !important;
        border: 2px solid #00e5ff !important;
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. FIRST PAGE: TIMED SPLASH SCREEN (2 SECONDS) ---
if 'intro_done' not in st.session_state:
    splash = st.empty()
    with splash.container():
        st.markdown('<p class="splash-text">HELLY</p>', unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center; color:#00e5ff;'>SIGNAL_INITIALIZING...</h2>", unsafe_allow_html=True)
        time.sleep(2)
    splash.empty()
    st.session_state.intro_done = True

# --- 5. LOGIC ENGINE ---
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
    return df.iloc[idx]['answer'] if similarity_scores[0][idx] > 0.2 else "Neural Signal Mismatch."

# --- 6. SIDEBAR ---
with st.sidebar:
    st.markdown('<p style="color:#00e5ff; font-weight:bold;">INTERFACE SETTINGS</p>', unsafe_allow_html=True)
    if st.button("CLEAR ACTIVE CACHE"):
        st.session_state.history = []
        st.rerun()
    st.markdown("---")
    st.write("**DEVELOPER:** Helly")
    st.write("**ENGINE:** NPCL V2.0")

# --- 7. SECOND PAGE: CHATBOT INTERFACE (YOUR ORIGINAL DESIGN) ---
st.markdown('<p class="voxa-header">NOVO CHATTERIX</p>', unsafe_allow_html=True)
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
    st.session_state.history.append({"q": final_query, "a": ans})
    st.rerun()

for item in reversed(st.session_state.history):
    st.markdown(f'''
    <div class="chat-card">
        <b style="color:#00e5ff">SIGNAL:</b> {item["q"]}<br><br>
        <b style="color:#b452ff">NOVO:</b> {item["a"]}
    </div>
    ''', unsafe_allow_html=True)
