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
    except:
        # Fallback data if file is missing
        return pd.DataFrame({"question": ["System Status"], "answer": ["Database signal active."]})

df = load_data()

# --- 4. UI CONFIGURATION (VOXA AESTHETIC REPLICA) ---
st.set_page_config(page_title="Nova Chatterix", layout="wide")

st.markdown("""
    <style>
    /* Importing a heavy Pixel font */
    @import url('https://fonts.googleapis.com/css2?family=Silkscreen:wght@400;700&display=swap');
    
    html, body, [class*="css"], .stText, .stMarkdown, .stButton, input, label {
        font-family: 'Silkscreen', sans-serif !important;
    }

    /* Background: Deep Obsidian with centered Cyan Aura as requested */
    .stApp {
        background: radial-gradient(circle at center, #001a1a 0%, #000000 100%) !important;
        color: #ffffff;
    }
    
    /* THE VOXA STYLE HEADER */
    .voxa-header {
        font-family: 'Silkscreen', sans-serif !important;
        font-size: clamp(3rem, 12vw, 10rem); /* Extremely large */
        font-weight: 700;
        color: #99ccff; /* The light icy blue/cyan from the image */
        text-align: center;
        text-transform: uppercase;
        margin-top: 50px;
        margin-bottom: -20px;
        line-height: 1;
        letter-spacing: -2px;
        /* This text-shadow creates the 'thick' blocky look and the neon glow */
        text-shadow: 
            5px 5px 0px #1a3a3a, 
            0 0 50px rgba(153, 204, 255, 0.4);
    }

    .orbital-line {
        height: 4px;
        background: linear-gradient(90deg, transparent, #99ccff, transparent);
        width: 60%;
        margin: 40px auto;
        box-shadow: 0 0 20px #99ccff;
    }

    /* Input and Button Styling */
    div.stButton > button {
        background: rgba(153, 204, 255, 0.1) !important;
        color: #99ccff !important;
        border: 2px solid #99ccff !important;
        font-weight: bold;
    }
    
    .stTextInput input {
        background-color: rgba(0, 0, 0, 0.5) !important;
        border: 2px solid #99ccff !important;
        color: #ffffff !important;
        font-size: 1.2rem;
    }

    .chat-card {
        background: rgba(153, 204, 255, 0.05);
        border: 1px solid rgba(153, 204, 255, 0.3);
        padding: 20px;
        margin-bottom: 15px;
        border-radius: 2px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 5. LOGIC ENGINE ---
def get_response(user_input):
    processed_input = preprocess_text(user_input)
    corpus = df['question'].apply(preprocess_text).tolist()
    corpus.append(processed_input)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    idx = similarity_scores.argmax()
    if similarity_scores[0][idx] > 0.2:
        return df.iloc[idx]['answer']
    return "Neural Signal Mismatch. Data not found."

# --- 6. SIDEBAR ---
with st.sidebar:
    st.title("SETTINGS")
    if st.button("CLEAR HISTORY"):
        st.session_state.history = []
        st.rerun()
    st.write("**Developer:** Helly Shah")
    st.markdown('<p style="color:#99ccff;">● SYSTEM: ONLINE</p>', unsafe_allow_html=True)

# --- 7. MAIN INTERFACE ---
# Title styled to match the image precisely
st.markdown('<p class="voxa-header">NOVA<br>CHATTERIX</p>', unsafe_allow_html=True)
st.markdown('<div class="orbital-line"></div>', unsafe_allow_html=True)

# Robot Positioning (Top Left)
if lottie_main:
    col_rob, _ = st.columns([1, 4])
    with col_rob:
        st_lottie(lottie_main, height=180, key="main_robot")

if 'history' not in st.session_state:
    st.session_state.history = []

# Questions Grid
st.markdown("### 📡 ACTIVE FREQUENCIES")
questions_list = df['question'].tolist()
cols = st.columns(3)
clicked_q = None

for i, q in enumerate(questions_list):
    if cols[i % 3].button(q, key=f"q_{i}"):
        clicked_q = q

# Input
with st.form(key='chat_form', clear_on_submit=True):
    user_query = st.text_input("Transmit Command:", placeholder="Enter signal...")
    submit = st.form_submit_button("TRANSMIT")

final_query = clicked_q if clicked_q else (user_query if submit else None)

if final_query:
    ans = get_response(final_query)
    st.session_state.history.append({"q": final_query, "a": ans})
    st.rerun()

# History
for item in reversed(st.session_state.history):
    st.markdown(f'''
    <div class="chat-card">
        <b style="color:#99ccff">SIGNAL:</b> {item["q"]}<br><br>
        <b>NOVA:</b> {item["a"]}
    </div>
    ''', unsafe_allow_html=True)
