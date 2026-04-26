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
        return pd.DataFrame({"question": ["System Status"], "answer": ["Database signal active."]})

df = load_data()

# --- 4. UI CONFIGURATION (PRECISE VOXA CLONE) ---
st.set_page_config(page_title="Nova Chatterix", layout="wide")

st.markdown("""
    <style>
    /* Professional Pixel font with high weight */
    @import url('https://fonts.googleapis.com/css2?family=VT323&display=swap');
    
    html, body, [class*="css"], .stText, .stMarkdown, .stButton, input, label {
        font-family: 'VT323', monospace !important;
    }

    /* Background: Deep Black with the precise Cyan/Blue center glow from the image */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #0a2d3d 0%, #000000 80%) !important;
        color: #ffffff;
    }
    
    /* THE VOXA STYLE HEADER: ULTRA BIG, THICK, AND GLOWING */
    .voxa-header {
        font-family: 'VT323', monospace !important;
        font-size: 140px !important; /* Massive Scale */
        font-weight: 900 !important;
        color: #add8e6 !important; /* The Icy Blue from the photo */
        text-align: center;
        text-transform: uppercase;
        line-height: 0.85;
        letter-spacing: -5px;
        margin-top: 40px;
        margin-bottom: 20px;
        /* Layered shadows to simulate the 'thick block' and neon glow */
        text-shadow: 
            3px 3px 0px #1a3a4a,
            6px 6px 0px #1a3a4a,
            0 0 30px rgba(173, 216, 230, 0.6),
            0 0 60px rgba(173, 216, 230, 0.3);
    }

    /* Styled horizontal line to match the aura */
    .orbital-line {
        height: 3px;
        background: linear-gradient(90deg, transparent, #add8e6, transparent);
        width: 70%;
        margin: 0 auto 50px auto;
        opacity: 0.8;
        box-shadow: 0 0 15px #add8e6;
    }

    /* Buttons and Inputs adjusted for the Icy Blue theme */
    div.stButton > button {
        background: rgba(173, 216, 230, 0.05) !important;
        color: #add8e6 !important;
        border: 2px solid #add8e6 !important;
        border-radius: 0px !important;
        font-size: 1.3rem !important;
    }
    
    div.stButton > button:hover {
        background: #add8e6 !important;
        color: #000 !important;
        box-shadow: 0 0 20px #add8e6;
    }

    .stTextInput input {
        background-color: rgba(0, 0, 0, 0.8) !important;
        border: 1px solid #add8e6 !important;
        color: #ffffff !important;
        font-size: 1.5rem !important;
        height: 60px;
    }

    .chat-card {
        background: rgba(10, 45, 61, 0.4);
        border: 1px solid #add8e6;
        padding: 25px;
        margin-bottom: 20px;
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
    st.markdown('<p style="color:#add8e6;">● SYSTEM: ONLINE</p>', unsafe_allow_html=True)

# --- 7. MAIN INTERFACE ---
# Title: Big, Thick, and exactly the VOXA color
st.markdown('<p class="voxa-header">NOVA<br>CHATTERIX</p>', unsafe_allow_html=True)
st.markdown('<div class="orbital-line"></div>', unsafe_allow_html=True)

# Robot Positioning
if lottie_main:
    col_rob, _ = st.columns([1, 4])
    with col_rob:
        st_lottie(lottie_main, height=200, key="main_robot")

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
    user_query = st.text_input("Transmit Command:", placeholder="ENTER SIGNAL...")
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
        <b style="color:#add8e6">SIGNAL:</b> {item["q"]}<br><br>
        <b>NOVA:</b> {item["a"]}
    </div>
    ''', unsafe_allow_html=True)
