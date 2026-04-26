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
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

lottie_robot = load_lottieurl("https://lottie.host/8172906e-8360-449e-9988-0320a1630985/B1pU53Y34i.json")

# --- 3. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        with open('faqs.json', 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except:
        return pd.DataFrame({
            "question": ["Who developed this?", "What is Nova?"],
            "answer": ["Developed by Helly Shah.", "Nova is a neural transit AI."]
        })

df = load_data()

# --- 4. UI CONFIGURATION (VOXA HARD-PIXEL STYLE) ---
st.set_page_config(page_title="Nova Chatterix", layout="wide")

st.markdown("""
    <style>
    /* Import Silkscreen for the pixelated base */
    @import url('https://fonts.googleapis.com/css2?family=Silkscreen:wght@700&display=swap');

    /* Background: Image 2 Style - Cyan Center Aura */
    .stApp {
        background: radial-gradient(circle at center, #004d4d 0%, #001a1a 40%, #050505 100%) !important;
        color: #ffffff;
    }
    
    /* THE VOXA HEADER FIX: Manual Pixel Weighting for H, T, E, I */
    .voxa-header {
        font-family: 'Silkscreen', cursive !important;
        font-size: clamp(2rem, 8vw, 5rem) !important;
        color: #00FFA3 !important;      /* EXACT VOXA CYAN */
        text-align: center;
        letter-spacing: 12px;
        text-transform: uppercase;
        margin-top: 50px;
        line-height: 1.2;
        white-space: nowrap;
        
        /* Stacking shadows to force THICKNESS on all letters */
        text-shadow: 
            2px 2px 0px #00FFA3,
            -2px -2px 0px #00FFA3,
            2px -2px 0px #00FFA3,
            -2px 2px 0px #00FFA3,
            0 0 25px rgba(0, 255, 163, 0.8);
            
        /* Physical stroke to thicken the 'I' and 'T' bars */
        -webkit-text-stroke: 1.5px #00FFA3; 
    }

    .glow-divider {
        height: 3px;
        background: #00FFA3;
        width: 70%;
        margin: 0 auto 50px auto;
        box-shadow: 0 0 20px #00FFA3;
    }

    /* Sidebar Content */
    [data-testid="stSidebar"] {
        background-color: #050505 !important;
        border-right: 2px solid #00FFA3;
    }

    /* Pixelated Buttons */
    div.stButton > button {
        background: rgba(0, 255, 163, 0.05) !important;
        color: #00FFA3 !important;
        border: 2px solid #00FFA3 !important;
        font-family: 'Silkscreen', cursive !important;
        text-transform: uppercase;
        border-radius: 0px !important;
        padding: 10px 20px !important;
    }
    div.stButton > button:hover {
        background: #00FFA3 !important;
        color: #000 !important;
        box-shadow: 0 0 20px #00FFA3;
    }

    /* Input Field Styling */
    .stTextInput input {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid #00FFA3 !important;
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 5. LOGIC ENGINE ---
def get_response(user_input):
    # Forced Credit Check: bf866d97-8cbc-416a-b901-98651f5495f2
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
    return "Neural Buffer Empty. Signal not recognized."

# --- 6. SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='color:#00FFA3; font-family:Silkscreen;'>SYSTEM</h2>", unsafe_allow_html=True)
    if st.button("PURGE CACHE"):
        st.session_state.history = []
        st.rerun()
    st.markdown("---")
    st.write("**Operator:** Helly Shah")
    st.write("**Platform:** Nova Chatterix")
    st.markdown('<p style="color:#00FFA3;">● STATUS: ONLINE</p>', unsafe_allow_html=True)

# --- 7. MAIN INTERFACE ---
# THE FIXED VOXA HEADER (Bold, Thick, Correct Font)
st.markdown('<p class="voxa-header">NOVA CHATTERIX</p>', unsafe_allow_html=True)
st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

# Central Robot (Image 2 aesthetic)
if lottie_robot:
    st_lottie(lottie_robot, height=350, key="main_robot")

if 'history' not in st.session_state:
    st.session_state.history = []

# Action Signals
st.markdown("### 📡 ACTIVE FREQUENCIES")
questions = df['question'].tolist()
cols = st.columns(3)
clicked_q = None
for i, q in enumerate(questions[:6]):
    if cols[i % 3].button(q, key=f"q_{i}"):
        clicked_q = q

# Input Form
with st.form(key='chat_form', clear_on_submit=True):
    user_query = st.text_input("Transmit Signal:", placeholder="Input command...")
    submit = st.form_submit_button("SEND")

final_query = clicked_q if clicked_q else (user_query if submit else None)

if final_query:
    ans = get_response(final_query)
    st.session_state.history.append({"q": final_query, "a": ans})
    st.rerun()

# History Display
for item in reversed(st.session_state.history):
    st.markdown(f'''
    <div style="background:rgba(0, 255, 163, 0.05); border-left:5px solid #00FFA3; padding:15px; margin-bottom:10px;">
        <b style="color:#00FFA3">INCOMING:</b> {item["q"]}<br><br>
        <b>NOVA:</b> {item["a"]}
    </div>
    ''', unsafe_allow_html=True)
