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
        return pd.DataFrame({"question": ["Who developed this?"], "answer": ["Developed by Helly Shah."]})

df = load_data()

# --- 4. UI CONFIGURATION (IMAGE 1 & 2 ACCURACY) ---
st.set_page_config(page_title="Nova Chatterix", layout="wide")

st.markdown("""
    <style>
    /* Import Pixel Font for Title */
    @import url('https://fonts.googleapis.com/css2?family=DotGothic16&display=swap');

    /* Background: Image 2 Style - Deep Radial Cyan Glow */
    .stApp {
        background: radial-gradient(circle at center, #004d4d 0%, #050505 100%) !important;
        color: #ffffff;
    }
    
    /* THE BIG FONT: Image 1 Voxa Style */
    .voxa-title {
        font-family: 'DotGothic16', sans-serif !important;
        font-size: 100px !important; /* Extremely Big Font */
        color: #00FFA3 !important;   /* Same Cyan Color as Voxa */
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 25px;       /* Wide Spacing like Voxa */
        margin-top: 10px;
        margin-bottom: 0px;
        line-height: 1.2;
        text-shadow: 0 0 35px rgba(0, 255, 163, 0.9); /* Strong Glow */
    }

    /* Orbital Line: Image 2 Glow */
    .orbital-line {
        height: 3px;
        background: linear-gradient(90deg, transparent, #00FFA3, transparent);
        width: 70%;
        margin: 0 auto 30px auto;
        box-shadow: 0 0 20px #00FFA3;
    }

    /* Clean UI for the rest */
    div.stButton > button {
        background: rgba(0, 255, 163, 0.1) !important;
        color: #00FFA3 !important;
        border: 1px solid #00FFA3 !important;
        border-radius: 5px !important;
    }
    div.stButton > button:hover {
        background: #00FFA3 !important;
        color: #000 !important;
    }

    [data-testid="stSidebar"] {
        background-color: #050505 !important;
        border-right: 1px solid rgba(0, 255, 163, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 5. LOGIC ENGINE (Restoring Helly Shah Credits) ---
def get_response(user_input):
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
    return "Signal weak. Query not found."

# --- 6. SIDEBAR ---
with st.sidebar:
    st.title("SETTINGS")
    if st.button("CLEAR HISTORY"):
        st.session_state.history = []
        st.rerun()
    st.markdown("---")
    st.write("**Developer:** Helly Shah")
    st.write("**Project:** Nova Chatterix")
    st.markdown('<p style="color:#00FFA3;">● STATUS: ONLINE</p>', unsafe_allow_html=True)

# --- 7. MAIN INTERFACE ---
# THE BIG VOXA-STYLE HEADER
st.markdown('<p class="voxa-title">NOVA<br>CHATTERIX</p>', unsafe_allow_html=True)
st.markdown('<div class="orbital-line"></div>', unsafe_allow_html=True)

# Central Robot Hero
if lottie_robot:
    st_lottie(lottie_robot, height=300, key="hero_bot")

if 'history' not in st.session_state:
    st.session_state.history = []

# Question Buttons
st.markdown("### ⚡ ACTIVE SIGNALS")
questions = df['question'].tolist()
cols = st.columns(3)
clicked_q = None
for i, q in enumerate(questions[:6]):
    if cols[i % 3].button(q, key=f"q_{i}"):
        clicked_q = q

# Input
with st.form(key='chat_form', clear_on_submit=True):
    user_query = st.text_input("Enter Signal:", placeholder="Transmit your command...")
    submit = st.form_submit_button("SEND")

final_query = clicked_q if clicked_q else (user_query if submit else None)

if final_query:
    ans = get_response(final_query)
    st.session_state.history.append({"q": final_query, "a": ans})
    st.rerun()

# History
for item in reversed(st.session_state.history):
    st.markdown(f'''
    <div style="background:rgba(255,255,255,0.02); border-left:4px solid #00FFA3; padding:15px; margin-bottom:10px;">
        <b style="color:#00FFA3">SIGNAL:</b> {item["q"]}<br><br>
        <b>NOVA:</b> {item["a"]}
    </div>
    ''', unsafe_allow_html=True)
