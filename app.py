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

# --- 2. ASSET LOADING (Stable & Fast) ---
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

# High-fidelity tech robot
lottie_main = load_lottieurl("https://lottie.host/8172906e-8360-449e-9988-0320a1630985/B1pU53Y34i.json")

# --- 3. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        with open('faqs.json', 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except:
        return pd.DataFrame({"question": ["System Online"], "answer": ["Please ensure faqs.json is in your repository."]})

df = load_data()

# --- 4. UI CONFIGURATION (IMAGE 1 & 2 AESTHETIC) ---
st.set_page_config(page_title="Nova Chatterix", layout="wide")

st.markdown("""
    <style>
    /* 1. Global Font: Pixel Style (Image 1) */
    @import url('https://fonts.googleapis.com/css2?family=DotGothic16&display=swap');
    
    html, body, [class*="css"], .stText, .stMarkdown, .stButton, input, label {
        font-family: 'DotGothic16', sans-serif !important;
    }

    /* 2. Background: Center-Glow Radial Gradient (Image 2) */
    .stApp {
        background: radial-gradient(circle at center, #003333 0%, #050505 100%);
        color: #ffffff;
    }
    
    /* 3. Voxa Pixel Header (Image 1) */
    .voxa-header {
        font-size: clamp(3rem, 10vw, 6rem);
        color: #00FFA3;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 15px;
        margin-top: 40px;
        text-shadow: 0 0 25px rgba(0, 255, 163, 0.8);
    }

    /* 4. Orbital Glow Line (Image 2) */
    .orbital-line {
        height: 2px;
        background: linear-gradient(90deg, transparent, #00FFA3, transparent);
        width: 80%;
        margin: 0 auto 30px auto;
        box-shadow: 0 0 15px #00FFA3;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #050505 !important;
        border-right: 1px solid rgba(0, 255, 163, 0.2);
    }

    /* Large Pixelated Buttons */
    div.stButton > button {
        background: rgba(0, 255, 163, 0.08) !important;
        color: #00FFA3 !important;
        border: 2px solid #00FFA3 !important;
        font-size: 1.1rem !important;
        padding: 12px !important;
        text-transform: uppercase;
        transition: 0.3s ease;
    }
    div.stButton > button:hover {
        background: #00FFA3 !important;
        color: #000 !important;
        box-shadow: 0 0 20px #00FFA3;
    }

    /* Form Fields */
    .stTextInput input {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid #00FFA3 !important;
        color: #ffffff !important;
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

# --- 6. SIDEBAR (Your Profile) ---
with st.sidebar:
    st.title("SETTINGS")
    if st.button("CLEAR HISTORY"):
        st.session_state.history = []
        st.rerun()
    st.markdown("---")
    st.write("**Developer:** Helly Shah")
    st.write("**Project:** Nova Chatterix")
    st.markdown('<p style="color:#00FFA3;">● SYSTEM: ONLINE</p>', unsafe_allow_html=True)

# --- 7. MAIN INTERFACE ---
st.markdown('<p class="voxa-header">NOVA CHATTERIX</p>', unsafe_allow_html=True)
st.markdown('<div class="orbital-line"></div>', unsafe_allow_html=True)

# Hero Section (Image 2 Style - No extra text)
if lottie_main:
    st_lottie(lottie_main, height=350, key="hero_robot")

if 'history' not in st.session_state:
    st.session_state.history = []

# Dynamic Signal Buttons (Restored)
st.markdown("### 📡 ACTIVE SIGNALS")
questions_list = df['question'].tolist()
cols = st.columns(3)
clicked_q = None

for i, q in enumerate(questions_list):
    if cols[i % 3].button(q, key=f"btn_{i}"):
        clicked_q = q

# Input Transmission
with st.form(key='chat_form', clear_on_submit=True):
    user_query = st.text_input("Transmit Command:", placeholder="Enter your query...")
    submit = st.form_submit_button("SEND")

final_query = clicked_q if clicked_q else (user_query if submit else None)

if final_query:
    ans = get_response(final_query)
    st.session_state.history.append({"q": final_query, "a": ans})
    st.rerun()

# Neural History
for item in reversed(st.session_state.history):
    st.markdown(f'''
    <div style="background:rgba(255,255,255,0.02); border-left:4px solid #00FFA3; padding:15px; margin-bottom:10px;">
        <b style="color:#00FFA3">SIGNAL:</b> {item["q"]}<br><br>
        <b>NOVA:</b> {item["a"]}
    </div>
    ''', unsafe_allow_html=True)
