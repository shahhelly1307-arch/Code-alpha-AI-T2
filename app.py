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

# --- 2. ASSET LOADING (Increased Stability) ---
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=10) # Added longer timeout to prevent failure
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# Selecting the precise orbital robot for the Image 2 feel
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

# --- 4. UI CONFIGURATION (IMAGE 1 & 2 AESTHETIC) ---
st.set_page_config(page_title="Nova Chatterix", layout="wide")

st.markdown("""
    <style>
    /* Global Pixel Font Application */
    @import url('https://fonts.googleapis.com/css2?family=DotGothic16&display=swap');
    
    html, body, [class*="css"], .stText, .stMarkdown, .stButton, input, label {
        font-family: 'DotGothic16', sans-serif !important;
    }

    /* Background from Image 2: Deep Obsidian Black with centered Cyan Aura */
    .stApp {
        background: radial-gradient(circle at center, #001f1f 0%, #050505 100%) !important;
        color: #ffffff;
    }
    
    /* Pixel Header (Image 1 Style) */
    .voxa-header {
        font-size: clamp(2.5rem, 8vw, 6rem);
        color: #00FFA3; /* THE EXACT BRIGHT CYAN FROM VOXA */
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 12px;
        margin-top: 30px;
        margin-bottom: 0px;
        text-shadow: 0 0 25px rgba(0, 255, 163, 0.8);
    }

    /* Orbital Line (Image 2 aesthetic) */
    .orbital-line {
        height: 2px;
        background: linear-gradient(90deg, transparent, #00FFA3, transparent);
        width: 80%;
        margin: 0 auto 30px auto;
        box-shadow: 0 0 15px #00FFA3;
    }

    /* Large Pixelated Buttons */
    div.stButton > button {
        background: rgba(0, 255, 163, 0.05) !important;
        color: #00FFA3 !important;
        border: 2px solid #00FFA3 !important;
        font-size: 1.1rem !important;
        text-transform: uppercase;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background: #00FFA3 !important;
        color: #000 !important;
        box-shadow: 0 0 20px #00FFA3;
    }

    /* Form Fields Styling */
    .stTextInput input {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid #00FFA3 !important;
        color: #ffffff !important;
    }

    /* Chat History Boxes */
    .chat-card {
        background: rgba(0, 255, 163, 0.03);
        border: 1px solid rgba(0, 255, 163, 0.2);
        padding: 20px;
        margin-bottom: 15px;
        border-radius: 4px;
        border-left: 4px solid #00FFA3;
    }

    [data-testid="stSidebar"] {
        background-color: #050505 !important;
        border-right: 1px solid rgba(0, 255, 163, 0.2);
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
    st.markdown("---")
    # restore Helly Shah's profile: User Summary
    st.write("**Developer:** Helly Shah")
    st.write("**Project:** Nova Chatterix")
    st.markdown('<p style="color:#00FFA3;">● SYSTEM: ONLINE</p>', unsafe_allow_html=True)

# --- 7. MAIN INTERFACE ---
# Pixelated Title (Image 1 style) and orbital line
st.markdown('<p class="voxa-header">NOVA CHATTERIX</p>', unsafe_allow_html=True)
st.markdown('<div class="orbital-line"></div>', unsafe_allow_html=True)

# THE KEY CHANGES: Robot moves to Top-Left, Status text is removed
if lottie_main:
    # Use a column layout for precise positioning
    col_rob, col_spacer = st.columns([1, 4])
    with col_rob:
        st_lottie(lottie_main, height=200, key="main_robot")

# Searching text is removed completely

if 'history' not in st.session_state:
    st.session_state.history = []

# Question Signals list loop (Image 2 style)
st.markdown("### 📡 ACTIVE FREQUENCIES")
questions_list = df['question'].tolist()
cols = st.columns(3)
clicked_q = None

# Using enumerate and modulus to evenly distribute buttons across columns
for i, q in enumerate(questions_list):
    if cols[i % 3].button(q, key=f"q_{i}"):
        clicked_q = q

# Input Transmission
with st.form(key='chat_form', clear_on_submit=True):
    user_query = st.text_input("Transmit Command:", placeholder="Enter your signal command...")
    submit = st.form_submit_button("TRANSMIT")

final_query = clicked_q if clicked_q else (user_query if submit else None)

if final_query:
    ans = get_response(final_query)
    st.session_state.history.append({"q": final_query, "a": ans})
    st.rerun()

# Neural History
for item in reversed(st.session_state.history):
    st.markdown(f'''
    <div class="chat-card">
        <b style="color:#00FFA3">SIGNAL:</b> {item["q"]}<br><br>
        <b>NOVA:</b> {item["a"]}
    </div>
    ''', unsafe_allow_html=True)
