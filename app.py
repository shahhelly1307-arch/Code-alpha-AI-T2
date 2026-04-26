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

# --- 2. DATA & ASSETS ---
@st.cache_data
def load_data():
    try:
        with open('faqs.json', 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame({"question": ["System Status"], "answer": ["Database signal active. Check faqs.json."]})

def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=10)
        return r.json() if r.status_code == 200 else None
    except: return None

df = load_data()
lottie_main = load_lottieurl("https://lottie.host/8172906e-8360-449e-9988-0320a1630985/B1pU53Y34i.json")

# --- 3. PAGE CONFIG & STYLING ---
st.set_page_config(page_title="Novo Chatterix", layout="wide")

# Global Font and Background Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Silkscreen:wght@400;700&display=swap');
    
    html, body, [class*="css"], .stText, .stMarkdown, .stButton, input, label {
        font-family: 'Silkscreen', cursive !important;
    }

    .stApp {
        background-color: #000000 !important;
        background-image: 
            linear-gradient(90deg, 
                rgba(0, 229, 255, 0.3) 0%, 
                rgba(180, 82, 255, 0.3) 50%, 
                rgba(0, 229, 255, 0.3) 100%) !important;
        background-attachment: fixed !important;
        background-size: cover;
    }

    .intro-text {
        font-size: 5rem;
        text-align: center;
        margin-top: 20%;
        background: linear-gradient(to right, #00e5ff, #b452ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: pulse 1.5s infinite;
    }

    @keyframes pulse {
        0% { opacity: 0.5; }
        50% { opacity: 1; }
        100% { opacity: 0.5; }
    }

    .voxa-header {
        font-size: clamp(2rem, 5vw, 6rem);
        font-weight: 700;
        background: linear-gradient(to right, #00e5ff, #b452ff, #00e5ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        letter-spacing: -2px;
    }

    .chat-card {
        background: rgba(0, 0, 0, 0.6);
        border: 1px solid #00e5ff;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. TIMED INTRO LOGIC ---
if 'initialized' not in st.session_state:
    placeholder = st.empty()
    with placeholder.container():
        st.markdown('<p class="intro-text">HELLY</p>', unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#00e5ff;'>INITIALIZING NEURAL LINK...</p>", unsafe_allow_html=True)
        time.sleep(2)
    placeholder.empty()
    st.session_state.initialized = True

# --- 5. MAIN CHATBOT INTERFACE ---
def get_response(user_input):
    processed_input = preprocess_text(user_input)
    corpus = df['question'].apply(preprocess_text).tolist()
    corpus.append(processed_input)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    idx = similarity_scores.argmax()
    return df.iloc[idx]['answer'] if similarity_scores[0][idx] > 0.2 else "Signal Lost."

# Sidebar
with st.sidebar:
    st.write("### SYSTEM BY HELLY")
    if st.button("RESET CACHE"):
        st.session_state.history = []
        st.rerun()

# Layout
st.markdown('<p class="voxa-header">NOVO CHATTERIX</p>', unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []

# Action Buttons
st.markdown("### 📡 ACTIVE FREQUENCIES")
questions_list = df['question'].tolist()
cols = st.columns(3)
clicked_q = None

for i, q in enumerate(questions_list):
    if cols[i % 3].button(q, key=f"q_{i}"):
        clicked_q = q

# Input
with st.form(key='chat_form', clear_on_submit=True):
    user_query = st.text_input("Transmit Command:", placeholder="AWAITING SIGNAL...")
    submit = st.form_submit_button("TRANSMIT")

final_query = clicked_q if clicked_q else (user_query if submit else None)

if final_query:
    ans = get_response(final_query)
    st.session_state.history.append({"q": final_query, "a": ans})
    st.rerun()

# Display Chat
for item in reversed(st.session_state.history):
    st.markdown(f'''
    <div class="chat-card">
        <b style="color:#00e5ff">SIGNAL:</b> {item["q"]}<br>
        <b style="color:#b452ff">NOVO:</b> {item["a"]}
    </div>
    ''', unsafe_allow_html=True)
    
