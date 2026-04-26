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

# High-quality floating robot animation
lottie_main = load_lottieurl("https://lottie.host/8172906e-8360-449e-9988-0320a1630985/B1pU53Y34i.json")

# --- 3. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        with open('faqs.json', 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame({"question": ["System Status"], "answer": ["Database signal active. Please check faqs.json file."]})

df = load_data()

# --- 4. THE NOVO CHATTERIX UI ---
st.set_page_config(page_title="Novo Chatterix", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=Silkscreen:wght@700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"], .stText, .stMarkdown, .stButton, input, label {
        font-family: 'Inter', sans-serif !important;
    }

    .stApp {
        background-color: #020205 !important; 
        background-image: 
            radial-gradient(circle at 20% 30%, rgba(0, 229, 255, 0.15) 0%, transparent 50%), 
            radial-gradient(circle at 80% 70%, rgba(180, 82, 255, 0.15) 0%, transparent 50%) !important;
        background-attachment: fixed !important;
    }

    /* Hero Section Container */
    .hero-container {
        text-align: center;
        padding-top: 50px;
        position: relative;
    }

    /* The Half Circle / Arch Glow */
    .arch-glow {
        position: absolute;
        width: 800px;
        height: 400px;
        border: 2px solid rgba(0, 229, 255, 0.3);
        border-radius: 50% 50% 0 0 / 100% 100% 0 0;
        bottom: -50px;
        left: 50%;
        transform: translateX(-50%);
        background: radial-gradient(circle at 50% 100%, rgba(0, 229, 255, 0.1), transparent 70%);
        box-shadow: 0 -10px 50px rgba(0, 229, 255, 0.1);
        z-index: 0;
    }

    /* Bot Name Styling from Image */
    .bot-title {
        font-size: 4rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0px;
        position: relative;
        z-index: 1;
        letter-spacing: -1px;
    }

    .bot-subtitle {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00e5ff, #b452ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: -10px;
        position: relative;
        z-index: 1;
    }

    /* Custom Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #00e5ff, #b452ff) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 12px 35px !important;
        font-weight: 600 !important;
        transition: 0.3s ease;
    }

    div.stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.5);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(2, 2, 5, 0.9) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    .chat-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 10px;
        backdrop-filter: blur(5px);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 5. HERO SECTION ---
st.markdown("""
    <div class="hero-container">
        <div class="arch-glow"></div>
        <h1 class="bot-title">The Future of</h1>
        <h2 class="bot-subtitle">Nova Chatterix</h2>
    </div>
    """, unsafe_allow_html=True)

# Animated Robot (Lottie) placed centrally in the "arch"
col_l, col_c, col_r = st.columns([1, 2, 1])
with col_c:
    if lottie_main:
        st_lottie(lottie_main, height=300, key="hero_robot")

# --- 6. LOGIC ENGINE ---
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

# --- 7. SIDEBAR ---
with st.sidebar:
    st.markdown("### SYSTEM SETTINGS")
    if st.button("RESET SESSION"):
        st.session_state.history = []
        st.rerun()
    st.write("**DEVELOPER:** Helly")
    st.write("**STATUS:** 🟢 ONLINE")

# --- 8. CHAT INTERFACE ---
if 'history' not in st.session_state:
    st.session_state.history = []

# Quick Frequency Buttons
st.markdown("<br>", unsafe_allow_html=True)
questions_list = df['question'].tolist()
cols = st.columns(3)
clicked_q = None

for i, q in enumerate(questions_list):
    if cols[i % 3].button(q, key=f"q_{i}"):
        clicked_q = q

# Chat Input
with st.form(key='chat_form', clear_on_submit=True):
    user_query = st.text_input("", placeholder="Ask Nova anything...")
    submit = st.form_submit_button("SEND COMMAND")

final_query = clicked_q if clicked_q else (user_query if submit else None)

if final_query:
    ans = get_response(final_query)
    st.session_state.history.append({"q": final_query, "a": ans})
    st.rerun()

# Display History
for item in reversed(st.session_state.history):
    st.markdown(f'''
    <div class="chat-card">
        <b style="color:#00e5ff">USER:</b> {item["q"]}<br><br>
        <b style="color:#b452ff">NOVA:</b> {item["a"]}
    </div>
    ''', unsafe_allow_html=True)
