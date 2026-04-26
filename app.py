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
        if r.status_code != 200: return None
        return r.json()
    except: return None

lottie_main = load_lottieurl("https://lottie.host/8172906e-8360-449e-9988-0320a1630985/B1pU53Y34i.json")

# --- 3. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        with open('faqs.json', 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except:
        return pd.DataFrame({
            "question": ["What is Nova AI?", "Who developed this interface?", "How does the matching engine work?", "What parameters define TF-IDF V2?", "What is the MASF Framework?", "Is this system real-time?"], 
            "answer": ["Nova AI is a neural-synapse chatbot.", "Developed by Helly Shah.", "It uses Cosine Similarity and TF-IDF vectors.", "Frequency and Inverse Document Frequency.", "Multi-Agent System Framework.", "Yes, signal processing is instantaneous."]
        })

df = load_data()

# --- 4. THE UI REPAIR (SCREEN SIZE & LEFT SIDE FIX) ---
st.set_page_config(page_title="Nova Chatterix", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Silkscreen:wght@700&display=swap');
    
    /* 1. REMOVE EXCESSIVE PADDING & REDUCE WIDTH */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 0rem !important;
        padding-left: 5% !important;
        padding-right: 5% !important;
        max-width: 95% !important; /* This stops the 'Too Big' feel */
    }

    /* 2. DUAL-SIDE GLOW (Adjusted for visibility) */
    [data-testid="stAppViewContainer"] {
        background-color: #000000 !important;
        background-image: 
            radial-gradient(circle at 8% 50%, rgba(0, 229, 255, 0.4) 0%, transparent 35%),
            radial-gradient(circle at 92% 50%, rgba(180, 82, 255, 0.3) 0%, transparent 35%) !important;
        background-attachment: fixed !important;
    }

    /* 3. FIX SIDEBAR OVERLAP */
    [data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.8) !important;
        border-right: 1px solid rgba(0, 229, 255, 0.2);
    }

    /* 4. CONTENT STYLING */
    * { font-family: 'Silkscreen', cursive !important; color: white; }

    .voxa-header {
        font-size: clamp(2rem, 5vw, 4.5rem) !important; 
        background: linear-gradient(to right, #00e5ff, #b452ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        filter: drop-shadow(0 0 10px rgba(0, 229, 255, 0.5));
        margin-top: -20px;
    }

    .orbital-line {
        height: 2px;
        background: linear-gradient(90deg, transparent, #00e5ff, transparent);
        width: 60%;
        margin: 0 auto 20px auto;
    }

    /* BUTTONS */
    div.stButton > button {
        background: rgba(0, 0, 0, 0.6) !important;
        color: #00e5ff !important;
        border: 2px solid #00e5ff !important;
        border-radius: 4px !important;
        width: 100%;
        padding: 10px;
        font-size: 0.75rem !important;
    }

    div.stButton > button:hover {
        box-shadow: 0 0 15px #00e5ff;
        background: rgba(0, 229, 255, 0.1) !important;
    }

    /* INPUT FIELD */
    .stTextInput input {
        background-color: rgba(10, 10, 10, 0.9) !important;
        border: 2px solid #00e5ff !important;
        height: 45px;
    }

    .chat-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(0, 229, 255, 0.2);
        border-left: 5px solid #00e5ff;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 4px;
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
    return "Neural Signal Mismatch. Signal lost."

# --- 6. INTERFACE ---
st.markdown('<p class="voxa-header">NOVA CHATTERIX</p>', unsafe_allow_html=True)
st.markdown('<div class="orbital-line"></div>', unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []

st.markdown("### 📡 ACTIVE FREQUENCIES")
cols = st.columns(3)
clicked_q = None

# Grid display
for i, q in enumerate(df['question'].tolist()[:6]):
    if cols[i % 3].button(q, key=f"q_{i}"):
        clicked_q = q

# Transmit Area
with st.form(key='chat_form', clear_on_submit=True):
    user_query = st.text_input("Transmit Command:", placeholder="AWAITING SIGNAL...")
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
        <span style="color:#00e5ff; font-weight:bold;">SIGNAL:</span> {item["q"]}<br>
        <span style="color:#b452ff; font-weight:bold;">NOVA:</span> {item["a"]}
    </div>
    ''', unsafe_allow_html=True)
