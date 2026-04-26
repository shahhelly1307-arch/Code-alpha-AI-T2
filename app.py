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
        return pd.DataFrame({
            "question": ["What is Nova AI?", "Who developed this interface?", "How does the matching engine work?", "What parameters define TF-IDF V2?", "What is the MASF Framework?", "Is this system real-time?"], 
            "answer": ["Nova AI is a neural-synapse chatbot.", "Developed by Helly Shah.", "It uses Cosine Similarity and TF-IDF vectors.", "Frequency and Inverse Document Frequency.", "Multi-Agent System Framework.", "Yes, signal processing is instantaneous."]
        })

df = load_data()

# --- 4. THE VOXA UI (LEFT-SIDE GLOW REPAIR) ---
st.set_page_config(page_title="Nova Chatterix", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Silkscreen:wght@700&display=swap');
    
    /* Global transparency for all Streamlit layers */
    [data-testid="stAppViewContainer"] {
        background-color: #000000 !important;
    }

    /* Target the Main Area and ensure the glow is at the VERY edges */
    [data-testid="stAppViewMainArea"] {
        background: 
            radial-gradient(circle at 2% 50%, rgba(0, 229, 255, 0.4) 0%, transparent 40%),
            radial-gradient(circle at 98% 50%, rgba(180, 82, 255, 0.3) 0%, transparent 40%) !important;
        background-color: #000000 !important;
    }

    /* Prevent sidebar from blocking the left-side glow */
    [data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.7) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(0, 229, 255, 0.2);
    }

    .main, .block-container {
        background: transparent !important;
    }

    * {
        font-family: 'Silkscreen', cursive !important;
    }

    .voxa-header {
        font-size: clamp(2.5rem, 6vw, 6rem) !important; 
        font-weight: 700 !important;
        background: linear-gradient(to right, #00e5ff, #b452ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        text-transform: uppercase;
        margin-bottom: 0px;
        filter: drop-shadow(0 0 10px rgba(0, 229, 255, 0.5));
    }

    .orbital-line {
        height: 2px;
        background: linear-gradient(90deg, transparent, #00e5ff, transparent);
        width: 70%;
        margin: 0 auto 30px auto;
    }

    div.stButton > button {
        background: rgba(0, 229, 255, 0.05) !important;
        color: #00e5ff !important;
        border: 2px solid #00e5ff !important;
        border-radius: 4px !important;
        transition: 0.3s;
        width: 100%;
    }

    div.stButton > button:hover {
        background: rgba(0, 229, 255, 0.2) !important;
        box-shadow: 0 0 20px #00e5ff;
        color: #fff !important;
    }

    .stTextInput input {
        background-color: rgba(0, 0, 0, 0.8) !important;
        border: 2px solid #00e5ff !important;
        color: #ffffff !important;
    }

    .chat-card {
        background: rgba(10, 10, 10, 0.7);
        border: 1px solid rgba(0, 229, 255, 0.3);
        border-left: 5px solid #00e5ff;
        padding: 15px;
        margin-bottom: 10px;
        backdrop-filter: blur(5px);
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

# --- 6. INTERFACE ---
st.markdown('<p class="voxa-header">NOVA CHATTERIX</p>', unsafe_allow_html=True)
st.markdown('<div class="orbital-line"></div>', unsafe_allow_html=True)

# Sidebar Credentials
with st.sidebar:
    st.markdown("### SYSTEM SETTINGS")
    st.write("**DEV:** Helly Shah")
    if st.button("CLEAR HISTORY"):
        st.session_state.history = []
        st.rerun()

if 'history' not in st.session_state:
    st.session_state.history = []

st.markdown("### 📡 ACTIVE FREQUENCIES")
cols = st.columns(3)
clicked_q = None

for i, q in enumerate(df['question'].tolist()):
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
        <span style="color:#00e5ff; font-weight:bold;">SIGNAL:</span> {item["q"]}<br>
        <span style="color:#b452ff; font-weight:bold;">NOVA:</span> {item["a"]}
    </div>
    ''', unsafe_allow_html=True)
