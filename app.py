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
    # Fixed non-breaking space errors and ensured clean indentation
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

lottie_main = load_lottieurl("https://lottie.host/8172906e-8360-449e-9988-0320a1630985/B1pU53Y34i.json")

# --- 3. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        with open('faqs.json', 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except Exception:
        # Fallback if file is missing
        return pd.DataFrame({
            "question": ["System Status", "What is Novo Chatterix?"], 
            "answer": ["Database signal active. Please check faqs.json file.", "A high-end NLP interface built for demo purposes."]
        })

df = load_data()

# --- 4. THE UI CONFIG ---
st.set_page_config(page_title="Novo Chatterix", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Silkscreen:wght@700&display=swap');
    
    html, body, [class*="css"], .stText, .stMarkdown, .stButton, input, label {
        font-family: 'Silkscreen', cursive !important;
    }

    .stApp {
        background-color: #050505 !important; 
        background-image: 
            radial-gradient(circle at 0% 0%, rgba(0, 229, 255, 0.2) 0%, transparent 60%), 
            radial-gradient(circle at 100% 100%, rgba(180, 82, 255, 0.2) 0%, transparent 60%),
            linear-gradient(135deg, #001214 0%, #11001c 100%) !important;
        background-attachment: fixed !important;
        background-size: cover;
        color: #ffffff;
    }
    
    .voxa-header {
        font-family: 'Silkscreen', cursive !important;
        font-size: clamp(2.5rem, 6vw, 8rem) !important; 
        font-weight: 700 !important;
        background: linear-gradient(to right, #00e5ff, #b452ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        text-transform: uppercase;
        white-space: nowrap; 
        letter-spacing: -3px;
        margin-top: 10px;
        margin-bottom: 0px;
        filter: drop-shadow(0 0 15px rgba(0, 229, 255, 0.4));
    }

    .orbital-line {
        height: 3px;
        background: linear-gradient(90deg, transparent, #00e5ff, transparent);
        width: 80%;
        margin: 0 auto 40px auto;
        box-shadow: 0 0 15px #00e5ff;
    }

    [data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.8) !important;
        border-right: 1px solid rgba(0, 229, 255, 0.3);
    }

    .sidebar-label {
        color: #00e5ff;
        font-size: 0.9rem;
        letter-spacing: 2px;
        font-weight: bold;
    }

    div.stButton > button {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #ffffff !important;
        border: 1px solid rgba(0, 229, 255, 0.5) !important;
        border-radius: 50px !important;
        font-size: 0.85rem !important;
        transition: 0.3s;
        padding: 10px 20px;
    }

    div.stButton > button:hover {
        background: rgba(0, 229, 255, 0.2) !important;
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.4);
        border: 1px solid #00e5ff !important;
    }
    
    .stTextInput input {
        background-color: rgba(20, 20, 20, 0.7) !important;
        border: 1px solid rgba(0, 229, 255, 0.5) !important;
        color: #ffffff !important;
        border-radius: 10px !important;
    }

    .chat-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(0, 229, 255, 0.2);
        border-left: 5px solid #00e5ff;
        padding: 20px;
        margin-bottom: 15px;
        backdrop-filter: blur(10px);
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 5. LOGIC ENGINE ---
def get_response(user_input):
    dev_query = user_input.lower()
    if any(x in dev_query for x in ["developed", "creator", "who made", "built by", "developer"]):
        return "This interface was developed by Helly as a professional demonstration of NLP and advanced UI design."
        
    processed_input = preprocess_text(user_input)
    # Re-build corpus for vectorization
    corpus = df['question'].apply(preprocess_text).tolist()
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Transform user input based on the same vectorizer
    user_vec = vectorizer.transform([processed_input])
    similarity_scores = cosine_similarity(user_vec, tfidf_matrix)
    
    idx = similarity_scores.argmax()
    if similarity_scores[0][idx] > 0.2:
        return df.iloc[idx]['answer']
    return "Neural Signal Mismatch. Data not found in current frequency."

# --- 6. SIDEBAR ---
with st.sidebar:
    st.markdown('<p class="sidebar-label">INTERFACE SETTINGS</p>', unsafe_allow_html=True)
    if st.button("CLEAR ACTIVE CACHE"):
        st.session_state.history = []
        st.rerun()
    
    st.markdown("---")
    st.markdown('<p class="sidebar-label">SYSTEM CREDENTIALS</p>', unsafe_allow_html=True)
    st.write("**DEVELOPER:** Helly")
    st.write("**ENGINE:** NPCL V2.0")
    
    st.markdown("---")
    st.markdown('<p style="color:#00e5ff; font-weight:bold;">● SYSTEM: ONLINE</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#b452ff; font-weight:bold;">● SIGNAL: ACTIVE</p>', unsafe_allow_html=True)

# --- 7. MAIN INTERFACE ---
st.markdown('<p class="voxa-header">NOVO CHATTERIX</p>', unsafe_allow_html=True)
st.markdown('<div class="orbital-line"></div>', unsafe_allow_html=True)

if lottie_main:
    col_rob, _ = st.columns([1, 4])
    with col_rob:
        st_lottie(lottie_main, height=150, key="main_robot")

if 'history' not in st.session_state:
    st.session_state.history = []

st.markdown("### 📡 ACTIVE FREQUENCIES")
questions_list = df['question'].tolist()
cols = st.columns(3)
clicked_q = None

# Quick-action buttons
for i, q in enumerate(questions_list):
    if cols[i % 3].button(q, key=f"q_{i}"):
        clicked_q = q

# Chat Input Form
with st.form(key='chat_form', clear_on_submit=True):
    user_query = st.text_input("Transmit Command:", placeholder="AWAITING SIGNAL...")
    submit = st.form_submit_button("TRANSMIT")

final_query = clicked_q if clicked_q else (user_query if submit else None)

if final_query:
    ans = get_response(final_query)
    st.session_state.history.insert(0, {"q": final_query, "a": ans})
    st.rerun()

# Display Chat History
for item in st.session_state.history:
    st.markdown(f'''
    <div class="chat-card">
        <b style="color:#00e5ff">SIGNAL:</b> {item["q"]}<br><br>
        <b style="color:#b452ff">NOVO:</b> {item["a"]}
    </div>
    ''', unsafe_allow_html=True)
