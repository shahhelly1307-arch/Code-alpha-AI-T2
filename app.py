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
        return pd.DataFrame({"question": ["System Status"], "answer": ["Database signal active. Please check faqs.json file."]})

df = load_data()

# --- 4. THE NOVO CHATTERIX UI & ANIMATION ---
st.set_page_config(page_title="Novo Chatterix", layout="wide")

# Initialize session state for the splash screen
if 'visited' not in st.session_state:
    st.session_state.visited = False

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=Silkscreen:wght@700&display=swap');
    
    html, body, [class*="css"], .stText, .stMarkdown, .stButton, input, label {
        font-family: 'Silkscreen', cursive !important;
    }

    /* SHARED BACKGROUND */
    .stApp {
        background-color: #020205 !important; 
        background-image: 
            radial-gradient(circle at 20% 30%, rgba(0, 229, 255, 0.15) 0%, transparent 50%), 
            radial-gradient(circle at 80% 70%, rgba(180, 82, 255, 0.15) 0%, transparent 50%),
            linear-gradient(135deg, #001214 0%, #11001c 100%) !important;
        background-attachment: fixed !important;
    }

    /* HERO / SPLASH STYLING */
    .hero-container {
        text-align: center;
        padding-top: 80px;
        position: relative;
    }

    .arch-glow {
        position: absolute;
        width: 700px;
        height: 350px;
        border: 3px solid rgba(0, 229, 255, 0.4);
        border-radius: 50% 50% 0 0 / 100% 100% 0 0;
        bottom: -20px;
        left: 50%;
        transform: translateX(-50%);
        background: radial-gradient(circle at 50% 100%, rgba(0, 229, 255, 0.1), transparent 70%);
        box-shadow: 0 -10px 40px rgba(0, 229, 255, 0.2);
    }

    .hero-title {
        font-size: 5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00e5ff, #b452ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 20px;
    }

    /* CHAT CARD STYLING */
    .chat-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(0, 229, 255, 0.2);
        border-left: 5px solid #00e5ff;
        padding: 20px;
        margin-bottom: 15px;
        backdrop-filter: blur(10px);
        border-radius: 4px;
    }
    
    .stTextInput input {
        background-color: rgba(20, 20, 20, 0.7) !important;
        border: 1px solid rgba(0, 229, 255, 0.5) !important;
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 5. SPLASH SCREEN LOGIC ---
placeholder = st.empty()

if not st.session_state.visited:
    with placeholder.container():
        st.markdown("""
            <div class="hero-container">
                <div class="arch-glow"></div>
                <h1 class="hero-title">NOVA CHATTERIX</h1>
            </div>
            """, unsafe_allow_html=True)
        
        # Central Animated Robot
        _, col_c, _ = st.columns([1, 2, 1])
        with col_c:
            if lottie_main:
                st_lottie(lottie_main, height=400, key="splash_robot")
        
        st.markdown("<h3 style='text-align:center; color:#00e5ff; opacity:0.6;'>INITIALIZING NEURAL LINK...</h3>", unsafe_allow_html=True)
        
        time.sleep(5)  # The 5-second delay
        st.session_state.visited = True
        st.rerun()

# --- 6. MAIN CHATBOT INTERFACE (LOADS AFTER 5 SECS) ---
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

with st.sidebar:
    st.markdown("### SYSTEM CONTROLS")
    if st.button("RESET INTERFACE"):
        st.session_state.history = []
        st.session_state.visited = False # Go back to splash
        st.rerun()
    st.write("**ENGINE:** NPCL V2.0")
    st.write("**STATUS:** 🟢 ACTIVE")

# Header for Chat Page
st.markdown("<h1 style='text-align:center; color:#00e5ff; letter-spacing: -2px;'>NOVA CHATTERIX</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid rgba(0, 229, 255, 0.2);'>", unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []

# Quick Questions
st.markdown("### 📡 FREQUENCY SELECT")
questions_list = df['question'].tolist()
cols = st.columns(3)
clicked_q = None

for i, q in enumerate(questions_list):
    if cols[i % 3].button(q, key=f"q_{i}"):
        clicked_q = q

# Input Form
with st.form(key='chat_form', clear_on_submit=True):
    user_query = st.text_input("Transmit Command:", placeholder="Awaiting Signal...")
    submit = st.form_submit_button("TRANSMIT")

final_query = clicked_q if clicked_q else (user_query if submit else None)

if final_query:
    ans = get_response(final_query)
    st.session_state.history.append({"q": final_query, "a": ans})
    st.rerun()

# Chat Logs
for item in reversed(st.session_state.history):
    st.markdown(f'''
    <div class="chat-card">
        <b style="color:#00e5ff">SIGNAL:</b> {item["q"]}<br><br>
        <b style="color:#b452ff">NOVA:</b> {item["a"]}
    </div>
    ''', unsafe_allow_html=True)
