import streamlit as st
import pd
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
        return pd.DataFrame({"question": ["System Status"], "answer": ["Database signal active."]})

df = load_data()

# --- 4. THE VOXA REPLICA UI ---
st.set_page_config(page_title="Nova Chatterix", layout="wide")

st.markdown("""
    <style>
    /* Importing a much chunkier pixel font */
    @import url('https://fonts.googleapis.com/css2?family=Silkscreen:wght@400;700&display=swap');
    
    html, body, [class*="css"], .stText, .stMarkdown, .stButton, input, label {
        font-family: 'Silkscreen', cursive !important;
    }

    /* BACKGROUND: Deep Black with the precise Icy Blue center glow */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #0a2d3d 0%, #000000 85%) !important;
        color: #ffffff;
    }
    
    /* THE VOXA HEADER: MASSIVE, THICK, SINGLE LINE */
    .voxa-header {
        font-family: 'Silkscreen', cursive !important;
        font-size: clamp(3rem, 7.5vw, 12rem) !important; 
        font-weight: 700 !important;
        color: #00e5ff !important; /* Brighter VOXA Cyan */
        text-align: center;
        text-transform: uppercase;
        white-space: nowrap; 
        letter-spacing: -4px;
        margin-top: 10px;
        margin-bottom: 0px;
        /* STACKED SHADOWS: This makes the font look "Thick" and "Blocky" */
        text-shadow: 
            2px 2px 0px #004d4d, 
            4px 4px 0px #004d4d, 
            6px 6px 0px #004d4d,
            0 0 40px rgba(0, 229, 255, 0.7);
    }

    .orbital-line {
        height: 4px;
        background: linear-gradient(90deg, transparent, #00e5ff, transparent);
        width: 85%;
        margin: 0 auto 40px auto;
        box-shadow: 0 0 20px #00e5ff;
    }

    /* SIDEBAR: Restoring Left Panel Visibility */
    [data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.95) !important;
        border-right: 2px solid #00e5ff;
    }

    .sidebar-label {
        color: #00e5ff;
        font-size: 1rem;
        letter-spacing: 2px;
        font-weight: bold;
    }

    div.stButton > button {
        background: rgba(0, 229, 255, 0.05) !important;
        color: #00e5ff !important;
        border: 2px solid #00e5ff !important;
        border-radius: 0px !important;
    }
    
    .stTextInput input {
        background-color: rgba(0, 0, 0, 0.8) !important;
        border: 2px solid #00e5ff !important;
        color: #ffffff !important;
    }

    .chat-card {
        background: rgba(10, 45, 61, 0.4);
        border: 1px solid #00e5ff;
        padding: 20px;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 5. LOGIC ENGINE ---
def get_response(user_input):
    dev_query = user_input.lower()
    if "developed" in dev_query or "creator" in dev_query or "who made" in dev_query:
        return "This project was developed by Helly as a technical demonstration of NLP and professional UI integration."
        
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

# --- 6. SIDEBAR (NPCL & STATUS) ---
with st.sidebar:
    st.markdown('<p class="sidebar-label">SETTINGS</p>', unsafe_allow_html=True)
    if st.button("CLEAR HISTORY"):
        st.session_state.history = []
        st.rerun()
    
    st.markdown("---")
    st.markdown('<p class="sidebar-label">NPCL CONTROL</p>', unsafe_allow_html=True)
    st.write("**DEVELOPER:** Helly Shah")
    st.write("**ENGINE:** NPCL V2.0")
    
    st.markdown("---")
    st.markdown('<p style="color:#00e5ff;">● SYSTEM: ONLINE</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#00e5ff;">● SIGNAL: ACTIVE</p>', unsafe_allow_html=True)

# --- 7. MAIN INTERFACE ---
st.markdown('<p class="voxa-header">NOVA CHATTERIX</p>', unsafe_allow_html=True)
st.markdown('<div class="orbital-line"></div>', unsafe_allow_html=True)

if lottie_main:
    col_rob, _ = st.columns([1, 4])
    with col_rob:
        st_lottie(lottie_main, height=180, key="main_robot")

if 'history' not in st.session_state:
    st.session_state.history = []

st.markdown("### 📡 ACTIVE FREQUENCIES")
questions_list = df['question'].tolist()
cols = st.columns(3)
clicked_q = None

for i, q in enumerate(questions_list):
    if cols[i % 3].button(q, key=f"q_{i}"):
        clicked_q = q

with st.form(key='chat_form', clear_on_submit=True):
    user_query = st.text_input("Transmit Command:", placeholder="ENTER SIGNAL...")
    submit = st.form_submit_button("TRANSMIT")

final_query = clicked_q if clicked_q else (user_query if submit else None)

if final_query:
    ans = get_response(final_query)
    st.session_state.history.append({"q": final_query, "a": ans})
    st.rerun()

for item in reversed(st.session_state.history):
    st.markdown(f'''
    <div class="chat-card">
        <b style="color:#00e5ff">SIGNAL:</b> {item["q"]}<br><br>
        <b>NOVA:</b> {item["a"]}
    </div>
    ''', unsafe_allow_html=True)
