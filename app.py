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

# Loading the animated robot for the left-side placement
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

# Applying the VOXA colors: Obsidian base with Blue/Cyan highlights
VOXA_CYAN = "#00FFA3" 
VOXA_OBSIDIAN = "#000a12"
VOXA_BLUE_GLOW = "#001f2d"

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DotGothic16&display=swap');
    
    html, body, [class*="css"], .stText, .stMarkdown, .stButton, input, label {{
        font-family: 'DotGothic16', sans-serif !important;
        color: #ffffff;
    }}

    /* Matches the background from your VOXA image */
    .stApp {{
        background: radial-gradient(circle at center, {VOXA_BLUE_GLOW} 0%, {VOXA_OBSIDIAN} 100%) !important;
        color: #ffffff;
    }}
    
    /* Pixelated header inspired by the image logo */
    .voxa-header {{
        font-family: 'DotGothic16', sans-serif !important;
        font-size: clamp(3rem, 10vw, 8rem) !important; 
        font-weight: 700 !important;
        color: {VOXA_CYAN} !important; 
        text-align: center;
        text-transform: uppercase;
        margin-top: 50px;
        margin-bottom: 20px;
        text-shadow: 0 0 30px rgba(0, 255, 163, 0.4);
        letter-spacing: -2px;
        white-space: nowrap;
    }}

    .orbital-line {{
        height: 3px;
        background: linear-gradient(90deg, transparent, {VOXA_CYAN}, transparent);
        width: 70%;
        margin: 0 auto 50px auto;
        box-shadow: 0 0 15px {VOXA_CYAN};
    }}

    /* Sidebar and button styling */
    [data-testid="stSidebar"] {{
        background-color: {VOXA_OBSIDIAN} !important;
        border-right: 1px solid rgba(0, 255, 163, 0.2);
    }}

    div.stButton > button {{
        background: rgba(0, 255, 163, 0.05) !important;
        color: {VOXA_CYAN} !important;
        border: 2px solid {VOXA_CYAN} !important;
        border-radius: 4px !important;
    }}

    .stTextInput input {{
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 2px solid {VOXA_CYAN} !important;
        color: #ffffff !important;
    }}

    .chat-card {{
        background: rgba(0, 255, 163, 0.03);
        border-left: 4px solid {VOXA_CYAN};
        padding: 20px;
        margin-bottom: 15px;
    }}
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
    return "Signal lost. Data not found in current sector."

# --- 6. SIDEBAR ---
with st.sidebar:
    st.title("INTERFACE SETTINGS")
    if st.button("CLEAR CACHE"):
        st.session_state.history = []
        st.rerun()
    st.write("**DEVELOPER:** Helly Shah")
    st.markdown(f'<p style="color:{VOXA_CYAN};">● SYSTEM: ONLINE</p>', unsafe_allow_html=True)

# --- 7. MAIN INTERFACE ---
st.markdown('<p class="voxa-header">NOVA CHATTERIX</p>', unsafe_allow_html=True)
st.markdown('<div class="orbital-line"></div>', unsafe_allow_html=True)

# PLACING THE ANIMATED ROBOT ON THE LEFT
if lottie_main:
    col_robot, col_empty = st.columns([1, 3]) 
    with col_robot:
        st_lottie(lottie_main, height=180, key="main_robot_left")

if 'history' not in st.session_state:
    st.session_state.history = []

# Questions Grid
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

# History display
for item in reversed(st.session_state.history):
    st.markdown(f'''
    <div class="chat-card">
        <b style="color:{VOXA_CYAN}">SIGNAL:</b> {item["q"]}<br><br>
        <b>NOVA:</b> {item["a"]}
    </div>
    ''', unsafe_allow_html=True)
