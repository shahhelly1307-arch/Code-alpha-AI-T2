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

lottie_robot = load_lottieurl("https://lottie.host/8172906e-8360-449e-9988-0320a1630985/B1pU53Y34i.json")

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

# Corrected vibrant colors
VOXA_CYAN = "#00FFA3" 
VOXA_OBSIDIAN = "#000a12"
VOXA_BLUE_GLOW = "#001f2d"

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DotGothic16&display=swap');
    
    html, body, [class*="css"], .stText, .stMarkdown, .stButton, input, label {{
        font-family: 'DotGothic16', sans-serif !important;
        color: {VOXA_CYAN} !important;
    }}

    .stApp {{
        background: radial-gradient(circle at center, {VOXA_BLUE_GLOW} 0%, {VOXA_OBSIDIAN} 100%) !important;
    }}
    
    .voxa-header {{
        font-size: clamp(3rem, 10vw, 7rem) !important; 
        font-weight: 700 !important;
        color: {VOXA_CYAN} !important; 
        text-align: center;
        text-transform: uppercase;
        margin-top: 20px;
        text-shadow: 0 0 20px rgba(0, 255, 163, 0.6);
        white-space: nowrap;
    }}

    .orbital-line {{
        height: 2px;
        background: linear-gradient(90deg, transparent, {VOXA_CYAN}, transparent);
        width: 80%;
        margin: 0 auto 40px auto;
        box-shadow: 0 0 10px {VOXA_CYAN};
    }}

    div.stButton > button {{
        background: transparent !important;
        color: {VOXA_CYAN} !important;
        border: 1px solid {VOXA_CYAN} !important;
        width: 100%;
        box-shadow: inset 0 0 5px {VOXA_CYAN};
        border-radius: 4px;
    }}

    .stTextInput input {{
        background-color: rgba(0, 255, 163, 0.05) !important;
        border: 1px solid {VOXA_CYAN} !important;
        color: white !important;
    }}

    .chat-card {{
        background: rgba(0, 255, 163, 0.03);
        border: 1px solid {VOXA_CYAN};
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 4px;
    }}

    [data-testid="stSidebar"] {{
        background-color: {VOXA_OBSIDIAN} !important;
        border-right: 1px solid {VOXA_CYAN};
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 5. CHAT LOGIC ---
def get_response(user_input):
    processed_input = preprocess_text(user_input)
    corpus = df['question'].apply(preprocess_text).tolist()
    corpus.append(processed_input)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    idx = similarity_scores.argmax()
    return df.iloc[idx]['answer'] if similarity_scores[0][idx] > 0.2 else "No signal found."

# --- 6. SIDEBAR (Robot ABOVE Settings) ---
with st.sidebar:
    # ROBOT AT THE TOP
    if lottie_robot:
        st_lottie(lottie_robot, height=200, key="sidebar_robot")
    
    st.markdown("---")
    st.title("SETTINGS")
    if st.button("CLEAR CACHE"):
        st.session_state.history = []
        st.rerun()
        
    st.write("**DEV:** Helly Shah")
    st.markdown(f'<p style="color:{VOXA_CYAN};">● SYSTEM: ONLINE</p>', unsafe_allow_html=True)

# --- 7. MAIN INTERFACE ---
st.markdown('<p class="voxa-header">NOVA CHATTERIX</p>', unsafe_allow_html=True)
st.markdown('<div class="orbital-line"></div>', unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []

st.markdown(f"### <span style='color:{VOXA_CYAN}'>📡 ACTIVE FREQUENCIES</span>", unsafe_allow_html=True)

# Frequency buttons
q_list = df['question'].tolist()
btn_cols = st.columns(3)
selected_q = None
for i, q in enumerate(q_list):
    if btn_cols[i % 3].button(q, key=f"btn_{i}"):
        selected_q = q

# Input
with st.form(key='chat_form', clear_on_submit=True):
    u_query = st.text_input("Transmit Command:", placeholder="ENTER SIGNAL...")
    submit = st.form_submit_button("TRANSMIT")

final_q = selected_q if selected_q else (u_query if submit else None)

if final_q:
    response = get_response(final_q)
    st.session_state.history.append({"q": final_q, "a": response})
    st.rerun()

# Chat History
for item in reversed(st.session_state.history):
    st.markdown(f'''
    <div class="chat-card">
        <b style="color:{VOXA_CYAN}">SIGNAL:</b> {item["q"]}<br>
        <b style="color:white">NOVA:</b> {item["a"]}
    </div>
    ''', unsafe_allow_html=True)
