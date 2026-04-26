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

# The robot animation for the left side
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

# Theme Colors based on your VOXA image
VOXA_CYAN = "#00FFA3" 
VOXA_OBSIDIAN = "#000a12"
VOXA_BLUE_GLOW = "#001f2d"

# Use double curly braces {{ }} for CSS to avoid f-string SyntaxErrors
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DotGothic16&display=swap');
    
    html, body, [class*="css"], .stText, .stMarkdown, .stButton, input, label {{
        font-family: 'DotGothic16', sans-serif !important;
        color: #ffffff;
    }}

    .stApp {{
        background: radial-gradient(circle at center, {VOXA_BLUE_GLOW} 0%, {VOXA_OBSIDIAN} 100%) !important;
    }}
    
    .voxa-header {{
        font-family: 'DotGothic16', sans-serif !important;
        font-size: clamp(3rem, 10vw, 8rem) !important; 
        font-weight: 700 !important;
        color: {VOXA_CYAN} !important; 
        text-align: center;
        text-transform: uppercase;
        margin-top: 20px;
        text-shadow: 0 0 30px rgba(0, 255, 163, 0.4);
        white-space: nowrap;
    }}

    .orbital-line {{
        height: 3px;
        background: linear-gradient(90deg, transparent, {VOXA_CYAN}, transparent);
        width: 80%;
        margin: 0 auto 40px auto;
        box-shadow: 0 0 15px {VOXA_CYAN};
    }}

    div.stButton > button {{
        background: rgba(0, 255, 163, 0.1) !important;
        color: {VOXA_CYAN} !important;
        border: 2px solid {VOXA_CYAN} !important;
        border-radius: 4px !important;
        width: 100%;
    }}

    .stTextInput input {{
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 2px solid {VOXA_CYAN} !important;
        color: #ffffff !important;
    }}

    .chat-card {{
        background: rgba(0, 255, 163, 0.05);
        border: 1px solid {VOXA_CYAN};
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 5px;
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
    return "No signal found in this sector."

# --- 6. SIDEBAR ---
with st.sidebar:
    st.title("SETTINGS")
    if st.button("CLEAR CACHE"):
        st.session_state.history = []
        st.rerun()
    st.write("**DEV:** Helly Shah")
    st.markdown(f'<p style="color:{VOXA_CYAN};">● ONLINE</p>', unsafe_allow_html=True)

# --- 7. MAIN INTERFACE ---
st.markdown('<p class="voxa-header">NOVA CHATTERIX</p>', unsafe_allow_html=True)
st.markdown('<div class="orbital-line"></div>', unsafe_allow_html=True)

# Create two columns for the robot on the left and chat on the right
col_robot, col_chat = st.columns([1, 2])

with col_robot:
    if lottie_robot:
        st_lottie(lottie_robot, height=400, key="robot_animation")

with col_chat:
    if 'history' not in st.session_state:
        st.session_state.history = []

    st.markdown(f"### <span style='color:{VOXA_CYAN}'>📡 ACTIVE FREQUENCIES</span>", unsafe_allow_html=True)
    
    # Frequency buttons
    q_list = df['question'].tolist()
    btn_cols = st.columns(2)
    selected_q = None
    for i, q in enumerate(q_list):
        if btn_cols[i % 2].button(q, key=f"btn_{i}"):
            selected_q = q

    # Input form
    with st.form(key='chat_form', clear_on_submit=True):
        u_query = st.text_input("Transmit Command:", placeholder="AWAITING SIGNAL...")
        submit = st.form_submit_button("TRANSMIT")

    final_q = selected_q if selected_q else (u_query if submit else None)

    if final_q:
        response = get_response(final_q)
        st.session_state.history.append({"q": final_q, "a": response})
        st.rerun()

    for item in reversed(st.session_state.history):
        st.markdown(f'''
        <div class="chat-card">
            <b style="color:{VOXA_CYAN}">SIGNAL:</b> {item["q"]}<br>
            <b>NOVA:</b> {item["a"]}
        </div>
        ''', unsafe_allow_html=True)
