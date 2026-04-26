import streamlit as st
import pandas as pd
import json
import nltk
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

# --- 2. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        with open('faqs.json', 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except:
        return pd.DataFrame({"question": ["System Status"], "answer": ["Database signal active."]})

df = load_data()

# --- 3. UI CONFIGURATION (VOXA AESTHETIC REPLICA) ---
st.set_page_config(page_title="Nova Chatterix", layout="wide")

# THE EXACT COLOR PALETTE FROM IMAGE 10
VOXA_BG = "radial-gradient(circle at center, #001f2d 0%, #000a12 100%)" # Deep Obsidian Blue
VOXA_CYAN = "#00FFA3" # Brighter, Neon Cyan from 'Try Voxa' text
VOXA_GLOW = "rgba(0, 255, 163, 0.4)"

st.markdown(f"""
    <style>
    /* 1. Global Font and Styling */
    @import url('https://fonts.googleapis.com/css2?family=DotGothic16&display=swap');
    
    html, body, [class*="css"], .stText, .stMarkdown, .stButton, input, label {{
        font-family: 'DotGothic16', sans-serif !important;
        color: #ffffff;
    }}

    /* 2. THE BACKGROUND: Deep Blue Radial Glow from Image 10 */
    .stApp {{
        background: {VOXA_BG} !important;
        color: #ffffff;
    }}
    
    /* 3. THE HEADER: Pixel, Huge, Neon Cyan (Image 10 style) */
    .voxa-header {{
        font-family: 'DotGothic16', sans-serif !important;
        font-size: clamp(3rem, 10vw, 8.5rem); /* Extremely large */
        color: {VOXA_CYAN} !important; 
        text-align: center;
        text-transform: uppercase;
        margin-top: 50px;
        margin-bottom: 20px;
        text-shadow: 0 0 30px {VOXA_GLOW}; /* Heavy neon glow */
        letter-spacing: -2px; /* Tight VOXA-style letter spacing */
        font-weight: 700;
    }}

    .orbital-line {{
        height: 3px;
        background: linear-gradient(90deg, transparent, {VOXA_CYAN}, transparent);
        width: 70%;
        margin: 0 auto 50px auto;
        box-shadow: 0 0 15px {VOXA_CYAN};
    }}

    /* Buttons & Inputs updated for the VOXA theme */
    div.stButton > button {{
        background: rgba(0, 255, 163, 0.05) !important;
        color: {VOXA_CYAN} !important;
        border: 2px solid {VOXA_CYAN} !important;
        font-size: 1.1rem !important;
        border-radius: 4px !important;
    }}
    div.stButton > button:hover {{
        background: {VOXA_CYAN} !important;
        color: #000 !important;
        box-shadow: 0 0 20px {VOXA_CYAN};
    }}

    .stTextInput input {{
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 2px solid {VOXA_CYAN} !important;
        color: #ffffff !important;
    }}

    .chat-card {{
        background: rgba(0, 255, 163, 0.03);
        border: 1px solid rgba(0, 255, 163, 0.2);
        padding: 20px;
        margin-bottom: 15px;
        border-radius: 4px;
        border-left: 4px solid {VOXA_CYAN};
    }}

    [data-testid="stSidebar"] {{
        background-color: #000a12 !important;
        border-right: 1px solid rgba(0, 255, 163, 0.2);
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 4. LOGIC ENGINE ---
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

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("SETTINGS")
    if st.button("CLEAR HISTORY"):
        st.session_state.history = []
        st.rerun()
    st.write("**Developer:** Helly Shah")
    st.markdown(f'<p style="color:{VOXA_CYAN};">● SYSTEM: ONLINE</p>', unsafe_allow_html=True)

# --- 6. MAIN INTERFACE ---
# Title is Huge, Pixelated, Neon Cyan, and sits on ONE single line
st.markdown('<p class="voxa-header">NOVA CHATTERIX</p>', unsafe_allow_html=True)
st.markdown('<div class="orbital-line"></div>', unsafe_allow_html=True)

# NOTE: No hand graphic is displayed.

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
    user_query = st.text_input("Transmit Command:", placeholder="ENTER SIGNAL...")
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
        <b style="color:{VOXA_CYAN}">SIGNAL:</b> {item["q"]}<br><br>
        <b>NOVA:</b> {item["a"]}
    </div>
    ''', unsafe_allow_html=True)
