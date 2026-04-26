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

# --- 2. ASSET LOADING (Stable & Error-Free) ---
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

# Clean tech robot matching the Image 2 vibe
lottie_main = load_lottieurl("https://lottie.host/8172906e-8360-449e-9988-0320a1630985/B1pU53Y34i.json")

# --- 3. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        # Load your external knowledge base
        with open('faqs.json', 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except:
        # Fallback dataset with the correct developer name
        return pd.DataFrame({
            "question": ["Who developed this interface?", "What is Nova AI?"],
            "answer": [
                "This project was developed by Helly Shah as a technical demonstration of NLP and professional UI integration.",
                "Nova is a next-gen neural interface assistant."
            ]
        })

df = load_data()

# --- 4. UI CONFIGURATION (STRICT IMAGE 1 & 2 THEME) ---
st.set_page_config(page_title="Nova Chatterix", layout="wide")

st.markdown("""
    <style>
    /* 1. Universal Pixel Font (Image 1 Style) */
    @import url('https://fonts.googleapis.com/css2?family=DotGothic16&display=swap');
    
    * {
        font-family: 'DotGothic16', sans-serif !important;
    }

    /* 2. Deep Obsidian Gradient Background (Image 2) */
    .stApp {
        background: radial-gradient(circle at center, #002b2b 0%, #050505 100%);
        color: #ffffff;
    }
    
    /* 3. Pixelated Title (Image 1) */
    .voxa-header {
        font-size: clamp(3rem, 10vw, 6rem);
        color: #00FFA3;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 15px;
        margin-top: 30px;
        text-shadow: 0 0 25px rgba(0, 255, 163, 0.7);
    }

    /* 4. Glowing Orbital Line (Image 2) */
    .orbital-line {
        height: 2px;
        background: linear-gradient(90deg, transparent, #00FFA3, transparent);
        width: 80%;
        margin: 0 auto 30px auto;
        box-shadow: 0 0 15px #00FFA3;
    }

    /* Pixelated Buttons and Inputs */
    div.stButton > button {
        background: rgba(0, 255, 163, 0.05) !important;
        color: #00FFA3 !important;
        border: 2px solid #00FFA3 !important;
        text-transform: uppercase;
        padding: 10px 20px !important;
    }
    
    div.stButton > button:hover {
        background: #00FFA3 !important;
        color: #000 !important;
        box-shadow: 0 0 20px #00FFA3;
    }

    .stTextInput input {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid #00FFA3 !important;
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] {
        background-color: #050505 !important;
        border-right: 1px solid rgba(0, 255, 163, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 5. UPDATED LOGIC (Helly Shah Restoration) ---
def get_response(user_input):
    # Hardcoded override for developer query to ensure Helly Shah is named
    if "developer" in user_input.lower() or "who developed" in user_input.lower():
        return "This project was developed by Helly Shah as a technical demonstration of NLP and professional UI integration."
    
    processed_input = preprocess_text(user_input)
    corpus = df['question'].apply(preprocess_text).tolist()
    corpus.append(processed_input)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    idx = similarity_scores.argmax()
    
    if similarity_scores[0][idx] > 0.2:
        return df.iloc[idx]['answer']
    return "Neural buffer empty. Signal not recognized."

# --- 6. SIDEBAR (Helly Shah Identity) ---
with st.sidebar:
    st.title("SETTINGS")
    if st.button("CLEAR HISTORY"):
        st.session_state.history = []
        st.rerun()
    st.markdown("---")
    st.write("**Developer:** Helly Shah")
    st.write("**Project:** Nova Chatterix")
    st.markdown('<p style="color:#00FFA3;">● SYSTEM: ONLINE</p>', unsafe_allow_html=True)

# --- 7. MAIN INTERFACE ---
st.markdown('<p class="voxa-header">NOVA CHATTERIX</p>', unsafe_allow_html=True)
st.markdown('<div class="orbital-line"></div>', unsafe_allow_html=True)

# Central Robot (Image 2 Landing)
if lottie_main:
    st_lottie(lottie_main, height=320, key="main_robot")

if 'history' not in st.session_state:
    st.session_state.history = []

# Restored Questions List
st.markdown("### ⚡ ACTIVE SIGNALS")
questions_list = df['question'].tolist()
cols = st.columns(3)
clicked_q = None

for i, q in enumerate(questions_list[:6]): # Show top 6 questions
    if cols[i % 3].button(q, key=f"q_{i}"):
        clicked_q = q

# Input Transmission
with st.form(key='chat_form', clear_on_submit=True):
    user_query = st.text_input("Transmit Command:", placeholder="Enter your query...")
    submit = st.form_submit_button("SEND")

final_query = clicked_q if clicked_q else (user_query if submit else None)

if final_query:
    ans = get_response(final_query)
    st.session_state.history.append({"q": final_query, "a": ans})
    st.rerun()

# History Display
for item in reversed(st.session_state.history):
    st.markdown(f'''
    <div style="background:rgba(255,255,255,0.02); border-left:4px solid #00FFA3; padding:15px; margin-bottom:10px;">
        <b style="color:#00FFA3">SIGNAL:</b> {item["q"]}<br><br>
        <b>NOVA:</b> {item["a"]}
    </div>
    ''', unsafe_allow_html=True)
