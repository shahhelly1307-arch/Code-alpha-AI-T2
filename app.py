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
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

# Stabilized high-tech robot for the central view
lottie_main = load_lottieurl("https://lottie.host/8172906e-8360-449e-9988-0320a1630985/B1pU53Y34i.json")

# --- 3. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        with open('faqs.json', 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except:
        return pd.DataFrame({"question": ["Hello"], "answer": ["Database signal missing. Please upload faqs.json"]})

df = load_data()

# --- 4. UI CONFIGURATION & CUSTOM THEME (Image 1 & 2 Merge) ---
st.set_page_config(page_title="Nova Chatterix", layout="wide")

st.markdown("""
    <style>
    /* Global Pixel Font: Image 1 style applied across app */
    @import url('https://fonts.googleapis.com/css2?family=DotGothic16&display=swap');
    
    html, body, [class*="css"], .stText, .stMarkdown, .stButton, input, label {
        font-family: 'DotGothic16', sans-serif !important;
    }

    /* Background from Image 2: Center-glow Gradient */
    .stApp {
        background: radial-gradient(circle at center, #001f1f 0%, #050505 100%);
        color: #ffffff;
    }
    
    /* Image 1 Title Styling: HUGE PIXEL FONT */
    .huge-nova-header {
        font-family: 'DotGothic16', sans-serif;
        font-size: clamp(3rem, 12vw, 7rem); /* Very large, scales dynamically */
        color: #00FFA3; /* The exact bright cyan color from Voxa */
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 15px; /* Spacing like Voxa */
        margin-top: 30px;
        margin-bottom: 0px;
        text-shadow: 0 0 25px rgba(0, 255, 163, 0.7); /* The glowing effect */
    }

    /* Orbital Line (Image 2 aesthetic) */
    .orbital-line {
        height: 2px;
        background: linear-gradient(90deg, transparent, #00FFA3, transparent);
        width: 80%;
        margin: 0 auto 30px auto;
        box-shadow: 0 0 15px #00FFA3;
    }

    /* Pixelated Large Buttons */
    div.stButton > button {
        background: rgba(0, 255, 163, 0.08) !important;
        color: #00FFA3 !important;
        border: 2px solid #00FFA3 !important;
        font-size: 1.1rem !important;
        text-transform: uppercase;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background: #00FFA3 !important;
        color: #000 !important;
        box-shadow: 0 0 20px #00FFA3;
    }

    /* Input Field Styling */
    .stTextInput input {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid #00FFA3 !important;
        color: #ffffff !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #050505 !important;
        border-right: 1px solid rgba(0, 255, 163, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 5. LOGIC ENGINE ---
def get_response(user_input):
    # Credits Fix: bf866d97-8cbc-416a-b901-98651f5495f2
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
    return "Neural signal weak. Data not recognized."

# --- 6. SIDEBAR (Helly Shah Profile) ---
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
# Huge Pixel Title (Image 1 style) and orbital line
st.markdown('<p class="huge-nova-header">NOVA CHATTERIX</p>', unsafe_allow_html=True)
st.markdown('<div class="orbital-line"></div>', unsafe_allow_html=True)

# Central Robot (Image 2 Landing View - Clean)
if lottie_main:
    st_lottie(lottie_main, height=350, key="main_robot")

if 'history' not in st.session_state:
    st.session_state.history = []

# Question Signals (bf866d97-8cbc-416a-b901-98651f5495f2 Restoration)
st.markdown("### 📡 ACTIVE SIGNALS")
questions_list = df['question'].tolist()
cols = st.columns(3)
clicked_q = None

for i, q in enumerate(questions_list[:6]):
    if cols[i % 3].button(q, key=f"q_{i}"):
        clicked_q = q

# Input
with st.form(key='chat_form', clear_on_submit=True):
    user_query = st.text_input("Enter Signal Command:", placeholder="Transmit your message...")
    submit = st.form_submit_button("TRANSMIT")

final_query = clicked_q if clicked_q else (user_query if submit else None)

if final_query:
    ans = get_response(final_query)
    st.session_state.history.append({"q": final_query, "a": ans})
    st.rerun()

# Neural History Display
for item in reversed(st.session_state.history):
    st.markdown(f'''
    <div style="background:rgba(255,255,255,0.02); border-left:4px solid #00FFA3; padding:15px; margin-bottom:10px;">
        <b style="color:#00FFA3">SIGNAL:</b> {item["q"]}<br><br>
        <b>NOVA:</b> {item["a"]}
    </div>
    ''', unsafe_allow_html=True)
