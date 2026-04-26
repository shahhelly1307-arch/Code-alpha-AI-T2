import streamlit as st
import pandas as pd
import json
import nltk
import time
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

# --- 2. ASSET LOADING (Animations) ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Futuristic Robot Animation
lottie_robot = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_96bovdur.json")

# --- 3. DATA LOADING ---
@st.cache_data
def load_data():
    # Ensure faqs.json exists in your repository
    try:
        with open('faqs.json', 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except FileNotFoundError:
        return pd.DataFrame({"question": ["Hello"], "answer": ["Hi there! Please upload faqs.json"]})

df = load_data()

# --- 4. UI CONFIGURATION & STYLING ---
st.set_page_config(page_title="Nova AI Interface", layout="wide")

st.markdown("""
    <style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .stApp {
        background: radial-gradient(circle at top right, #0a1f1a, #050505);
        color: #e0e0e0;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid #00FFA3;
    }

    /* Animated Chat Blocks */
    .q-block {
        background: rgba(0, 255, 163, 0.05);
        border: 1px solid rgba(0, 255, 163, 0.2);
        padding: 20px;
        border-radius: 15px;
        margin-top: 15px;
        animation: fadeIn 0.5s ease forwards;
    }

    .a-block {
        background: rgba(255, 255, 255, 0.03);
        border-left: 4px solid #00FFA3;
        padding: 20px;
        border-radius: 0 15px 15px 0;
        margin-top: 5px;
        margin-bottom: 20px;
        animation: fadeIn 0.8s ease forwards;
    }

    /* Floating Effect for Buttons */
    div.stButton > button {
        background: transparent !important;
        color: #00FFA3 !important;
        border: 1px solid #00FFA3 !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px #00FFA3;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 5. THE CHAT ENGINE ---
def get_response(user_input):
    processed_input = preprocess_text(user_input)
    corpus = df['question'].apply(preprocess_text).tolist()
    corpus.append(processed_input)
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    idx = similarity_scores.argmax()
    confidence = similarity_scores[0][idx]
    
    if confidence > 0.2:
        return df.iloc[idx]['answer'], confidence
    else:
        return "I'm sorry, I couldn't find a high-confidence match for that query in the database.", 0

# --- 6. SIDEBAR & SETTINGS ---
with st.sidebar:
    st_lottie(lottie_robot, height=150, key="robot")
    st.title("📂 System Control")
    
    # Settings Expander (The "Gear" Equivalent)
    with st.expander("⚙️ Settings"):
        if st.button("🗑️ Clear Chat History"):
            st.session_state.history = []
            st.rerun()
            
    st.markdown("---")
    st.write("**Framework:** MASF AI")
    st.write("**Engine:** TF-IDF V2")
    st.markdown('<div style="color:#00FFA3; font-family:monospace; font-size:0.8rem;">● SYSTEM ONLINE</div>', unsafe_allow_html=True)

# --- 7. MAIN INTERFACE ---
st.markdown('<h1 style="color: #00FFA3; text-align: center; font-family: Courier New;">🤖 NOVA NEURAL INTERFACE</h1>', unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []

# Quick Commands
st.markdown("### ⚡ SUGGESTED SIGNALS")
questions = df['question'].tolist()
cols = st.columns(3)
clicked_q = None

for i, q in enumerate(questions[:6]): # Limit to top 6
    if cols[i % 3].button(q, key=f"btn_{i}"):
        clicked_q = q

# Input Field
with st.form(key='chat_form', clear_on_submit=True):
    user_query = st.text_input("Transmit Message:", placeholder="Type your query here...")
    submit = st.form_submit_button("SEND")

final_query = clicked_q if clicked_q else (user_query if submit else None)

if final_query:
    ans, conf = get_response(final_query)
    st.session_state.history.append({"q": final_query, "a": ans, "c": conf})
    st.rerun()

# Display History with Animation
for item in reversed(st.session_state.history):
    st.markdown(f'<div class="q-block">📡 <b>Signal:</b> {item["q"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="a-block">🤖 <b>Nova:</b> {item["a"]}</div>', unsafe_allow_html=True)
