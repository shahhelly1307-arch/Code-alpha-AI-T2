import streamlit as st
import pandas as pd
import json
import nltk
import time
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
    with open('faqs.json', 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

df = load_data()

# --- 3. UI CONFIGURATION & EMERALD OBSIDIAN THEME ---
st.set_page_config(page_title="Nova AI Interface", layout="wide")

st.markdown("""
    <style>
    /* Main App Background - Radial Gradient */
    .stApp {
        background: radial-gradient(circle at top right, #0a1f1a, #050505);
        color: #e0e0e0;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid #00FFA3;
    }

    /* Glassmorphic Chat Blocks with Hover Effect */
    .q-block {
        background: rgba(0, 255, 163, 0.05);
        border: 1px solid rgba(0, 255, 163, 0.2);
        padding: 20px;
        border-radius: 15px;
        margin-top: 15px;
        transition: transform 0.3s ease;
    }
    .q-block:hover {
        transform: translateX(10px);
        border-color: #00FFA3;
    }

    .a-block {
        background: rgba(255, 255, 255, 0.03);
        border-left: 4px solid #00FFA3;
        padding: 20px;
        border-radius: 0 15px 15px 0;
        margin-top: 5px;
        margin-bottom: 20px;
    }

    /* Interactive Emerald Buttons */
    div.stButton > button {
        background: transparent !important;
        color: #00FFA3 !important;
        border: 1px solid #00FFA3 !important;
        border-radius: 8px !important;
        transition: all 0.4s ease !important;
        width: 100%;
    }
    div.stButton > button:hover {
        background: #00FFA3 !important;
        color: #000000 !important;
        box-shadow: 0 0 15px #00FFA3;
    }

    /* Status Boxes in Sidebar */
    .status-box {
        background-color: rgba(0, 255, 163, 0.05);
        border: 1px solid #00FFA3;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-family: monospace;
        color: #00FFA3;
        font-size: 0.8rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. THE CHAT ENGINE ---
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
        return "I'm sorry, I couldn't find a high-confidence match for that query in the MASF database.", 0

# --- 5. SIDEBAR BRANDING ---
with st.sidebar:
    st.title("📂 Project Details")
    st.markdown("---")
    st.write("**Framework:** MASF AI")
    st.write("**Developer:** Helly")
    st.markdown("---")
    st.markdown('<div class="status-box">SYSTEM: ONLINE</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="status-box">ENGINE: TF-IDF V2</div>', unsafe_allow_html=True)

# --- 6. MAIN INTERFACE ---
st.markdown('<h1 style="color: #00FFA3; text-align: center; font-family: Courier New;">🤖 NOVA NEURAL INTERFACE</h1>', unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []

# Quick Command Buttons
st.markdown("### ⚡ SYSTEM COMMANDS")
questions = df['question'].tolist()
cols = st.columns(3)
clicked_q = None

for i, q in enumerate(questions):
    if cols[i % 3].button(q):
        clicked_q = q

# --- 7. INPUT HANDLING & NEURAL TYPING ---
with st.form(key='chat_form', clear_on_submit=True):
    user_query = st.text_input("Enter your signal query:")
    submit = st.form_submit_button("Send Signal")

# Logic for handling input (both button clicks and typing)
final_query = clicked_q if clicked_q else (user_query if submit else None)

if final_query:
    with st.spinner("🧠 Neural Engine Processing..."):
        time.sleep(0.6) # Simulates neural processing delay
        ans, conf = get_response(final_query)
        st.session_state.history.append({"q": final_query, "a": ans, "c": conf})
        st.rerun()

# Display Chat History
for item in reversed(st.session_state.history):
    st.markdown(f'<div class="q-block"><b>User:</b> {item["q"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="a-block"><b>Nova AI:</b> {item["a"]}</div>', unsafe_allow_html=True)
