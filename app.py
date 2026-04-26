import streamlit as st
import pandas as pd 
import json
import nltk
import requests
from streamlit_lottie import st_lottie
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer

# --- SETUP ---
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

@st.cache_data
def load_data():
    try:
        with open('faqs.json', 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except:
        return pd.DataFrame({
            "question": ["What is Nova AI?", "Who developed this interface?", "How does the matching engine work?", "What parameters define TF-IDF V2?", "What is the MASF Framework?", "Is this system real-time?"], 
            "answer": ["Nova AI is a neural-synapse chatbot.", "Developed by Helly Shah.", "It uses Cosine Similarity and TF-IDF vectors.", "Frequency and Inverse Document Frequency.", "Multi-Agent System Framework.", "Yes, signal processing is instantaneous."]
        })

df = load_data()

# --- THE UI (VIBRANT SIDE GLOWS) ---
st.set_page_config(page_title="Nova Chatterix", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Silkscreen:wght@700&display=swap');
    
    /* FORCE GLOBAL BLACK */
    [data-testid="stAppViewContainer"] {
        background-color: #000000 !important;
    }

    /* THE VOXA MIX: HEAVY BLUE (LEFT) AND PURPLE (RIGHT) */
    [data-testid="stAppViewMainArea"] {
        background: 
            radial-gradient(circle at -10% 50%, #00e5ff 0%, transparent 50%),
            radial-gradient(circle at 110% 50%, #b452ff 0%, transparent 50%) !important;
        background-color: #000000 !important;
    }

    /* FIXING PADDING AND TRANSPARENCY */
    .block-container {
        padding-top: 2rem !important;
        max-width: 90% !important;
        background: transparent !important;
    }

    [data-testid="stHeader"], .main {
        background: transparent !important;
    }

    * { font-family: 'Silkscreen', cursive !important; color: white; }

    /* HEADER */
    .voxa-header {
        font-size: clamp(2rem, 5vw, 5rem) !important; 
        background: linear-gradient(to right, #00e5ff, #b452ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-top: -10px;
        filter: drop-shadow(0 0 20px rgba(0, 229, 255, 0.6));
    }

    .orbital-line {
        height: 3px;
        background: linear-gradient(90deg, transparent, #00e5ff, #b452ff, transparent);
        width: 60%;
        margin: 0 auto 30px auto;
    }

    /* BUTTONS */
    div.stButton > button {
        background: rgba(0, 0, 0, 0.8) !important;
        color: #00e5ff !important;
        border: 2px solid #00e5ff !important;
        border-radius: 2px !important;
        width: 100%;
        padding: 12px;
    }

    div.stButton > button:hover {
        box-shadow: 0 0 25px #00e5ff;
        background: #00e5ff !important;
        color: black !important;
    }

    /* INPUT */
    .stTextInput input {
        background-color: rgba(0, 0, 0, 0.9) !important;
        border: 2px solid #b452ff !important;
        color: white !important;
        height: 50px;
    }

    /* CHAT CARDS */
    .chat-card {
        background: rgba(0, 0, 0, 0.6);
        border: 1px solid #b452ff;
        border-left: 8px solid #00e5ff;
        padding: 20px;
        margin-bottom: 15px;
        backdrop-filter: blur(5px);
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIC ---
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
    return "Neural Signal Mismatch."

# --- MAIN ---
st.markdown('<p class="voxa-header">NOVA CHATTERIX</p>', unsafe_allow_html=True)
st.markdown('<div class="orbital-line"></div>', unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []

st.markdown("### 📡 ACTIVE FREQUENCIES")
cols = st.columns(3)
clicked_q = None

for i, q in enumerate(df['question'].tolist()[:6]):
    if cols[i % 3].button(q, key=f"q_{i}"):
        clicked_q = q

with st.form(key='chat_form', clear_on_submit=True):
    user_query = st.text_input("Transmit Command:", placeholder="AWAITING SIGNAL...")
    submit = st.form_submit_button("TRANSMIT")

final_query = clicked_q if clicked_q else (user_query if submit else None)

if final_query:
    ans = get_response(final_query)
    st.session_state.history.append({"q": final_query, "a": ans})
    st.rerun()

for item in reversed(st.session_state.history):
    st.markdown(f'''
    <div class="chat-card">
        <span style="color:#00e5ff; font-weight:bold;">SIGNAL:</span> {item["q"]}<br><br>
        <span style="color:#b452ff; font-weight:bold;">NOVA:</span> {item["a"]}
    </div>
    ''', unsafe_allow_html=True)
