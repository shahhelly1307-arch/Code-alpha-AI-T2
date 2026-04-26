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

lottie_main = load_lottieurl("https://lottie.host/8172906e-8360-449e-9988-0320a1630985/B1pU53Y34i.json")

# --- 3. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        with open('faqs.json', 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except:
        return pd.DataFrame({
            "question": ["System Status", "Who are you?", "Capabilities"], 
            "answer": ["Database signal active.", "I am Nova, your neural interface.", "I process NLP queries in real-time."]
        })

df = load_data()

# --- 4. THE VOXA REPLICA UI ---
st.set_page_config(page_title="Nova Chatterix", layout="wide")

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Silkscreen:wght@700&display=swap');
    
    html, body, [class*="css"], .stText, .stMarkdown, .stButton, input, label {{
        font-family: 'Silkscreen', cursive !important;
    }}

    /* BACKGROUND: User Specified Colors */
    .stApp {{
        background-color: #B7C3D3 !important;
        background: linear-gradient(to bottom, #B7C3D3, #AEBCCD) !important;
        background-attachment: fixed !important;
        color: #1a1a1a; /* Darker text for light background */
    }}
    
    /* HEADER: Cyan-Purple Text Gradient */
    .voxa-header {{
        font-family: 'Silkscreen', cursive !important;
        font-size: clamp(2.5rem, 6vw, 8rem) !important; 
        font-weight: 700 !important;
        background: linear-gradient(to right, #008ba3, #6a1b9a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        text-transform: uppercase;
        white-space: nowrap; 
        letter-spacing: -3px;
        margin-top: 10px;
        margin-bottom: 0px;
        filter: drop-shadow(0 0 5px rgba(0, 0, 0, 0.2));
    }}

    .orbital-line {{
        height: 3px;
        background: linear-gradient(90deg, transparent, #008ba3, transparent);
        width: 80%;
        margin: 0 auto 40px auto;
    }}

    /* SIDEBAR styling */
    [data-testid="stSidebar"] {{
        background-color: #AEBCCD !important;
        border-right: 2px solid #008ba3;
    }}

    .sidebar-label {{
        color: #004d40;
        font-size: 0.9rem;
        letter-spacing: 2px;
        font-weight: bold;
    }}

    /* BUTTONS */
    div.stButton > button {{
        background: rgba(255, 255, 255, 0.5) !important;
        color: #004d40 !important;
        border: 2px solid #004d40 !important;
        border-radius: 0px !important;
        font-size: 0.85rem !important;
        transition: 0.3s;
        width: 100%;
    }}

    div.stButton > button:hover {{
        background: #004d40 !important;
        color: #fff !important;
    }}
    
    .stTextInput input {{
        background-color: rgba(255, 255, 255, 0.7) !important;
        border: 2px solid #004d40 !important;
        color: #000000 !important;
        border-radius: 0px;
    }}

    .chat-card {{
        background: rgba(255, 255, 255, 0.4);
        border: 1px solid #004d40;
        border-left: 8px solid #004d40;
        padding: 20px;
        margin-bottom: 15px;
        color: #1a1a1a;
        backdrop-filter: blur(5px);
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 5. LOGIC ENGINE ---
def get_response(user_input):
    dev_query = user_input.lower()
    if any(word in dev_query for word in ["developed", "creator", "who made"]):
        return "This project was developed by Helly as a technical demonstration of NLP and professional UI integration."
        
    processed_input = preprocess_text(user_input)
    corpus = df['question'].apply(preprocess_text).tolist()
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus + [processed_input])
    
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    idx = similarity_scores.argmax()
    
    if similarity_scores[0][idx] > 0.2:
        return df.iloc[idx]['answer']
    return "Neural Signal Mismatch. Data not found in current frequency."

# --- 6. SIDEBAR ---
with st.sidebar:
    st.markdown('<p class="sidebar-label">INTERFACE SETTINGS</p>', unsafe_allow_html=True)
    if st.button("CLEAR CACHE"):
        st.session_state.history = []
        st.rerun()
    
    st.markdown("---")
    st.markdown('<p class="sidebar-label">CREDENTIALS</p>', unsafe_allow_html=True)
    st.write("**DEVELOPER:** Helly Shah")
    st.write("**ENGINE:** NPCL V2.0")
    
    st.markdown("---")
    st.markdown('<p style="color:#004d40;">● SYSTEM: ONLINE</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#004d40;">● SIGNAL: ACTIVE</p>', unsafe_allow_html=True)

# --- 7. MAIN INTERFACE ---
st.markdown('<p class="voxa-header">NOVA CHATTERIX</p>', unsafe_allow_html=True)
st.markdown('<div class="orbital-line"></div>', unsafe_allow_html=True)

if lottie_main:
    col_rob, col_empty = st.columns([1, 4])
    with col_rob:
        st_lottie(lottie_main, height=150, key="main_robot")

if 'history' not in st.session_state:
    st.session_state.history = []

st.markdown("### 📡 ACTIVE FREQUENCIES")
questions_list = df['question'].tolist()
cols = st.columns(3)
clicked_q = None

for i, q in enumerate(questions_list[:6]):
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
        <b style="color:#004d40">SIGNAL:</b> {item["q"]}<br><br>
        <b style="color:#4a148c">NOVA:</b> {item["a"]}
    </div>
    ''', unsafe_allow_html=True)
