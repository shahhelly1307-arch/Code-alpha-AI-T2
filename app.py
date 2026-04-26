import streamlit as st
import pd as pd
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
            "question": ["System Status", "Who made this?", "Capabilities"], 
            "answer": ["Database signal active.", "Developed by Helly.", "Neural NLP processing active."]
        })

df = load_data()

# --- 4. THE VOXA-STYLE UI ---
st.set_page_config(page_title="Nova Chatterix", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Silkscreen:wght@700&display=swap');
    
    html, body, [class*="css"], .stText, .stMarkdown, .stButton, input, label {
        font-family: 'Silkscreen', cursive !important;
    }

    /* THE VOXA BACKGROUND: TRUE BLACK WITH SIDE GLOWS */
    .stApp {
        background-color: #000000 !important;
        background-image: 
            /* Left side Cyan glow */
            radial-gradient(circle at -5% 50%, rgba(0, 229, 255, 0.25) 0%, transparent 45%),
            /* Right side Purple glow */
            radial-gradient(circle at 105% 50%, rgba(180, 82, 255, 0.2) 0%, transparent 45%) !important;
        background-attachment: fixed !important;
        color: #ffffff;
    }
    
    /* HEADER: Cyan-Purple Text Gradient */
    .voxa-header {
        font-family: 'Silkscreen', cursive !important;
        font-size: clamp(2.5rem, 6vw, 8rem) !important; 
        font-weight: 700 !important;
        background: linear-gradient(to right, #00e5ff, #b452ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        text-transform: uppercase;
        white-space: nowrap; 
        letter-spacing: -3px;
        margin-top: 10px;
        margin-bottom: 0px;
        filter: drop-shadow(0 0 15px rgba(0, 229, 255, 0.4));
    }

    .orbital-line {
        height: 2px;
        background: linear-gradient(90deg, transparent, #00e5ff, transparent);
        width: 80%;
        margin: 0 auto 40px auto;
        opacity: 0.6;
    }

    /* SIDEBAR styling */
    [data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.8) !important;
        border-right: 1px solid rgba(0, 229, 255, 0.3);
    }

    /* BUTTONS */
    div.stButton > button {
        background: rgba(0, 229, 255, 0.05) !important;
        color: #00e5ff !important;
        border: 1px solid #00e5ff !important;
        border-radius: 4px !important;
        font-size: 0.8rem !important;
        transition: 0.3s;
        width: 100%;
    }

    div.stButton > button:hover {
        background: rgba(0, 229, 255, 0.2) !important;
        box-shadow: 0 0 15px #00e5ff;
        color: #fff !important;
    }
    
    /* INPUT BOX */
    .stTextInput input {
        background-color: rgba(15, 15, 15, 0.9) !important;
        border: 1px solid #00e5ff !important;
        color: #ffffff !important;
        border-radius: 4px;
    }

    /* CHAT CARDS */
    .chat-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(0, 229, 255, 0.2);
        border-left: 4px solid #00e5ff;
        padding: 20px;
        margin-bottom: 15px;
        backdrop-filter: blur(10px);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 5. LOGIC ENGINE ---
def get_response(user_input):
    dev_query = user_input.lower()
    if any(x in dev_query for x in ["developed", "creator", "who made"]):
        return "This project was developed by Helly as a technical demonstration of NLP and professional UI integration."
        
    processed_input = preprocess_text(user_input)
    corpus = df['question'].apply(preprocess_text).tolist()
    corpus.append(processed_input)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    idx = similarity_scores.argmax()
    if similarity_scores[0][idx] > 0.2:
        return df.iloc[idx]['answer']
    return "Neural Signal Mismatch. Data not found in current frequency."

# --- 6. SIDEBAR ---
with st.sidebar:
    st.markdown('<p style="color:#00e5ff; font-weight:bold;">SYSTEM SETTINGS</p>', unsafe_allow_html=True)
    if st.button("RESET INTERFACE"):
        st.session_state.history = []
        st.rerun()
    
    st.markdown("---")
    st.write("**DEV:** Helly Shah")
    st.write("**CORE:** NPCL V2.0")
    st.markdown('<p style="color:#00e5ff;">● ONLINE</p>', unsafe_allow_html=True)

# --- 7. MAIN INTERFACE ---
st.markdown('<p class="voxa-header">NOVA CHATTERIX</p>', unsafe_allow_html=True)
st.markdown('<div class="orbital-line"></div>', unsafe_allow_html=True)

if lottie_main:
    col_rob, _ = st.columns([1, 5])
    with col_rob:
        st_lottie(lottie_main, height=120, key="main_robot")

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
        <span style="color:#00e5ff; font-weight:bold;">SIGNAL:</span> {item["q"]}<br><br>
        <span style="color:#b452ff; font-weight:bold;">NOVA:</span> {item["a"]}
    </div>
    ''', unsafe_allow_html=True)
