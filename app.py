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
        return pd.DataFrame({"question": ["System Status"], "answer": ["Database signal active."]})

df = load_data()

# --- 4. UI CONFIG ---
st.set_page_config(page_title="Nova Chatterix", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Silkscreen:wght@700&display=swap');

html, body, [class*="css"], .stText, .stMarkdown, .stButton, input, label {
    font-family: 'Silkscreen', cursive !important;
}

/* BACKGROUND */
.stApp {
    background: radial-gradient(circle at 50% 40%, #0d3b4f 0%, #020202 85%) !important;
    color: #e6f7ff;
}

/* HEADER */
.voxa-header {
    font-size: clamp(2.5rem, 6vw, 8rem);
    color: #66e0ff;
    text-align: center;
    letter-spacing: -3px;
    text-shadow: 
        2px 2px 0px #003344,
        4px 4px 0px #003344,
        0 0 40px rgba(102, 224, 255, 0.8);
}

/* GLOW LINE */
.orbital-line {
    height: 4px;
    background: linear-gradient(90deg, transparent, #66e0ff, transparent);
    width: 80%;
    margin: 0 auto 40px auto;
    box-shadow: 0 0 25px #66e0ff;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #050505 !important;
    border-right: 1px solid #66e0ff;
}

.sidebar-label {
    color: #66e0ff;
    font-size: 0.9rem;
    letter-spacing: 2px;
}

/* BUTTON */
div.stButton > button {
    background: rgba(102, 224, 255, 0.08);
    color: #66e0ff;
    border: 1px solid #66e0ff;
    border-radius: 0;
}

/* INPUT */
.stTextInput input {
    background-color: rgba(0, 0, 0, 0.85);
    border: 1px solid #66e0ff;
    color: #e6f7ff;
}

/* CHAT BOX */
.chat-card {
    background: rgba(13, 59, 79, 0.35);
    border: 1px solid #66e0ff;
    padding: 20px;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# --- 5. LOGIC ---
def get_response(user_input):
    dev_query = user_input.lower()
    if "developed" in dev_query or "creator" in dev_query or "who made" in dev_query:
        return "This project was developed by Helly."

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

# --- 6. SIDEBAR ---
with st.sidebar:
    st.markdown('<p class="sidebar-label">SETTINGS</p>', unsafe_allow_html=True)

    if st.button("CLEAR HISTORY"):
        st.session_state.history = []
        st.rerun()

    st.markdown("---")
    st.markdown('<p class="sidebar-label">SYSTEM</p>', unsafe_allow_html=True)
    st.write("**DEVELOPER:** Helly")
    st.write("**ENGINE:** NPCL V2.0")

    st.markdown("---")
    st.markdown('<p style="color:#66e0ff;">● SYSTEM ONLINE</p>', unsafe_allow_html=True)

# --- 7. MAIN UI ---
st.markdown('<p class="voxa-header">NOVA CHATTERIX</p>', unsafe_allow_html=True)
st.markdown('<div class="orbital-line"></div>', unsafe_allow_html=True)

if lottie_main:
    col_rob, _ = st.columns([1, 4])
    with col_rob:
        st_lottie(lottie_main, height=150)

if 'history' not in st.session_state:
    st.session_state.history = []

st.markdown("### 📡 ACTIVE FREQUENCIES")

questions_list = df['question'].tolist()
cols = st.columns(3)
clicked_q = None

for i, q in enumerate(questions_list):
    if cols[i % 3].button(q):
        clicked_q = q

with st.form(key='chat_form', clear_on_submit=True):
    user_query = st.text_input("Transmit Command:", placeholder="ENTER SIGNAL...")
    submit = st.form_submit_button("TRANSMIT")

final_query = clicked_q if clicked_q else (user_query if submit else None)

if final_query:
    ans = get_response(final_query)
    st.session_state.history.append({"q": final_query, "a": ans})
    st.rerun()

for item in reversed(st.session_state.history):
    st.markdown(f'''
    <div class="chat-card">
        <b style="color:#66e0ff">SIGNAL:</b> {item["q"]}<br><br>
        <b>NOVA:</b> {item["a"]}
    </div>
    ''', unsafe_allow_html=True)
