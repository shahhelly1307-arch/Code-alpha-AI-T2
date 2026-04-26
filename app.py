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
# Pre-download required data to avoid runtime issues
@st.cache_resource
def setup_nlp():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('punkt_tab')

setup_nlp()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    # Remove non-alphanumeric and lemmatize
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

# Placeholder Lottie animation (Robot head)
# A replacement for the 'gear' icon and to add a sleek AI touch
lottie_main = load_lottieurl("https://lottie.host/8172906e-8360-449e-9988-0320a1630985/B1pU53Y34i.json")

# --- 3. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        # Assuming you have a faqs.json file with question/answer pairs
        with open('faqs.json', 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except:
        # Fallback if no JSON data is found
        return pd.DataFrame({"question": ["System Status"], "answer": ["Database signal active."]})

df = load_data()

# --- 4. THE NOVA REPLICA UI ---
st.set_page_config(page_title="Nova Chatterix", layout="wide")

# This is where the primary visual styling happens
# Using extensive custom CSS to overwrite Streamlit's default look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Silkscreen:wght@700&display=swap');
    
    # /* GLOBAL FONT (Silkscreen, bold) */
    html, body, [class*="css"], .stText, .stMarkdown, .stButton, input, label {
        font-family: 'Silkscreen', cursive !important;
    }

    # /* THE COSMIC BACKGROUND: Pure #02030A mixed with soft cyan and purple accents */
    .stApp {
        background-color: #02030A !important;
        # /* Creates a subtle 'glow' effect blending blue and purple across the field */
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(0, 229, 255, 0.1) 0%, transparent 40%),
            radial-gradient(circle at 80% 80%, rgba(180, 82, 255, 0.08) 0%, transparent 40%) !important;
        background-attachment: fixed !important;
        color: #ffffff;
    }
    
    # /* MAIN HEADER: Nova Chatterix with a sleek blue-purple gradient */
    .voxa-header {
        font-family: 'Silkscreen', cursive !important;
        font-size: clamp(2.5rem, 6vw, 8rem) !important; /* Scalable size */
        font-weight: 700 !important;
        # /* Vertical gradient for text */
        background: linear-gradient(to right, #00e5ff, #b452ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        text-transform: uppercase;
        white-space: nowrap; 
        letter-spacing: -3px;
        margin-top: 10px;
        margin-bottom: 0px;
        # /* Add text drop shadow for separation */
        filter: drop-shadow(0 0 15px rgba(0, 229, 255, 0.4));
    }

    # /* The 'orbital' line directly below the main title */
    .orbital-line {
        height: 3px;
        background: linear-gradient(90deg, transparent, #00e5ff, transparent);
        width: 80%;
        margin: 0 auto 40px auto;
        box-shadow: 0 0 15px #00e5ff;
    }

    # /* SIDEBAR styling - Keeping it dark with neon cyan accents */
    [data-testid="stSidebar"] {
        background-color: rgba(2, 3, 10, 0.95) !important;
        border-right: 2px solid #00e5ff;
    }

    .sidebar-label {
        color: #00e5ff;
        font-size: 0.9rem;
        letter-spacing: 2px;
        font-weight: bold;
    }

    # /* NEON CYAN BUTTONS (Awaiting frequencies and Transmit) */
    div.stButton > button {
        background: rgba(0, 229, 255, 0.05) !important;
        color: #00e5ff !important;
        border: 2px solid #00e5ff !important;
        border-radius: 0px !important;
        font-size: 0.85rem !important;
        transition: 0.3s;
    }

    div.stButton > button:hover {
        background: rgba(0, 229, 255, 0.2) !important;
        box-shadow: 0 0 20px #00e5ff;
        color: #fff !important;
    }
    
    # /* NEON CYAN INPUT FIELD (Awaiting signal...) */
    .stTextInput input {
        background-color: rgba(0, 0, 0, 0.8) !important;
        border: 2px solid #00e5ff !important;
        color: #ffffff !important;
    }

    # /* The 'gear' icon alternative: Lottie Robot alignment */
    .robot-container {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
    }
    .gear-text {
        color: #00e5ff;
        font-size: 1.2rem;
        margin-left: 10px;
        font-weight: bold;
    }
    
    # /* Chat card styling - soft blend of background color with neon border */
    .chat-card {
        background: rgba(0, 229, 255, 0.03);
        border: 1px solid #00e5ff;
        border-left: 5px solid #00e5ff;
        padding: 20px;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 5. LOGIC ENGINE ---
# Your core response mechanism remains intact
def get_response(user_input):
    # Specialized 'Helly' developer query handling
    dev_query = user_input.lower()
    if "developed" in dev_query or "creator" in dev_query or "who made" in dev_query:
        return "This project was developed by Helly as a technical demonstration of NLP and professional UI integration."
        
    # Standard TF-IDF fallback for known FAQs
    processed_input = preprocess_text(user_input)
    corpus = df['question'].apply(preprocess_text).tolist()
    corpus.append(processed_input)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    idx = similarity_scores.argmax()
    if similarity_scores[0][idx] > 0.2: # Match threshold
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
    # Neon status markers
    st.markdown('<p style="color:#00e5ff;">● SYSTEM: ONLINE</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#00e5ff;">● SIGNAL: ACTIVE</p>', unsafe_allow_html=True)

# --- 7. MAIN INTERFACE ---
# Title Area
st.markdown('<p class="voxa-header">NOVA CHATTERIX</p>', unsafe_allow_html=True)
st.markdown('<div class="orbital-line"></div>', unsafe_allow_html=True)

# Lottie (replaces the 'gear' icon area)
if lottie_main:
    col_rob, _ = st.columns([1, 4])
    with col_rob:
        # Keep the robot head close to the header text
        st_lottie(lottie_main, height=150, key="main_robot")

# Initialize chat history
if 'history' not in st.session_state:
    st.session_state.history = []

# Questions Area (Buttons)
st.markdown("### 📡 ACTIVE FREQUENCIES")
questions_list = df['question'].tolist()
cols = st.columns(3)
clicked_q = None

# Create buttons for each FAQ
for i, q in enumerate(questions_list):
    if cols[i % 3].button(q, key=f"q_{i}"):
        clicked_q = q

# Input Form Area (Clear on Submit)
with st.form(key='chat_form', clear_on_submit=True):
    user_query = st.text_input("Transmit Command:", placeholder="AWAITING SIGNAL...")
    submit = st.form_submit_button("TRANSMIT")

# Logic Handling - prioritizing clicked questions, then manual input
final_query = clicked_q if clicked_q else (user_query if submit else None)

if final_query:
    # Process and append to history
    ans = get_response(final_query)
    st.session_state.history.append({"q": final_query, "a": ans})
    # Force rerun to clear form input and display history
    st.rerun()

# Display Chat History (in reverse order)
for item in reversed(st.session_state.history):
    st.markdown(f'''
    <div class="chat-card">
        <b style="color:#00e5ff">SIGNAL:</b> {item["q"]}<br><br>
        <b>NOVA:</b> {item["a"]}
    </div>
    ''', unsafe_allow_html=True)
