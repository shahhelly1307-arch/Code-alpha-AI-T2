import streamlit as st
import pandas as pd
import json
import nltk
import requests
import time
from streamlit_lottie import st_lottie
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer

# --- 1. SESSION STATE FOR INTRO ---
if 'intro_done' not in st.session_state:
    st.session_state.intro_done = False

# --- 2. INTRO PAGE (HTML/CSS + VOICE + SMOOTH TRANSITION) ---
if not st.session_state.intro_done:
    st.set_page_config(page_title="Nova Chatterix", layout="wide")
    
    intro_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://fonts.googleapis.com/css2?family=Silkscreen:wght@700&display=swap" rel="stylesheet">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                background-color: #01050a;
                height: 100vh;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                font-family: 'Silkscreen', cursive;
                overflow: hidden;
                transition: opacity 1.5s ease-in-out, filter 1.5s ease-in-out;
            }
            .fade-out {
                opacity: 0;
                filter: blur(15px) brightness(1.5);
            }
            .container {
                display: flex;
                flex-direction: column;
                align-items: center;
                position: relative;
                width: 100%;
            }
            .robot-box {
                margin-bottom: -130px; 
                z-index: 10;
                animation: hover 4s ease-in-out infinite;
            }
            .eye-pupil { animation: eyeMovement 6s ease-in-out infinite; }
            .eye-lid { transform-origin: center; animation: blink 4s ease-in-out infinite; }
            .waving-arm { transform-origin: 145px 135px; animation: bigWave 0.8s ease-in-out 2; }
            .arc-svg {
                width: 950px;
                height: 350px;
                z-index: 5;
                filter: drop-shadow(0 0 20px rgba(92, 225, 255, 0.4));
            }
            .nova-brand {
                margin-top: -80px;
                font-size: 3.5rem;
                letter-spacing: 12px;
                text-transform: uppercase;
                background: linear-gradient(to bottom, #ffffff, #5ce1ff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                opacity: 0.9;
            }
            .init-btn {
                margin-top: 30px;
                padding: 12px 30px;
                background: transparent;
                border: 2px solid #5ce1ff;
                color: #5ce1ff;
                font-family: 'Silkscreen', cursive;
                cursor: pointer;
                border-radius: 5px;
                z-index: 20;
                transition: 0.3s;
            }
            .init-btn:hover {
                background: #5ce1ff;
                color: #01050a;
                box-shadow: 0 0 20px #5ce1ff;
            }
            @keyframes hover { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-25px); } }
            @keyframes bigWave { 0%, 100% { transform: rotate(0deg); } 50% { transform: rotate(-25deg) scale(1.1); } }
            @keyframes eyeMovement { 0%, 10%, 100% { transform: translate(0, 0); } 20%, 40% { transform: translate(3px, -1px); } 50%, 70% { transform: translate(-3px, 1px); } 80% { transform: translate(0, -2px); } }
            @keyframes blink { 0%, 45%, 55%, 100% { transform: scaleY(1); } 50% { transform: scaleY(0.1); } }
        </style>
    </head>
    <body>
    <div class="container">
        <div class="robot-box">
            <svg width="280" height="280" viewBox="0 0 200 200">
                <defs>
                    <linearGradient id="bodyGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                        <stop offset="0%" stop-color="#b85cff" /><stop offset="60%" stop-color="#7a7aff" /><stop offset="100%" stop-color="#5ce1ff" />
                    </linearGradient>
                    <filter id="glow"><feGaussianBlur stdDeviation="6" result="blur" /><feComposite in="SourceGraphic" in2="blur" operator="over" /></filter>
                </defs>
                <circle cx="100" cy="35" r="4" fill="#b85cff" /><rect x="98" y="38" width="4" height="12" fill="#b85cff" />
                <path d="M55 85 C 55 40, 145 40, 145 85 C 145 130, 125 170, 100 170 C 75 170, 55 130, 55 85" fill="url(#bodyGrad)" stroke="white" stroke-width="0.5" />
                <rect x="65" y="75" width="70" height="42" rx="21" fill="#0c121d" stroke="#5ce1ff" stroke-width="1" />
                <g class="eye-lid">
                    <circle cx="82" cy="95" r="8" fill="white" /><circle class="eye-pupil" cx="82" cy="95" r="3.5" fill="#0c121d" />
                    <circle cx="118" cy="95" r="8" fill="white" /><circle class="eye-pupil" cx="118" cy="95" r="3.5" fill="#0c121d" />
                </g>
                <path d="M92 122 Q100 128 108 122" stroke="white" stroke-width="2" fill="none" stroke-linecap="round" />
                <path d="M56 130 C 25 135, 25 165, 58 155" fill="#7a7aff" stroke="white" stroke-width="1" />
                <g class="waving-arm"><path d="M144 130 C 175 135, 175 165, 142 155" fill="#7a7aff" stroke="white" stroke-width="1" /></g>
                <ellipse cx="100" cy="170" rx="35" ry="12" fill="#5ce1ff" filter="url(#glow)" opacity="0.7" />
            </svg>
        </div>
        <svg class="arc-svg" viewBox="0 0 800 300">
            <path d="M40 260 Q400 20 760 260" stroke="#5ce1ff" stroke-width="12" fill="none" opacity="0.2" filter="blur(15px)" />
            <path d="M40 260 Q400 20 760 260" stroke="url(#arcGrad)" stroke-width="5" fill="none" stroke-linecap="round" />
            <defs><linearGradient id="arcGrad"><stop offset="0%" stop-color="transparent"/><stop offset="50%" stop-color="#5ce1ff"/><stop offset="100%" stop-color="transparent"/></linearGradient></defs>
        </svg>
        <h1 class="nova-brand">Nova Chatterix</h1>
        <button class="init-btn" onclick="startSequence()">CONNECT INTERFACE</button>
    </div>

    <script>
        function startSequence() {
            // 1. Voice Greeting (Soft Female)
            if ('speechSynthesis' in window) {
                const msg = new SpeechSynthesisUtterance();
                msg.text = "Hello Chatterix. System is now online.";
                msg.volume = 1; msg.rate = 0.85; msg.pitch = 1.2;
                let voices = window.speechSynthesis.getVoices();
                let femaleVoice = voices.find(v => v.name.includes('Female') || v.name.includes('Google UK English Female') || v.name.includes('Samantha'));
                if (femaleVoice) msg.voice = femaleVoice;
                window.speechSynthesis.speak(msg);
            }
            
            // 2. Smooth Radiation Fade Out
            setTimeout(() => {
                document.body.classList.add('fade-out');
            }, 1000);
        }
    </script>
    </body>
    </html>
    """
    
    st.components.v1.html(intro_html, height=800)
    # Give user time to click, hear the voice, and see the fade
    time.sleep(4.5) 
    st.session_state.intro_done = True
    st.rerun()

else:
    # --- 3. NLP SETUP ---
    @st.cache_resource
    def setup_nlp():
        try: nltk.data.find('tokenizers/punkt'); 
        except: nltk.download('punkt')
        try: nltk.data.find('corpora/wordnet'); 
        except: nltk.download('wordnet')
        try: nltk.data.find('tokenizers/punkt_tab'); 
        except: nltk.download('punkt_tab')

    setup_nlp()
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        tokens = nltk.word_tokenize(text.lower())
        return " ".join([lemmatizer.lemmatize(token) for token in tokens if token.isalnum()])

    # --- 4. ASSET LOADING ---
    def load_lottieurl(url: str):
        try:
            r = requests.get(url, timeout=10)
            return r.json() if r.status_code == 200 else None
        except: return None

    lottie_main = load_lottieurl("https://lottie.host/8172906e-8360-449e-9988-0320a1630985/B1pU53Y34i.json")

    # --- 5. DATA LOADING ---
    @st.cache_data
    def load_data():
        try:
            with open('faqs.json', 'r') as f: data = json.load(f)
            return pd.DataFrame(data)
        except:
            return pd.DataFrame({"question": ["Status", "Who"], "answer": ["Active", "I am NOVA."] })

    df = load_data()

    # --- 6. THE NOVA CHATTERIX UI ---
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Silkscreen:wght@700&display=swap');
        html, body, [class*="css"], .stText, .stMarkdown, .stButton, input, label { font-family: 'Silkscreen', cursive !important; }
        .stApp {
            background-color: #050505 !important; 
            background-image: radial-gradient(circle at 0% 0%, rgba(0, 229, 255, 0.2) 0%, transparent 60%), radial-gradient(circle at 100% 100%, rgba(180, 82, 255, 0.2) 0%, transparent 60%), linear-gradient(135deg, #001214 0%, #11001c 100%) !important;
            background-attachment: fixed !important;
            color: #ffffff;
        }
        .voxa-header { font-size: clamp(2.5rem, 6vw, 8rem) !important; background: linear-gradient(to right, #00e5ff, #b452ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; text-transform: uppercase; letter-spacing: -3px; margin-top: 10px; filter: drop-shadow(0 0 15px rgba(0, 229, 255, 0.4)); }
        .orbital-line { height: 3px; background: linear-gradient(90deg, transparent, #00e5ff, transparent); width: 80%; margin: 0 auto 40px auto; box-shadow: 0 0 15px #00e5ff; }
        [data-testid="stSidebar"] { background-color: rgba(0, 0, 0, 0.8) !important; border-right: 1px solid rgba(0, 229, 255, 0.3); }
        div.stButton > button { background: rgba(255, 255, 255, 0.05) !important; color: #ffffff !important; border: 1px solid rgba(0, 229, 255, 0.5) !important; border-radius: 50px !important; transition: 0.3s; padding: 10px 20px; }
        div.stButton > button:hover { background: rgba(0, 229, 255, 0.2) !important; box-shadow: 0 0 20px rgba(0, 229, 255, 0.4); border: 1px solid #00e5ff !important; }
        .chat-card { background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(0, 229, 255, 0.2); border-left: 5px solid #00e5ff; padding: 20px; margin-bottom: 15px; border-radius: 4px; }
        .sidebar-robot-container { animation: sideHover 4s ease-in-out infinite; display: flex; justify-content: center; }
        @keyframes sideHover { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-10px); } }
        </style>
        """, unsafe_allow_html=True)

    # --- 7. LOGIC ENGINE ---
    def get_response(user_input):
        dev_query = user_input.lower()
        if any(x in dev_query for x in ["developed", "creator", "who made", "built by", "developer"]):
            return "This interface was developed by Helly as a professional demonstration of NLP and advanced UI design."
        processed_input = preprocess_text(user_input)
        corpus = df['question'].apply(preprocess_text).tolist()
        vectorizer = TfidfVectorizer(); tfidf_matrix = vectorizer.fit_transform(corpus)
        user_vec = vectorizer.transform([processed_input])
        similarity_scores = cosine_similarity(user_vec, tfidf_matrix)
        idx = similarity_scores.argmax()
        return df.iloc[idx]['answer'] if similarity_scores[0][idx] > 0.2 else "Neural Signal Mismatch. Data not found."

    # --- 8. SIDEBAR ---
    with st.sidebar:
        st.markdown('<div class="sidebar-robot-container"><svg width="120" height="120" viewBox="0 0 200 200"><rect x="65" y="75" width="70" height="42" rx="21" fill="#0c121d" stroke="#5ce1ff" stroke-width="1" /><circle cx="82" cy="95" r="5" fill="white" /><circle cx="118" cy="95" r="5" fill="white" /></svg></div>', unsafe_allow_html=True)
        st.markdown('**INTERFACE SETTINGS**')
        if st.button("CLEAR CACHE"): st.session_state.history = []; st.rerun()
        st.markdown("---")
        st.write("**DEVELOPER:** Helly")
        st.write("**ENGINE:** NOVA-V2")

    # --- 9. MAIN INTERFACE ---
    st.markdown('<p class="voxa-header">NOVA CHATTERIX</p>', unsafe_allow_html=True)
    st.markdown('<div class="orbital-line"></div>', unsafe_allow_html=True)

    if 'history' not in st.session_state: st.session_state.history = []

    st.markdown("### 📡 ACTIVE FREQUENCIES")
    questions_list = df['question'].tolist()
    cols = st.columns(3)
    clicked_q = None
    for i, q in enumerate(questions_list):
        if cols[i % 3].button(q, key=f"q_{i}"): clicked_q = q

    with st.form(key='chat_form', clear_on_submit=True):
        user_query = st.text_input("Transmit Command:", placeholder="AWAITING SIGNAL...")
        submit = st.form_submit_button("TRANSMIT")

    final_query = clicked_q if clicked_q else (user_query if submit else None)
    if final_query:
        ans = get_response(final_query)
        st.session_state.history.insert(0, {"q": final_query, "a": ans})
        st.rerun()

    for item in st.session_state.history:
        st.markdown(f'<div class="chat-card"><b style="color:#00e5ff">SIGNAL:</b> {item["q"]}<br><br><b style="color:#b452ff">NOVA:</b> {item["a"]}</div>', unsafe_allow_html=True)
