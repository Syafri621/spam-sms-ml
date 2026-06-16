import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# ============================================================
# SETUP
# ============================================================
nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

st.set_page_config(
    page_title="Spam SMS Detector",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
.header-box {
    background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
    color: white; text-align: center;
    padding: 1.5rem 1rem; border-radius: 14px;
    margin-bottom: 1.5rem;
}
.header-box h1 { margin: 0; font-size: 2rem; }
.header-box p  { margin: 0.3rem 0 0; opacity: 0.8; font-size: 1rem; }

.result-spam {
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    border-left: 6px solid #dc2626;
    padding: 1.2rem 1.5rem; border-radius: 10px; margin: 1rem 0;
}
.result-ham {
    background: linear-gradient(135deg, #dcfce7, #bbf7d0);
    border-left: 6px solid #16a34a;
    padding: 1.2rem 1.5rem; border-radius: 10px; margin: 1rem 0;
}
.result-spam h2, .result-ham h2 { margin: 0 0 0.4rem; font-size: 1.4rem; }
.result-spam p,  .result-ham p  { margin: 0; }

.badge-spam { background:#dc2626; color:white; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
.badge-ham  { background:#16a34a; color:white; padding:3px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }

.info-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 1rem;
}
.disclaimer {
    background: #fefce8; border: 1px solid #fbbf24;
    border-radius: 8px; padding: 0.7rem 1rem;
    font-size: 0.82rem; color: #78350f;
}
stTextArea textarea { font-size: 1rem !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PREPROCESSING (sama dengan notebook)
# ============================================================
stemmer    = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    model     = joblib.load('model_spam.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

try:
    model, vectorizer = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="header-box">
    <h1>📱 Spam SMS Detector</h1>
    <p>Klasifikasi pesan SMS menggunakan NLP + Linear SVM</p>
</div>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.header("ℹ️ Tentang Aplikasi")
    st.info(
        "Aplikasi ini mendeteksi apakah sebuah pesan SMS merupakan "
        "**SPAM** atau **HAM (normal)** menggunakan teknik NLP (TF-IDF) "
        "dan algoritma **Linear SVM**."
    )

    st.markdown("---")
    st.header("📊 Performa Model")
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", "~98%")
    col2.metric("F1-Score", "~97%")

    st.markdown("---")
    st.header("🔧 Pipeline NLP")
    st.markdown("""
    1. Lowercase
    2. Hapus URL & angka
    3. Hapus tanda baca
    4. Tokenisasi
    5. Hapus stopwords
    6. Stemming
    7. TF-IDF (unigram + bigram)
    """)

    st.markdown("---")
    st.header("📝 Contoh Pesan")
    examples = {
        "🚫 Contoh SPAM 1": "WINNER!! You have been selected as a winner of £1000 prize. Call now FREE!",
        "🚫 Contoh SPAM 2": "URGENT: Your account has been suspended. Verify now or lose access! Click here",
        "✅ Contoh HAM 1" : "Hey, are you coming to the meeting tomorrow? Let me know.",
        "✅ Contoh HAM 2" : "Hi mom, I'll be home late tonight. Don't wait up for me.",
    }
    selected = st.selectbox("Coba contoh pesan:", list(examples.keys()))
    if st.button("Gunakan contoh ini", use_container_width=True):
        st.session_state['example_text'] = examples[selected]

# MAIN — INPUT
if not model_loaded:
    st.error(
        "Model tidak ditemukan! Jalankan notebook terlebih dahulu "
        "untuk menghasilkan `model_spam.pkl` dan `tfidf_vectorizer.pkl`, "
        "lalu letakkan file tersebut di folder yang sama dengan app.py."
    )
    st.stop()

default_text = st.session_state.get('example_text', '')

st.subheader("✍️ Masukkan Pesan SMS")
user_input = st.text_area(
    label="Ketik atau paste pesan SMS di sini:",
    value=default_text,
    height=140,
    placeholder="Contoh: Congratulations! You've won a FREE prize. Call now to claim...",
    label_visibility="collapsed"
)

col_btn1, col_btn2 = st.columns([3, 1])
with col_btn1:
    detect_btn = st.button("🔍 Deteksi Sekarang", type="primary", use_container_width=True)
with col_btn2:
    if st.button("🗑️ Reset", use_container_width=True):
        st.session_state['example_text'] = ''
        st.rerun()

# PREDIKSI
if detect_btn:
    if not user_input.strip():
        st.warning("⚠️ Pesan tidak boleh kosong!")
    else:
        with st.spinner("Menganalisis pesan..."):
            cleaned     = preprocess_text(user_input)
            vectorized  = vectorizer.transform([cleaned])
            prediction  = model.predict(vectorized)[0]

            # LinearSVC tidak punya predict_proba; gunakan decision function
            decision_score = model.decision_function(vectorized)[0]

            # Konversi ke probabilitas kasar menggunakan sigmoid
            import numpy as np
            prob_spam = 1 / (1 + np.exp(-decision_score))
            prob_ham  = 1 - prob_spam

        st.markdown("---")
        st.subheader("🎯 Hasil Deteksi")

        if prediction == 1:
            st.markdown(f"""
            <div class="result-spam">
                <h2>🚫 SPAM TERDETEKSI</h2>
                <p>Pesan ini teridentifikasi sebagai <strong>SPAM</strong>.
                Harap berhati-hati dan jangan mengklik tautan apapun.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-ham">
                <h2>✅ PESAN NORMAL (HAM)</h2>
                <p>Pesan ini teridentifikasi sebagai <strong>pesan normal (bukan spam)</strong>.</p>
            </div>
            """, unsafe_allow_html=True)

        # Confidence score
        st.subheader("📊 Tingkat Keyakinan Model")
        c1, c2 = st.columns(2)
        c1.metric("Ham (Normal)", f"{prob_ham*100:.1f}%",
                  delta="✅ Aman" if prob_ham > 0.5 else None)
        c2.metric("Spam", f"{prob_spam*100:.1f}%",
                  delta="🚫 Waspada" if prob_spam > 0.5 else None)

        # Progress bar visual
        st.markdown("**Distribusi Probabilitas:**")
        spam_pct = int(prob_spam * 100)
        ham_pct  = 100 - spam_pct
        st.markdown(f"""
        <div style="background:#e2e8f0;border-radius:8px;overflow:hidden;height:28px;margin:8px 0">
            <div style="width:{ham_pct}%;background:#16a34a;height:100%;display:inline-block;transition:width 0.5s"></div>
            <div style="width:{spam_pct}%;background:#dc2626;height:100%;display:inline-block;transition:width 0.5s"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:0.85rem;margin-top:2px">
            <span style="color:#16a34a">✅ Ham: {ham_pct}%</span>
            <span style="color:#dc2626">🚫 Spam: {spam_pct}%</span>
        </div>
        """, unsafe_allow_html=True)

        # Detail preprocessing
        with st.expander("🔍 Detail Preprocessing Teks"):
            st.markdown("**Teks Asli:**")
            st.code(user_input, language=None)
            st.markdown("**Teks Setelah Preprocessing:**")
            st.code(cleaned if cleaned else "(kosong setelah preprocessing)", language=None)
            st.caption(f"Jumlah token setelah preprocessing: {len(cleaned.split())} kata")

        st.markdown("""
        <div class="disclaimer">
            ⚠️ <strong>Catatan:</strong> Hasil deteksi dihasilkan oleh model Machine Learning dan dapat
            mengandung kesalahan. Gunakan penilaian Anda sendiri untuk memvalidasi hasil.
        </div>
        """, unsafe_allow_html=True)


st.markdown("---")
with st.expander("📋 Deteksi Beberapa Pesan Sekaligus (Batch)"):
    st.markdown("Masukkan beberapa pesan, **satu per baris**:")
    batch_input = st.text_area("Batch input:", height=150, key="batch",
                               placeholder="Pesan 1...\nPesan 2...\nPesan 3...")
    if st.button("🔍 Deteksi Semua", key="batch_btn"):
        if batch_input.strip():
            messages = [m.strip() for m in batch_input.strip().split('\n') if m.strip()]
            results  = []
            for msg in messages:
                c = preprocess_text(msg)
                v = vectorizer.transform([c])
                p = model.predict(v)[0]
                results.append({
                    'Pesan'  : msg[:60] + ('...' if len(msg) > 60 else ''),
                    'Label'  : '🚫 SPAM' if p == 1 else '✅ HAM',
                    'Status' : 'SPAM' if p == 1 else 'HAM'
                })
            import pandas as pd
            result_df = pd.DataFrame(results)
            st.dataframe(result_df[['Pesan', 'Label']], use_container_width=True)
            spam_count = sum(1 for r in results if r['Status'] == 'SPAM')
            st.info(f"Hasil: **{spam_count} SPAM** dan **{len(results)-spam_count} HAM** dari {len(results)} pesan.")
        else:
            st.warning("Masukkan minimal 1 pesan!")

st.markdown("---")
st.caption("🎓 Proyek Machine Learning — Klasifikasi Spam SMS | Metodologi CRISP-DM | NLP + Linear SVM")
