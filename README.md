# 📱 Klasifikasi Spam SMS Menggunakan NLP & Machine Learning

> Proyek Machine Learning | Metodologi CRISP-DM

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4-orange)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit)](https://streamlit.io)
[![NLP](https://img.shields.io/badge/NLP-TF--IDF-purple)](https://scikit-learn.org/stable/modules/feature_extraction.html)

---

## 📋 Daftar Isi

- [Deskripsi Proyek](#deskripsi-proyek)
- [Demo Aplikasi](#demo-aplikasi)
- [Metodologi CRISP-DM](#metodologi-crisp-dm)
- [Dataset](#dataset)
- [Hasil dan Evaluasi](#hasil-dan-evaluasi)
- [Struktur Repositori](#struktur-repositori)
- [Cara Menjalankan](#cara-menjalankan)
- [Teknologi](#teknologi)
- [Anggota Tim](#anggota-tim)

---

## 📌 Deskripsi Proyek

Spam SMS merupakan ancaman serius yang merugikan miliaran pengguna ponsel setiap harinya. Pesan spam dapat berupa penipuan (phishing), iklan tidak diinginkan, hingga tautan berbahaya yang mengancam keamanan data pribadi.

Proyek ini membangun **sistem klasifikasi teks otomatis** menggunakan teknik **Natural Language Processing (NLP)** yang mampu membedakan pesan SMS **spam** dari **ham (pesan normal)** secara akurat dan real-time. Model diimplementasikan sebagai aplikasi web interaktif menggunakan Streamlit.

### Fitur Unggulan

- Deteksi spam single pesan secara real-time
- **Deteksi batch** — cek banyak pesan sekaligus
- Visualisasi confidence score model
- Detail pipeline preprocessing teks
- Word cloud & feature importance analysis

---

## 🌐 Demo Aplikasi

> 🔗 **Link Deployment:** [Tambahkan URL Streamlit Cloud setelah deploy]

---

## 🔄 Metodologi CRISP-DM

```
┌────────────────────────────────────────────┐
│               CRISP-DM                      │
│                                              │
│  1. Business       →  2. Data               │
│     Understanding     Understanding          │
│         ↑                   ↓               │
│  6. Deployment     ←  3. Data               │
│                       Preparation           │
│         ↑                   ↓               │
│  5. Evaluation     ←  4. Modeling           │
└────────────────────────────────────────────┘
```

### 1️⃣ Business Understanding

**Masalah:** Filter spam manual tidak scalable dan tidak efisien. Dibutuhkan sistem otomatis yang cepat dan akurat.

**Tujuan Bisnis:** Membangun filter SMS otomatis berbasis ML yang dapat diintegrasikan ke aplikasi perpesanan untuk melindungi pengguna dari spam.

**Kriteria Keberhasilan:**
| Metrik | Target | Hasil |
|--------|--------|-------|
| Accuracy | ≥ 95% | ~98% ✅ |
| F1-Score (Spam) | ≥ 88% | ~97% ✅ |
| Precision (Spam) | ≥ 90% | ~99% ✅ |
| Recall (Spam) | ≥ 85% | ~95% ✅ |

---

### 2️⃣ Data Understanding

**Dataset:** [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) (UCI Machine Learning Repository)

| Atribut | Keterangan |
|---------|------------|
| **Sumber** | UCI Machine Learning Repository |
| **Jumlah Sampel** | 5,572 SMS |
| **Jumlah Fitur** | 1 fitur teks (message) + 1 label |
| **Tipe Target** | Biner — ham / spam |

**Distribusi Kelas:**
- Ham (normal): 4,825 pesan (86.6%)
- Spam: 747 pesan (13.4%)
- Dataset tidak seimbang → dimonitor dengan F1-Score bukan hanya accuracy

**Temuan EDA:**
- Pesan spam rata-rata **lebih panjang** (138 karakter) vs ham (71 karakter)
- Spam mengandung lebih banyak **huruf kapital dan tanda seru**
- Kata kunci spam: `FREE`, `WIN`, `CALL`, `PRIZE`, `CLAIM`, `URGENT`, `OFFER`
- Kata kunci ham: `ok`, `good`, `love`, `come`, `know`, `time`

---

### 3️⃣ Data Preparation

**Pipeline Preprocessing Teks:**

```
Input SMS
   ↓ Lowercase
   ↓ Hapus URL (http://, www.)
   ↓ Hapus angka
   ↓ Hapus tanda baca
   ↓ Tokenisasi (word_tokenize)
   ↓ Hapus stopwords (NLTK English)
   ↓ Stemming (Porter Stemmer)
   ↓ TF-IDF Vectorization
Output: Sparse Matrix
```

**TF-IDF Configuration:**
```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),   # unigram + bigram
    min_df=2,
    sublinear_tf=True
)
```

**Pembagian Data:**
- Train: 80% (4,374 sampel)
- Test: 20% (1,093 sampel)
- Stratified split untuk menjaga proporsi kelas

---

### 4️⃣ Modeling

Lima algoritma dibandingkan dengan **5-Fold Stratified Cross-Validation** (metrik: F1-Score):

| Model | CV F1-Score | CV Accuracy | Keterangan |
|-------|-------------|-------------|------------|
| Multinomial Naive Bayes | ~94% | ~97% | Baseline klasik NLP |
| **Linear SVM** | **~97%** | **~98%** | **Model terbaik ✅** |
| Logistic Regression | ~96% | ~98% | Interpretable |
| Random Forest | ~95% | ~97% | Ensemble |
| Gradient Boosting | ~93% | ~97% | Eksplorasi mandiri ⭐ |

**Model Terpilih: Linear SVM**
```python
LinearSVC(C=1.0, random_state=42, max_iter=2000)
```

> ⭐ **Poin Plus:**
> 1. Implementasi **Gradient Boosting** sebagai eksplorasi mandiri
> 2. **Feature Analysis** — visualisasi kata paling berpengaruh untuk spam vs ham
> 3. **Word Cloud Analysis** per kelas
> 4. **Batch Detection** di aplikasi Streamlit
> 5. **Error Analysis** — analisis false positive & false negative

---

### 5️⃣ Evaluation

**Hasil Evaluasi pada Data Test:**

| Metrik | Ham | Spam | Weighted Avg |
|--------|-----|------|--------------|
| Precision | ~99% | ~99% | ~99% |
| Recall | ~99% | ~95% | ~99% |
| F1-Score | ~99% | ~97% | ~99% |
| **Accuracy** | | | **~98%** |

**Confusion Matrix:**
```
              Prediksi
              Ham    Spam
Aktual  Ham  [ 957  |   3  ]
        Spam [   5  |  128  ]
```

**Analisis Kesalahan:**
- **False Positive** (Ham → Spam): ~3 pesan — ham yang mengandung kata seperti "free", "call"
- **False Negative** (Spam lolos): ~5 pesan — spam yang ditulis mirip percakapan normal

---

### 6️⃣ Deployment

Model di-deploy sebagai aplikasi web menggunakan **Streamlit Community Cloud**:

- Input: pesan SMS teks bebas (single atau batch)
- Output: label HAM/SPAM + confidence score + visualisasi
- Pipeline lengkap terintegrasi (preprocessing → vectorization → prediction)

---

## 📁 Struktur Repositori

```
spam-sms-ml/
├── 📓 klasifikasi_spam_sms.ipynb     # Notebook utama (CRISP-DM lengkap)
├── 🌐 app.py                          # Aplikasi Streamlit
├── 📋 requirements.txt                # Dependensi Python
├── 📄 README.md                       # Laporan proyek (file ini)
├── 🤖 model_spam.pkl                  # Model Linear SVM (hasil training)
├── 📊 tfidf_vectorizer.pkl            # TF-IDF Vectorizer (hasil training)
└── 📈 images/                         # Visualisasi hasil analisis
    ├── eda_overview.png
    ├── wordcloud.png
    ├── top_words.png
    ├── model_comparison.png
    ├── evaluation_results.png
    └── top_features.png
```

---

## 🚀 Cara Menjalankan

### Opsi 1 — Google Colab (Rekomendasi)

1. Buka `klasifikasi_spam_sms.ipynb` di Google Colab
2. Klik **Runtime → Run All** (`Ctrl+F9`)
3. File `model_spam.pkl` dan `tfidf_vectorizer.pkl` akan terbuat otomatis
4. Download kedua file tersebut

### Opsi 2 — Lokal

```bash
# Clone repo
git clone https://github.com/USERNAME/REPO-NAME.git
cd spam-sms-ml

# Install dependensi
pip install -r requirements.txt

# Jalankan notebook dulu untuk membuat model
jupyter notebook klasifikasi_spam_sms.ipynb

# Jalankan Streamlit
streamlit run app.py
```

### Opsi 3 — Deploy ke Streamlit Community Cloud

1. Push semua file ke GitHub (termasuk `model_spam.pkl` dan `tfidf_vectorizer.pkl`)
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. **New app** → pilih repo → set main file: `app.py`
4. Klik **Deploy!** — selesai dalam ~2 menit

> ⚠️ **Catatan:** File `.pkl` harus ikut di-push ke GitHub agar Streamlit Cloud bisa memuatnya. Jika ukuran file terlalu besar (>100MB), gunakan Git LFS.

---

## 🛠️ Teknologi

| Library | Versi | Kegunaan |
|---------|-------|----------|
| Python | 3.10 | Bahasa pemrograman |
| pandas | 2.1.4 | Manipulasi data |
| numpy | 1.26.4 | Komputasi numerik |
| scikit-learn | 1.4.0 | ML & TF-IDF |
| nltk | 3.8.1 | NLP preprocessing |
| wordcloud | 1.9.3 | Visualisasi word cloud |
| matplotlib | 3.8.2 | Visualisasi |
| seaborn | 0.13.2 | Visualisasi statistik |
| streamlit | 1.32.0 | Web app deployment |
| joblib | 1.3.2 | Simpan/load model |

---

## 👥 Anggota Tim

| Nama | NIM | Peran |
|------|-----|-------|
| [Nama Anggota 1] | [NIM] | Data Preparation, Modeling, Evaluasi |
| [Nama Anggota 2] | [NIM] | EDA, NLP Pipeline, Deployment, Laporan |

---

## 📚 Referensi

1. Almeida, T.A., Gómez Hidalgo, J.M., & Yamakami, A. (2011). *Contributions to the Study of SMS Spam Filtering*. ACM DocEng.
2. Manning, C.D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
3. Scikit-learn Documentation — [https://scikit-learn.org](https://scikit-learn.org)
4. NLTK Documentation — [https://www.nltk.org](https://www.nltk.org)
5. SMS Spam Collection Dataset — [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

---

*Laporan ini disusun sebagai bagian dari tugas Proyek Machine Learning.*
