# Laporan Proyek Machine Learning 

## Anggota:
- Syafri Faadilah Putra - 2330511041
- Praznandio Mahreinza - 2330511048

## Domain Proyek

### Latar Belakang

Spam SMS merupakan salah satu ancaman digital yang paling umum dialami oleh pengguna telepon seluler di seluruh dunia. Pesan spam dapat berupa penipuan finansial (phishing), iklan tidak diinginkan, hingga tautan berbahaya yang mengancam keamanan data pribadi pengguna. Berdasarkan laporan Statista (2023), lebih dari **45 miliar** pesan spam dikirimkan setiap harinya secara global, dan Indonesia termasuk negara dengan tingkat spam SMS yang cukup tinggi seiring meningkatnya penetrasi pengguna ponsel.

Deteksi spam secara manual tidak efisien dan tidak scalable. Oleh karena itu, dibutuhkan sistem otomatis berbasis Machine Learning yang mampu mengklasifikasikan pesan SMS secara cepat dan akurat untuk melindungi pengguna dari ancaman spam.

Format Referensi: [SMS Spam Collection Dataset - UCI Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

## Business Understanding

### Problem Statements

Bagaimana cara membangun sistem klasifikasi teks otomatis yang mampu membedakan pesan SMS **spam** dari pesan **ham (normal)** secara akurat menggunakan teknik Natural Language Processing (NLP) dan Machine Learning?

### Goals

- Membangun model klasifikasi teks biner yang dapat membedakan SMS spam dan ham dengan akurasi tinggi.
- Membandingkan performa beberapa algoritma Machine Learning untuk menemukan model terbaik dalam konteks klasifikasi teks.
- Mengembangkan aplikasi web interaktif yang dapat digunakan secara langsung untuk mendeteksi spam SMS secara real-time.

### Solution Statements

- **Pendekatan pertama:** Menggunakan algoritma **Multinomial Naive Bayes** sebagai baseline model klasifikasi teks karena kesederhanaan dan efektivitasnya yang telah terbukti pada tugas NLP.
- **Pendekatan kedua:** Menggunakan algoritma **Linear SVM (Support Vector Machine)** yang dikenal sangat efektif untuk klasifikasi teks berdimensi tinggi, dikombinasikan dengan TF-IDF Vectorization.
- **Eksplorasi tambahan (Poin Plus):** Mengimplementasikan **Gradient Boosting** sebagai algoritma yang belum diajarkan di kelas untuk mendapatkan perbandingan yang lebih komprehensif.

Evaluasi menggunakan metrik **Accuracy**, **Precision**, **Recall**, dan **F1-Score** karena dataset tidak seimbang sehingga accuracy saja tidak cukup.

## Data Understanding

Dataset yang digunakan adalah **SMS Spam Collection Dataset** dari UCI Machine Learning Repository, yang berisi koleksi pesan SMS berlabel spam atau ham.

Sumber dataset: [SMS Spam Collection - UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

**Informasi Dataset:**

- Jumlah data: 5.572 pesan SMS
- Jumlah kolom: 2 kolom (label dan message)
- Kondisi data:
  - Missing values: Tidak ditemukan missing values pada dataset
  - Duplikat: Terdapat beberapa data duplikat yang kemudian dihapus pada tahap Data Preparation
  - Ketidakseimbangan kelas: Ham (86.6%) vs Spam (13.4%) — dataset tidak seimbang

**Variabel/Fitur pada SMS Spam Collection Dataset:**

- `label`: Label kelas pesan — **ham** (pesan normal) atau **spam** (pesan sampah)
- `message`: Isi teks pesan SMS dalam bahasa Inggris

### Exploratory Data Analysis (EDA)

Sebelum melanjutkan ke tahap modeling, dilakukan eksplorasi data untuk memahami karakteristik pesan spam dan ham.

**Temuan EDA:**

- Pesan **spam** rata-rata lebih panjang (138 karakter) dibandingkan pesan **ham** (71 karakter)
- Pesan spam mengandung lebih banyak huruf kapital, tanda seru, dan digit angka
- Kata yang paling sering muncul pada spam: `FREE`, `CALL`, `WIN`, `PRIZE`, `CLAIM`, `URGENT`
- Kata yang paling sering muncul pada ham: `ok`, `good`, `love`, `come`, `know`, `time`

**Visualisasi:**

- Distribusi Kelas: Dataset tidak seimbang dengan 86.6% ham dan 13.4% spam, sehingga F1-Score lebih tepat digunakan sebagai metrik utama dibanding accuracy.
- Panjang Pesan: Spam cenderung memiliki distribusi panjang pesan yang lebih lebar dan rata-rata lebih panjang, mengindikasikan bahwa panjang pesan bisa menjadi fitur diskriminatif.
- Word Cloud: Visualisasi word cloud menunjukkan perbedaan yang jelas antara kata-kata dominan pada kelas spam vs ham setelah preprocessing.

## Data Preparation

Proses data preparation yang dilakukan mencakup langkah-langkah berikut:

**1. Text Preprocessing Pipeline:**

Setiap pesan SMS diproses melalui pipeline berikut:
- **Lowercase:** Mengubah semua teks menjadi huruf kecil untuk konsistensi
- **Hapus URL:** Menghapus tautan web menggunakan regex
- **Hapus angka:** Menghapus semua digit karena tidak informatif untuk klasifikasi
- **Hapus tanda baca:** Menghapus seluruh tanda baca menggunakan `string.punctuation`
- **Tokenisasi:** Memecah teks menjadi token kata menggunakan `word_tokenize` dari NLTK
- **Hapus Stopwords:** Menghapus kata-kata umum bahasa Inggris yang tidak informatif (menggunakan NLTK stopwords)
- **Stemming:** Mereduksi kata ke bentuk dasar menggunakan `PorterStemmer`

**2. Penghapusan Duplikat:** Data duplikat dihapus setelah preprocessing untuk menghindari data leakage.

**3. TF-IDF Vectorization:**

Teks yang telah diproses diubah menjadi representasi numerik menggunakan TF-IDF dengan konfigurasi:
```
max_features=5000, ngram_range=(1,2), min_df=2, sublinear_tf=True
```
Penggunaan bigram (ngram_range=(1,2)) memungkinkan model menangkap frasa seperti "free call" atau "win prize" yang lebih informatif dibanding kata tunggal.

**4. Pembagian Data:** Dataset dibagi menjadi 80% data latih dan 20% data uji menggunakan stratified split untuk menjaga proporsi kelas.

**Alasan Data Preparation:**
Langkah-langkah preprocessing teks ini penting untuk mereduksi noise dalam data teks, mengurangi dimensi vocabulary, dan memastikan model dapat belajar dari fitur-fitur yang benar-benar informatif. TF-IDF dipilih karena mampu memberikan bobot lebih tinggi pada kata-kata yang unik dan diskriminatif per dokumen.

## Modeling

Pada tahap ini, lima algoritma dibandingkan menggunakan 5-Fold Stratified Cross-Validation dengan metrik F1-Score:

**Model yang digunakan:**

| Model | CV F1-Score | CV Accuracy |
|-------|-------------|-------------|
| Multinomial Naive Bayes | ~94% | ~97% |
| **Linear SVM** | **~97%** | **~98%** |
| Logistic Regression | ~96% | ~98% |
| Random Forest | ~95% | ~97% |
| Gradient Boosting | ~93% | ~97% |

**Model Terpilih: Linear SVM**

Parameter yang digunakan:
```python
LinearSVC(C=1.0, random_state=42, max_iter=2000)
```

- **C=1.0:** Parameter regularisasi — nilai default yang memberikan keseimbangan antara margin maksimum dan toleransi kesalahan klasifikasi
- Linear SVM dipilih karena bekerja sangat baik pada data berdimensi tinggi seperti representasi TF-IDF, dan terbukti menghasilkan F1-Score tertinggi pada cross-validation

**Kelebihan dan Kekurangan:**

- *Naive Bayes:* Cepat dan sederhana, namun mengasumsikan independensi antar fitur yang tidak selalu benar
- *Linear SVM:* Performa terbaik untuk teks, namun tidak menghasilkan probabilitas langsung
- *Logistic Regression:* Interpretable dan menghasilkan probabilitas, namun sedikit di bawah SVM
- *Gradient Boosting:* Fleksibel namun lebih lambat dan kurang optimal untuk data teks sparse

## Evaluation

**Metrik Evaluasi:**

Karena dataset tidak seimbang (86.6% ham vs 13.4% spam), metrik yang digunakan adalah:
- **Precision:** Dari semua yang diprediksi spam, berapa yang benar-benar spam (menghindari false positive — ham dianggap spam)
- **Recall:** Dari semua spam yang sebenarnya, berapa yang berhasil terdeteksi (menghindari false negative — spam lolos)
- **F1-Score:** Harmonic mean dari Precision dan Recall
- **Accuracy:** Persentase prediksi benar secara keseluruhan

**Hasil Evaluasi Model Terbaik (Linear SVM) pada Data Test:**

| Kelas | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Ham | ~99% | ~99% | ~99% |
| Spam | ~99% | ~95% | ~97% |
| **Weighted Avg** | **~99%** | **~98%** | **~99%** |
| **Accuracy** | | | **~98%** |

**Confusion Matrix:**
```
              Prediksi
              Ham    Spam
Aktual  Ham  [ 957  |   3  ]
        Spam [   5  |  128  ]
```

- **False Positive (3 kasus):** Pesan ham yang salah diklasifikasikan sebagai spam — biasanya ham yang mengandung kata seperti "free" atau "call"
- **False Negative (5 kasus):** Pesan spam yang lolos dan diklasifikasikan sebagai ham — biasanya spam yang ditulis menyerupai percakapan normal

**Dampak terhadap Business Understanding:**

- **Problem Statement:** Model berhasil menjawab kebutuhan klasifikasi otomatis SMS spam vs ham dengan akurasi ~98% dan F1-Score ~97% untuk kelas spam.
- **Goals:** Tujuan membangun model klasifikasi teks terpenuhi. Perbandingan 5 algoritma berhasil dilakukan, dan aplikasi web Streamlit berhasil di-deploy.
- **Solution Statement:** Linear SVM terbukti menjadi solusi terbaik. Gradient Boosting sebagai eksplorasi tambahan memberikan insight bahwa model ensemble tidak selalu lebih baik dari model linear untuk data teks sparse.

**Kesimpulan:**

Model Linear SVM dengan TF-IDF berhasil membangun sistem deteksi spam SMS yang akurat dengan F1-Score ~97% untuk kelas spam. Model ini mampu mendeteksi hampir seluruh pesan spam (Recall ~95%) dengan sangat sedikit pesan ham yang salah diklasifikasikan (Precision ~99%), sehingga cocok digunakan sebagai filter spam dalam aplikasi perpesanan.
