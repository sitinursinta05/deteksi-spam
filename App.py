import streamlit as st
import joblib
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
import re
import string
import time

# Download tokenizer punkt dari NLTK
def safe_nltk_download(resource):
    for _ in range(3):
        try:
            nltk.download(resource)
            break
        except:
            time.sleep(1)

safe_nltk_download('punkt')

# Cek apakah tokenizer tersedia
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    safe_nltk_download('punkt')

# Inisialisasi stemmer
stemmer = PorterStemmer()

# Fungsi Preprocessing (tanpa Sastrawi, tapi tetap bersih)
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)  # Hapus URL
    text = re.sub(r"\d+", "", text)              # Hapus angka
    text = text.translate(str.maketrans("", "", string.punctuation))  # Hapus tanda baca
    text = re.sub(r"\s+", " ", text).strip()     # Normalisasi spasi
    tokens = word_tokenize(text)
    stemmed = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed)

# Load model dan vectorizer
try:
    model = joblib.load('RandomForest.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
except Exception as e:
    st.error("❌ Gagal memuat model atau vectorizer. Pastikan file 'RandomForest.pkl' dan 'tfidf_vectorizer.pkl' tersedia.")
    st.stop()

# Konfigurasi halaman
st.set_page_config(page_title="Email Spam Detector", page_icon="📧", layout="centered")

# Navigasi sidebar
menu = st.sidebar.radio("Navigasi", ["🏠 Beranda", "📧 Deteksi Spam"])

# Halaman Beranda
if menu == "🏠 Beranda":
    st.title("📬 Selamat Datang di Aplikasi Deteksi Spam")
    st.markdown("""
        Aplikasi ini menggunakan model **Random Forest** untuk memprediksi apakah sebuah email merupakan **Spam** atau **Bukan Spam (Ham)**.

        ### 🔍 Fitur Utama:
        - Preprocessing otomatis dan ringan (tanpa Sastrawi)
        - Deteksi cepat dengan model Machine Learning
        - Visualisasi hasil prediksi

        Pilih menu **📧 Deteksi Spam** untuk mulai menggunakan aplikasi.
    """)

# Halaman Deteksi Spam
elif menu == "📧 Deteksi Spam":
    st.title("📧 Email Spam Classifier")
    st.write("Masukkan isi email di bawah ini untuk mengecek apakah termasuk **SPAM** atau **BUKAN SPAM (HAM)**.")

    user_input = st.text_area("✉️ Masukkan Isi Email", height=200)

    if st.button("🔍 Cek Email"):
        if not user_input.strip():
            st.warning("⚠️ Silakan masukkan isi email terlebih dahulu.")
        else:
            # Proses deteksi
            preprocessed_text = preprocess(user_input)
            vectorized_input = vectorizer.transform([preprocessed_text])
            prediction = model.predict(vectorized_input)[0]
            prediction_proba = model.predict_proba(vectorized_input)[0]

            if prediction == 1:
                label = "📮 SPAM"
                confidence = prediction_proba[1] * 100
                st.error(f"Hasil Prediksi: **{label}**")
            else:
                label = "✅ BUKAN SPAM (HAM)"
                confidence = prediction_proba[0] * 100
                st.success(f"Hasil Prediksi: **{label}**")

            st.info(f"Kepercayaan model: **{confidence:.2f}%**")
