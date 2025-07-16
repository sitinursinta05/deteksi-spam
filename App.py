import streamlit as st
import joblib
from nltk.tokenize import word_tokenize
import nltk
import re
import string
import time
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Unduh resource NLTK
def safe_nltk_download(resource):
    max_tries = 3
    for i in range(max_tries):
        try:
            nltk.download(resource)
            break
        except PermissionError:
            time.sleep(1)
    else:
        pass

safe_nltk_download('punkt')

# Inisialisasi stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi preprocessing disamakan dengan proses training
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)  # Hapus URL
    text = re.sub(r"\d+", "", text)              # Hapus angka
    text = text.translate(str.maketrans("", "", string.punctuation))  # Hapus tanda baca
    text = re.sub(r"\s+", " ", text).strip()     # Hapus spasi berlebih
    tokens = word_tokenize(text)
    stemmed = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed)

# Load model dan vectorizer
model = joblib.load('RandomForest.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Konfigurasi halaman
st.set_page_config(page_title="Email Spam Detector", page_icon="📧", layout="centered")

# Navigasi
menu = st.sidebar.radio("Navigasi", ["🏠 Beranda", "📧 Deteksi Spam"])

# Halaman Beranda
if menu == "🏠 Beranda":
    st.title("📬 Selamat Datang di Aplikasi Deteksi Spam")
    st.markdown("""
        Aplikasi ini menggunakan model **Random Forest** untuk mendeteksi apakah sebuah email merupakan **Spam** atau **Bukan Spam (Ham)**.

        ### ✨ Fitur:
        - Prediksi akurat menggunakan model terlatih.
        - Tampilan sederhana dan mudah digunakan.
        - Menampilkan akurasi deteksi berdasarkan input email.

        Silakan pilih menu **📧 Deteksi Spam** di sebelah kiri untuk mulai menggunakan aplikasi ini.
    """)

# Halaman Deteksi Spam
elif menu == "📧 Deteksi Spam":
    st.title("📧 Email Spam Classifier")
    st.write("Masukkan isi email di bawah ini, dan sistem akan memprediksi apakah email tersebut merupakan **Spam** atau **Bukan Spam (Ham)**.")

    user_input = st.text_area("✉️ Isi Email", height=200)

    if st.button("🔍 Cek Email"):
        if not user_input.strip():
            st.warning("Silakan masukkan isi email terlebih dahulu.")
        else:
            # Proses prediksi
            preprocessed_text = preprocess(user_input)
            vectorized_input = vectorizer.transform([preprocessed_text])
            prediction = model.predict(vectorized_input)[0]
            prediction_proba = model.predict_proba(vectorized_input)[0]

            # Tampilkan debug opsional
            # st.write("Teks setelah preprocessing:", preprocessed_text)
            # st.write("Probabilitas prediksi:", prediction_proba)

            if prediction == 1:
                label = "📮 SPAM"
                confidence = prediction_proba[1] * 100
            else:
                label = "✅ BUKAN SPAM (HAM)"
                confidence = prediction_proba[0] * 100

            st.success(f"Hasil Prediksi: **{label}**")
            st.info(f"Akurasi deteksi berdasarkan input ini: **{confidence:.2f}%**")
