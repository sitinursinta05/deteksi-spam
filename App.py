import streamlit as st
import joblib
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
import time

# Unduh resource NLTK dengan aman
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

# Download tokenizers yang dibutuhkan
safe_nltk_download('punkt')
safe_nltk_download('punkt_tab')

# Cek ketersediaan tokenizer
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    safe_nltk_download('punkt')

# Load model dan vectorizer
model = joblib.load('RandomForest.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing fungsi
stemmer = PorterStemmer()
def preprocess(text):
    tokens = word_tokenize(text)
    stemmed = ' '.join([stemmer.stem(word) for word in tokens])
    return stemmed

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

            if prediction == 1:
                label = "📮 SPAM"
                confidence = prediction_proba[1] * 100
            else:
                label = "✅ BUKAN SPAM (HAM)"
                confidence = prediction_proba[0] * 100

            st.success(f"Hasil Prediksi: **{label}**")
            st.info(f"Akurasi deteksi berdasarkan input ini: **{confidence:.2f}%**")
