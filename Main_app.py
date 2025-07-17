import streamlit as st
from streamlit_option_menu import option_menu

# ====== Halaman-Halaman ======

def Dashboard():
    st.markdown("# Dashboard 🎈")
    st.write("Selamat datang di aplikasi klasifikasi varietas padi!")

def Introduction():
    st.markdown("# Introduction")
    st.title("Latar Belakang")
    st.write("""
             Beras merupakan komoditas pangan utama dan sumber karbohidrat penting bagi masyarakat Indonesia. 
             Konsumsinya terus meningkat dan mencapai 30,34 juta ton pada tahun 2024, menjadikannya sangat vital bagi ketahanan pangan nasional. 
             Namun, terdapat tantangan seperti pemalsuan atau pencampuran varietas beras dapat mengganggu stabilitas pasar dan menurunkan kepercayaan konsumen.
             Sementara itu, metode identifikasi manual berbasis visual sering kali memakan waktu, membutuhkan keahlian khusus, dan rawan kesalahan. 
             Untuk mengatasinya, teknologi berbasis kecerdasan buatan (AI) dapat menjadi solusi yang menjanjikan. 
             Identifikasi otomatis dengan AI dapat meningkatkan akurasi, efisiensi, dan keandalan klasifikasi varietas beras, serta mendukung pengelolaan dan distribusi yang lebih baik.
    """)

def Dataset_Information():
    st.markdown("# Dataset Information")
    st.write("""
        Dataset terdiri dari gambar biji padi yang telah dikategorikan berdasarkan varietasnya.
        Terdapat data latih, validasi, dan pengujian dalam format citra.
    """)

def Preprocessing():
    st.markdown("# Preprocessing 🔧")
    st.write("""
        Tahapan preprocessing meliputi:
        1. Penghapusan background (rembg)
        2. Konversi grayscale
        3. CLAHE
        4. Cropping objek
        5. Konversi RGB
        6. Resize
        7. Normalisasi pixel
    """)

def Model_Training():
    st.markdown("# Model Training 🧠")
    st.write("""
        Model dilatih menggunakan arsitektur **DenseNet201** dengan pendekatan Transfer Learning dan Non-Transfer Learning.
        Hyperparameter yang digunakan:
        - Optimizer: Adam
        - Learning rate: 0.001
        - Batch size: 32
        - Epoch: 30
    """)

def Model_Evaluation():
    st.markdown("# Model Evaluation 📊")
    st.write("""
        Evaluasi model menggunakan metrik:
        - Accuracy
        - Precision
        - Recall
        - F1-score
        - Confusion Matrix

        Hasil evaluasi menunjukkan bahwa metode Transfer Learning menghasilkan akurasi lebih tinggi dibanding Non-Transfer Learning.
    """)

def Prediction():
    st.markdown("# Prediction 🔍")
    st.write("Upload gambar biji padi untuk prediksi varietasnya.")

    uploaded_file = st.file_uploader("Upload gambar...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Gambar yang diunggah", use_column_width=True)
        st.success("Model prediksi akan ditampilkan di sini (simulasi).")
        st.write("Prediksi: **Varietas IR64**")  # Ganti bagian ini jika ada model asli

# ====== Menu Sidebar (Option Menu) ======

with st.sidebar:
    selected = option_menu(
        menu_title="Rice Variety Classification",
        options=[
            "Dashboard", 
            "Introduction", 
            "Dataset Information", 
            "Preprocessing", 
            "Model Training", 
            "Model Evaluation", 
            "Prediction"
        ],
        icons=["house", "info-circle", "bar-chart", "tools", "cpu", "clipboard-check", "search"],
        menu_icon="",
        default_index=0,
    )

# ====== Pemanggilan Fungsi Halaman Berdasarkan Menu ======

page_names_to_funcs = {
    "Dashboard": Dashboard,
    "Introduction": Introduction,
    "Dataset Information": Dataset_Information,
    "Preprocessing": Preprocessing,
    "Model Training": Model_Training,
    "Model Evaluation": Model_Evaluation,
    "Prediction": Prediction,
}

page_names_to_funcs[selected]()
