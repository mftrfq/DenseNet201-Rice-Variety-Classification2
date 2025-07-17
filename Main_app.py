import streamlit as st

# ====== Halaman Utama ======

def Dashboard():
    st.markdown("# Dashboard 🎈")
    st.write("Selamat datang di aplikasi klasifikasi varietas padi!")

def Introduction():
    st.markdown("# Introduction")
    st.write("""
        Aplikasi ini dibuat menggunakan arsitektur Transfer Learning untuk mengklasifikasikan varietas padi.
        Dataset telah diproses melalui tahapan preprocessing, training, evaluasi, dan prediksi.
    """)

def Dataset_Information():
    st.markdown("# Dataset Information")
    st.write("""
        Dataset terdiri dari gambar biji padi yang telah dikategorikan berdasarkan varietasnya.
        Terdapat data latih, validasi, dan pengujian dalam format citra.
    """)

# ====== Halaman Tambahan ======

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
        st.write("Prediksi: **Varietas IR64**")  # Gantilah bagian ini jika ingin memuat model asli

# ====== Routing Halaman ======

page_names_to_funcs = {
    "Dashboard": Dashboard,
    "Introduction": Introduction,
    "Dataset Information": Dataset_Information,
    "Preprocessing": Preprocessing,
    "Model Training": Model_Training,
    "Model Evaluation": Model_Evaluation,
    "Prediction": Prediction,
}

# ====== Sidebar Navigasi ======

st.sidebar.title("Navigasi Halaman")
selected_page = st.sidebar.selectbox("Pilih Halaman", page_names_to_funcs.keys())

# ====== Tampilkan Halaman ======
page_names_to_funcs[selected_page]()
