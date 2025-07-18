import streamlit as st
from streamlit_option_menu import option_menu

# ====== Halaman-Halaman ======

def Dashboard():
    st.markdown("# Dashboard 🎈")
    st.write("Selamat datang di aplikasi klasifikasi varietas padi!")

def Introduction():
    st.title("Latar Belakang")
    st.write("""
             Beras merupakan komoditas pangan utama dan sumber karbohidrat penting bagi masyarakat Indonesia. 
             Konsumsinya terus meningkat dan mencapai 30,34 juta ton pada tahun 2024, menjadikannya sangat vital bagi ketahanan pangan nasional. 
             Namun, terdapat tantangan seperti pemalsuan atau pencampuran varietas beras dapat mengganggu stabilitas pasar dan menurunkan kepercayaan konsumen.
             Sementara itu, metode identifikasi manual berbasis visual sering kali memakan waktu, membutuhkan keahlian khusus, dan rawan kesalahan. 
             Untuk mengatasinya, teknologi berbasis kecerdasan buatan (AI) dapat menjadi solusi yang menjanjikan. 
             Identifikasi otomatis dengan AI dapat meningkatkan akurasi, efisiensi, dan keandalan klasifikasi varietas beras, serta mendukung pengelolaan dan distribusi yang lebih baik.
    """)
    st.header("Rumusan Masalah")
    st.subheader("Permasalahan")
    st.write("""
            Perbedaan karakteristik visual biji beras sering kali menjadi tantangan utama dalam mengklasifikasikan varietas beras secara akurat. 
            Variasi dalam bentuk, ukuran, tekstur, dan warna biji beras sering kali sangat kecil sehingga sulit dibedakan oleh metode konvensional berbasis pengamatan manual. 
            Dengan demikian, diperlukan pendekatan berbasis teknologi kecerdasan buatan yang mampu mengenali pola-pola visual kompleks pada data citra beras dalam jumlah besar 
            sekaligus dapat mengklasifikasi secara otomatis dan memberikan hasil klasifikasi dengan akurasi yang tinggi
    """)
    st.subheader("Solusi Permasalahan")
    st.write("""
            Dalam mengatasi permasalahan yang telah disebutkan di atas, solusi yang diusulkan dalam penelitian ini adalah dengan mengimplementasikan metode Deep learning berbasis Transfer Learning 
            yaitu dengan arsitektur DenseNet-201 untuk mengklasifikasikan varietas beras secara akurat. Dengan pendekatan ini, diharapkan dapat dihasilkan model klasifikasi yang efektif 
            dalam mendeteksi varietas beras secara optimal sehingga dapat mencegah pemalsuan atau pencampuran varietas beras serta mendukung stabilitas, pengelolaan dan distribusi beras yang lebih efisien
    """)
    st.subheader("Pertanyaan Penelitian")
    st.write("""
            Bagaimana tingkat Accuracy, Precision, Recall, dan F1-Score dari model DenseNet-201 berbasis Transfer Learning dibandingkan dengan Non-Transfer Learning dalam mengklasifikasikan varietas beras?
    """)

    st.header("Tujuan dan Manfaat Penelitian") 
    st.subheader("Tujuan Penelitian")
    st.write("""
            Mengetahui tingkat Accuracy, Precision, recall dan F1-Score dari model DenseNet-201 yang berbasis Transfer Learning dan Non-Transfer Learning dalam mengklasifikasikan varietas beras
    """)
    st.subheader("Manfaat Penelitian")
    st.write("""
            Manfaat dari penelitian ini adalah memberikan kontribusi dalam pengembangan ilmu pengetahuan, khususnya di bidang teknologi pangan dan image classification, 
            dan menghasilkan sistem klasifikasi yang akurat untuk mengidentifikasi varietas beras yang dapat digunakan sebagai solusi inovatif untuk mengatasi keterbatasan metode manual 
            serta dapat digunakan sebagai bahan rujukan untuk penelitian terkait selanjutnya
    """)

    st.header("Batasan Masalah")
    st.write("""
            1.	Penelitian ini hanya mencakup klasifikasi tiga varietas beras, yaitu Ciherang, IR64, dan Mentik Susu, berdasarkan karakteristik visual biji beras. 
            2.	Dataset yang digunakan dalam penelitian ini merupakan data primer berupa data citra biji beras yang terdiri dari 6000 data dengan 3 kelas yaitu Ciherang, IR64, dan Mentik Susu dengan masing-masing kelasnya terdiri dari 2000 citra. 
            3.	Penelitian ini difokuskan pada klasifikasi varietas beras berdasarkan citra digital biji beras.
            4.	Algoritma yang digunakan dalam penelitian ini adalah pre-trained model DenseNet-201.
    """)
            

    st.header("Metodologi")
    st.image("Images/arsitektursistem.png", caption="Arsitektur sistem")

def Dataset_Information():
    st.markdown("# Informasi Dataset")
    st.write("""
             Dataset yang digunakan merupakan data primer yang diperoleh melalui pemotretan langsung menggunakan kamera ponsel  dalam jarak 11.5 cm dari object 
             dengan tingkat zoom maksimal dalam kondisi pencahayaan luar ruangan pada siang hari. Proses pengumpulan data dilakukan mulai dari tanggal 1 November 2024 
             hingga 30 November 2024. <br>     Dataset yang terkumpul berjumlah 6000 data citra biji beras yang terbagi ke dalam 3 kelas yaitu Ciherang, IR64 dan Mentik Susu 
             di mana masing-masing kelas terdiri dari 2.000  data dengan resolusi 3024×3024.
             
    """)
    st.markdown("## Sampel Data")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("Images/ciherang.jpg", caption="Varietas Ciherang", use_column_width=True)
    with col2:
        st.image("Images/ir64.jpg", caption="Varietas IR64", use_column_width=True)
    with col3:
        st.image("Images/mentik.jpg", caption="Varietas Mentik Susu", use_column_width=True)


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
