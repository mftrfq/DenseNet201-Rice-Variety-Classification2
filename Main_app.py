import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import pandas as pd

# ====== Halaman-Halaman ======

def Dashboard():
    st.markdown("# Dashboard")
    st.write("Selamat datang")

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
    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.image("Images/arsitektursistem.png", caption="Arsitektur sistem")

def Dataset_Information():
    st.markdown("# Informasi Dataset")
    st.write("""
             Dataset yang digunakan merupakan data primer yang diperoleh melalui pemotretan langsung menggunakan kamera ponsel  dalam jarak 11.5 cm dari object 
             dengan tingkat zoom maksimal dalam kondisi pencahayaan luar ruangan pada siang hari. Proses pengumpulan data dilakukan mulai dari tanggal 1 November 2024 
             hingga 30 November 2024. <br>     Dataset yang terkumpul berjumlah 6000 data citra biji beras yang terbagi ke dalam 3 kelas yaitu Ciherang, IR64 dan Mentik Susu 
             di mana masing-masing kelas terdiri dari 2.000  data dengan resolusi 3024×3024.
             
    """)
    st.markdown("### Sampel Data")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("Images/Ciherang.jpg", caption="Varietas Ciherang", use_container_width=True)
    with col2:
        st.image("Images/IR64.jpg", caption="Varietas IR64", use_container_width=True)
    with col3:
        st.image("Images/Mentik.jpg", caption="Varietas Mentik Susu", use_container_width=True)

def Preprocessing():
    st.markdown("# Preprocessing")
    st.subheader("1. Background Removing")
    st.write("""
            Penghapusan latar belakang dilakukan menggunakan library rembg yang memanfaatkan pre-trained model U2-Net. Dengan cara kerjanya adalah: 
            1. Input RGB 
            2. Two-Level U-Structure (Encoder & Decoder) 
            3. Probability Maps 
            4. Fushion & Thresholding 
            5. Output Masking (RGBA) 
    """)
    st.subheader("Arsitektur U2-Net")
    rembg1 = Image.open("Images/u2netarch.png").resize((600, 400))
    rembg2 = Image.open("Images/RSU.png").resize((600, 400))
    col1, col2 = st.columns(2)
    with col1:
        st.image(rembg1, caption="Arsitektur U2-Net", use_container_width=True)
    with col2:
        st.image(rembg2, caption="Komponen Decoder-Encoder", use_container_width=True)
    st.markdown("### Hasil")
    st.image("Images/bgremoved.png", caption="Background Removed")
    
    st.subheader("2. Grayscale Conversion")
    st.write("""
            Grayscale Conversion dilakukan dengan mengambil nilai intensitas warna dari setiap pixel dalam gambar RGB 
            dan menghitung nilai abu-abu menggunakan weighted sum dari nilai merah (R), hijau (G), dan biru (B).
    """)
    st.latex(r"""
             gray = 0.299R + 0.587G + 0.114B 
    """)
    st.write("""
            Setelah nilai intensitas abu-abu dihitung, setiap pixel dalam gambar asli (yang berisi tiga nilai R, G, dan B) digantikan oleh satu nilai grey
    """)
    st.subheader("")
    st.write("""
    """)
    st.subheader("")
    st.write("""
    """)
    
def Model_Training():
    st.markdown("# Model Training")
    st.write("""
        Model dilatih menggunakan arsitektur **DenseNet201** dengan pendekatan Transfer Learning yang memanfaatkan pre-trained weight dari ImageNet.
        Hyperparameter yang digunakan:
        - Optimizer: Adam
        - Learning rate: 0.001
        - Batch size: 32
        - Epoch: 30
    """)
    st.subheader("Arsitektur DenseNet-201")
    st.image("Images/densenet201arch.png", caption = "Arsitektur DenseNet-201")
    st.markdown("### DenseNet-201 Layers")
    data = {
        "Layers": [
            "Input",
            "Convolution",
            "Pooling",
            "Dense Block 1",
            "Transition Layer 1",
            " ",
            "Dense Block 2",
            "Transition Layer 2",
            " ",
            "Dense Block 3",
            "Transition Layer 3",
            " ",
            "Dense Block 4",
            "Classification Layer",
            " "
        ],
        "Details": [
            "--",
            "Convolution 7×7, Stride 2",
            "Max Pool 3×3, Stride 2",
            "[(Conv 1×1 @ Conv 3×3)] × 6",
            "Convolution 1×1",
            "Average Pool 2×2, Stride 2",
            "[(Conv 1×1 @ Conv 3×3)] × 12",
            "Convolution 1×1",
            "Average Pool 2×2, Stride 2",
            "[(Conv 1×1 @ Conv 3×3)] × 48",
            "Convolution 1×1",
            "Average Pool 2×2, Stride 2",
            "[(Conv 1×1 @ Conv 3×3)] × 32",
            "Global Average Pool 7×7",
            "1000D Fully connected, Softmax"
        ],
        "Output Shape": [
            "224×224×3",
            "112×112×64",
            "56×56×64",
            "56×56×256",
            "56×56×128",
            "28×28×128",
            "28×28×512",
            "28×28×256",
            "14×14×256",
            "14×14×1792",
            "14×14×896",
            "7×7×896",
            "7×7×1920",
            "1920",
            "1000"
        ]
    }  
    df = pd.DataFrame(data)
    st.table(df)

def Model_Evaluation():
    st.markdown("# Model Evaluation")
    st.write("""
            Evaluasi model dilakukan untuk performa model yang telah dilatih dalam mengklasifikasikan data yang tidak terlihat sebelumnya yang diukur 
            melalui nilai nilai metric score yang didapatkan berdasarkan confusion matrix
    """)
    st.image("Images/confusionmtx.png", caption = "Confusion Matrix")
    st.write("### Classification Report")
    st.write("""
            Sehingga melalui confusion matrix dapat diketahui beberapa evaluation metric seperti Accuracy, Precision, recall, dan F1-Score sebagai berikut
    """)
    data = {
    "Class": ["ciherang", "ir64", "mentik", "accuracy", "macro avg", "weighted avg"],
    "Precision": [0.87, 0.92, 0.99, None, 0.93, 0.93],
    "Recall":    [0.92, 0.89, 0.97, None, 0.93, 0.93],
    "F1-Score":  [0.89, 0.90, 0.98, 0.93, 0.93, 0.93],
    "Support":   [200, 200, 200, 600, 600, 600]
    }
    
    df = pd.DataFrame(data)
    st.subheader("Hasil Evaluasi Model")
    st.table(df)

def Prediction():
    st.markdown("# Prediction")
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
