import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image, ImageOps
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
import cv2
from rembg import remove
from io import BytesIO
from collections import Counter
import os
import gdown

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Rice Variety Classification",
    page_icon="🌾",
    initial_sidebar_state='auto'
)

@st.cache_resource
def load_model():
    drive_id = "14T6m4berh-Z_WjMFaQ07sQDthquWjkyk"
    filename = "TL_model_30epoch.keras"
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?id={drive_id}"
        gdown.download(url, filename, quiet=False)
    model = tf.keras.models.load_model(filename)
    return model

model = load_model()
class_names = ['ciherang', 'ir64', 'mentik']
label_colors = {
    'ciherang': (255, 0, 0),
    'ir64': (0, 0, 255),
    'mentik': (0, 255, 0),
}

def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image) / 255.0
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape, verbose=0)
    return prediction

# ====== Halaman-Halaman ======
def Introduction():
    st.title("KLASIFIKASI VARIETAS BERAS MENGGUNAKAN TRANSFER LEARNING DENGAN ARSITEKTUR DENSENET-201")
    st.divider()
    st.header("Latar Belakang")
    st.write("""
             Beras merupakan komoditas pangan utama dan sumber karbohidrat penting bagi masyarakat Indonesia. 
             Konsumsinya terus meningkat dan mencapai 30,34 juta ton pada tahun 2024, menjadikannya sangat vital bagi ketahanan pangan nasional. 
             Namun, terdapat tantangan seperti pemalsuan atau pencampuran varietas beras dapat mengganggu stabilitas pasar dan menurunkan kepercayaan konsumen.
             Sementara itu, metode identifikasi manual berbasis visual sering kali memakan waktu, membutuhkan keahlian khusus, dan rawan kesalahan. 
             Untuk mengatasinya, teknologi berbasis kecerdasan buatan (AI) dapat menjadi solusi yang menjanjikan. 
             Identifikasi otomatis dengan AI dapat meningkatkan akurasi, efisiensi, dan keandalan klasifikasi varietas beras, serta mendukung pengelolaan dan distribusi yang lebih baik.
    """)
    # st.divider()
    
    st.header("Rumusan Masalah")
    st.markdown("#### - Permasalahan")
    st.write("""
            Perbedaan karakteristik visual biji beras sering kali menjadi tantangan utama dalam mengklasifikasikan varietas beras secara akurat. 
            Variasi dalam bentuk, ukuran, tekstur, dan warna biji beras sering kali sangat kecil sehingga sulit dibedakan oleh metode konvensional berbasis pengamatan manual. 
            Dengan demikian, diperlukan pendekatan berbasis teknologi kecerdasan buatan yang mampu mengenali pola-pola visual kompleks pada data citra beras dalam jumlah besar 
            sekaligus dapat mengklasifikasi secara otomatis dan memberikan hasil klasifikasi dengan akurasi yang tinggi
    """)
    st.markdown("#### - Solusi Permasalahan")
    st.write("""
            Dalam mengatasi permasalahan yang telah disebutkan di atas, solusi yang diusulkan dalam penelitian ini adalah dengan mengimplementasikan metode Deep learning berbasis Transfer Learning 
            yaitu dengan arsitektur DenseNet-201 untuk mengklasifikasikan varietas beras secara akurat. Dengan pendekatan ini, diharapkan dapat dihasilkan model klasifikasi yang efektif 
            dalam mendeteksi varietas beras secara optimal sehingga dapat mencegah pemalsuan atau pencampuran varietas beras serta mendukung stabilitas, pengelolaan dan distribusi beras yang lebih efisien
    """)
    st.markdown("#### - Pertanyaan Penelitian")
    st.write("""
            Bagaimana tingkat Accuracy, Precision, Recall, dan F1-Score dari model DenseNet-201 berbasis Transfer Learning dibandingkan dengan Non-Transfer Learning dalam mengklasifikasikan varietas beras?
    """)
    # st.divider()

    st.header("Tujuan dan Manfaat Penelitian") 
    st.markdown("#### - Tujuan Penelitian")
    st.write("""
            Mengetahui tingkat Accuracy, Precision, recall dan F1-Score dari model DenseNet-201 yang berbasis Transfer Learning dan Non-Transfer Learning dalam mengklasifikasikan varietas beras
    """)
    st.markdown("#### - Manfaat Penelitian")
    st.write("""
            Manfaat dari penelitian ini adalah memberikan kontribusi dalam pengembangan ilmu pengetahuan, khususnya di bidang teknologi pangan dan image classification, 
            dan menghasilkan sistem klasifikasi yang akurat untuk mengidentifikasi varietas beras yang dapat digunakan sebagai solusi inovatif untuk mengatasi keterbatasan metode manual 
            serta dapat digunakan sebagai bahan rujukan untuk penelitian terkait selanjutnya
    """)
    # st.divider()

    st.header("Batasan Masalah")
    st.write("""
            1.	Penelitian ini hanya mencakup klasifikasi tiga varietas beras, yaitu Ciherang, IR64, dan Mentik Susu, berdasarkan karakteristik visual biji beras. 
            2.	Dataset yang digunakan dalam penelitian ini merupakan data primer berupa data citra biji beras yang terdiri dari 6000 data dengan 3 kelas yaitu Ciherang, IR64, dan Mentik Susu dengan masing-masing kelasnya terdiri dari 2000 citra. 
            3.	Penelitian ini difokuskan pada klasifikasi varietas beras berdasarkan citra digital biji beras.
            4.	Algoritma yang digunakan dalam penelitian ini adalah pre-trained model DenseNet-201.
    """)
    # st.divider()
       
    st.header("Metodologi")
    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.image("Images/systemarch.png", caption="Arsitektur sistem")

def Dataset_Information():
    st.markdown("# Informasi Dataset")
    st.divider()
    st.write("""
             Dataset yang digunakan merupakan data primer yang diperoleh melalui pemotretan langsung menggunakan kamera ponsel  dalam jarak 11.5 cm dari object 
             dengan tingkat zoom maksimal dalam kondisi pencahayaan luar ruangan pada siang hari. Proses pengumpulan data dilakukan mulai dari tanggal 1 November 2024 
             hingga 30 November 2024. Dataset yang terkumpul berjumlah 6000 data citra biji beras yang terbagi ke dalam 3 kelas yaitu Ciherang, IR64 dan Mentik Susu 
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
    st.divider()
    st.subheader("1. Background Removing")
    st.write("""
            Penghapusan latar belakang dilakukan menggunakan library rembg yang memanfaatkan pre-trained model U2-Net. 
    """)
    st.markdown("#### Arsitektur U2-Net")
    u2net1 = Image.open("Images/u2netarch.png").resize((600, 400))
    u2net2 = Image.open("Images/RSU.png").resize((600, 400))
    st.write("""
            Cara kerjanya adalah: 
            1. **Input RGB**  
            2. **Two-Level U-Structure** (Encoder & Decoder)  
            3. **Probability Maps**  
            4. **Fusion & Thresholding**  
            5. **Output Masking (RGBA)**
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.image(u2net1, caption="Arsitektur U2-Net", use_container_width=True)
    with col2:
        st.image(u2net2, caption="Komponen Decoder-Encoder", use_container_width=True)
    # st.image("Images/bgremoved.png", caption="Background Removed")
    
    rembg1 = Image.open("Images/ciherang_rembg.png").resize((400, 400))
    rembg2 = Image.open("Images/ir64_rembg.png").resize((400, 400))
    rembg3 = Image.open("Images/mentik_rembg.png").resize((400, 400))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(rembg1, caption="Ciherang", use_container_width=True)
    with col2:
        st.image(rembg2, caption="IR64", use_container_width=True)
    with col3:
        st.image(rembg3, caption="Mentik Susu", use_container_width=True)
    st.divider()

    st.subheader("2. Grayscale Conversion")
    st.write("""
            Grayscale Conversion dilakukan dengan mengambil nilai intensitas warna dari setiap pixel dalam gambar RGB 
            dan menghitung nilai abu-abu menggunakan weighted sum dari nilai merah (R), hijau (G), dan biru (B).
    """)
    st.latex(r"""
             gray = 0.299R + 0.587G + 0.114B 
    """)
    st.write("""
            Setelah nilai intensitas abu-abu dihitung, setiap pixel dalam gambar asli (yang berisi tiga nilai R, G, dan B) digantikan oleh satu nilai gray
    """)
    # st.image("Images/grayscale.png", caption= "Grayscale Image")
    gray1 = Image.open("Images/ciherang_gray.png").resize((400, 400))
    gray2 = Image.open("Images/ir64_gray.png").resize((400, 400))
    gray3 = Image.open("Images/mentik_gray.png").resize((400, 400))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(gray1, caption="Ciherang", use_container_width=True)
    with col2:
        st.image(gray2, caption="IR64", use_container_width=True)
    with col3:
        st.image(gray3, caption="Mentik Susu", use_container_width=True)
    st.divider()

    st.subheader("3. Image Cropping")
    st.write("""
            Proses cropping dilakukan dengan metode thresholding dan kontur. Pada tahap awal, gambar diubah menjadi binary mask menggunakan dengan nilai ambang 10
            , di mana pixel  di atas 10 akan diubah menjadi 255 (putih), sedangkan nilai di bawah atau sama dengan 10 menjadi 0 (hitam) untuk memisahkan objek dari latar belakang. 
            Selanjutnya, kontur yang terdeteksi dari binary mask digunakan untuk membuat bounding box. Bounding box ini kemudian diperbesar dengan padding 1.5 yang selanjutnya 
            akan dilakukan array slicing berdasarkan koordinat yang didapat dari bounding box
    """)
    # st.image("Images/cropped.png", caption = "Cropped Image")
    crop1 = Image.open("Images/ciherang_crop.png").resize((400, 400))
    crop2 = Image.open("Images/ir64_crop.png").resize((400, 400))
    crop3 = Image.open("Images/mentik_crop.png").resize((400, 400))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(crop1, caption="Ciherang", use_container_width=True)
    with col2:
        st.image(crop2, caption="IR64", use_container_width=True)
    with col3:
        st.image(crop3, caption="Mentik Susu", use_container_width=True)
    st.divider()

    st.subheader("4. RGB Conversion")
    st.write("""
            Mengubah citra dari format grayscale kembali ke format RGB. Hal ini dilakukan dikarenakan Model DenseNet menerima input dengan 3 channel warna.
            Proses RGB Conversion dilakukan dengan mereplikasi nilai intensitas channel grayscale ke ketiga channel RGB.
    """)
    st.markdown("**Pixel sebelum RGB Conversion:**")
    st.code("""
            [[128,  64, 200,  90,  30],
             [  0, 255, 100, 150,  60],
             [ 75,  80,  95,  40,  20],
             [ 10, 220, 180,  70, 110],
             [ 55,  33,  99, 121,  88]]
    """, language='python')

    st.markdown("**Pixel setelah RGB Conversion:**")
    st.code("""
            [[[128,128,128], [ 64, 64, 64], [200,200,200], [ 90, 90, 90], [ 30, 30, 30]],
             [[  0,  0,  0], [255,255,255], [100,100,100], [150,150,150], [ 60, 60, 60]],
             [[ 75, 75, 75], [ 80, 80, 80], [ 95, 95, 95], [ 40, 40, 40], [ 20, 20, 20]],
             [[ 10, 10, 10], [220,220,220], [180,180,180], [ 70, 70, 70], [110,110,110]],
             [[ 55, 55, 55], [ 33, 33, 33], [ 99, 99, 99], [121,121,121], [ 88, 88, 88]]]
    """, language='python')
    st.divider()
    
    st.subheader("5. Image Resizing")
    st.write("""
            Image resizing dilakukan supaya data yang digunakan memiliki ukuran yang sesuai dengan dimensi input yang diperlukan oleh model DenseNet-201, 
            sehingga ukuran citra akan diubah menjadi sesuai dengan input size model DenseNet-201 yaitu 224×224
    """)
    
    resize1 = Image.open("Images/before_resize.png").resize((300, 300))
    resize2 = Image.open("Images/after_resize.png").resize((224, 224))

    col1, col2 = st.columns(2)
    with col1:
        st.image(resize1, caption = "Citra sebelum Image Resizing")
    with col2:
        st.image(resize2, caption = "Citra setelah Image Resizing")
    st.divider()
    
    st.subheader("6. Pixel Normalization")
    st.write("""
            Pixel normalization dilakukan dengan mengubah nilai pixel ke rentang tertentu yaitu 0 hingga 1, 
            dengan cara membagi nilai setiap pixel dengan nilai maksimum (255). 
            Proses ini membantu dalam meningkatkan stabilitas dan efisiensi pelatihan model, 
            karena model dapat belajar dari data tanpa bias yang disebabkan oleh skala pixel yang berbeda
    """)
    st.markdown("**Pixel sebelum Normalization:**")
    st.code("""
            [[[128,128,128], [ 64, 64, 64], [200,200,200], [ 90, 90, 90], [ 30, 30, 30]],
             [[  0,  0,  0], [255,255,255], [100,100,100], [150,150,150], [ 60, 60, 60]],
             [[ 75, 75, 75], [ 80, 80, 80], [ 95, 95, 95], [ 40, 40, 40], [ 20, 20, 20]],
             [[ 10, 10, 10], [220,220,220], [180,180,180], [ 70, 70, 70], [110,110,110]],
             [[ 55, 55, 55], [ 33, 33, 33], [ 99, 99, 99], [121,121,121], [ 88, 88, 88]]]
    """, language='python')
    
    st.markdown("**Pixel setelah Normalization:**")
    st.code("""
            [[[0.502,0.502,0.502], [0.251,0.251,0.251], [0.784,0.784,0.784], [0.353,0.353,0.353], [0.118,0.118,0.118]],
             [[0.000,0.000,0.000], [1.000,1.000,1.000], [0.392,0.392,0.392], [0.588,0.588,0.588], [0.235,0.235,0.235]],
             [[0.294,0.294,0.294], [0.314,0.314,0.314], [0.373,0.373,0.373], [0.157,0.157,0.157], [0.078,0.078,0.078]],
             [[0.039,0.039,0.039], [0.863,0.863,0.863], [0.706,0.706,0.706], [0.275,0.275,0.275], [0.431,0.431,0.431]],
             [[0.216,0.216,0.216], [0.129,0.129,0.129], [0.388,0.388,0.388], [0.475,0.475,0.475], [0.345,0.345,0.345]]]
    """, language='python')
    
def Model_Training():
    st.markdown("# Model Training")
    st.divider()
    st.write("""
        Model dilatih menggunakan arsitektur **DenseNet-201** dengan pendekatan Transfer Learning yang memanfaatkan pre-trained weight dari ImageNet.
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
            "Conv 1×1 & Conv 3×3 × 6",
            "Convolution 1×1",
            "Average Pool 2×2, Stride 2",
            "Conv 1×1 & Conv 3×3 × 12",
            "Convolution 1×1",
            "Average Pool 2×2, Stride 2",
            "Conv 1×1 & Conv 3×3 × 48",
            "Convolution 1×1",
            "Average Pool 2×2, Stride 2",
            "Conv 1×1 & Conv 3×3 × 32",
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

    st.subheader("Model History")
    st.image("Images/model_history.png", caption = "Model Training History")

def Model_Evaluation():
    st.markdown("# Model Evaluation")
    st.divider()
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
    st.table(df)

# def Prediction():
#     st.header("Prediction")
#     st.divider()
#     with st.sidebar:
#         st.sidebar.header("Select Image Source")
#         img_source = st.radio("Image Source", ["Upload image", "Sample image"])

#     sample_images = {
#         "Ciherang": [
#             r'Images/sampel ciherang_1.png',
#             r'Images/sampel ciherang_2.png',
#             r'Images/sampel ciherang_3.png'
#         ],
#         "IR64": [
#             r'Images/sampel ir64_1.png',
#             r'Images/sampel ir64_2.png',
#             r'Images/sampel ir64_3.png'
#         ],
#         "Mentik": [
#             r'Images/sampel mentik_1.png',
#             r'Images/sampel mentik_2.png',
#             r'Images/sampel mentik_3.png'
#         ]
#     }

#     if img_source == "Sample image":
#         st.sidebar.header("Select Class")
#         selected_class = st.sidebar.selectbox("Rice Variety", list(sample_images.keys()))
#         st.markdown(f"#### {selected_class} Samples")
#         columns = st.columns(3)
#         selected_image = None
#         for i, image_path in enumerate(sample_images[selected_class]):
#             with columns[i % 3]:
#                 image = Image.open(image_path)
#                 st.image(image, caption=f"Sample {i + 1}", use_container_width=True)
#                 if st.button(f"Gunakan Sample {i + 1}", key=image_path):
#                     selected_image = image_path

#         if selected_image:
#             image = Image.open(selected_image).convert('RGB')
#             st.image(image, caption=selected_image, use_container_width=True)
#             predictions = import_and_predict(image, model)
#             confidence = np.max(predictions) * 100
#             pred_class = class_names[np.argmax(predictions)]
#             st.header("🔎HASIL")
#             st.warning(f"Varietas: {pred_class.upper()}")
#             st.info(f"Confidence: {confidence:.2f}%")
#         else:
#             st.info("Pilih salah satu sample untuk prediksi")

#     else:
#         file = st.file_uploader("Upload gambar...", type=["jpg", "png", "jpeg"])
#         if file is None:
#             st.info("Silakan upload gambar beras")
#         else:
#             try:
#                 file_bytes = file.read()
#                 image_buffer = BytesIO(file_bytes)
#                 image = Image.open(image_buffer).convert('RGB')
#                 st.image(image, caption="Gambar yang diunggah", use_container_width=True)

#                 rembg_buffer = BytesIO(file_bytes)
#                 output_bytes = remove(rembg_buffer.read())
#                 img_no_bg = Image.open(BytesIO(output_bytes)).convert("RGB")
#                 img_np = np.array(img_no_bg)

#                 gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
#                 _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#                 num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

#                 object_count = 0
#                 for i in range(1, num_labels):
#                     area = stats[i, cv2.CC_STAT_AREA]
#                     if area >= 300:
#                         object_count += 1

#                 if object_count <= 1:
#                     output_bytes = remove(file_bytes)
#                     img_no_bg = Image.open(BytesIO(output_bytes)).convert("RGB")
#                     img_np = np.array(img_no_bg)
#                     gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
#                     _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
#                     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                     if contours:
#                         x, y, w, h = cv2.boundingRect(contours[0])
#                         side = int(max(w, h) * 1.5)
#                         cx = x + w // 2
#                         cy = y + h // 2
#                         x1 = max(0, cx - side // 2)
#                         y1 = max(0, cy - side // 2)
#                         x2 = min(img_np.shape[1], x1 + side)
#                         y2 = min(img_np.shape[0], y1 + side)
#                         crop_gray = gray[y1:y2, x1:x2]
#                         crop_rgb = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2RGB)

#                         resized = cv2.resize(crop_rgb, (224, 224))
#                         x_input = tf.expand_dims(resized / 255.0, axis=0)
        
#                         pred = model.predict(x_input, verbose=0)
#                         score = tf.nn.softmax(pred[0])
#                         label = class_names[np.argmax(score)]
#                         confidence = np.max(score) * 100
        
#                         st.header("🔎 HASIL")
#                         st.warning(f"Varietas: {label.upper()}")
#                         st.info(f"Confidence: {confidence:.2f}%")
#                     else:
#                         st.error("Objek tidak terdeteksi setelah proses cropping.")

#                 else:
#                     st.info("HASIL PREDIKSI")
#                     draw_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#                     variety_counter = Counter()

#                     for i in range(1, num_labels):
#                         x, y, w, h, area = stats[i]
#                         cx, cy = centroids[i]
#                         if area < 300:
#                             continue

#                         side = int(max(w, h) * 1.5)
#                         cx_int, cy_int = int(cx), int(cy)
#                         x1 = max(0, cx_int - side // 2)
#                         y1 = max(0, cy_int - side // 2)
#                         side = min(side, min(img_np.shape[1] - x1, img_np.shape[0] - y1))

#                         crop = img_np[y1:y1 + side, x1:x1 + side]
#                         resized = cv2.resize(crop, (224, 224))
#                         x_input = tf.expand_dims(resized / 255.0, axis=0)

#                         pred = model.predict(x_input, verbose=0)
#                         score = tf.nn.softmax(pred[0])
#                         label = class_names[np.argmax(score)]
#                         color = label_colors.get(label, (0, 255, 255))

#                         cv2.rectangle(draw_img, (x1, y1), (x1 + side, y1 + side), color, 2)
#                         cv2.putText(draw_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
#                                     fontScale=1.2, color=color, thickness=2)
#                         variety_counter[label] += 1

#                     st.image(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB), caption="Hasil Prediksi", use_container_width=True)
#                     st.header("🔎 RINGKASAN")
#                     st.markdown(f"Jumlah beras teridentifikasi: {sum(variety_counter.values())}")
#                     for variety, total in variety_counter.items():
#                         st.markdown(f"{variety.upper()}: {total} biji")

#             except Exception as e:
#                 st.error("Gagal memproses gambar.")
#                 st.error(str(e))

# def Prediction():
#     st.subheader("Prediction")

#     uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         img_array = np.array(image)

#         # Ubah ke grayscale
#         gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

#         # Threshold untuk binerisasi
#         _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
#         contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         # Hitung kontur dengan filter noise
#         min_contour_area = 500  # filter kontur kecil (noise)
#         valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

#         # Ambil kontur terbesar untuk analisis area
#         if valid_contours:
#             max_contour = max(valid_contours, key=cv2.contourArea)
#             max_area = cv2.contourArea(max_contour)
#         else:
#             st.warning("Tidak ada objek valid terdeteksi.")
#             return

#         img_area = gray.shape[0] * gray.shape[1]
#         area_ratio = max_area / img_area

#         st.write(f"Area Ratio: {area_ratio:.3f} | Jumlah Kontur Valid: {len(valid_contours)}")

#         if area_ratio > 0.20 and len(valid_contours) == 1:
#             st.write("🔍 Terdeteksi sebagai **Single Grain**")

#             x, y, w, h = cv2.boundingRect(max_contour)
#             cropped = gray[y:y+h, x:x+w]

#             # Resize dan konversi ke RGB (setelah cropping)
#             resized = cv2.resize(cropped, (224, 224))
#             rgb_img = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
#             rgb_img = rgb_img / 255.0
#             rgb_img = np.expand_dims(rgb_img, axis=0)

#             prediction = model.predict(rgb_img)
#             class_idx = np.argmax(prediction)
#             class_label = class_names[class_idx]
#             confidence = prediction[0][class_idx]

#             st.image(cropped, caption="Cropped Grain (Grayscale)", use_column_width=True)
#             st.success(f"Predicted Class: {class_label} ({confidence:.2%})")

#         else:
#             st.write("🔍 Terdeteksi sebagai **Multiple Grain**")

#             img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

#             for cnt in valid_contours:
#                 x, y, w, h = cv2.boundingRect(cnt)
#                 roi = gray[y:y+h, x:x+w]

#                 resized = cv2.resize(roi, (224, 224))
#                 rgb_roi = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
#                 rgb_roi = rgb_roi / 255.0
#                 rgb_roi = np.expand_dims(rgb_roi, axis=0)

#                 prediction = model.predict(rgb_roi)
#                 class_idx = np.argmax(prediction)
#                 class_label = class_names[class_idx]
#                 confidence = prediction[0][class_idx]

#                 # Gambar kotak dan label
#                 cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
#                 cv2.putText(img_rgb, f"{class_label} ({confidence:.2%})", (x, y-10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

#             st.image(img_rgb, caption="Deteksi Multiple Grain", use_column_width=True)

# def Prediction():
#     st.subheader("Prediction")

#     uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file).convert("RGBA")
#         img_np = np.array(image)

#         # Background removal
#         removed = remove(img_np)
#         removed_rgb = cv2.cvtColor(removed, cv2.COLOR_RGBA2RGB)

#         # Grayscale
#         gray = cv2.cvtColor(removed_rgb, cv2.COLOR_RGB2GRAY)

#         # Threshold
#         _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

#         # Connected Components
#         num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

#         min_area = 500
#         valid_components = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > min_area]

#         # Single Grain
#         if len(valid_components) == 1:
#             st.info("🔍 Terdeteksi sebagai **Single Grain**")
#             i = valid_components[0]
#             x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
#                          stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

#             # Buat crop kotak (square box)
#             size = max(w, h)
#             center_x, center_y = x + w // 2, y + h // 2
#             half = size // 2
#             x1 = max(center_x - half, 0)
#             y1 = max(center_y - half, 0)
#             x2 = min(center_x + half, gray.shape[1])
#             y2 = min(center_y + half, gray.shape[0])

#             crop = gray[y1:y2, x1:x2]
#             resized = cv2.resize(crop, (224, 224))
#             rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
#             normalized = rgb / 255.0
#             input_tensor = np.expand_dims(normalized, axis=0)

#             pred = model.predict(input_tensor)
#             class_idx = np.argmax(pred)
#             confidence = pred[0][class_idx]
#             label = class_names[class_idx]

#             st.image(crop, caption="Cropped Single Grain (Grayscale)", use_column_width=True)
#             st.success(f"Predicted Class: {label} ({confidence:.2%})")

#         elif len(valid_components) > 1:
#             st.info("🔍 Terdeteksi sebagai **Multiple Grain**")

#             img_copy = removed_rgb.copy()

#             for i in valid_components:
#                 x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
#                              stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

#                 # Buat crop kotak (square box)
#                 size = max(w, h)
#                 center_x, center_y = x + w // 2, y + h // 2
#                 half = size // 2
#                 x1 = max(center_x - half, 0)
#                 y1 = max(center_y - half, 0)
#                 x2 = min(center_x + half, gray.shape[1])
#                 y2 = min(center_y + half, gray.shape[0])

#                 crop = gray[y1:y2, x1:x2]
#                 resized = cv2.resize(crop, (224, 224))
#                 rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
#                 normalized = rgb / 255.0
#                 input_tensor = np.expand_dims(normalized, axis=0)

#                 pred = model.predict(input_tensor)
#                 class_idx = np.argmax(pred)
#                 confidence = pred[0][class_idx]
#                 label = class_names[class_idx]

#                 # Gambar kotak dan label
#                 cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(img_copy, f"{label} ({confidence:.2%})", (x1, y1 - 5),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

#             st.image(img_copy, caption="Multiple Grain Detection with Bounding Boxes", use_column_width=True)

#         else:
#             st.warning("❌ Tidak ada objek valid terdeteksi.")


# def Prediction():
#     st.subheader("Prediction")

#     uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file).convert("RGBA")
#         img_np = np.array(image)

#         # Step 1: Background removal
#         removed = remove(img_np)
#         removed_rgb = cv2.cvtColor(removed, cv2.COLOR_RGBA2RGB)

#         # Step 2: Convert to grayscale
#         gray = cv2.cvtColor(removed_rgb, cv2.COLOR_RGB2GRAY)

#         # Step 3: Thresholding
#         _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

#         # Step 4: Connected Component Analysis
#         num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

#         # Filter komponen kecil (area < 500)
#         min_area = 500
#         valid_components = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > min_area]

#         # Deteksi Single vs Multiple
#         if len(valid_components) == 1:
#             st.info("🔍 Terdeteksi sebagai **Single Grain**")

#             i = valid_components[0]
#             x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
#                          stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            
#             crop = gray[y:y+h, x:x+w]
#             resized = cv2.resize(crop, (224, 224))
#             rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
#             normalized = rgb / 255.0
#             input_tensor = np.expand_dims(normalized, axis=0)

#             pred = model.predict(input_tensor)
#             class_idx = np.argmax(pred)
#             confidence = pred[0][class_idx]
#             label = class_names[class_idx]

#             st.image(crop, caption="Cropped Single Grain (Grayscale)", use_column_width=True)
#             st.success(f"Predicted Class: {label} ({confidence:.2%})")

#         elif len(valid_components) > 1:
#             st.info("🔍 Terdeteksi sebagai **Multiple Grain**")

#             img_copy = removed_rgb.copy()

#             for i in valid_components:
#                 x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
#                              stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                
#                 crop = gray[y:y+h, x:x+w]
#                 resized = cv2.resize(crop, (224, 224))
#                 rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
#                 normalized = rgb / 255.0
#                 input_tensor = np.expand_dims(normalized, axis=0)

#                 pred = model.predict(input_tensor)
#                 class_idx = np.argmax(pred)
#                 confidence = pred[0][class_idx]
#                 label = class_names[class_idx]

#                 # Draw rectangle & label
#                 cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 cv2.putText(img_copy, f"{label} ({confidence:.2%})", (x, y - 5),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

#             st.image(img_copy, caption="Detected Multiple Grains", use_column_width=True)

#         else:
#             st.warning("❌ Tidak ada objek valid terdeteksi.")

# def Prediction():
#     st.header("Prediction")
#     st.divider()
#     with st.sidebar:
#         st.sidebar.header("Select Image Source")
#         img_source = st.radio("Image Source", ["Upload image", "Sample image"])

#     sample_images = {
#         "Ciherang": [
#             r'Images/sampel ciherang_1.png',
#             r'Images/sampel ciherang_2.png',
#             r'Images/sampel ciherang_3.png'
#         ],
#         "IR64": [
#             r'Images/sampel ir64_1.png',
#             r'Images/sampel ir64_2.png',
#             r'Images/sampel ir64_3.png'
#         ],
#         "Mentik": [
#             r'Images/sampel mentik_1.png',
#             r'Images/sampel mentik_2.png',
#             r'Images/sampel mentik_3.png'
#         ]
#     }

#     if img_source == "Sample image":
#         st.sidebar.header("Select Class")
#         selected_class = st.sidebar.selectbox("Rice Variety", list(sample_images.keys()))
#         st.markdown(f"#### {selected_class} Samples")
#         columns = st.columns(3)
#         selected_image = None
#         for i, image_path in enumerate(sample_images[selected_class]):
#             with columns[i % 3]:
#                 image = Image.open(image_path)
#                 st.image(image, caption=f"Sample {i + 1}", use_container_width=True)
#                 if st.button(f"Gunakan Sample {i + 1}", key=image_path):
#                     selected_image = image_path

#         if selected_image:
#             image = Image.open(selected_image).convert('RGB')
#             st.image(image, caption=selected_image, use_container_width=True)
#             predictions = import_and_predict(image, model)
#             confidence = np.max(predictions) * 100
#             pred_class = class_names[np.argmax(predictions)]
#             st.header("🔎HASIL")
#             st.warning(f"Varietas: {pred_class.upper()}")
#             st.info(f"Confidence: {confidence:.2f}%")
#         else:
#             st.info("Pilih salah satu sample untuk prediksi")

#     else:
#         file = st.file_uploader("Upload gambar...", type=["jpg", "png", "jpeg"])
#         if file is None:
#             st.info("Silakan upload gambar beras")
#         else:
#             try:
#                 file_bytes = file.read()
#                 image = Image.open(BytesIO(file_bytes)).convert('RGB')
#                 st.image(image, caption="Gambar yang diunggah", use_container_width=True)

#                 # Remove background
#                 output_bytes = remove(file_bytes)
#                 img_no_bg = Image.open(BytesIO(output_bytes)).convert("RGB")
#                 img_np = np.array(img_no_bg)

#                 # Deteksi objek
#                 gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
#                 _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#                 num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

#                 # Hitung objek besar (area minimal 1000)
#                 large_components = sum(1 for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= 1000)

#                 if large_components <= 1:
#                     # === Objek Tunggal ===
#                     gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
#                     _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
#                     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#                     if contours:
#                         x, y, w, h = cv2.boundingRect(contours[0])
#                         side = int(max(w, h) * 1.5)
#                         cx = x + w // 2
#                         cy = y + h // 2
#                         x1 = max(0, cx - side // 2)
#                         y1 = max(0, cy - side // 2)
#                         x2 = min(img_np.shape[1], x1 + side)
#                         y2 = min(img_np.shape[0], y1 + side)

#                         crop_gray = gray[y1:y2, x1:x2]
#                         crop_rgb = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2RGB)
#                         resized = cv2.resize(crop_rgb, (224, 224))
#                         x_input = tf.expand_dims(resized / 255.0, axis=0)

#                         pred = model.predict(x_input, verbose=0)
#                         score = tf.nn.softmax(pred[0])
#                         label = class_names[np.argmax(score)]
#                         confidence = np.max(score) * 100

#                         st.header("🔎 HASIL")
#                         st.warning(f"Varietas: {label.upper()}")
#                         st.info(f"Confidence: {confidence:.2f}%")
#                     else:
#                         st.error("Objek tidak terdeteksi setelah proses cropping.")

#                 else:
#                     # === Multiple Grain ===
#                     st.info("HASIL PREDIKSI")
#                     draw_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#                     variety_counter = Counter()

#                     for i in range(1, num_labels):
#                         x, y, w, h, area = stats[i]
#                         cx, cy = centroids[i]
#                         if area < 1000:
#                             continue

#                         side = int(max(w, h) * 1.5)
#                         cx_int, cy_int = int(cx), int(cy)
#                         x1 = max(0, cx_int - side // 2)
#                         y1 = max(0, cy_int - side // 2)
#                         side = min(side, min(img_np.shape[1] - x1, img_np.shape[0] - y1))

#                         crop = img_np[y1:y1 + side, x1:x1 + side]
#                         resized = cv2.resize(crop, (224, 224))
#                         x_input = tf.expand_dims(resized / 255.0, axis=0)

#                         pred = model.predict(x_input, verbose=0)
#                         score = tf.nn.softmax(pred[0])
#                         label = class_names[np.argmax(score)]
#                         color = label_colors.get(label, (0, 255, 255))

#                         cv2.rectangle(draw_img, (x1, y1), (x1 + side, y1 + side), color, 2)
#                         cv2.putText(draw_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
#                                     fontScale=1.2, color=color, thickness=2)
#                         variety_counter[label] += 1

#                     st.image(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB), caption="Hasil Prediksi", use_container_width=True)
#                     st.header("🔎 RINGKASAN")
#                     st.markdown(f"Jumlah beras teridentifikasi: {sum(variety_counter.values())}")
#                     for variety, total in variety_counter.items():
#                         st.markdown(f"{variety.upper()}: {total} biji")

#             except Exception as e:
#                 st.error("Gagal memproses gambar.")
#                 st.error(str(e))

# def Prediction():
#     st.header("Prediction")
#     st.divider()
#     with st.sidebar:
#         st.sidebar.header("Select Image Source")
#         img_source = st.radio("Image Source", ["Upload image", "Sample image"])

#     sample_images = {
#         "Ciherang": [
#             r'Images/sampel ciherang_1.png',
#             r'Images/sampel ciherang_2.png',
#             r'Images/sampel ciherang_3.png'
#         ],
#         "IR64": [
#             r'Images/sampel ir64_1.png',
#             r'Images/sampel ir64_2.png',
#             r'Images/sampel ir64_3.png'
#         ],
#         "Mentik": [
#             r'Images/sampel mentik_1.png',
#             r'Images/sampel mentik_2.png',
#             r'Images/sampel mentik_3.png'
#         ]
#     }

#     if img_source == "Sample image":
#         st.sidebar.header("Select Class")
#         selected_class = st.sidebar.selectbox("Rice Variety", list(sample_images.keys()))
#         st.markdown(f"#### {selected_class} Samples")
#         columns = st.columns(3)
#         selected_image = None
#         for i, image_path in enumerate(sample_images[selected_class]):
#             with columns[i % 3]:
#                 image = Image.open(image_path)
#                 st.image(image, caption=f"Sample {i + 1}", use_container_width=True)
#                 if st.button(f"Gunakan Sample {i + 1}", key=image_path):
#                     selected_image = image_path

#         if selected_image:
#             image = Image.open(selected_image).convert('RGB')
#             st.image(image, caption=selected_image, use_container_width=True)
#             predictions = import_and_predict(image, model)
#             confidence = np.max(predictions) * 100
#             pred_class = class_names[np.argmax(predictions)]
#             st.header("🔎HASIL")
#             st.warning(f"Varietas: {pred_class.upper()}")
#             st.info(f"Confidence: {confidence:.2f}%")
#         else:
#             st.info("Pilih salah satu sample untuk prediksi")

#     else:
#         file = st.file_uploader("Upload gambar...", type=["jpg", "png", "jpeg"])
#         if file is None:
#             st.info("Silakan upload gambar beras")
#         else:
#             try:
#                 file_bytes = file.read()
#                 image = Image.open(BytesIO(file_bytes)).convert('RGB')
#                 st.image(image, caption="Gambar yang diunggah", use_container_width=True)

#                 # Preprocessing untuk deteksi objek
#                 output_bytes = remove(file_bytes)
#                 img_no_bg = Image.open(BytesIO(output_bytes)).convert("RGB")
#                 img_np = np.array(img_no_bg)
#                 gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
#                 _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#                 num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

#                 # Hitung objek dengan area minimal 300
#                 object_count = 0
#                 for i in range(1, num_labels):
#                     area = stats[i, cv2.CC_STAT_AREA]
#                     if area >= 300:
#                         object_count += 1

#                 if object_count <= 1:
#                     # === Objek Tunggal ===
#                     gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
#                     _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
#                     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#                     if contours:
#                         x, y, w, h = cv2.boundingRect(contours[0])
#                         side = int(max(w, h) * 1.5)
#                         cx = x + w // 2
#                         cy = y + h // 2
#                         x1 = max(0, cx - side // 2)
#                         y1 = max(0, cy - side // 2)
#                         x2 = min(img_np.shape[1], x1 + side)
#                         y2 = min(img_np.shape[0], y1 + side)
#                         crop_gray = gray[y1:y2, x1:x2]
#                         crop_rgb = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2RGB)

#                         resized = cv2.resize(crop_rgb, (224, 224))
#                         x_input = tf.expand_dims(resized / 255.0, axis=0)

#                         pred = model.predict(x_input, verbose=0)
#                         score = tf.nn.softmax(pred[0])
#                         label = class_names[np.argmax(score)]
#                         confidence = np.max(score) * 100

#                         st.header("🔎 HASIL")
#                         st.warning(f"Varietas: {label.upper()}")
#                         st.info(f"Confidence: {confidence:.2f}%")
#                     else:
#                         st.error("Objek tidak terdeteksi.")
#                 else:
#                     # === Multiple Grain ===
#                     st.info("HASIL PREDIKSI")
#                     draw_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#                     variety_counter = Counter()

#                     for i in range(1, num_labels):
#                         area = stats[i, cv2.CC_STAT_AREA]
#                         if area < 300:
#                             continue
#                         x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
#                                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

#                         side = int(max(w, h) * 1.5)
#                         cx = x + w // 2
#                         cy = y + h // 2
#                         x1 = max(0, cx - side // 2)
#                         y1 = max(0, cy - side // 2)
#                         x2 = min(img_np.shape[1], x1 + side)
#                         y2 = min(img_np.shape[0], y1 + side)

#                         crop = img_np[y1:y2, x1:x2]
#                         resized = cv2.resize(crop, (224, 224))
#                         x_input = tf.expand_dims(resized / 255.0, axis=0)

#                         pred = model.predict(x_input, verbose=0)
#                         score = tf.nn.softmax(pred[0])
#                         label = class_names[np.argmax(score)]
#                         color = label_colors.get(label, (0, 255, 255))

#                         # Visualisasi kotak prediksi
#                         cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, 2)
#                         cv2.putText(draw_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
#                         variety_counter[label] += 1

#                     st.image(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB), caption="Hasil Prediksi", use_container_width=True)
#                     st.header("🔎 RINGKASAN")
#                     st.markdown(f"Jumlah beras teridentifikasi: {sum(variety_counter.values())}")
#                     for variety, total in variety_counter.items():
#                         st.markdown(f"{variety.upper()}: {total} biji")

#             except Exception as e:
#                 st.error("Gagal memproses gambar.")
#                 st.error(str(e))

def Prediction():
    st.header("Prediction")
    st.divider()
    with st.sidebar:
        st.sidebar.header("Select Image Source")
        img_source = st.radio("Image Source", ["Upload image", "Sample image"])

    sample_images = {
        "Ciherang": [
            r'Images/sampel ciherang_1.png',
            r'Images/sampel ciherang_2.png',
            r'Images/sampel ciherang_3.png'
        ],
        "IR64": [
            r'Images/sampel ir64_1.png',
            r'Images/sampel ir64_2.png',
            r'Images/sampel ir64_3.png'
        ],
        "Mentik": [
            r'Images/sampel mentik_1.png',
            r'Images/sampel mentik_2.png',
            r'Images/sampel mentik_3.png'
        ]
    }

    if img_source == "Sample image":
        st.sidebar.header("Select Class")
        selected_class = st.sidebar.selectbox("Rice Variety", list(sample_images.keys()))
        st.markdown(f"#### {selected_class} Samples")
        columns = st.columns(3)
        selected_image = None
        for i, image_path in enumerate(sample_images[selected_class]):
            with columns[i % 3]:
                image = Image.open(image_path)
                st.image(image, caption=f"Sample {i + 1}", use_container_width=True)
                if st.button(f"Gunakan Sample {i + 1}", key=image_path):
                    selected_image = image_path

        if selected_image:
            image = Image.open(selected_image).convert('RGB')
            st.image(image, caption=selected_image, use_container_width=True)
            predictions = import_and_predict(image, model)
            confidence = np.max(predictions) * 100
            pred_class = class_names[np.argmax(predictions)]
            st.header("🔎HASIL")
            st.warning(f"Varietas: {pred_class.upper()}")
            st.info(f"Confidence: {confidence:.2f}%")
        else:
            st.info("Pilih salah satu sample untuk prediksi")

    else:
        file = st.file_uploader("Upload gambar...", type=["jpg", "png", "jpeg"])
        if file is None:
            st.info("Silakan upload gambar beras")
        else:
            try:
                file_bytes = file.read()
                image = Image.open(BytesIO(file_bytes)).convert('RGB')
                st.image(image, caption="Gambar yang diunggah", use_container_width=True)

                # === Preprocessing: remove background
                output_bytes = remove(file_bytes)
                img_no_bg = Image.open(BytesIO(output_bytes)).convert("RGB")
                img_np = np.array(img_no_bg)
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

                # === Hitung objek dengan area minimal 300
                object_count = 0
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] >= 300:
                        object_count += 1

                if object_count <= 1:
                    # === SINGLE GRAIN ===
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        x, y, w, h = cv2.boundingRect(contours[0])
                        side = int(max(w, h) * 1.5)
                        cx = x + w // 2
                        cy = y + h // 2
                        x1 = max(0, cx - side // 2)
                        y1 = max(0, cy - side // 2)
                        x2 = min(img_np.shape[1], x1 + side)
                        y2 = min(img_np.shape[0], y1 + side)

                        crop_gray = gray[y1:y2, x1:x2]
                        crop_rgb = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2RGB)
                        resized = cv2.resize(crop_rgb, (224, 224))
                        x_input = tf.expand_dims(resized / 255.0, axis=0)

                        pred = model.predict(x_input, verbose=0)
                        score = tf.nn.softmax(pred[0])
                        label = class_names[np.argmax(score)]
                        confidence = np.max(score) * 100

                        st.header("🔎 HASIL")
                        st.warning(f"Varietas: {label.upper()}")
                        st.info(f"Confidence: {confidence:.2f}%")
                    else:
                        st.error("Objek tidak terdeteksi.")
                else:
                    # === MULTIPLE GRAIN ===
                    st.info("HASIL PREDIKSI")
                    draw_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    variety_counter = Counter()

                    for i in range(1, num_labels):
                        area = stats[i, cv2.CC_STAT_AREA]
                        if area < 300:
                            continue

                        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                                     stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

                        # === SQUARE BOUNDING BOX seperti permintaan kamu
                        side = int(max(w, h) * 1.5)
                        cx = x + w // 2
                        cy = y + h // 2
                        x1 = max(0, cx - side // 2)
                        y1 = max(0, cy - side // 2)
                        x2 = min(img_np.shape[1], x1 + side)
                        y2 = min(img_np.shape[0], y1 + side)

                        crop = img_np[y1:y2, x1:x2]
                        resized = cv2.resize(crop, (224, 224))
                        x_input = tf.expand_dims(resized / 255.0, axis=0)

                        pred = model.predict(x_input, verbose=0)
                        score = tf.nn.softmax(pred[0])
                        label = class_names[np.argmax(score)]
                        color = label_colors.get(label, (0, 255, 255))

                        # === Visualisasi bounding box square
                        cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(draw_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        variety_counter[label] += 1

                    st.image(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB), caption="Hasil Deteksi", use_container_width=True)
                    st.header("🔎 RINGKASAN")
                    st.markdown(f"Jumlah beras teridentifikasi: {sum(variety_counter.values())}")
                    for variety, total in variety_counter.items():
                        st.markdown(f"{variety.upper()}: {total} biji")

            except Exception as e:
                st.error("Gagal memproses gambar.")
                st.error(str(e))


def About_us():
    st.header("Klasifikasi Varietas Beras Menggunakan Transfer Learning dengan Arsitektur DenseNet-201")
    st.markdown("#### Peneliti:")
    st.markdown("""
                1. Moch. Miftachur Rifqi Al Husain
                2. Kurniawan Eka Permana, S.Kom., M.Sc.
                3. Andharini Dwi Cahyani, S.Kom., M.Kom., Ph.D.
    """)
    st.markdown("#### Institusi: Universitas Trunojoyo Madura")
    st.markdown("#### Tahun Penelitian: 2025")
    
# ====== Menu Sidebar (Option Menu) ======
with st.sidebar:
    selected = option_menu(
        menu_title="Option Menu",
        options=[
            "Introduction", 
            "Dataset Information", 
            "Preprocessing", 
            "Model Training", 
            "Model Evaluation", 
            "Prediction",
            "About Us"
        ],
        icons=["info-circle", "bar-chart", "tools", "cpu", "clipboard-check", "search", "balloon-heart"], menu_icon="list", default_index=0,
    )

# ====== Pemanggilan Fungsi Halaman Berdasarkan Menu ======

page_names_to_funcs = {
    "Introduction": Introduction,
    "Dataset Information": Dataset_Information,
    "Preprocessing": Preprocessing,
    "Model Training": Model_Training,
    "Model Evaluation": Model_Evaluation,
    "Prediction": Prediction,
    "About Us": About_us
}

page_names_to_funcs[selected]()
