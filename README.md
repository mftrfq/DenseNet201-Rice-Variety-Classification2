# 🌾Rice Variety Classification Using Transfer Learning with DenseNet-201 Architecture
## 🧠 Overview

This repository presents the full implementation and dataset from my undergraduate thesis titled "Rice Variety Classification Using Transfer Learning with DenseNet-201 Architecture."
The project focuses on developing a deep learning–based image classification model to identify Indonesian rice varieties (Ciherang, IR64, and Mentik Susu) using transfer learning on the DenseNet-201 architecture.

The model was trained and evaluated on a primary, self-collected dataset of high-resolution rice grain images captured under controlled natural lighting conditions. The work also includes comprehensive benchmarking against several state-of-the-art convolutional neural networks (CNNs) — ResNet50, VGG16, MobileNetV2, and EfficientNetB3 — to assess comparative performance in accuracy, robustness, and generalization.

Although the related manuscript could not be successfully published due to an overly complex review process with repeated revision requests, this repository remains a complete and open resource for research and educational purposes in agricultural AI and image-based crop analysis.

---
🧩 Key Features

Transfer Learning with DenseNet-201 fine-tuned for rice variety classification

Comprehensive Benchmarking with ResNet50, VGG16, MobileNetV2, and EfficientNetB3

Data Augmentation using rotation, brightness variation, and flips for improved generalization

Evaluation Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix, and ROC Analysis

Open-Set Recognition Experiments to handle unknown rice varieties

---
📊 Dataset

The dataset used in this research is a primary dataset collected between November 1–30, 2024, consisting of 6,000 high-resolution (3024×3024) images of three verified Indonesian rice varieties.
All samples were photographed using an iPhone XR (12 MP) camera at a fixed distance of 11.5 cm under natural sunlight, with a blue background and proper labeling verified by local agricultural authorities.

📁 Dataset Access: https://drive.google.com/drive/folders/1fsYYLagCB7Ms1eB1w_eWa89SCX8RlEN1?usp=sharing


🚀 Live Demo
You can try the trained DenseNet-201 model directly via the interactive Streamlit web app:
🔗 https://densenet201-rice-variety-classification2.streamlit.app/

---

Language: Python

Frameworks: TensorFlow / Keras, NumPy, Pandas, Matplotlib, scikit-learn

Environment: Google Colab, Streamlit

Statistical Tests: Cochran’s Q Test, McNemar Test for model comparison
