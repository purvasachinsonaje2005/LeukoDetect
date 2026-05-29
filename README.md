# LeukoDetect: Leukemia Detection Using Deep Learning

<div align="center">

## AI-Powered Leukemia Detection System

Using DenseNet121 and Medical Image Analysis

</div>

---

# Overview

LeukoDetect is a deep learning-based leukemia detection system that classifies microscopic blood smear images into leukemia-infected and healthy categories using the DenseNet121 architecture.

The project uses transfer learning and medical image analysis techniques to assist in early leukemia detection through an interactive Streamlit dashboard.

---

# Features

✅ Single Image Leukemia Prediction
✅ Bulk Image Prediction
✅ DenseNet121 Transfer Learning Model
✅ Interactive Streamlit Dashboard
✅ Real-Time Prediction
✅ Classification Charts & Analytics
✅ Upload Multiple Images Together
✅ Downloadable Prediction Results

---

# Dataset

### C-NMC 2019 ALL Challenge Dataset

The dataset contains:

* ALL (Acute Lymphoblastic Leukemia) Images
* HEM (Healthy Cell) Images

---

# Deep Learning Architecture

## DenseNet121

DenseNet121 is a densely connected convolutional neural network where each layer receives feature maps from all previous layers.

### Advantages

* Better Feature Reuse
* Reduced Vanishing Gradient Problem
* Efficient Training
* Improved Accuracy
* Reduced Overfitting

---

# System Workflow

1. Input Blood Smear Image Acquisition
2. Image Preprocessing
3. Data Augmentation
4. Feature Extraction Using DenseNet121
5. Classification Layer
6. Model Training & Optimization
7. Performance Evaluation

---

# Model Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 91%   |
| Precision | 1.00  |
| Recall    | 0.91  |
| F1-Score  | 0.95  |

---

# Technologies Used

| Technology | Purpose               |
| ---------- | --------------------- |
| Python     | Programming Language  |
| TensorFlow | Deep Learning         |
| Keras      | Model Building        |
| Streamlit  | Dashboard             |
| NumPy      | Numerical Computation |
| Matplotlib | Data Visualization    |
| OpenCV     | Image Processing      |
| Pillow     | Image Handling        |

---

# Project Structure

```plaintext id="ppbupg"
LeukoDetect/
│
├── app.py
├── leukemia_densenet.h5
├── requirements.txt
├── README.md
│
├── pages/
│   └── bulk_prediction.py
│
├── assets/
│
├── outputs/
│
└── sample_images/
```

---

# Installation

##  Clone Repository

```bash id="fwl8h9"
git clone https://github.com/yourusername/LeukoDetect.git
```

---

## Install Requirements

```bash id="q7nt5w"
pip install -r requirements.txt
```

---

## Run Application

```bash id="rxem5o"
streamlit run app.py
```

---

# Dashboard Preview

## Main Dashboard

Add screenshot here:

<img width="1917" height="923" alt="Screenshot 2026-05-26 233643" src="https://github.com/user-attachments/assets/5ba10791-39d7-4119-b9c5-7d2621bf68d7" />


---

## Single Image Prediction

<img width="1919" height="924" alt="Screenshot 2026-05-26 234854" src="https://github.com/user-attachments/assets/ac7f335c-a610-40cf-ac97-abac63ab2ddc" />

---

## Bulk Prediction

<img width="1481" height="916" alt="Screenshot 2026-05-26 235536" src="https://github.com/user-attachments/assets/f97b9713-fd4a-4936-aa00-3bb5c44f3559" />

<img width="1482" height="759" alt="Screenshot 2026-05-26 235620" src="https://github.com/user-attachments/assets/bdc28121-f23e-4a96-9492-fc965d8a615c" />



---

## Accuracy Visualization

<img width="1486" height="836" alt="Screenshot 2026-05-26 235709" src="https://github.com/user-attachments/assets/68efb1e5-6ffa-4642-8282-75055ca3e0f3" />

---

# Output Examples

| Input Image       | Prediction        |
| ----------------- | ----------------- |
| Blood Smear Image | Leukemia Detected |
| Blood Smear Image | Normal Cell       |

---

# Visualizations Included

* Training Accuracy Curve
* Classification Metrics Graph
* Prediction Confidence Analysis
* Bulk Prediction Statistics

---

# Future Scope

* Multi-Class Leukemia Detection
* Explainable AI using Grad-CAM
* Cloud Deployment
* Real-Time Hospital Integration
* Mobile Healthcare Application

---

# Authors

### Team Members

* Purva Sonaje
* Swamini Patil
* Vaishnavi Borase
* Akash Shinde

### Guided By

Prof. Priyanka Lanjewar

---

# Research Paper

### Title

**LeukoDetect: Leukemia Detection Using Deep Learning for Medical Image Analysis**

---

# ⭐ GitHub Repository

If you like this project, give it a ⭐ on GitHub.

---

# 📧 Contact

📩 [purvasonaje@gmail.com](mailto:purvasonaje@gmail.com)
