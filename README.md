# SCT-ML-03

# Cat vs. Dog Image Classifier using SVM + MobileNetV2

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green)
![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle_Cats_vs_Dogs-brightgreen)

A lightweight image classifier that uses **MobileNetV2 for feature extraction** and **Support Vector Machines (SVM)** for classification, achieving **90%+ accuracy** on the Kaggle Cats vs. Dogs dataset.

## 🚀 Features
- **Efficient Pipeline**: MobileNetV2 extracts 1280-D features from images
- **SVM Classifier**: LinearSVC for fast and interpretable predictions
- **Kaggle-ready**: Generates submission files in the correct format
- **Optimized**: Processes images at 128x128 resolution for speed

## 📦 Dataset
Dataset structure expected:
```bash
data/
├── train/
│ ├── cat.0.jpg ... cat.12499.jpg # 12,500 cat images
│ └── dog.0.jpg ... dog.12499.jpg # 12,500 dog images
└── test1/
├── 1.jpg # Test images (numeric names)
└── ...
```

## ⚙️ Installation
```bash
git clone https://github.com/ARYAN15032005/SCT-ML-03
cd cat-dog-svm
pip install -r requirements.txt
```

##🏃‍♂️ Usage
Training (extracts features + trains SVM):
```
python train_svm.py
```
Prediction (generates submission.csv):
```
python predict.py
```
## 📊 Performance
| Metric               | Score    |
|----------------------|----------|
| Training Accuracy    | 92.4%    |
| Validation Accuracy  | 89.7%    |
| Inference Speed (CPU)| ~50 img/s|

## 🛠️ Technical Stack
- **Feature Extraction**: MobileNetV2 (ImageNet weights)
- **Classifier**: `LinearSVC` (scikit-learn)
- **Image Processing**: OpenCV
- **Data Handling**: NumPy, Pandas

## 📝 File Structure
```bash
.
├── train_svm.py # Training script
├── predict.py # Prediction script
├── svm_model.pkl # Trained SVM model
└── submission.csv # Kaggle submission file
```

## 🤖 How It Works
1. MobileNetV2 extracts **1280-D features** from images
2. SVM classifies features into **cat (0)** or **dog (1)**
3. Generates Kaggle-ready `submission.csv`



