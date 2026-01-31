# 🎭 Emotion-Based Human–Computer Interaction (HCI)

<div align="center">

**A comprehensive deep learning pipeline for facial expression recognition enabling emotion-aware human-computer interaction systems**

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture & Models](#-architecture--models)
- [Performance Results](#-performance-results)
- [Visualizations](#-visualizations)
- [Dataset](#-dataset)
- [Installation & Usage](#-installation--usage)
- [Project Structure](#-project-structure)
- [Contributors](#-contributors)

---

## 🎯 Overview

This project implements a state-of-the-art **Facial Expression Recognition (FER)** system that classifies **8 distinct facial emotions** using multiple deep learning architectures. The system is designed to enable emotion-aware Human–Computer Interaction (HCI) applications, supporting:

- 🤖 **Adaptive Learning Platforms** - Personalized educational experiences based on student emotions
- 💚 **Mental Wellness Monitoring** - Real-time emotion tracking for healthcare applications
- 🖥️ **Intelligent User Interfaces** - Emotion-responsive software that adapts to user states
- 📱 **Accessibility Tools** - Assistive technologies for emotion communication

The project demonstrates a comprehensive comparison of CNN-based, attention-based, and transformer-based architectures, achieving robust performance through advanced preprocessing, class imbalance handling, and transfer learning techniques.

---

## ✨ Key Features

### 🔬 **Advanced Model Architectures**
- **Baseline CNN** - Custom convolutional neural network for baseline comparison
- **CBAM-CNN** - Convolutional Block Attention Module (Channel + Spatial Attention)
- **EfficientNet-B0** - State-of-the-art efficient CNN with transfer learning
- **Vision Transformer (ViT-B/16)** - Transformer-based architecture with self-attention mechanisms

### 🛠️ **Robust Preprocessing Pipeline**
- Automatic face detection and cropping using Haar Cascades
- Contrast enhancement via CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Standardized image resizing and normalization
- Data augmentation for improved generalization

### 📊 **Comprehensive Evaluation**
- Quantitative metrics: Accuracy, Precision, Recall, F1-Score
- Class-wise performance analysis
- ROC curves and AUC scores
- Confusion matrix visualization
- Saliency maps for model interpretability

### ⚖️ **Class Imbalance Handling**
- Weighted loss functions
- Weighted random sampling
- Class-weighted evaluation metrics

---

## 🏗️ Architecture & Models

### 1. **Baseline CNN**
A custom convolutional neural network with:
- 4 convolutional blocks with increasing filters (32 → 64 → 128 → 256)
- Global Average Pooling
- Fully connected classifier

### 2. **CBAM-CNN (Convolutional Block Attention Module)**
Enhanced CNN architecture featuring:
- **Channel Attention** - Focuses on "what" features are important
- **Spatial Attention** - Focuses on "where" important features are located
- Batch normalization and dropout for regularization

### 3. **EfficientNet-B0**
Transfer learning approach using:
- Pretrained ImageNet weights
- Progressive fine-tuning (last MBConv block + classifier)
- Optimized architecture for efficiency and accuracy

### 4. **Vision Transformer (ViT-B/16)** 🆕
Transformer-based architecture featuring:
- **Self-Attention Mechanisms** - Captures long-range dependencies in facial features
- **Patch-based Processing** - Divides images into patches for transformer processing
- **Pretrained ImageNet Weights** - Leverages large-scale pretraining
- **Progressive Fine-tuning** - Last 4 transformer blocks + classifier
- **Advanced Training Techniques**:
  - Gradient clipping for stability
  - Cosine annealing with warm restarts
  - Enhanced data augmentation (rotation, color jitter, affine transforms)

---

## 📈 Performance Results

| Model | Test Accuracy | Architecture Type | Key Innovation |
|-------|---------------|-------------------|----------------|
| **Baseline CNN** | 47.25% | Custom CNN | Baseline comparison |
| **CBAM-CNN** | 55.98% | CNN + Attention | Channel & Spatial Attention |
| **EfficientNet-B0** | 66.49% | Transfer Learning | Pretrained ImageNet weights |
| **Vision Transformer (ViT-B/16)** | **TBD** 🚀 | Transformer | Self-attention mechanisms |

> **Note:** Vision Transformer results will be available after training. The model is expected to achieve competitive or improved accuracy due to its ability to capture long-range spatial dependencies in facial features.

### 📊 Performance Insights

- **EfficientNet-B0** currently achieves the best performance (66.49%), demonstrating the effectiveness of transfer learning
- **CBAM-CNN** shows significant improvement over baseline (+8.73%), highlighting the value of attention mechanisms
- **Vision Transformer** introduces a fundamentally different architecture that may capture complementary features

---

## 📊 Visualizations

### Confusion Matrix
The confusion matrix provides detailed class-wise classification performance, showing where the model excels and where it struggles across all 8 emotion categories.

![Confusion Matrix](confusion_matrix.png)

### ROC Curve
The Receiver Operating Characteristic (ROC) curve illustrates the trade-off between true positive rate and false positive rate, with micro-average AUC scores demonstrating overall model discriminative ability.

![ROC Curve](roc_curve.png)

### Model Interpretability - Saliency Maps

Saliency maps visualize the facial regions that most strongly influence the model's predictions, providing transparency and explainability crucial for real-world deployment.

#### Happy Emotion
The saliency map highlights key facial features (eyes, mouth) that contribute to happiness recognition.

![Saliency – Happy](saliency_happy.png)

#### Anger Emotion
The saliency map shows attention to facial regions (brows, mouth) important for anger detection.

![Saliency – Anger](saliency_anger.png)

---

## 📦 Dataset

The project uses two well-established facial expression datasets:

- **FER-2013** - Large-scale dataset with 35,887 grayscale images
- **CK+ (Extended Cohn–Kanade)** - High-quality dataset with 593 sequences from 123 subjects

### Emotion Classes (8 total)
1. **Angry** 😠
2. **Disgust** 🤢
3. **Fear** 😨
4. **Happy** 😊
5. **Neutral** 😐
6. **Sad** 😢
7. **Surprise** 😲
8. **Contempt** 😏

> **Note:** Datasets are not included in this repository due to license restrictions. Please download them separately and organize according to the expected directory structure.

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.7 or higher
- CUDA-capable GPU (recommended for training)
- Jupyter Notebook

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd emotion-hci
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare datasets**
   - Download FER-2013 and CK+ datasets
   - Organize them in the expected directory structure
   - Update paths in the notebook if needed

4. **Run the notebook**
```bash
jupyter notebook emotion_based_hci.ipynb
```

### Usage Workflow

1. **Data Preprocessing** (Cell 0)
   - Face detection and cropping
   - Contrast enhancement
   - Dataset preparation

2. **Model Training**
   - Train Baseline CNN (Cell 5)
   - Train CBAM-CNN (Cell 6)
   - Train EfficientNet-B0 (Cell 7)
   - Train Vision Transformer (Cell 16) 🆕

3. **Evaluation & Analysis**
   - Model comparison (Cell 18) 🆕
   - Performance metrics
   - Visualizations

---

## 📁 Project Structure

```
emotion-hci/
│
├── emotion_based_hci.ipynb    # Main implementation notebook
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── confusion_matrix.png        # Confusion matrix visualization
├── roc_curve.png              # ROC curve visualization
├── saliency_happy.png         # Saliency map for happy emotion
└── saliency_anger.png         # Saliency map for anger emotion
```

---

## 👥 Contributors

This project was developed as a group coursework project.

### Team Members

- **Achuth Reddy Bangaru**
  - Model design and implementation
  - Training and evaluation of CNN, CBAM, EfficientNet, and Vision Transformer models
  - Performance analysis and optimization
  - Model interpretability (saliency maps)

- **Praveen LS** ([@praveen-ls](https://github.com/praveen-ls))
  - Data preprocessing pipeline
  - Dataset handling and organization
  - Experimentation support

---

## 🔮 Future Enhancements

- [ ] Real-time emotion detection from webcam
- [ ] Ensemble methods combining multiple models
- [ ] Mobile deployment optimization
- [ ] Additional emotion classes
- [ ] Multi-modal emotion recognition (audio + video)
- [ ] Transfer learning from emotion-specific pretrained models

---

## 📄 License

This project is available for educational and research purposes.

---

## 🙏 Acknowledgments

- FER-2013 dataset creators
- CK+ dataset creators
- PyTorch and torchvision communities
- All open-source contributors whose work made this project possible

---

<div align="center">

**Built with ❤️ for advancing emotion-aware human-computer interaction**

⭐ Star this repo if you find it helpful!

</div>
