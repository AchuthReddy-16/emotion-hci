# Emotion-Based Human–Computer Interaction (HCI)

## Overview
This project implements an end-to-end Facial Expression Recognition (FER) pipeline
to enable emotion-aware Human–Computer Interaction systems.

The system classifies **8 facial emotions** using deep learning models and focuses on
robustness, interpretability, and real-world applicability.

Emotion-aware HCI systems can support adaptive learning platforms, mental wellness
monitoring, and intelligent user interfaces.

---

## Models Implemented
- Baseline Convolutional Neural Network (CNN)
- CBAM-Attention CNN (Channel + Spatial Attention)
- EfficientNet-B0 (Transfer Learning with ImageNet weights)

---

## Key Features
- Face preprocessing with cropping and contrast enhancement
- Handling of class imbalance during training
- Comparison of baseline, attention-based, and transfer-learning models
- Quantitative evaluation using accuracy and class-wise analysis
- Designed for emotion-aware human-centered applications

---

## Results
| Model | Test Accuracy |
|------|---------------|
| CNN | 47.25% |
| CBAM-CNN | 55.98% |
| EfficientNet-B0 | **66.49%** |

EfficientNet-B0 achieved the best performance, demonstrating the effectiveness of
transfer learning for facial emotion recognition.

---

## Dataset
- FER-2013
- CK+ (Extended Cohn–Kanade)

Datasets are not included in this repository due to license restrictions.

---

## Files
- `emotion_based_hci.ipynb` — complete implementation (training, evaluation, analysis)
- `requirements.txt` — dependencies

---

## 👥 Team and Contribution
This was a group coursework project.

- **Achuth Reddy Bangaru** – Model design and implementation, training and evaluation
  of CNN, CBAM, and EfficientNet models, performance analysis
- **Praveen LS** ([@praveen-ls](https://github.com/praveen-ls)) – Data preprocessing,
  dataset handling, and experimentation support

---

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook emotion_based_hci.ipynb
