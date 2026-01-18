# Emotion-Based Human–Computer Interaction (HCI)

## Overview
This project implements an end-to-end Facial Expression Recognition (FER) pipeline
to enable emotion-aware Human–Computer Interaction systems.

The system classifies 8 facial emotions using deep learning models and focuses on
robustness and interpretability.

## Models Implemented
- Baseline CNN
- CBAM-Attention CNN
- EfficientNet-B0 (Transfer Learning)

## Results
| Model | Test Accuracy |
|------|---------------|
| CNN | 47.25% |
| CBAM-CNN | 55.98% |
| EfficientNet-B0 | 66.49% |

## Dataset
- FER-2013
- CK+ (Extended Cohn–Kanade)

Datasets are not included in this repository due to license restrictions.

## Files
- project_cv_final.ipynb: full implementation
- CV_HCI_REPORT.pdf: detailed project report

## Team
Group project with primary contribution by you and one teammate.

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook project_cv_final.ipynb
