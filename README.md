# Automated Deforestation Detection Using Deep Learning: A Multi-Temporal ResNet-50 Approach on Sentinel-2 Satellite Imagery

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Sentinel-2](https://img.shields.io/badge/Sentinel--2-ESA-003366?style=for-the-badge)](https://sentinel.esa.int/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.style=for-the-badge)](https://creativecommons.org/licenses/by/4.0/)

An automated, lightweight deep learning pipeline for satellite-based land cover classification and multi-temporal deforestation monitoring. Built using transfer learning with a pre-trained **ResNet-50** architecture on the **EuroSAT RGB** benchmark dataset, this project achieves a **98.52% test accuracy** and provides an accessible, reproducible framework for regional environmental change detection.

---

## 📌 Key Highlights

- **Overall Test Accuracy:** `98.52%` across 2,700 unseen evaluation images.
- **Two-Phase Training Strategy:** Backbone feature extraction (Epochs 1–5) followed by full-model fine-tuning (Epochs 6–10).
- **Multi-Temporal Change Detection:** Grid-based spatial inference over 160 spatial patches ($64 \times 64$ pixels per patch) detecting regional forest-to-non-forest transitions.
- **Accessible & Lightweight:** Runs end-to-end on PyTorch with Google Colab (NVIDIA T4 GPU acceleration).

---

## 🏗️ Model Architecture

The core classifier adapts a pre-trained **ResNet-50** deep convolutional network. The original 1,000-class ImageNet classification head is replaced with a linear layer mapping 2,048 feature dimensions to the 10 target EuroSAT land cover categories.

![ResNet-50 Architecture](Sentinel-2%20Land%20Cover%20Classification%20Pipeline_2.png)

### Two-Phase Training Strategy:
1. **Phase 1 (Feature Extraction):** Frozen ResNet-50 backbone, $\alpha = 0.001$, Adam optimizer (Epochs 1–5).
2. **Phase 2 (Full Fine-Tuning):** Unfrozen model layers, reduced learning rate $\alpha = 0.0001$ for spatial adaptation (Epochs 6–10).

---

## 🔄 Multi-Temporal Deforestation Detection Workflow

To track environmental changes over time, multi-temporal Sentinel-2 imagery is processed through a spatial grid-based change detection pipeline.

![Deforestation Detection Workflow](Satellite%20Deforestation%20Detection%20Workflow_2.png)

1. **Grid Extraction:** Division of Sentinel-2 rasters from two timeframes ($T_1$ and $T_2$) into non-overlapping $64 \times 64$ pixel spatial patches.
2. **Batch Inference:** Independent land cover prediction across all spatial grid cells.
3. **Transition Mask Generation:** Automated flagging of grid patches transitioning strictly from **Forest** at $T_1$ to non-forest categories (e.g., *AnnualCrop*, *Pasture*, *Industrial*, *Residential*) at $T_2$.

---

## 📊 Experimental Results & Evaluation

### 1. Training Dynamics & Loss Trajectory
The model demonstrated stable convergence, dropping training loss from `0.5848` in Phase 1 to `0.0277` by the end of Phase 2 fine-tuning. Peak validation accuracy reached **`98.41%`**.

![Training Curves](training_curves-9852_3.png)

### 2. Confusion Matrix Analysis
Evaluated on the 2,700-image test set, the model showed high precision across all 10 classes with minimal confusion between spectrally distinct categories.

![Confusion Matrix](confusion_matrix-9852_3.png)

#### Per-Class Performance Summary

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **AnnualCrop** | 0.98 | 0.99 | 0.98 | 295 |
| **Forest** | **1.00** | **0.99** | **0.99** | **305** |
| **HerbVeg** | 0.97 | 0.97 | 0.97 | 299 |
| **Highway** | 0.98 | 0.98 | 0.98 | 248 |
| **Industrial** | 0.98 | 1.00 | 0.99 | 243 |
| **Pasture** | 0.97 | 0.98 | 0.97 | 194 |
| **PermCrop** | 0.98 | 0.97 | 0.98 | 264 |
| **Residential** | 1.00 | 0.99 | 1.00 | 324 |
| **River** | 0.98 | 0.99 | 0.99 | 240 |
| **SeaLake** | 1.00 | 0.99 | 0.99 | 288 |
| **Overall Accuracy** | — | — | **0.99 (98.52%)** | **2700** |

---

## 🌲 Regional Change Detection Application

When applied to a real-world multi-temporal Sentinel-2 scene divided into 160 spatial patches, the pipeline successfully isolated **2 deforested patches** while filtering out seasonal crop rotation patterns.

![Deforestation Detection Result](deforestation_detection_result-9852_2.jpg)

---

## 📁 Repository Structure

```text
├── data/                       # Dataset processing scripts
├── models/                     # ResNet-50 model definitions
├── notebooks/                  # Training & evaluation Jupyter notebooks
├── assets/                     # Architecture & result images
│   ├── Sentinel-2 Land Cover Classification Pipeline_2.png
│   ├── Satellite Deforestation Detection Workflow_2.png
│   ├── training_curves-9852_3.png
│   ├── confusion_matrix-9852_3.png
│   └── deforestation_detection_result-9852_2.jpg
├── README.md                   # Project documentation
└── requirements.txt            # Environment dependencies
```

---

## ⚙️ Quickstart Guide

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA-enabled GPU (recommended)

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/deforestation-detection-resnet50.git
cd deforestation-detection-resnet50

# Install dependencies
pip install -r requirements.txt
```

### Dataset Access
This project utilizes the public **EuroSAT RGB** dataset available on Hugging Face:
[Hugging Face EuroSAT RGB Dataset](https://huggingface.co/datasets/blanchon/EuroSAT_RGB)

---

## 📄 License

This project is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** License. See the [LICENSE](LICENSE) file for details.

---

## ✍️ Author & Correspondence

**Piyush Manna**  
*Department of Computer Science, Bal Kalyan Vidhya Mandir, Biratnagar, Nepal*  
📧 Email: `w3bpiyush@gmail.com`