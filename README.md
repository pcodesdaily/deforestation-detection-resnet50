# Automated Deforestation Detection Using Fine-Tuned ResNet-50

A PyTorch project for satellite land-cover classification and multi-temporal deforestation analysis using **ResNet-50** and **Sentinel-2 imagery**.

The project has two parts: a land-cover classifier trained on the **EuroSAT RGB** dataset, and a grid-based workflow that compares satellite imagery from two timeframes to identify potential forest-to-non-forest transitions. The model achieved **98.52% test accuracy** on 2,700 held-out EuroSAT images. In the regional change-detection experiment, the workflow flagged **2 patches out of 160** for further consideration as deforestation events.

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Sentinel-2](https://img.shields.io/badge/Sentinel--2-ESA-003366?style=for-the-badge)](https://sentinel.esa.int/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey?style=for-the-badge)](https://creativecommons.org/licenses/by/4.0/)

---

## Contents

- [Overview](#overview)
- [Results at a Glance](#results-at-a-glance)
- [How It Works](#how-it-works)
- [Model and Training](#model-and-training)
- [Classification Results](#classification-results)
- [Regional Change Detection](#regional-change-detection)
- [Repository Structure](#repository-structure)
- [Run the Notebook](#run-the-notebook)
- [Dataset](#dataset)
- [Limitations](#limitations)
- [Research Paper](#research-paper)
- [License](#license)
- [Author](#author)

---

## Overview

Manual comparison of satellite images can take considerable time, particularly when a region needs to be checked across multiple dates. This project explores a practical way to support that work with a fine-tuned image classifier and a simple temporal comparison pipeline.

The classifier uses a pre-trained **ResNet-50** model adapted to the 10 land-cover classes in EuroSAT RGB. After training, the model is applied to satellite images from two timeframes. Each image is divided into spatial patches, each patch is classified, and the resulting land-cover maps are compared cell by cell. A patch is flagged when its predicted class changes from **Forest** in the earlier image (\(T_1\)) to an approved non-forest class in the later image (\(T_2\)).

This is a regional change-detection experiment, not a claim that the model has been validated for every landscape or satellite acquisition condition.

## Results at a Glance

| Evaluation | Result |
| --- | ---: |
| EuroSAT RGB dataset size | 27,000 images |
| Land-cover categories | 10 |
| Held-out test set | 2,700 images |
| Overall test accuracy | **98.52%** |
| Peak validation accuracy | **98.41%** |
| Training procedure | 2 phases, 10 epochs in total |
| Regional grid evaluated | 160 spatial patches |
| Patches flagged as deforested | **2** |

The test accuracy measures land-cover classification on the held-out EuroSAT test set. The regional experiment is a separate application of the classifier to multi-temporal Sentinel-2 imagery; its result should not be read as a general false-positive rate or proof of performance across other regions.

---

## How It Works

![Deforestation Detection Workflow](https://raw.githubusercontent.com/pcodesdaily/deforestation-detection-resnet50/refs/heads/main/images/Satellite%20Deforestation%20Detection%20Workflow.png)

The workflow follows four steps:

1. **Prepare the imagery.** Obtain satellite rasters for the same region at two timeframes, \(T_1\) and \(T_2\), and divide them into non-overlapping spatial patches.
2. **Classify each patch.** Run each patch through the fine-tuned ResNet-50 model to predict one of the 10 EuroSAT land-cover categories.
3. **Compare the predictions.** Compare the two classification maps cell by cell. A transition is flagged only when a patch predicted as **Forest** at \(T_1\) is predicted as an approved non-forest category at \(T_2\), such as *AnnualCrop*, *Pasture*, *Industrial*, or *Residential*.
4. **Review the change mask.** Count the flagged patches and compare the detections with independent forest-loss information where available.

The grid-based approach makes the comparison straightforward, but its output depends on the quality of the input imagery and the classifier's predictions. Seasonal changes and differences between acquisition conditions can still affect the result.

---

## Model and Training

![ResNet-50 Architecture](https://raw.githubusercontent.com/pcodesdaily/deforestation-detection-resnet50/refs/heads/main/images/Sentinel-2%20Land%20Cover%20Classification%20Pipeline.png)

The classifier uses a ResNet-50 network initialized with ImageNet weights. Its original 1,000-class classification layer is replaced with a linear layer that maps 2,048 features to the 10 EuroSAT categories.

```python
model.fc = nn.Linear(in_features=2048, out_features=10)
```

### Two-phase training

| Phase | Setup | Learning rate | Epochs |
| --- | --- | ---: | ---: |
| Phase 1: Feature extraction | Freeze the ResNet-50 backbone and train the classification head | `0.001` | 1–5 |
| Phase 2: Full-model fine-tuning | Unfreeze the network and fine-tune all layers | `0.0001` | 6–10 |

Both phases use the Adam optimizer and Cross-Entropy Loss. The learning rate is reduced for Phase 2 so the pre-trained features can adapt to satellite imagery with smaller parameter updates.

### Input preparation

- EuroSAT RGB images begin at **64 × 64 pixels** and are resized to **224 × 224 pixels** for ResNet-50.
- Images are normalized using the ImageNet channel means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`.
- Random horizontal and vertical flips (`p = 0.5`) are applied to the training subset.
- Validation and test images are not augmented.
- The dataset split is **80% training (21,600 images), 10% validation (2,700 images), and 10% testing (2,700 images)**.
- The split uses `torch.Generator().manual_seed(42)` for reproducibility, with a batch size of `32`.

### Land-cover classes

The model predicts one of these 10 EuroSAT categories:

`AnnualCrop` · `Forest` · `HerbVeg` · `Highway` · `Industrial` · `Pasture` · `PermCrop` · `Residential` · `River` · `SeaLake`

---

## Classification Results

The fine-tuned model achieved **98.52% overall test accuracy** on 2,700 held-out images. Peak validation accuracy was **98.41%**.

![Training Curves](https://raw.githubusercontent.com/pcodesdaily/deforestation-detection-resnet50/refs/heads/main/images/training_curves-9852.png)

Training loss decreased from `0.5848` at the start of Phase 1 to `0.0277` at the end of Phase 2.

![Confusion Matrix](https://raw.githubusercontent.com/pcodesdaily/deforestation-detection-resnet50/refs/heads/main/images/confusion_matrix-9852.png)

### Per-class performance

| Class | Precision | Recall | F1-score | Support |
| :--- | :---: | :---: | :---: | ---: |
| AnnualCrop | 0.98 | 0.99 | 0.98 | 295 |
| Forest | 1.00 | 0.99 | 0.99 | 305 |
| HerbVeg | 0.97 | 0.97 | 0.97 | 299 |
| Highway | 0.98 | 0.98 | 0.98 | 248 |
| Industrial | 0.98 | 1.00 | 0.99 | 243 |
| Pasture | 0.97 | 0.98 | 0.97 | 194 |
| PermCrop | 0.98 | 0.97 | 0.98 | 264 |
| Residential | 1.00 | 0.99 | 1.00 | 324 |
| River | 0.98 | 0.99 | 0.99 | 240 |
| SeaLake | 1.00 | 0.99 | 0.99 | 288 |
| **Overall accuracy** | — | — | **0.99 (98.52%)** | **2700** |

The remaining classification errors were concentrated in some visually similar vegetation categories, including Pasture, Herbaceous Vegetation, and Annual Crop. Their appearance can overlap depending on vegetation density and growing season, so overall accuracy alone does not describe every class-level difficulty.

---

## Regional Change Detection

![Deforestation Detection Result](https://raw.githubusercontent.com/pcodesdaily/deforestation-detection-resnet50/refs/heads/main/images/deforestation_detection_result-9852.png)

The regional experiment compared two Sentinel-2 timeframes using **160 spatial patches**, each based on a **64 × 64-pixel** grid. The pipeline flagged **2 patches** where the predicted class changed from Forest in the earlier raster to a non-forest category in the later raster. The workflow was designed to filter out seasonal crop-rotation patterns that could otherwise be mistaken for forest loss.

These results show how the classifier can be used in a temporal comparison workflow. They do not, by themselves, establish the system's false-positive rate or confirm that the same performance will hold across different geographic regions and environmental conditions. The research paper discusses these limitations in more detail.

---

## Repository Structure

```text
├── images/                                     # Architecture diagrams, curves, and evaluation outputs
│   ├── Satellite Deforestation Detection Workflow.png
│   ├── Sentinel-2 Land Cover Classification Pipeline.png
│   ├── confusion_matrix-9852.png
│   ├── deforestation_detection_result-9852.png
│   └── training_curves-9852.png
├── Automated Deforestation Detection.docx      # Research manuscript (Editable Word Document)
├── Automated Deforestation Detection.pdf       # Research manuscript (PDF Version)
├── code.ipynb                                  # Complete PyTorch training, evaluation, & inference notebook
└── README.md                                   # Project documentation
```

---

## Run the Notebook

The complete training, evaluation, and inference workflow is provided in `code.ipynb`.

### Option 1: Google Colab

1. Open `code.ipynb` in Google Colab.
2. Select a runtime with GPU access if available.
3. Run the notebook cells in order.

The experiments described in the research paper were run in Google Colab with an **NVIDIA T4 GPU**.

### Option 2: Jupyter Notebook

Open `code.ipynb` in a local Jupyter Notebook environment with the dependencies installed.

### Dependencies

The notebook imports the following third-party packages:

- `torch` — model training, tensor operations, and GPU support
- `torchvision` — the pre-trained ResNet-50 model and image transforms
- `datasets` — loading the EuroSAT RGB dataset from Hugging Face
- `numpy` — array operations and the change-detection mask
- `matplotlib` — plots and saved result figures
- `seaborn` — confusion-matrix visualization
- `scikit-learn` — accuracy, classification report, and confusion matrix metrics
- `Pillow` — opening, converting, and cropping satellite images

Install them with:

```bash
pip install torch torchvision datasets numpy matplotlib seaborn scikit-learn Pillow
```

The notebook also imports `os` from Python's standard library, so no separate installation is needed for it. `requests` is not imported by the provided notebook.

---

## Dataset

The classifier is trained and evaluated on the public **EuroSAT RGB** dataset, which contains **27,000 labelled RGB image patches** across 10 land-cover classes.

- **Dataset:** [EuroSAT RGB on Hugging Face](https://huggingface.co/datasets/blanchon/EuroSAT_RGB)
- **Image dimensions in the dataset:** 64 × 64 pixels
- **Input dimensions for ResNet-50:** 224 × 224 pixels after resizing
- **Classes:** 10

The regional change-detection experiment uses multi-temporal Sentinel-2 imagery. The repository's research manuscript describes the methodology, results, and limitations of that experiment.

---

## Limitations

A few points matter when interpreting the results:

- **RGB input only:** The classifier was trained on EuroSAT RGB and does not use Sentinel-2's Near-Infrared (NIR) or Short-Wave Infrared (SWIR) bands.
- **Patch size:** Each 64 × 64-pixel patch represents approximately 640 m × 640 m at a ground sampling distance of 10 metres per pixel. Small or selective logging activity may be missed.
- **Geographic transfer:** EuroSAT mainly represents European landscapes. Performance may change when the model is applied to tropical regions or other landscapes that differ from its training data.
- **Image and seasonal variation:** Clouds, cloud shadows, haze, harvesting, and seasonal vegetation changes can affect comparisons between dates.
- **Scope of evaluation:** The 98.52% result is the held-out EuroSAT classification accuracy. The regional experiment involved 160 patches and flagged 2 potential deforestation patches; it does not establish general performance across regions.

For these reasons, flagged patches should be treated as detections for review, rather than as a substitute for independent validation.

---

## Research Paper

The full manuscript is included in the repository in both editable and PDF formats:

- `Automated Deforestation Detection.docx`
- `Automated Deforestation Detection.pdf`

It covers the related work, dataset and preprocessing, model training, quantitative evaluation, regional change-detection experiment, and methodological limitations.

## License

This project is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** License. See the manuscript files for complete details.

[Read the CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/)

---

## Author

**Piyush Manna**  
*Department of Computer Science, Bal Kalyan Vidhya Mandir, Biratnagar, Nepal*  
Email: `w3bpiyush@gmail.com`
