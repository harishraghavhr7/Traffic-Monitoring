# Traffic Density Estimation using CSRNet on the TRANCOS Dataset

This repository contains a complete, production-quality implementation of **CSRNet** (CVPR 2018) in PyTorch to estimate vehicle traffic density maps and total vehicle counts from the **TRANCOS** dataset.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Dataset Setup](#dataset-setup)
5. [Usage](#usage)
   - [Training](#training)
   - [Evaluation](#evaluation)
   - [Prediction / Inference](#prediction--inference)
6. [Methodology & Architecture](#methodology--architecture)
   - [CSRNet Architecture](#csrnet-architecture)
   - [Density Map Generation](#density-map-generation)
   - [Google Colab & Drive Support](#google-colab--drive-support)

---

## Project Overview

Vehicle crowd counting is a crucial component in modern intelligent transportation systems. In highly congested environments, traditional bounding-box-based object detection pipelines (e.g. YOLO, Faster R-CNN) often fail due to severe vehicle overlaps, occlusions, and scale variations. 

This project solves this by training **CSRNet (Congested Scene Recognition Network)** to predict continuous traffic density maps from point annotations. Integrating (summing) the values in the predicted density map gives the estimated total vehicle count in the scene.

---

## Repository Structure

```text
.
├── density_estimation/
│   └── TrafficDensity/
│       ├── datasets/
│       │   └── trancos_dataset.py  # Custom PyTorch dataset loader
│       ├── models/
│       │   └── csrnet.py           # CSRNet model architecture definition
│       ├── utils/
│       │   ├── density_map.py      # Gaussian density map generation utilities
│       │   ├── metrics.py          # MAE, RMSE, and count calculation functions
│       │   └── visualize.py        # Heatmap and image overlay generation
│       ├── train.py                # Main training script (supports CPU/GPU, checkpoints, and Colab)
│       ├── evaluate.py             # Evaluation script for computing metrics on test split
│       ├── predict.py              # Single-image inference and visualization script
│       ├── TrafficDensity_Colab.ipynb # Ready-to-run Google Colab Notebook
│       ├── requirements.txt        # Package dependencies
│       └── README.md               # Subdirectory documentation
└── README.md                   # Main landing page documentation (this file)
```

---

## Installation

Ensure you have Python 3.8+ installed. Install the package dependencies using `pip`:

```bash
pip install -r density_estimation/TrafficDensity/requirements.txt
```

---

## Dataset Setup

The TRANCOS dataset is expected to be placed under a root dataset directory (e.g. `density_estimation/TRANCOS - edited/`) containing two splits:

```text
density_estimation/TRANCOS - edited/
├── train_data/
│   ├── images/       # *.jpg traffic scene images (e.g., image-1-000001.jpg)
│   └── txt/          # *.txt containing coordinate lists (x y) per vehicle center
└── test_data/
    ├── images/       # *.jpg test traffic images
    └── txt/          # *.txt containing test coordinate lists
```

---

## Usage

### Training

To train the CSRNet model from scratch on the TRANCOS dataset, run:

```bash
python density_estimation/TrafficDensity/train.py --dataset_path "density_estimation/TRANCOS - edited/train_data" --epochs 100 --batch_size 8 --lr 1e-5
```

**Key Arguments:**
* `--dataset_path`: Path to the TRANCOS `train_data` split folder (defaults to `density_estimation/TRANCOS - edited/train_data`).
* `--checkpoint_path`: Folder to save training checkpoints (saved as `best_model.pth` and `last_model.pth` under `density_estimation/TrafficDensity/checkpoints/`).
* `--epochs`: Total training epochs.
* `--batch_size`: Number of images per training batch.
* `--lr`: Learning rate for Adam optimizer.
* `--resume`: Path to a checkpoint file to resume training (e.g. `--resume density_estimation/TrafficDensity/checkpoints/last_model.pth`).
* `--no_pretrained`: Skip downloading VGG-16 weights (useful for debugging).
* `--dry_run`: Run a quick dry-run with 4 samples to verify the pipeline.
* `--colab`: Enable Google Drive mounting and change save path to Google Drive.

### Evaluation

To evaluate a trained model checkpoint on the TRANCOS test dataset split:

```bash
python density_estimation/TrafficDensity/evaluate.py --dataset_path "density_estimation/TRANCOS - edited/test_data" --model_path "density_estimation/TrafficDensity/checkpoints/best_model.pth"
```

This script will run inference over the test images and output the following metrics:
1. **Average MAE (Mean Absolute Error):** Measures the average absolute error in vehicle counts.
2. **Average RMSE (Root Mean Squared Error):** Captures the variance of prediction errors.
3. **Average Count Error (Mean Bias):** Indicates if the network is systematically overestimating ($+$) or underestimating ($-$) counts.

### Prediction / Inference

To run prediction on a single query image and visualize the results:

```bash
python density_estimation/TrafficDensity/predict.py --image_path "density_estimation/TRANCOS - edited/test_data/images/image-1-000333.jpg" --model_path "density_estimation/TrafficDensity/checkpoints/best_model.pth" --show
```

**Arguments:**
* `--image_path`: Path to the input image file (Required).
* `--model_path`: Path to the trained checkpoint (Defaults to `density_estimation/TrafficDensity/checkpoints/best_model.pth`).
* `--save_path`: Destination path to save the 4-panel visualization chart (Defaults to `density_estimation/TrafficDensity/prediction_result.png`).
* `--show`: Launch an interactive desktop window using Matplotlib GUI to view the comparison.

---

## Methodology & Architecture

### CSRNet Architecture

CSRNet consists of two main components:
1. **Frontend (VGG-16):** The first 10 convolutional layers of a pretrained VGG-16 network (containing 3 MaxPool2d layers). It generates feature maps that are $1/8$ of the original input size.
2. **Backend (Dilated Convolutions):** 6 dilated convolutional layers (dilation rate = 2, kernel size = 3, padding = 2) that expand the receptive field to aggregate multi-scale context without downsampling the resolution.
3. **Output:** A $1 \times 1$ convolutional layer mapping 64 channels to a single-channel density map.

### Density Map Generation

Ground truth density maps are generated by applying a 2D Gaussian filter to a coordinate grid of vehicle centers:
$$F(x) = \sum_{i=1}^{N} \dots * G_{\sigma}(x)$$

**Sum Preservation Scaling:** Because the network produces outputs at $1/8$ of the input resolution, during image resizing, the ground truth density map is downscaled by a factor of 8. The density map is scaled so that its integral sum remains exactly equal to the vehicle count.

### Google Colab & Drive Support

To execute in Google Colab, add the `--colab` flag when launching `train.py`. This will trigger Google Drive mounting to `/content/drive` and direct checkpoint saving to your Drive, ensuring that checkpoints are preserved even if the Colab runtime disconnects:

```python
# Colab execution command:
!python density_estimation/TrafficDensity/train.py --colab --google_drive_dir "/content/drive/MyDrive/TrafficDensity/checkpoints"
```
You can find the ready-to-run Jupyter notebook inside this repository at `density_estimation/TrafficDensity/TrafficDensity_Colab.ipynb`.
