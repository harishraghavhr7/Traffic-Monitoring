# Traffic Density Estimation using CSRNet on the TRANCOS Dataset

This repository contains a complete, production-quality implementation of **CSRNet** (CVPR 2018) in PyTorch to estimate vehicle traffic density maps and total vehicle counts from the **TRANCOS** dataset.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Dataset Setup](#dataset-setup)
5. [Usage](#usage)
   - [Training](#training)
   - [Evaluation](#evaluation)
   - [Prediction](#prediction)
6. [Methodology](#methodology)
   - [CSRNet Architecture](#csrnet-architecture)
   - [Density Map Generation](#density-map-generation)
   - [Google Colab Support](#google-colab-support)

---

## Project Overview

Vehicle crowd counting is a crucial component in traffic management systems. In highly congested environments, traditional bounding-box-based object detection fails due to severe vehicle overlaps and occlusion. 

This project trains **CSRNet (Congested Scene Recognition Network)** to predict continuous traffic density maps from point annotations. Integrating (summing) the values in the predicted density map gives the estimated total vehicle count in the scene.

---

## Project Structure

```text
TrafficDensity/
├── datasets/
│   └── trancos_dataset.py  # Custom PyTorch dataset loader
├── models/
│   └── csrnet.py           # CSRNet model architecture definition
├── utils/
│   ├── density_map.py      # Gaussian density map generation utilities
│   ├── metrics.py          # MAE and RMSE calculation functions
│   └── visualize.py        # Heatmap and image overlay generation
├── train.py                # Main training script (supports resuming and Colab)
├── evaluate.py             # Evaluation script for computing metrics on test split
├── predict.py              # Single-image inference and visualization script
├── requirements.txt        # Package dependencies
└── README.md               # Documentation
```

---

## Installation

Ensure you have Python 3.8+ installed. Install the package dependencies using `pip`:

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

The TRANCOS dataset is expected to be placed under a root dataset directory (e.g. `TRANCOS - edited/`) containing two splits:

```text
TRANCOS - edited/
├── train_data/
│   ├── images/       # *.jpg traffic scene images
│   ├── txt/          # *.txt containing coordinate lists (x y) per vehicle center
│   ├── ground-truth/ # (Optional) original mat files
│   └── dots/         # (Optional) original dots png files
└── test_data/
    ├── images/       # *.jpg test traffic images
    └── txt/          # *.txt containing test coordinate lists
```

---

## Usage

### Training

To train the CSRNet model from scratch on the TRANCOS dataset, run:

```bash
python train.py --dataset_path "../TRANCOS - edited/train_data" --epochs 100 --batch_size 4 --lr 1e-5
```

**Key Arguments:**
* `--dataset_path`: Path to the TRANCOS `train_data` split folder (defaults to `../TRANCOS - edited/train_data`).
* `--checkpoint_path`: Folder to save training checkpoints (saved as `best_model.pth` and `last_model.pth`).
* `--epochs`: Total training epochs.
* `--batch_size`: Number of images per training batch.
* `--lr`: Learning rate for Adam optimizer.
* `--resume`: Path to a checkpoint file to resume training (e.g. `--resume checkpoints/last_model.pth`).
* `--colab`: If set, mounts Google Drive automatically.
* `--google_drive_dir`: Target path on Google Drive to save checkpoints when running in Colab.

### Evaluation

To evaluate a trained model checkpoint on the TRANCOS test dataset split:

```bash
python evaluate.py --dataset_path "../TRANCOS - edited/test_data" --model_path "checkpoints/best_model.pth"
```

This script will run inference over the test images and output the following metrics:
1. **Average MAE (Mean Absolute Error):** Counts the average absolute error in vehicle counts.
2. **Average RMSE (Root Mean Squared Error):** Captures variance of prediction errors.
3. **Average Count Error (Mean Bias):** Indicates if the network is systematically overestimating or underestimating counts.

### Prediction

To run prediction on a single query image and visualize the results:

```bash
python predict.py --image_path "../TRANCOS - edited/test_data/images/image-1-000001.jpg" --model_path "checkpoints/best_model.pth" --save_path "prediction_result.png"
```

The script will:
* Load the image and process it to be compatible with CSRNet.
* Output the **Estimated Vehicle Count** directly to the terminal.
* Save a 4-panel comparison image (`prediction_result.png`) showing:
  1. Original image
  2. Greyscale density map
  3. Jet color heatmap
  4. Overlay of the heatmap on the original image

---

## Methodology

### CSRNet Architecture

CSRNet consists of two main parts:
1. **Frontend (VGG-16):** The first 10 convolutional layers of a pretrained VGG-16 network (containing 3 MaxPool2d layers). It generates feature maps that are $1/8$ of the original input size.
2. **Backend (Dilated Convolutions):** 6 dilated convolutional layers (dilation rate = 2, kernel size = 3, padding = 2) that expand the receptive field to aggregate context without downsampling the resolution.
3. **Output:** A $1 \times 1$ convolutional layer mapping 64 channels to a single-channel density map.

### Density Map Generation

Ground truth density maps are generated by applying a 2D Gaussian filter to a coordinate grid of vehicle centers:
$$F(x) = \sum_{i=1}^{N} \delta(x - x_i) * G_{\sigma}(x)$$

**Sum Preservation Scaling:** Because the network produces output at $1/8$ of the input resolution, during image resizing, the ground truth density map is downsampled by a factor of 8. The density map is scaled so that its integral sum remains exactly equal to the vehicle count.

### Google Colab Support

To execute in Google Colab, add the `--colab` flag when launching `train.py`. This will trigger Google Drive mounting to `/content/drive` and direct checkpoint saving to your Drive, ensuring that checkpoints are preserved even if the Colab runtime disconnects:

```python
# Colab execution command:
!python train.py --colab --google_drive_dir "/content/drive/MyDrive/TrafficDensity/checkpoints"
```
