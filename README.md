# PyTorch Image Caption Generator (RTX GPU Enabled)

A modular, production-ready PyTorch codebase designed for training and serving an Image Captioning model (ResNet50 CNN + LSTM RNN) on the Flickr8k dataset, with future support for Beam Search inference, Hindi translation, FastAPI backend service, and a Next.js web application.

---

## 🚀 Project Overview

This repository establishes the clean, production-grade foundation of the Image Caption Generator. It features:
* **Python 3.11+** compliance.
* **Modular layout** with dedicated configuration, processing, modeling, training, and inference directories.
* **RTX GPU-ready setup** with verification utility.
* Strict separation of concerns (configuration separated from model logic).

---

## 📁 Project Structure

```
image-caption-generator/
├── data/                      # Dataset directories
│   └── Flickr8k/              # Placeholder for Flickr8k images & captions
├── outputs/                   # Training artifacts
│   ├── checkpoints/           # Model weight checkpoints (.pth)
│   └── logs/                  # TensorBoard execution logs
├── notebooks/                 # Jupyter notebooks for EDA & visualization
├── src/                       # Main source code
│   ├── config/                # Path resolution and hyperparameter settings
│   ├── utils/                 # Utility scripts (GPU checker, helpers)
│   ├── data_processing/       # Dataset builder, vocabulary & prep pipelines
│   ├── feature_extraction/    # ResNet50 CNN feature extractor
│   ├── model/                 # LSTM Decoder model definition
│   ├── training/              # Training loop & trainer class
│   └── inference/             # Beam Search & translation modules
├── tests/                     # Automated unit testing suite
├── .gitignore                 # ML-tailored git exclusion rules
├── requirements.txt           # Python dependency declarations
└── README.md                  # Project documentation
```

---

## 🛠️ Quick Start & Environment Setup

### Option A: Automated Setup (Windows)
We provide an automated setup batch script that handles virtual environment creation, PyTorch (CUDA 12.1) installation, all dependency setup, and runs the GPU verification check.

Simply run:
```cmd
setup.bat
```

### Option B: Manual Setup (Cross-Platform)

#### 1. Prerequisites
Ensure you have **Python 3.11+** installed.

#### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install RTX GPU Ready PyTorch
For systems with NVIDIA RTX GPUs, install PyTorch compiled with CUDA support:

```bash
# Recommended for CUDA 12.1 (modern RTX cards)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

*(Alternatively, for CPU-only training/inference)*
```bash
pip install torch torchvision
```

#### 4. Install Remaining Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚡ Verification

Validate your PyTorch and GPU configuration by running the built-in test utility:

```bash
python src/utils/check_gpu.py
```

### Expected Output
If your CUDA device (RTX GPU) is configured correctly, you will see output similar to:
```
============================================================
PyTorch & RTX GPU Verification Setup
============================================================
Python version: 3.11.x ...
PyTorch version: 2.1.x+cu121
CUDA Available: True
CUDA Built-in Version: 12.1
Number of GPUs found: 1
  Device 0: NVIDIA GeForce RTX 3060/4070 (or similar)
    Compute Capability: 8.x

Running test matrix multiplication on GPU...
Tensor multiplication succeeded in x.xxxx ms.
[SUCCESS] PyTorch RTX GPU support is properly configured and active!
============================================================
```

---

## 🗺️ Roadmap & Next Steps

1. **`data_processing`**: Implement custom PyTorch `Dataset` and `DataLoader` for Flickr8k, text tokenization, and vocabulary building.
2. **`feature_extraction`**: Freeze and extract feature maps using pre-trained ResNet50 CNN.
3. **`model`**: Define the LSTM Decoder network that consumes CNN features.
4. **`training`**: Implement multi-epoch training with TensorBoard logging, teacher forcing, and checkpoint saving.
5. **`inference`**: Add a Beam Search inference helper and integrate translation APIs for Hindi captions.
6. **FastAPI & Next.js**: Wrap inference in an API endpoint and build a premium frontend dashboard to upload images and generate captions.
