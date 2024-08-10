# WSVM: Weakly Supervised Object Detection for Automatic Tooth-marked Tongue Recognition

WSVM is a fully automated method for tongue detection that focuses on accurately extracting the tongue foreground from raw clinical images. This repository proposes a weakly supervised object detection method based on Vision Transformer (ViT) and Multiple Instance Learning (MIL) to recognize tooth-marked tongues, and validates the effectiveness of WSVM in detecting and distinguishing tooth-marked tongues.

## Table of Contents

- [WSVM: Weakly Supervised Object Detection for Automatic Tooth-marked Tongue Recognition](#wsvm-weakly-supervised-object-detection-for-automatic-tooth-marked-tongue-recognition)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
  - [Dataset and Pre-trained Model](#dataset-and-pre-trained-model)
    - [Dataset](#dataset)
    - [Pretrained Model](#pretrained-model)
  - [Getting Started](#getting-started)
    - [Training the Model](#training-the-model)
    - [Testing the Model](#testing-the-model)
  - [Project Structure](#project-structure)
  - [Acknowledgements](#acknowledgements)

## Installation

### Prerequisites

- Python 3.19.9 
- PyTorch 2.2.2, CUDA 12.3
- Required Python packages (specified in `environment.yaml`)

### Setup

Clone the repository:
```bash
git clone https://github.com/yc-zh/WSVM.git
cd WSVM
```

Create a virtual environment and install dependencies:
```bash
conda env create -f environment.yaml
conda activate WSVM
```
## Dataset and Pre-trained Model
### Dataset
Download the dataset from the following link: [Tongue Image Dataset](https://www.kaggle.com/datasets/clearhanhui/biyesheji?resource=download) and put it in the `data/tongue` directory.

### Pretrained Model
The pre-trained model weights can be downloaded from the following link: [Pre-trained model](https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA) and put it in the `vit_weights` directory.

## Getting Started
### Training the Model

To train the model, run the following command:
```bash
python train.py
```

### Testing the Model

To test the model, use:
```bash
python test.py
```

## Project Structure

```
WSVM/
├── data/                     # Directory for storing datasets
├── models/                   # fine-tuned model files
├── tongue_extraction/        # Scripts for tongue foreground extraction
├── vision_transformer/       # Vision Transformer related code
├── vit_weights/              # Pre-trained ViT weights
├── environment.yaml          # Conda environment configuration file
├── README.md                 # Project documentation
├── test.py                   # Testing script
├── train.py                  # Training script
└── utils.py                  # Utility functions
```

## Acknowledgements
Thanks [deep-learning-for-image-processing](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing), [SAM](https://github.com/facebookresearch/segment-anything) and [YOLOv8](https://github.com/ultralytics/ultralytics) for their public code and released models.
