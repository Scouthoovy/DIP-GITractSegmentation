# Medical Image Preprocessing Pipeline for GI Tract Segmentation

[![Build Status](https://img.shields.io/...)](…)
[![License: MIT](https://img.shields.io/...)](LICENSE)

This project implements a comprehensive preprocessing pipeline for gastrointestinal (GI) MRI images. The pipeline addresses common challenges in medical image analysis, with the intent of preparing high-quality data for training segmentation models and an end of goal enhancing the performance of deep learning segmentation applications on medical images.

## Table of Contents
- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Installation](#isntallation)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

It's important that segmentation of the GI tract is accurate for streamlining the delivery of high-quality care to patients. In practice, radiation oncologist often manually segment out the positions of the stomach and intestines in order to avoid vital areas when directing x-ray beams during radiation therapy. This manual process is time-consuming; both for the pracitioner and the patient. By automating this process, we elimiate a lot of downtime for the patient while increasing the efficiency of the oncologist's workflow. Thus, we aim to develop a robust method for preparing high-quality image data that can feed into deep learning models, potentially enhancing their segmentation ability.

The key steps in this automated process are as follows:

* Wavelet denoising to suppress noise while maintain organ structures and sharpness
* Min-max intensity normalization to standardize the intensity range.
* Contrast Limited Adaptive Histogram Equalization to enhance contrast without over-amplification of noise.

## Dependencies
The following Python libraries are utilized to build this project:

* SimpleITK
* NumPy
* Scikit-Image
* Pandas
* Tensorflow
* PyWavelets
* OpenCV
* Scikit-Learn
* Matplotlib
* glob
* os
* random
* collections.defaultdict

## Installation
## Code Structure
The code structure of the project is organized into 5 main files:
* main.py
* utilities.py
* preprocessing.py
* model.py
* metrics.py
* create_dataset.py

## Data & Format

The dataset can found online at https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/data.

Once downloaded, the structure of the data follows this format:

data/ 

└── train/ 

└── case101/ 

├── case101_day20/

│   └── scans/

│       ├── slice_0001.png

│       ├── slice_0002.png

│       └── ...

├── case101_day22/

│   └── scans/

│       └── ...

└── ...

This format is what the code will expect as well.

## Execution
### Configuration
modify the config:




