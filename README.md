# Medical Image Preprocessing Pipeline for GI Tract Segmentation

This project implements a comprehensive preprocessing pipeline for gastrointestinal (GI) MRI images. The pipeline addresses common challenges in medical image analysis, with the intent of preparing high-quality data for training segmentation models and an end of goal enhancing the performance of deep learning segmentation applications on medical images.

## Introduction

It's important that segmentation of the GI tract is accurate for streamlining the delivery of high-quality care to patients. In practice, radiation oncologist often manually segment out the positions of the stomach and intestines in order to avoid vital areas when directing x-ray beams during radiation therapy. This manual process is time-consuming; both for the pracitioner and the patient. By automating this process, we elimiate a lot of downtime for the patient while increasing the efficiency of the oncologist's workflow. Thus, we aim to develop a robust method for preparing high-quality image data that can feed into deep learning models, potentially enhancing their segmentation ability.

The key steps in this automated process are as follows:

* Conversion of PNG slices to NifTI format.
* Bias field correction to mitigate intenstiy inhomogeneities.
* Intensity normalization to standardize the intensity range.
* Intra-case atlas creation to reduce variation across scans.

## Dependencies
The follwoing Python libraries are utilized to build this project:

* SimpleITK: Allows for multi-dimensional image analysis. Used to process our NIFTI images and conduct image registration.
* NumPy: Allows for efficent mathematical operations.
* Pillow: Allows for reading PNG image slices.
* Scikit-Image: Used for resizing images.

## Installation
## Code Structure
The code structure of the project is organized into 5 main files:
* main.py
* utils.py
* preprocessing.py
* atlas_creation.py
* validation.py

## Data Format
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

## Execution
### Configuration
modify the config:




