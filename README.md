# Medical Image Preprocessing Pipeline for GI Tract Segmentation

This project implements a comprehensive preprocessing pipeline for gastrointestinal (GI) MRI images. The pipeline addresses common challenges in medical image analysis, with the intent of preparing high-quality data for training segmentation models and an end of goal enhancing the performance of deep learning segmentation applications on medical images.

## Table of Contents
- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Installation](#isntallation)
- [Configuration](#configuration)
- [Execution](#execution)
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
Each .py file contains necessary functions for running the experiment.

* utilities.py: contains functions for decoding RLE strings, plotting images, extracting masks from tf.Datasets, and building 3D volumes from slices.
* preprocessing.py: contains functions for applying the three main transformations: intensity normalization, CLAHE, and wavelet denoising.
* metrics.py: contains functions for calculating mean Dice coefficent, mean 3D Hausdorff distance, a combined metric of the two.
* model.py contains functions for building a U-Net model architecture.
* create_dataset.py: contains functions to load, filter, and split images into datasets.
* main.py: contains main() method for running full program. 

### Configuration

Simply running main.py will conduct the full pipeline, from loading the data to training the model and calculating the Dice coefficient and 3D Hausdorff distance. It will also save both trained models at the end.
If you wish to change any parameters, it would be best to do so within the main.py file.

### Results

We should expect to see:
* mean Dice coefficient for both unprocessed and processed data models.
* mean 3D Hausdorff distance for both unprocessed and processed data models.
* Weighted Dice + 3D Hausdorff distance metric for both unprocessed and processed data models.
* .h5 saves for both unprocessed and processed data models.

## Contact
Brayden Yarish - braydenyarish@gmail.com \\
The University of Winnipeg - MSc Student







