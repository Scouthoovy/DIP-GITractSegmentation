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
## Installation
## Code Structure
## Data Format
## Execution



