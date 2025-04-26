import cv2
from skimage.restoration import denoise_wavelet
import SimpleITK as sitk
import numpy as np

def preprocess_pipeline(images, bias_cor=False, wavelet=False, CLAHE=False, normalization=False):
  """
  Preprocess images with optional steps for including methods
  
  Parameters
    images (np.ndarray): array of images to preprocess
    bias_cor (boolean, optional): If True, apply bias field correction, with default False
    wavelet (boolean, optional): If True, apply wavelet denoising, with default False
    CLAHE (boolean, optional): If True, apply CLAHE, with default False
    normalizatoin (boolean, optional): If True, apply normalization, with default False
    
  Returns
    images (np.ndaray): preprocessed images
  """
  images = images.astype(np.float32)

  if bias_cor:
    print(f"Applying Bias Correction...")
    images = apply_bias_correction(images)
  if wavelet:
    print(f"Applying Wavelet Denoising...")
    images = apply_wavelet_denoising(images)
  if normalization:
    print(f"Applying Min-Max Normalization...")
    images = apply_normalization(images)
  if CLAHE:
    print(f"Applying CLAHE...")
    images = apply_clahe(images)

  print(f"Preprocessing complete!")
  return images

def apply_wavelet_denoising(images, wavelet='db1', mode='soft', method='BayesShrink'):
  """
  Apply wavelet-based denoising to images
  
  Parameters
    images (np.ndarray): array of images to denoise
    wavelet(str, optional): wavelet type to use, with default 'db1'
    mode (str, optional): Denoising mode with default 'soft'
    method (str, optional): thresholding method, with default 'BayesShrink'
    
  Returns
    images (np.ndarray): Denoised images
  """
  for index, image in enumerate(images):
    image = denoise_wavelet(image,wavelet=wavelet, mode=mode, method=method)
    images[index] = image
    if index % 500 == 0:
      print(f"Denoised {index}/{len(images)} images")

  return images

def apply_clahe(images, clip_limit=2.0, grid_size=(8,8)):
  """
  Apply CLAHE equalization to images
  
  Parameters
    images (np.ndarray): Array of images to equalize
    clip_limit (float, optional): Threshold for contrast clipping, with default 2.0
    grid_size (tuple of int, optional): Size of grid for localized histogram equalization, with default (8,8)
    
  Returns
    images (np.ndarray): Equalized images

  """
  
  #images = np.clip(images, 0, 255)
  images = images.astype(np.uint8)
  clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
  for index, image in enumerate(images):
    image = clahe.apply(image)
    images[index] = image
    if index % 500 == 0:
      print(f"Equalized {index}/{len(images)} images")

  images = images.astype(np.float32)
  return images

def apply_normalization(images):
  """
  Apply min-max normalization to images
  
  Parameters
    images (np.ndarray): Array of images to normalize
    
  Returns
    images (np.ndarray): Normalized images
  """
  for index, image in enumerate(images):
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    images[index] = image
    if index % 500 == 0:
      print(f"Normalized {index}/{len(images)} images")

  return images

def apply_bias_correction(images, shrink=4):
  """
  Apply N4 bias field correction to images
  
  Parameters
    images (np.ndarray): Array of images to corrected_image
    shrink (int, optional): Shrink factor for intial N4 correction, with default 4
    
  Returns
    images (np.ndarray): Bias-corrected images
  """
  images = np.clip(images, 0, 255)
  images = images.astype(np.uint8)
  for index, image in enumerate(images):
    ret, thresh1 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    raw_image = sitk.GetImageFromArray(image.astype(np.float32))
    input_image = sitk.GetImageFromArray(image.astype(np.float32))
    mask_image = sitk.GetImageFromArray(thresh1)

    input_image = sitk.Shrink(input_image, [shrink]*input_image.GetDimension())
    mask_image = sitk.Shrink(mask_image, [shrink]*input_image.GetDimension())

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image = corrector.Execute(input_image, mask_image)

    log_bias_field = corrector.GetLogBiasFieldAsImage(raw_image)
    corrected_image_og_resolution = raw_image / sitk.Exp(log_bias_field)
    images[index] = sitk.GetArrayFromImage(corrected_image_og_resolution)

    if index % 500 == 0:
      print(f"Corrected {index}/{len(images)} images")

  #images = images.astype(np.float32)
  return images
