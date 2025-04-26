import matplotlib.pyplot as plt
import random
from collections import defaultdict
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
from skimage.restoration import denoise_wavelet
import cv2

def get_predicted_masks(model, ds):
  """
  Generate predicted segmentation masks from a model

  Parameters
    model (tf.keras.Model): Trained model for inference
    ds (tf.data.Dataset): Dataset with (image, mask) batches

  Returns
    all_predictions (np.ndarray): Array of predicted mask labels
  """
  all_predictions = []
  for images, masks in ds:
    preds = model.predict(images)
    pred_masks = np.argmax(preds, axis=-1)
    all_predictions.extend(pred_masks)

  return np.array(all_predictions)

def extract_masks(dataset):
  """
  Extract 2D mask arrays from one-hot encoded masks in a dataset

  Parameters
    dataset (tf.data.Dataset): Dataset with (image, mask) pairs, where mask is one-hot.

  Returns
    masks (np.ndarray): Array of mask label arrays
  """
  masks = []
  for image, mask in dataset:
    mask = np.argmax(mask, axis=-1)
    masks.extend(mask)
  masks = np.array(masks)
  return masks

def build_volumes(masks, id_map):
  """
  Assemble 3D volumes from 2D slice masks using slice identifiers

  Parameters
    masks (np.ndarray): Array of 2D masks, shape (n_slices, H, W)
    id_map (dict): Mapping from slice ID string to index in masks

  Returns
    volumes (dict): Mapping case_id to 3D volume array of shape (Z, H, W)
  """
  by_case = defaultdict(list)
  for id, index in id_map.items():
    case_id, day, slice_str, slice_num = id.split('_')
    z = int(slice_num)
    by_case[case_id].append((z, masks[index]))

  volumes = {}
  for case_id, slice_list in by_case.items():
    min_z = min(z for z, mask in slice_list)
    max_z = max(z for z, mask in slice_list)
    Z = max_z - min_z + 1
    H, W = masks.shape[1:]

    volume = np.zeros((Z, H, W), dtype=np.uint8)
    for z, mask in slice_list:
      volume[z-min_z] = mask

    volumes[case_id] = volume

  return volumes

def augment_data(image, mask):
  """
  Apply random augmentations to image and mask tensors

  Parameters
    image (tf.Tensor): Input image tensor
    mask (tf.Tensor): Corresponding mask tensor

  Returns
    image (tf.Tensor): Augmented image tensor
    mask (tf.Tensor): Augmented mask tensor
  """
  if tf.random.uniform(()) > 0.5:
    image = tf.image.flip_left_right(image)
    mask = tf.image.flip_left_right(mask)

  if tf.random.uniform(()) > 0.5:
    image = tf.image.flip_up_down(image)
    mask = tf.image.flip_up_down(mask)

  image = tf.image.random_brightness(image, 0.1)
  image = tf.image.random_contrast(image, 0.9, 1.1)
  image = tf.clip_by_value(image, 0, 1)

  return image, mask

def plot_from_image_index(image_index, image_arr):
  plt.imshow(image_arr[image_index], cmap="gray")
  plt.axis('off')
  plt.show()

def plot_from_mask_index(mask_index, mask_arr):
  plt.imshow(mask_arr[mask_index], cmap="gray")
  plt.axis('off')
  plt.show()

def plot_image_mask_overlay(image_index, image_arr, mask_arr):
  mask = np.ma.masked_where(mask_arr[image_index] == 0, mask_arr[image_index])

  plt.imshow(image_arr[image_index], cmap="gray")
  plt.imshow(mask, alpha=0.5, cmap='jet')
  plt.axis('off')
  plt.show()

def plot_random_image_mask_overlay(image_arr, mask_arr):
  plt.figure(figsize=(15, 15))
  image_index = random.randint(0, len(image_arr)-1)
  print(f"Image index is: {image_index}")
  mask = np.ma.masked_where(mask_arr[image_index] == 0, mask_arr[image_index])

  plt.subplot(1, 3, 1)
  plt.imshow(image_arr[image_index], cmap="gray")
  plt.title('Original Image')
  plt.axis('off')

  plt.subplot(1, 3, 2)
  plt.imshow(mask_arr[image_index], cmap="gray")
  plt.title('Segmentation Mask')
  plt.axis('off')

  plt.subplot(1, 3, 3)
  plt.imshow(image_arr[image_index], cmap="gray")
  plt.imshow(mask, alpha=0.5, cmap='jet')
  plt.title('Mask Overlaid on Image')
  plt.axis('off')

  plt.savefig('/content/drive/MyDrive/DIP_Project/saved_images_final/random_image_mask_overlay_2.png', bbox_inches='tight')
  plt.show()

def plot_image_mask_overlay_from_tensor(dataset, num_images):
  plt.figure(figsize=(15, 15))
  for n, image_mask_tuple in enumerate(dataset.take(num_images)):
    image = image_mask_tuple[0]
    mask = image_mask_tuple[1]
    mask = np.ma.masked_where(mask == 0, mask)

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(image_mask_tuple[1], cmap="gray")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(image, cmap="gray")
    plt.imshow(mask, alpha=0.5, cmap='jet')
    plt.axis('off')

    plt.show()

def plot_image_preprocess(image_arr):
  plt.figure(figsize=(15, 15))
  image_index = random.randint(0, len(image_arr)-1)
  image = image_arr[image_index]

  plt.subplot(1, 5, 1)
  plt.imshow(image, cmap="gray")
  plt.axis('off')

  image = np.clip(image, 0, 255)
  image = image.astype(np.uint8)
  raw_image = sitk.GetImageFromArray(image.astype(np.float32))
  mask_image = sitk.LiThreshold(raw_image, 0, 1)

  original_size = raw_image.GetSize()
  shrink_factor = [2] * raw_image.GetDimension()

  small_image = sitk.Shrink(raw_image, shrink_factor)
  small_mask = sitk.Shrink(mask_image, shrink_factor)

  corrector = sitk.N4BiasFieldCorrectionImageFilter()
  small_image_corrected = corrector.Execute(small_image, small_mask)

  log_bias_field = corrector.GetLogBiasFieldAsImage(small_image)
  log_bias_field_rs = sitk.Resample(log_bias_field, raw_image)
  corrected_image = raw_image / sitk.Exp(log_bias_field_rs)
  image = sitk.GetArrayFromImage(corrected_image)

  print(f"After bias correction: min={image.min()}, max={image.max()}")

  plt.subplot(1, 5, 2)
  plt.imshow(image, cmap="gray")
  plt.axis('off')

  image = image.astype(np.float32)

  image = denoise_wavelet(image)
  plt.subplot(1, 5, 3)
  plt.imshow(image, cmap="gray")
  plt.axis('off')

  image = np.clip(image, 0 , 255)
  image = image.astype(np.uint8)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  image = clahe.apply(image)
  plt.subplot(1, 5, 4)
  plt.imshow(image, cmap="gray")
  plt.axis('off')

  image = image.astype(np.float32)
  image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
  plt.subplot(1, 5, 5)
  plt.imshow(image, cmap="gray")
  plt.axis('off')

  plt.show()

def rle_decode(rle_string, shape):
  """
  Decode a run-length encoded string into a binary mask

  Parameters
    rle_string (str): Run-length encoding
    shape (tuple): (height, width) of the mask to return

  Returns
    mask (np.ndarray): Binary mask array
  """
  s = rle_string.split()
  starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
  starts -= 1
  ends = starts + lengths
  img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
  for lo, hi in zip(starts, ends):
      img[lo:hi] = 1
  return img.reshape(shape)