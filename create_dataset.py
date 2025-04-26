import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import tensorflow as tf
import glob
import os
import cv2
import pandas as pd

def create_train_images(IMAGE_DIR, SIZE_X, SIZE_Y):
  """
  Load and resize images with mapping
  
  Parameters
    IMAGE_DIR (str): Directory containing case subdirectories
    SIZE_X (int): Target image width
    SIZE_Y (int): Target image height
    
  Returns
    image_id_map (dict): mapping image IDs to list indicies
    train_images (np.ndarray): Array of loaded resized images
    train_images_dims (list of tuples): Original dimensions (height, width) per image
    case_ids (list of str): Case identifier for each image
  """
  image_id_map = {}
  train_images = []
  train_images_dims = []
  case_ids = []

  i=0
  for case_path in glob.glob(os.path.join(IMAGE_DIR, '*/') ):
    case_id = case_path.split('/')[-2]
    print(f"Case {i+1}/85, case id: {case_id}")
    i+=1

    for caseday_path in glob.glob(os.path.join(case_path, '*/') ):
      day_id = caseday_path.split('/')[-2].split('_')[-1]

      for image_path in glob.glob(os.path.join(caseday_path, 'scans/*.png')):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (SIZE_X, SIZE_Y))

        filename = os.path.basename(image_path)  # Gets "slice_0063_266_266_1.50_1.50.png"
        parts = filename.split('_')
        slice_num = parts[1]
        image_width = int(parts[2])
        image_height = int(parts[3])

        image_id = f"{case_id}_{day_id}_slice_{slice_num}"
        index = len(train_images)
        image_id_map[image_id] = index

        train_images.append(image)
        case_ids.append(case_id)
        train_images_dims.append((image_height, image_width))

  train_images = np.array(train_images)

  print(f"Total images loaded: {len(train_images)}")

  return image_id_map, train_images, train_images_dims, case_ids

def create_train_masks(MASK_CSV, SIZE_X, SIZE_Y, image_id_map, train_images, train_images_dims, class_map):
  """
  Generate unified masks aligned to training images from RLE annotations
  
  Parameters
    MASK_CSV (str): Path to CSV with segmentation RLE annotations
    SIZE_X (int): Target mask width
    SIZE_Y (int): Target mask height
    image_id_map (dict): Mapping image IDs to indicies in train_images
    train_images (np.ndarray): Array of images
    train_images_dims (list of tuple): Original dimensions for each image
    class_map (dict): Mapping of class names to integer labels.
    
  Returns
    mask_id_map (dict): Mapping image IDs to indices in unified_masks
    unified_masks_aligned (np.ndarray): Masks array matching order of train_images
  
  """

  train_mask_csv = pd.read_csv(MASK_CSV)
  mask_id_map = {}
  id_groups = train_mask_csv.groupby('id')

  unified_masks = []
  i=0
  for image_id, group in id_groups:
    unified_mask = np.zeros((SIZE_X, SIZE_Y), dtype=np.uint8)
    dim_index = image_id_map[image_id]
    original_dim = train_images_dims[dim_index]

    mask_id_map[image_id] = len(unified_masks)

    print(f"Slice {i+1}/38496, slice id: {image_id}")
    i+=1

    for index, row in group.iterrows():
      class_name = row['class']
      class_index = class_map[class_name]

      if pd.notna(row['segmentation']):
        binary_mask = rle_decode(row['segmentation'], original_dim)
        binary_mask = cv2.resize(binary_mask, (SIZE_X, SIZE_Y), interpolation=cv2.INTER_NEAREST)
        unified_mask = np.where(binary_mask == 1, class_index, unified_mask)

    unified_masks.append(unified_mask)

  unified_masks = np.array(unified_masks)

  unified_mask_aligned = []

  for image in train_images:
    unified_mask_aligned.append(np.zeros((SIZE_X, SIZE_Y)))

  for image_id, mask_index in mask_id_map.items():
    image_index = image_id_map[image_id]
    unified_mask_aligned[image_index] = unified_masks[mask_index]

  unified_masks_aligned = np.array(unified_mask_aligned)
  print(f"Created unified masks for {len(unified_masks_aligned)} images")

  return mask_id_map, unified_masks_aligned

def remove_images_without_masks(train_images, train_masks, case_ids, image_id_map):
  """
  Filter our images lacking any mask and update mappings
  
  Parameters
    train_images (np.ndarray): Array of training images
    train_masks (np.ndarray): Array of training masks
    case_ids (list of str): Case ID for each image
    image_id_map (dict): Original mapping of image IDs to indices
    
  Returns
    train_images (np.ndarray): Filtered images array
    train_masks (np.ndarray): Filtered masks array
    case_ids (list of str): Filtered case IDs
    image_id_map (dict) Updated mapping for fitlered images

  """
  valid_mask = np.any(train_masks != 0, axis=(1,2))
  train_images = train_images[valid_mask]
  train_masks = train_masks[valid_mask]
  case_ids = [case_ids[i] for i, keep in enumerate(valid_mask) if keep]
  
  old_indices = np.nonzero(valid_mask)[0]
  old_to_new = {old : new for new, old in enumerate(old_indices)}

  image_id_map = {key : old_to_new[old_index] for key, old_index in image_id_map.items() if old_index in old_to_new}

  print(f"Remaining samples after filtering: {len(train_images)}")

  return train_images, train_masks, case_ids, image_id_map

def create_train_test_split(train_images, train_masks, case_ids, train_size, image_id_map):
  """
  Split dataset into training and test sets with grouping by case IDs
  
  Parameters
    train_images (np.ndarray): Array of training images
    train_masks (np.ndarray): Array of training masks
    case_ids (list of str): Case ID for each image
    train_size (float): Proportion fo data to include in training split
    image_id_map (dict): Original mapping of image IDs to indices
    
  Returns
    train_images (np.ndarray): Training subset of images.
    train_masks (np.ndarray): Training subset of masks.
    test_images (np.ndarray): Test subset of images.
    test_masks (np.ndarray): Test subset of masks.
    train_ids (list of str): Case IDs in training set.
    test_ids (list of str): Case IDs in test set.
    train_id_map (dict): Mapping image IDs to new training indices.
    test_id_map (dict): Mapping image IDs to new test indices

  """
  splitter = GroupShuffleSplit(train_size=train_size, n_splits=1, random_state=42)
  train_idx, test_idx = next(splitter.split(train_images, groups=case_ids))

  train_images, test_images = train_images[train_idx], train_images[test_idx]
  train_masks, test_masks = train_masks[train_idx], train_masks[test_idx]

  train_ids = [f"{case_ids[i]}" for i in train_idx]
  test_ids = [f"{case_ids[i]}" for i in test_idx]

  old_to_new_train = {old : new for new, old in enumerate(train_idx)}
  old_to_new_test = {old : new for new, old in enumerate(test_idx)}

  train_id_map = {key : old_to_new_train[old_index] for key, old_index in image_id_map.items() if old_index in old_to_new_train}
  test_id_map = {key : old_to_new_test[old_index] for key, old_index in image_id_map.items() if old_index in old_to_new_test}


  print(f"Split into {len(train_images)} training images and {len(test_images)} test images.")
  print(f"Intersection in train and test: {len(set(train_ids) & set(test_ids))}")

  return train_images, train_masks, test_images, test_masks, train_ids, test_ids, train_id_map, test_id_map

def create_tf_dataset(images, masks, augment=False):
  """
  Build a TensorFlow dataset from image and mask arrays, with optional data augmentation
  
  Parameters
    images (np.ndarray): Array of images
    masks (np.ndarray): Array of integer mask labels
    augment (boolean, optional): If True, apply data augmentation, with default False
    
  Returns
    dataset (tf.data.Dataset): Batched dataset of (image, one-hot mask) pairs.
  """
  images = np.expand_dims(images, axis=-1)
  images = images.astype(np.float32)
  dataset = tf.data.Dataset.from_tensor_slices((images, masks))
  dataset = dataset.map(lambda image, mask: (image, tf.one_hot(mask, 4)))

  if augment:
    dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)

  dataset = dataset.batch(16)
  return dataset

