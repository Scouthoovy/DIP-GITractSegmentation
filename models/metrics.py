import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from scipy.spatial.distance import directed_hausdorff

def mean_dice(y_true_slices, y_pred_slices):
  """
  Compute mean and standard deviation of Dice coefficient across all slices.
  
  Parameters
    y_true_slices (iterable of np.ndarray): True binary masks for each slice
    y_pred_slices (iterable of np.ndarray): Predicted binary masks for each slice
    
  Returns
    mean_dice (float): Mean dice coefficient
    std_dice (float): Standard deviation of Dice coefficients
  """
  scores = []
  for true_mask, pred_mask in zip(y_true_slices, y_pred_slices):
    t = true_mask.astype(bool)
    p = pred_mask.astype(bool)
    intersection = np.logical_and(t, p).sum()
    total = t.sum() + p.sum()
    dice = (2.*intersection) / total
    scores.append(dice)
  return np.mean(scores), np.std(scores)

def mean_hd(y_true_vol, y_pred_vol):
  """
  Compute mean and std of Hausdorff distance across volumes

  Parameters
    y_true_vol (dict): Mapping case_id to true volume mask as np.ndarray
    y_pred_vol (dict): Mapping case_id to predicted volume mask as np.ndarray
    
  Returns
    mean_hd (float): Mean Hausdorff distance across cases
    std_hd (float): Standard deviation of Hausdorff distances across cases
  """
  scores = {}
  for case_id, true_vol in y_true_vol.items():
    pred_vol = y_pred_vol[case_id]
    scores[case_id] = hausdorff_distance(true_vol, pred_vol)
  
  scores = np.array(list(scores.values()))
  mean_score = np.mean(scores)
  std_score = np.std(scores)

  return mean_score, std_score

def hausdorff_distance(y_true_vol, y_pred_vol):
  """
  Compute normalized Hausdorff distance between two volumes
  
  Parameters
    y_true_vol (np.ndarray): Ground truth binary volume mask
    y_pred_vol (np.ndarray): Predicted binary volume mask
  
  Returns
    hd (float): Normalized Hausdorff distance
  """
  A = np.argwhere(y_pred_vol)
  B = np.argwhere(y_true_vol)

  dims = np.array(y_pred_vol.shape) - 1
  A = A.astype(float) / dims
  B = B.astype(float) / dims

  d1 = directed_hausdorff(A, B)[0]
  d2 = directed_hausdorff(B, A)[0]

  return max(d1, d2) / np.sqrt(3)

    

def get_callbacks(model_name="unet"):
  """
  Create list of Keras callbacks for training
  
  Parameters
    model_name (str): Name for model, log, and check directories, with default 'unet'
    
  Returns
    callbacks (list): list of keras callbacks
  """
    
  log_dir = f"logs/{model_name}"


  callbacks = [
      TensorBoard(
          log_dir=log_dir,
          histogram_freq=1,
          write_graph=True,
          write_images=True,
          update_freq='epoch'
      ),
      ModelCheckpoint(
          filepath=f'models/{model_name}_best.h5',
          monitor='val_class_iou',
          mode='max',
          save_best_only=True,
          verbose=1
      ),

      EarlyStopping(
          monitor='val_class_iou',
          mode='max',
          patience=10,
          verbose=1,
          restore_best_weights=True
      ),

      ReduceLROnPlateau(
          monitor='val_class_iou',
          factor=0.1,
          patience=5,
          min_lr=1e-6,
          verbose=1
      )
  ]

  return callbacks
