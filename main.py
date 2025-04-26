from utilities import rle_decode, plot_random_image_mask_overlay, augment_data, build_volumes, extract_masks, get_predicted_masks
from preprocessing import preprocess_pipeline, apply_wavelet_denoising, apply_normalization, apply_clahe
from create_dataset import create_train_images, create_train_masks, remove_images_without_masks, create_tf_dataset, create_train_test_split
from model import conv_block, encoder_block, decoder_block, build_unet
from metrics import mean_dice, mean_hd, hausdorff_distance, get_callbacks
import tensorflow as tf




def main():
	IMAGE_DIRECTORY = 'path/to/images'
	MASK_CSV_PATH = 'path/to/masks.csv'
	SIZE_X = 256
	SIZE_Y = 256
	class_map = {
		'background' : 0,
		'large_bowel' : 1,
		'small_bowel' : 2,
		'stomach' : 3
	}


	image_id_map, train_images, train_images_dims, case_ids = create_train_images(IMAGE_DIRECTORY, SIZE_X, SIZE_Y)
	mask_id_map, train_masks = create_train_masks(MASK_CSV_PATH, SIZE_X, SIZE_Y, image_id_map, train_images, train_images_dims, class_map)
	train_images_reduced, train_masks_reduced, case_ids_reduced, image_id_map_reduced = remove_images_without_masks(train_images, train_masks, case_ids, image_id_map)
	X_train, y_train, X_test, y_test, train_case_ids, test_case_ids, train_id_map, test_id_map = create_train_test_split(train_images_reduced, train_masks_reduced, case_ids_reduced, 0.8, image_id_map_reduced)
	X_val, y_val, X_test, y_test, val_case_ids, test_case_ids, val_id_map, test_id_map = create_train_test_split(X_test, y_test, test_case_ids, 0.5, test_id_map)

	X_train_processed = preprocess_pipeline(X_train, bias_cor=False, wavelet=True, CLAHE=True, normalization=True)
	X_val_processed = preprocess_pipeline(X_val, bias_cor=False, wavelet=True, CLAHE=True, normalization=True)
	X_test_processed = preprocess_pipeline(X_test, bias_cor=False, wavelet=True, CLAHE=True, normalization=True)

	train_ds_processed = create_tf_dataset(X_train_processed, y_train, augment=False)
	train_ds_unprocessed = create_tf_dataset(X_train, y_train, augment=False)
	val_ds_unprocessed = create_tf_dataset(X_val, y_val, augment=False)
	val_ds_processed = create_tf_dataset(X_val_processed, y_val, augment=False)
	test_ds_unprocessed = create_tf_dataset(X_test, y_test, augment=False)
	test_ds_processed = create_tf_dataset(X_test_processed, y_test, augment=False)
	
	### Model Trained with Un-processed Images ###
	model_unprocessed = build_unet((SIZE_X, SIZE_Y, 1), len(class_map))
	model_unprocessed.compile(optimizer='adam', loss=tf.keras.losses.CategoricalFocalCrossentropy(), metrics=['accuracy',
																										tf.keras.metrics.CategoricalAccuracy(),
																										tf.keras.metrics.OneHotMeanIoU(num_classes=4, name='mean_iou'),
																										tf.keras.metrics.OneHotIoU(num_classes=4, target_class_ids=[0], name='bg_iou'),
																										tf.keras.metrics.OneHotIoU(num_classes=4, target_class_ids=[1,2,3], name='class_iou')])
	### Model trained with Processed Images ###
	model_processed = build_unet((SIZE_X, SIZE_Y, 1), len(class_map))
	model_processed.compile(optimizer='adam', loss=tf.keras.losses.CategoricalFocalCrossentropy(), metrics=['accuracy',
																										tf.keras.metrics.CategoricalAccuracy(),
																										tf.keras.metrics.OneHotMeanIoU(num_classes=4, name='mean_iou'),
																										tf.keras.metrics.OneHotIoU(num_classes=4, target_class_ids=[0], name='bg_iou'),
																										tf.keras.metrics.OneHotIoU(num_classes=4, target_class_ids=[1,2,3], name='class_iou')])			
	history_unprocessed = model_unprocessed.fit(train_ds_unprocessed, validation_data=val_ds_unprocessed, epochs=100, verbose=1, callbacks = get_callbacks(model_name='unet_unprocessed_v2'))
	history_processed = model_processed.fit(train_ds_processed, validation_data=val_ds_processed, epochs=100, verbose=1, callbacks = get_callbacks(model_name='unet_processed_v2'))

	unprocessed_preds = get_predicted_masks(model_unprocessed, test_ds_unprocessed)
	processed_preds = get_predicted_masks(model_processed, test_ds_processed)

	unprocessed_volumes = build_volumes(unprocessed_preds, test_id_map)
	processed_volumes = build_volumes(processed_preds, test_id_map)
	test_volumes = build_volumes(y_test, test_id_map)

	dice_unprocessed, dice_std_unprocessed = mean_dice(extract_masks(test_ds_unprocessed), unprocessed_preds)
	dice_processed, dice_std_processed = mean_dice(extract_masks(test_ds_processed), processed_preds)

	hd_unprocessed, hd_std_unprocessed = mean_hd(test_volumes, unprocessed_volumes)
	hd_processed, hd_std_processed = mean_hd(test_volumes, processed_volumes)

	print(f"The mean Dice score for the unprocessed U-Net model is: {dice_unprocessed} pm {dice_std_unprocessed}")
	print(f"The mean Dice score for the processed U-Net model is: {dice_processed} pm {dice_std_processed}")

	print(f"The mean 3D Hausdorff distance for the unprocessed U-Net model is {hd_unprocessed} pm {hd_std_unprocessed}")
	print(f"The mean 3D Hausdorff distance for the processed U-Net model is {hd_processed} pm {hd_std_processed}")

	print(f"The weighted dice-hausdorff metric for unprocessed is: {0.4*dice_unprocessed + 0.6*(1-hd_unprocessed)} and for processed is: {0.4*dice_processed + 0.6*(1-hd_processed)}"
	
	model_unprocessed.save('models/unet_unprocessed.h5')
	model_processed.save('models/unet_processed.h5')
	
	print(f"Complete!")


if __name__ == "__main__":
    main()
																										