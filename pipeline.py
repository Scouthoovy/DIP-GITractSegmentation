def run_pipeline(config):
    """Main pipeline function."""

    base_dir = config['base_dir']
    output_dir = config['output_dir']
    csv_path = config['csv_path']
    case_id = config['case_id']
    height = config['height']
    width = config['width']
    depth = config['depth']

    target_height = config.get('target_height', 128)  # Get from config or default to 128
    target_width = config.get('target_width', 128)
    target_depth = config.get('target_depth', 128)

    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)

    # --- Preprocessing ---
    if config['run_preprocessing']:
        print("--- Preprocessing ---")
        case_day_dirs = [os.path.join(base_dir, case_id, d, "scans")
                        for d in os.listdir(os.path.join(base_dir, case_id)) if "day" in d]

        for day_dir in case_day_dirs:
            print(f"Processing: {day_dir}")
            day_name = os.path.basename(os.path.dirname(day_dir))
            nifti_temp = os.path.join(output_dir, f"{case_id}_{day_name}_temp.nii.gz")
            nifti_bias_corrected = os.path.join(output_dir, f"{case_id}_{day_name}_bias_corrected.nii.gz")
            nifti_normalized = os.path.join(output_dir, f"{case_id}_{day_name}_normalized.nii.gz")
            nifti_cropped = os.path.join(output_dir, f"{case_id}_{day_name}_cropped.nii.gz")

            png_to_nifti(day_dir, nifti_temp)
            correct_bias_field(nifti_temp, nifti_bias_corrected)
            normalize_intensity(nifti_bias_corrected, nifti_normalized, method="minmax")
            crop_roi(nifti_normalized, nifti_cropped)

            # Clean up temporary file
            if os.path.exists(nifti_temp):
                os.remove(nifti_temp)

    # --- Atlas Creation ---
    if config['run_atlas_creation']:
        print("--- Atlas Creation ---")
        case_day_nifti_dirs = [os.path.join(output_dir, f"{case_id}_{os.path.basename(os.path.dirname(d))}_normalized.nii.gz")
                            for d in case_day_dirs]
        atlas_output_path = os.path.join(output_dir, f"{case_id}_atlas.nii.gz")
        create_intra_case_atlas(case_day_nifti_dirs, atlas_output_path)

    # --- Validation ---
    if config['run_validation']:
        print("--- Validation ---")
        atlas_path = os.path.join(output_dir, f"{case_id}_atlas.nii.gz")
        sample_day_name = os.path.basename(os.listdir(os.path.join(base_dir, case_id))[0])
        sample_day_nifti_path_input = os.path.join(output_dir, f"{case_id}_{sample_day_name}_normalized.nii.gz")
        sample_nifti_path = os.path.join(output_dir, f"{case_id}_sample_registered.nii.gz")

        # Register a sample day to the atlas
        register_image(sample_day_nifti_path_input, atlas_path, sample_nifti_path)

        # Load segmentations 
        segmentations = load_segmentations(csv_path, case_id, target_height, target_width, target_depth)

        # Find a matching segmentation
        sample_segmentation_key = None
        for key in segmentations.keys():
            if case_id in key:
                sample_segmentation_key = key
                break

        if sample_segmentation_key:
            sample_segmentation = segmentations[sample_segmentation_key]
            # Load the registered sample and atlas as SimpleITK images
            registered_image_sitk = sitk.ReadImage(sample_nifti_path)
            atlas_image_sitk = sitk.ReadImage(atlas_path)
            
            atlas_array = sitk.GetArrayFromImage(atlas_image_sitk)  # Get NumPy array
            resized_atlas_array = resize(atlas_array, (target_height, target_width, target_depth),
                                        order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
            atlas_image_sitk = sitk.GetImageFromArray(resized_atlas_array)  # Create SimpleITK image

            # Calculate Dice Score
            atlas_array = sitk.GetArrayViewFromImage(atlas_image_sitk)
            atlas_mask = (atlas_array > 0).astype(np.uint8)  # Simple threshold
            atlas_mask_sitk = sitk.GetImageFromArray(atlas_mask)

            dice_score = calculate_dice_sitk(sitk.GetImageFromArray(sample_segmentation.astype(np.uint8)), atlas_mask_sitk)
            print(f"Dice Score: {dice_score}")
        else:
            print(f"Warning: No matching segmentation found for {case_id}")