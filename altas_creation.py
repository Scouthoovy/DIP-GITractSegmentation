# --- atlas_creation.py ---
def create_intra_case_atlas(case_day_nifti_paths, output_path):
    """Creates an atlas from multiple 'day' scans of the same case (using preprocessed NIfTIs)."""
    num_days = len(case_day_nifti_paths)
    if num_days < 2:
        raise ValueError("At least two preprocessed NIfTI files are needed to create an atlas.")

    # 1. Load Volumes
    volumes = [sitk.ReadImage(nifti_path, sitk.sitkFloat32) for nifti_path in case_day_nifti_paths]

    # 2. Initial Template (e.g., first day)
    atlas = sitk.Image(volumes[0].GetSize(), sitk.sitkFloat32)  # Create a new SimpleITK image
    atlas.CopyInformation(volumes[0])  # Copy geometry information
    atlas = volumes[0] # Assign the first volume

    # 3. Iterative Registration and Averaging
    for iteration in range(1):
        print(f"Atlas Creation - Iteration {iteration + 1}/5")
        sum_of_registered = sitk.Image(atlas.GetSize(), sitk.sitkFloat32)
        sum_of_registered.CopyInformation(atlas)

        for i in range(num_days):
            if i != 0:  # Skip initial template
                moving_image = volumes[i]  # Use the SimpleITK image directly

                registration_method = sitk.ImageRegistrationMethod()
                registration_method.SetMetricAsMattesMutualInformation()
                registration_method.SetInitialTransform(sitk.TranslationTransform(atlas.GetDimension()))
                registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
                registration_method.SetInterpolator(sitk.sitkLinear)

                final_transform = registration_method.Execute(atlas, moving_image)
                registered_image = sitk.Resample(moving_image, atlas, final_transform, sitk.sitkLinear, 0.0)
                sum_of_registered = sum_of_registered + registered_image
            else:
                sum_of_registered = sum_of_registered + volumes[i]

        atlas = sitk.Cast(sum_of_registered / num_days, sitk.sitkFloat32) # Ensure atlas is float32

    # 4. Save
    sitk.WriteImage(atlas, output_path)