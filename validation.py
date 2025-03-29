def rle_decode(rle_string, original_height, original_width, original_depth):
    """
    Decodes a Run-Length Encoded (RLE) string into a 3D NumPy array.

    Args:
        rle_string (str): The RLE string to decode.
        original_height (int): The original height of the image.
        original_width (int): The original width of the image.
        original_depth (int): The original depth (number of slices) of the image.

    Returns:
        numpy.ndarray: A 3D NumPy array representing the decoded segmentation mask.
    """

    if not isinstance(rle_string, str):
        return np.zeros((original_height, original_width, original_depth), dtype=np.uint8)

    s = rle_string.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(original_height * original_width * original_depth, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((original_height, original_width, original_depth))

def load_segmentations(csv_path, case_id, target_height, target_width, target_depth):
    """
    Loads segmentation masks from a CSV file (id, class, segmentation format) and resizes them.
    Handles missing segmentation data and multiple classes.

    Args:
        csv_path (str): Path to the CSV file.
        case_id (str): The case ID to filter segmentations for.
        target_height (int): Target height for resized segmentations.
        target_width (int): Target width for resized segmentations.
        target_depth (int): Target depth for resized segmentations.

    Returns:
        dict: A dictionary where keys are image IDs and values are dictionaries
              containing the resized segmentation masks for each class.
    """

    segmentations = {}
    original_height = 266
    original_width = 266
    original_depth = 266

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}. Please check the path.")
        return segmentations
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file at {csv_path} is empty.")
        return segmentations
    except Exception as e:
        print(f"An unexpected error occurred while reading the CSV: {e}")
        return segmentations

    # --- Filter by case_id ---
    df = df[df['id'].str.contains(case_id)]

    # --- Group by 'id' ---
    for img_id, group_df in df.groupby('id'):
        segmentations[img_id] = {}  # Initialize a dictionary for each image

        for _, row in group_df.iterrows():
            class_name = row['class']
            rle_string = str(row['segmentation']).strip()

            if rle_string:
                rle_string = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', rle_string)

                print(f"Decoding RLE for {img_id}, class {class_name}: {rle_string}")
                decoded_segmentation = rle_decode(rle_string, original_height, original_width, original_depth)
                print(f"Decoded segmentation shape: {decoded_segmentation.shape}")
                print(f"Decoded segmentation max: {np.max(decoded_segmentation)}")

                resized_segmentation = resize(decoded_segmentation,
                                               (target_height, target_width, target_depth),
                                               order=0,
                                               preserve_range=True,
                                               anti_aliasing=False).astype(np.uint8)
                segmentations[img_id][class_name] = resized_segmentation
            else:
                print(f"Warning: No valid segmentation data for {img_id}, class {class_name}. Skipping.")

    return segmentations

def register_image(moving_image_path, fixed_image_path, output_path):
    """Registers a moving image to a fixed image using SimpleITK."""
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation()
    registration_method.SetInitialTransform(sitk.TranslationTransform(fixed_image.GetDimension()))
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
    registration_method.SetInterpolator(sitk.sitkLinear)

    final_transform = registration_method.Execute(fixed_image, moving_image)
    registered_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0)
    sitk.WriteImage(registered_image, output_path)
    return registered_image

def calculate_dice_sitk(image1, image2):
    """Calculates Dice score using SimpleITK."""
    image1 = sitk.Cast(image1, sitk.sitkUInt8)
    image2 = sitk.Cast(image2, sitk.sitkUInt8)
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(image1, image2)
    return overlap_measures_filter.GetDiceCoefficient()