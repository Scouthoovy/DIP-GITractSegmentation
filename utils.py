def load_png_slices(slice_dir):
    """Loads PNG slices from a directory and reconstructs a 3D volume."""
    slice_paths = sorted([os.path.join(slice_dir, f) for f in os.listdir(slice_dir) if f.endswith(".png")])
    slices = [np.array(Image.open(path)) for path in slice_paths]
    volume = np.stack(slices, axis=-1)
    return volume

def load_png_slices_to_sitk(slice_dir):
    """Loads PNG slices from a directory and returns a SimpleITK image."""
    slice_paths = sorted([os.path.join(slice_dir, f) for f in os.listdir(slice_dir) if f.endswith(".png")])
    slices = [sitk.ReadImage(path) for path in slice_paths]
    volume = sitk.JoinSeries(slices)
    return volume

def load_nifti(nifti_path):
    """Loads a NIfTI file using nibabel and returns the data."""
    img = nib.load(nifti_path)
    return img.get_fdata(), img.affine

def save_nifti(data, affine, output_path):
    """Saves a numpy array as a NIfTI file."""
    img = nib.Nifti1Image(data, affine)
    nib.save(img, output_path)