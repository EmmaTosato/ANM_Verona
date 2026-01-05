# process3d.py
import os
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm
from preprocessing.config import ConfigLoader


def list_data(input_dir, augmented=False):
    """
    Returns a list of all .nii.gz files in the given directory.
    If augmented=True, searches subfolders for each subject.
    """
    if augmented:
        files_path = sorted(glob.glob(os.path.join(input_dir, '*', '*.nii.gz')))
    else:
        files_path = sorted(glob.glob(os.path.join(input_dir, '*.nii.gz')))

    return files_path

def preprocess_fc_maps(maps_files, output_dir,mask_path, threshold = 0.2,  augmented=False, normalization=None):
    """
        Preprocess FC maps: load image, apply thresholding and masking, optional normalization, and save as .npy.
        """
    os.makedirs(output_dir, exist_ok=True)

    # Load the mask
    mask = nib.load(mask_path).get_fdata()
    mask = mask != 0

    # Loop over all files
    for file_path in tqdm(maps_files, desc="Preprocessing FC maps"):
        try:
            # Load the file
            img = nib.load(file_path)
            data = img.get_fdata()
            #affine = img.affine

            # Threshold the data
            if threshold is not None:
                data[data < threshold] = 0

            # Masking
            data[~mask] = 0

            # Normalization
            if normalization:
                nonzero = data[data != 0]
                if nonzero.size > 0:
                    min_val = nonzero.min()
                    max_val = nonzero.max()
                    if max_val != min_val:
                        data[data != 0] = (data[data != 0] - min_val) / (max_val - min_val)

            # Saving
            if augmented:
                subj_id = os.path.basename(os.path.dirname(file_path))
                filename = os.path.basename(file_path).replace('.nii.gz', '')
                subject_folder = os.path.join(output_dir, subj_id)
                os.makedirs(subject_folder, exist_ok=True)
                save_path = os.path.join(str(subject_folder), f"{filename}.processed.npy")
                #save_affine_path = os.path.join(subject_folder, f"{filename}.affine.npy")
            else:
                filename = os.path.basename(file_path).replace('.FDC.nii.gz', '')
                save_path = os.path.join(output_dir, f"{filename}.processed.npy")
                #save_affine_path = os.path.join(output_dir, f"{filename}.affine.npy")

            np.save(save_path, data.astype(np.float32))
            #np.save(save_affine_path, affine)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue


# Main block
if __name__ == "__main__":
    # Load configuration parameters
    loader = ConfigLoader()
    args = loader.args

    # List input files from the specified directory
    files = list_data(args["dir_FCmaps"], augmented=args["augmentation"])

    # Run the preprocessing pipeline
    preprocess_fc_maps(
        files,
        output_dir=args['dir_FC3Dmaps_processed'],
        mask_path=args["gm_mask_path"],
        threshold=args["thresholding"],
        augmented=args["augmentation"],
        normalization=args["normalization"]
    )


