import os
import glob
import numpy as np
import nibabel as nib
import argparse
from tqdm import tqdm

# --- Define argparse globally ---
parser = argparse.ArgumentParser(description="Preprocess FC maps with masking, thresholding and optional normalization.")
parser.add_argument('--input_dir', type=str, required=True, help='Path to input folder containing .nii.gz files.')
parser.add_argument('--output_dir', type=str, required=True, help='Path where .npy processed files will be saved.')
parser.add_argument('--mask_path', type=str, required=True, help='Path to GM mask .nii or .nii.gz file.')
parser.add_argument('--threshold', type=float, default=0.2, help='Threshold to apply to voxel values.')
parser.add_argument('--augmented', action='store_true', help='Flag indicating if data is organized with subfolders per subject.')
parser.add_argument('--normalization', action='store_true', help='Apply MinMax normalization on non-zero voxels.')

# --- Function that lists all .nii.gz files from a directory ---
def list_data(input_dir, augmented=False):
    if augmented:
        files_path = sorted(glob.glob(os.path.join(input_dir, '*', '*.nii.gz')))
    else:
        files_path = sorted(glob.glob(os.path.join(input_dir, '*.nii.gz')))

    return files_path

# --- Function for preprocessing the data: loading, thresholding, masking and saving ---
def preprocess_fc_maps(files, output_dir,mask_path, threshold = 0.2,  augmented=False, normalization=None):
    os.makedirs(output_dir, exist_ok=True)

    # Load the mask
    mask = nib.load(mask_path).get_fdata()
    mask = mask != 0

    # Loop over all files
    for file_path in tqdm(files, desc="Preprocessing FC maps"):
        try:
            # Load the file
            img = nib.load(file_path)
            data = img.get_fdata()
            #affine = img.affine

            # Threshold the data
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
                save_path = os.path.join(subject_folder, f"{filename}.processed.npy")
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


# --- Main block ---
if __name__ == "__main__":
    args = parser.parse_args()

    # List all files
    files = list_data(args.input_dir, augmented=args.augmented)

    # Preprocess the files
    preprocess_fc_maps(
        files,
        output_dir=args.output_dir,
        mask_path=args.mask_path,
        threshold=args.threshold,
        augmented=args.augmented,
        normalization=args.normalization
    )


