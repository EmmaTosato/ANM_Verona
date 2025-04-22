# --- Libraries ---
import os
import pandas as pd
import glob
import random
import shutil
import argparse
import csv
from sklearn.model_selection import train_test_split

# --- Variables ---
parser = argparse.ArgumentParser(description="Generate augmented FC maps")
parser.add_argument('--n_augmentations', type=int, default=10, help="How many maps per subject")
parser.add_argument('--subset_size', type=int, default=17, help="How many HCP per augmentation")
parser.add_argument('--labels_file', type=str, default="./data_utils/labels.xlsx")
parser.add_argument('--hcp_root', type=str, default="./hcp_subjects/")
parser.add_argument('--output_root', type=str, default="./data/FCmaps_mean/")
parser.add_argument('--csv_out', type=str, default="./data_utils/all_augmented_info.csv")
args = parser.parse_args()

# --- Config ---
# Seed for reproducibility
random.seed(42)
hcp_subjects = 173
df_labels = pd.read_excel(args.labels_file)
subjects = df_labels['ID'].tolist()

# --- Track info in CSV ---
csv_rows = [("subject", "augmentation", "hcp_subset")]

# --- Loop over each subject ---
for subject in subjects:
    print(f"Augmenting subject: {subject}")

    # Output directory
    subj_out_dir = os.path.join(args.output_root)
    os.makedirs(subj_out_dir, exist_ok=True)

    # Every subset of 17 HCPs needs to be unique
    used_hcp = set()

    for aug in range(1, args.n_augmentations + 1):
        # A list of HCPs not yet used for this subject
        available_hcp = list(set(hcp_subjects) - used_hcp)

        # If we run out of unique HCPs, reset the used set (theorically this should not happen)
        if len(available_hcp) < args.subset_size:
            used_hcp = set()
            available_hcp = list(set(hcp_subjects))

        # Randomly sample HCPs for this augmentation
        subset = random.sample(available_hcp, args.subset_size)
        used_hcp.update(subset)

        # Create a temporary folder
        aug_dir = os.path.join("./tmp_aug", subject, f"aug_{aug}")
        os.makedirs(aug_dir, exist_ok=True)

        for hcp_id in subset:
            sca_files = glob.glob(os.path.join(args.hcp_root, hcp_id, "*.SCA_result.nii.gz"))
            for file in sca_files:
                shutil.copy(file, aug_dir)

        merged_file = os.path.join(aug_dir, f"merged.{subject}.nii.gz")
        mean_file = os.path.join(subj_out_dir, f"{subject}_FDC_{aug}.nii.gz")

        os.system(f"fslmerge -t {merged_file} {aug_dir}/*.SCA_result.nii.gz")
        os.system(f"fslmaths {merged_file} -Tmean {mean_file}")

        csv_rows.append((subject, aug, ",".join(subset)))

# --- Save final tracking CSV ---
with open(args.csv_out, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)

print("Completed augmentation for all subjects.")