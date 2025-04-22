import os
import random
import argparse
import glob
import csv
import subprocess

# Args
parser = argparse.ArgumentParser(description="Generate augmented FC maps for a single patient")
parser.add_argument("--subject_id", type=str, required=True)
parser.add_argument("--dataset_dir", type=str, required=True)
parser.add_argument("--hcp_list", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--csv_out", type=str, required=True)
parser.add_argument("--n_augmentations", type=int, default=10)
parser.add_argument("--subset_size", type=int, default=17)
args = parser.parse_args()

# Read HCP list
with open(args.hcp_list, "r") as f:
    hcp_pool = [line.strip() for line in f.readlines()]
random.seed(42)

# Generate fixed HCP subsets
hcp_subsets = [random.sample(hcp_pool, args.subset_size) for _ in range(args.n_augmentations)]

# Setup paths
subject_dir = os.path.join(args.dataset_dir, args.subject_id)
subject_outdir = os.path.join(args.output_dir, args.subject_id)
os.makedirs(subject_outdir, exist_ok=True)

# Initialize CSV tracking
csv_exists = os.path.exists(args.csv_out)
with open(args.csv_out, "a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    if not csv_exists:
        writer.writerow(["subject", "augmentation", "hcp_subset"])

    for i, hcp_subset in enumerate(hcp_subsets, start=1):
        # Filter only files from selected HCPs
        files_to_merge = []
        hcp_ids_used = []
        for hcp_id in hcp_subset:
            matches = glob.glob(os.path.join(subject_dir, f"*{hcp_id}*.SCA_result.nii.gz"))
            if matches:
                files_to_merge.extend(matches)
                hcp_ids_used.append(hcp_id)

        if len(files_to_merge) == 0:
            print(f"[WARNING] No matching SCA files found for {args.subject_id} aug{i}")
            continue

        merged_file = os.path.join(subject_outdir, f"{args.subject_id}.merged.aug{i}.nii.gz")
        mean_file = os.path.join(subject_outdir, f"{args.subject_id}.FDC.aug{i}.nii.gz")

        # Merge and average
        merge_cmd = ["fslmerge", "-t", merged_file] + files_to_merge
        mean_cmd = ["fslmaths", merged_file, "-Tmean", mean_file]

        print(f"Running fslmerge for {args.subject_id} aug{i}")
        subprocess.run(merge_cmd, check=True)
        subprocess.run(mean_cmd, check=True)

        # Write tracking
        writer.writerow([args.subject_id, i, ",".join(hcp_ids_used)])
