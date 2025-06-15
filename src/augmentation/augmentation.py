# Libraries
import os
import random
import argparse
import csv
import time
import subprocess

# Argument parsing
parser = argparse.ArgumentParser(description="Generate augmented FC maps for a single patient")
parser.add_argument("--subject_id", type=str, required=True)
parser.add_argument("--dataset_dir", type=str, required=True)
parser.add_argument("--hcp_list", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--csv_out", type=str, required=True)
parser.add_argument("--n_augmentations", type=int, default=10)
parser.add_argument("--subset_size", type=int, default=17)
parser.add_argument("--index", type=int, required=True)
args = parser.parse_args()

# Read HCP list
with open(args.hcp_list, "r") as f:
    hcp_pool = [line.strip() for line in f if line.strip()]

# Ensure reproducibility
random.seed(42)

# Check: enough HCPs to create disjoint subsets
total_needed = args.n_augmentations * args.subset_size
if len(hcp_pool) < total_needed:
    raise ValueError(f"Not enough HCPs ({len(hcp_pool)}) for {args.n_augmentations} disjoint subsets of size {args.subset_size}.")

# Generate disjoint HCP subsets
shuffled_hcp = random.sample(hcp_pool, total_needed)
hcp_subsets = [
    shuffled_hcp[i * args.subset_size : (i + 1) * args.subset_size]
    for i in range(args.n_augmentations)
]

# Validate again that that subsets are disjoint
flat_list = [hcp for subset in hcp_subsets for hcp in subset]
if len(flat_list) != len(set(flat_list)):
    raise ValueError("HCP subsets are not disjoint!")

# Define subject paths
subject_dir = os.path.join(args.dataset_dir, args.subject_id)
subject_outdir = os.path.join(args.output_dir, args.subject_id)
os.makedirs(subject_outdir, exist_ok=True)

# Prepare CSV writer
csv_exists = os.path.exists(args.csv_out)
with open(args.csv_out, "a", newline="") as csvfile:
    writer = csv.writer(csvfile)

    # If the scv file doesn't exit, prepare it
    if not csv_exists:
        writer.writerow(["subject", "augmentation", "hcp_subset", "missing_hcps"])

    # Begin preprocessing this subject
    start_time = time.time()

    # Loop over each augmentation
    for i, hcp_subset in enumerate(hcp_subsets, start=1):
        files_to_merge = []
        hcp_ids_used = []

        for hcp_id in hcp_subset:
            file_name = f"{args.subject_id}.{hcp_id}.SCA_result.nii.gz"
            file_path = os.path.join(str(subject_dir), str(file_name))
            if os.path.exists(file_path):
                files_to_merge.append(file_path)
                hcp_ids_used.append(hcp_id)

        if len(files_to_merge) < args.subset_size:
            print(f"[ERROR] Augmentation {i} NOT succeed: only {len(files_to_merge)}/{args.subset_size} files found")
            continue

        merged_file = os.path.join(str(subject_outdir), f"{args.subject_id}.merged.aug{i}.nii.gz")
        mean_file = os.path.join(str(subject_outdir), f"{args.subject_id}.FDC.aug{i}.nii.gz")

        # Try and except
        try:
            subprocess.run(["fslmerge", "-t", merged_file] + files_to_merge, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Subject {args.subject_id}")
            print(f"[ERROR] Augmentation {i} NOT succeed: fslmerge failed with error {e}")
            continue

        try:
            subprocess.run(["fslmaths", merged_file, "-Tmean", mean_file], check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Subject {args.subject_id}")
            print(f"[ERROR] Augmentation {i} NOT succeed: fslmaths failed with error {e}")
            continue

        if not os.path.exists(mean_file):
            print(f"[ERROR] Subject {args.subject_id}")
            print(f"[ERROR] Augmentation {i} NOT succeed: output file {mean_file} was not created")
            continue

        # Remove merged files
        if os.path.exists(merged_file):
            os.remove(merged_file)

        # Write to csv
        hcp_ids_used = sorted(hcp_ids_used)
        missing_hcps = [hcp for hcp in hcp_subset if hcp not in hcp_ids_used]
        writer.writerow([
            args.subject_id,
            i,
            ",".join(hcp_ids_used),
            ",".join(missing_hcps) if missing_hcps else ""
        ])

    # Print every subject
    end_time = time.time()
    elapsed = round(end_time - start_time, 2)
    print(f"\nProcessed subject: {args.subject_id} with time {elapsed}\n")



