# utils.py
import sys
import umap
import numpy as np
import os

def run_umap(x_train, x_test=None, n_neighbors=15, n_components=2, min_dist=0.1, metric='euclidean', random_state=42):
    """
    Fits UMAP on x_train. If x_test is provided, applies transform and returns both.
    Otherwise, returns embedding of x_train
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric=metric,
        n_epochs=1000,
        learning_rate=1.0,
        init='spectral',
        min_dist=min_dist,
        spread=1.0,
        low_memory=False,
        set_op_mix_ratio=1.0,
        local_connectivity=1,
        repulsion_strength=1.0,
        negative_sample_rate=5,
        transform_queue_size=4.0,
        random_state=random_state
    )

    x_train_embedded = reducer.fit_transform(x_train)

    if x_test is not None:
        x_test_embedded = reducer.transform(x_test)
        return x_train_embedded, x_test_embedded

    return x_train_embedded


def log_to_file(log_path):
    """
    Redirects stdout to a file.
    Useful for saving all print() outputs during script execution.
    """
    sys.stdout = open(log_path, "w")

def reset_stdout():
    """
    Restores stdout to its default state (the terminal).
    """
    sys.stdout.close()
    sys.stdout = sys.__stdout__

def resolve_split_csv_path(split_dir, group1, group2):
    """
    Resolves the correct split CSV file based on group1 and group2 names,
    supporting both possible name orders.
    """
    fname1 = f"{group1}_{group2}_splitted.csv"
    fname2 = f"{group2}_{group1}_splitted.csv"
    path1 = os.path.join(split_dir, fname1)
    path2 = os.path.join(split_dir, fname2)
    if os.path.exists(path1):
        print(f"\nUsing split file: {path1}\n")
        return path1
    elif os.path.exists(path2):
        print(f"\nUsing split file: {path2}\n")
        return path2
    else:
        raise FileNotFoundError(f"No split CSV found for {group1}, {group2} in {split_dir}")

import os

def build_output_path(base_dir, job_type, dataset_type, umap, umap_all=False):
    """
    Builds the output directory path based on base_dir, dataset_type, job_type and umap flags.
    If job_type is empty, returns only up to 'umap' or dataset folder.
    If both umap and umap_all are True, appends 'all' to the job_type.
    """
    if umap and not job_type:
        return os.path.join(base_dir, dataset_type, "umap")

    prefix = "umap_" if umap else ""
    suffix = "_all" if umap and umap_all else ""

    return os.path.join(base_dir, dataset_type, f"{prefix}{job_type}{suffix}" if job_type else "")

