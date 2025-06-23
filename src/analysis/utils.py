# utils.py
import sys
import umap
import numpy as np

np.random.seed(42)

def run_umap(x_input, n_neighbors=15, n_components=2, min_dist=0.1, metric='euclidean'):
    """
    Computes UMAP embedding on input data.
    Returns the embedded 2D coordinates.
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
        random_state=42
    )
    return reducer.fit_transform(x_input)


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
