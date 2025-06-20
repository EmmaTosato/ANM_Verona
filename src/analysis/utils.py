import os


def threshold_prefix(threshold):
    """Return prefix string and log name based on threshold value."""
    if threshold in [0.1, 0.2]:
        return f"{threshold} Threshold", f"log_{threshold}_threshold"
    return "No Threshold", "log_no_threshold"


def ensure_dir(path):
    """Create directory if it doesn't exist and return the path."""
    os.makedirs(path, exist_ok=True)
    return path

