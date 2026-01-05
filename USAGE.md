# Usage Guide

This document explains how to use the ANM Verona codebase for data processing and analysis.

## 1. Environment Setup

Ensure you have the required dependencies installed. It is recommended to use a virtual environment or conda environment.

```bash
pip install -r requirements.txt
```

## 2. Configuration

All configuration files are located in `src/config/`.

### Machine Learning Config (`ml_config.json`, `ml_grid.json`)
*   **`ml_config.json`**: Controls the ML analysis pipeline (`src/analysis/run_all_classifications.py`).
    *   `dataset_type`: Inputs to use (e.g., "networks", "fdc").
    *   `task_type`: "classification" or "regression".
    *   `umap`: Boolean to enable UMAP dimensionality reduction.
    *   `seeds`: List of random seeds for reproducibility.
*   **`ml_grid.json`**: Defines hyperparameter search spaces for models (RandomForest, SVM, etc.).
*   **`ml_paths.json`**: Points to input data directories (e.g., `data/dataframes`) and metadata.

### CNN Config (`cnn_config.json`, `cnn_grid.json`)
*   **`cnn_config.json`**: Controls the Deep Learning training pipeline (`src/training/run_train.py`).
    *   `paths`: Directories for augmented data, checkpoints, and splits.
    *   `training`: Hyperparameters like `epochs`, `batch_size`, `lr`, `model_type` (resnet, densenet, vgg16).
    *   `experiment`: Defines the comparison groups (e.g., `group1: "PSP"`, `group2: "CBS"`).
*   **`cnn_grid.json`**: Hyperparameters for tuning CNNs.

## 3. Data Preprocessing

The raw `.nii.gz` W-score maps usually need to be processed into `.npy` format for efficient loading during CNN training.

### Processing 3D Maps
Run the `process3d.py` script:

```bash
python src/preprocessing/process3d.py
```
This script uses settings from `src/preprocessing/config.py` (which loads `ml_config.json` and `ml_paths.json`) to threshold, mask, and save the maps.

### Data Augmentation
To generate augmented data using the HCP-based pipeline:
```bash
python src/augmentation/augmentation.py --subject_id <ID> --dataset_dir <DIR> ...
```
*(See `src/augmentation/augmentation.sh` for batch processing usage).*

## 4. Running Analyses

### Machine Learning (Classification)
To run the full suite of ML classifiers (SVM, RF, GBM, KNN) on the tabular data:

```bash
python src/analysis/run_all_classifications.py
```
This script iterates through the seeds defined in `ml_config.json` and outputs results to `results/ml_analysis/`.

### CNN Training
To train a 3D CNN model:

```bash
python src/training/run_train.py
```
This script:
1.  Reads `src/config/cnn_config.json`.
2.  Loads data defined in `data_dir` (or `data_dir_augmented`).
3.  Runs cross-validation or a single training run.
4.  Saves best models and logs to `results/runs/`.

To run **Hyperparameter Tuning** for CNNs:
```bash
python src/training/hyper_tuning.py
```

## 5. Results

*   **ML Results**: Check `results/ml_analysis/` for CSV summaries, confusion matrices, and UMAP plots.
*   **CNN Results**: Check `results/runs/` for training logs, loss curves, and model checkpoints (`.pt` files).
