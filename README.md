# Atrophy Network Mapping Reveals Convergent Network Architecture in Tauopathies


## Overview

This project implements the analysis pipeline for the study *"Atrophy network mapping reveals convergent network architecture and clinical relevance, but limited diagnostic specificity in tauopathies"*.

We investigate the network of brain atrophy in three major neurodegenerative tauopathies:
*   **Alzheimer’s Disease (AD)**
*   **Corticobasal Syndrome (CBS)**
*   **Progressive Supranuclear Palsy (PSP)**

By utilizing **Atrophy Network Mapping (ANM)** on 3T T1-weighted MRI data, this codebase aims to explore whether these distinct clinical syndromes share convergent underlying network failures despite their differences in regional vulnerability and clinical presentation. The project utilizes both **Machine Learning (ML)** and **Deep Learning (CNN)** approaches to analyze diagnostic specificity and clinical relevance.

## Dataset

Data were obtained from the **Alzheimer’s Disease Neuroimaging Initiative (ADNI)** and the **4-Repeat Tauopathy Neuroimaging Initiative (4RTNI)**.

*   **Total Participants**: 249
*   **Cohorts**:
    *   **AD**: 72 biologically confirmed (A+T+) cases.
    *   **CBS**: 51 expert consensus diagnosed cases.
    *   **PSP**: 68 expert consensus diagnosed cases.
    *   **CN**: 58 healthy controls (propensity score matched for age, sex, and education).

## Methodology: W-Score Maps & ANM

The core of the analysis relies on **W-score maps**, which provide a voxel-wise measure of accumulated gray matter (GM) loss (atrophy) normalized against a healthy control population.

### 1. W-Score Calculation
To disentangle pathological atrophy from normal aging effects:
1.  **Normative Model**: A General Linear Model (GLM) is fitted on Healthy Controls: $GM_{density} \sim Age + Sex + Education$.
2.  **W-Score**: For each patient, the observed GM density is compared to the model prediction:
    $$W = \frac{GM_{observed} - GM_{predicted}}{SD_{residuals}}$$
    *   Values $W < -1.96$ indicate significant atrophy (p < 0.05).

### 2. Analytical Workflow

**Objective**: To query the link between functional connectivity and consciousness, we evaluated whether ANM-derived disconnection patterns could support diagnostic classification and reflect clinical severity.

**Features**:
*   **UMAP Embedding**: Dimensionality reduction of voxel-wise W-scores.
*   **Network Metrics**: Atlas-based metrics from Yeo’s 7 cortical networks + subcortical regions.

#### A. Diagnostic Classification
To assess whether atrophy patterns map onto clinical diagnoses:

1.  **Unsupervised Clustering** (Data-driven):
    *   **Algorithm**: K-Means (k=3), initialized to find natural groupings.
    *   **Goal**: Test if natural clusters align with diagnostic labels (AD, CBS, PSP) or clinical stages (e.g., CDR scores).
2.  **Supervised Machine Learning** (Pairwise Classification):
    *   **Algorithm**: Random Forest (RF).
    *   **Features**: Combined Voxel (UMAP) + Network metrics.
    *   **Task**: Pairwise differentiation between diagnostic groups.
3.  **Deep Learning** (High-complexity):
    *   **Algorithm**: 3D Convolutional Neural Networks (CNNs).
    *   **Input**: Raw voxel-wise W-scores.

#### B. Clinical Correlates
To test clinical relevance, we associated ANM features with clinical severity (MMSE, CDR-SB) using **OLS regression** both on the full sample and within diagnostic groups.

## Project Structure

```
ANM_Verona/
├── data/                   # Raw and processed data (ignored by git)
│   ├── FCmaps/             # Original W-score maps
│   ├── FCmaps_processed/   # Preprocessed .npy files for training
│   └── dataframes/         # Tabular data (features, networks)
├── resources/              # Metadata and configuration resources
│   ├── split/              # ML train/test split CSVs
│   ├── split_cnn/          # CNN train/test split CSVs
│   ├── metadata/           # Patient metadata (labels, demographics)
│   └── masks/              # Brain masks (GM mask, etc.)
├── results/                # Analysis outputs
│   ├── ml_analysis/        # ML classification results
│   ├── runs/               # CNN training checkpoints and logs
│   └── tuning/             # Hyperparameter tuning results
├── src/                    # Source code
│   ├── cnn/                # 3D CNN model architectures and dataset loaders
│   ├── training/           # CNN training loops and runners
│   ├── analysis/           # ML analysis scripts (classification, clustering)
│   ├── preprocessing/      # Data processing scripts (3D mapping, splitting)
│   ├── augmentation/       # Data augmentation logic
│   └── config/             # Configuration JSONs
└── notebooks/              # Jupyter notebooks for exploration and viz
```

## Documentation

For detailed instructions on how to run the code, see **[USAGE.md](USAGE.md)**.
