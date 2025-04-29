#!/bin/bash

# ---- Pre-processing -----
# Set your real paths below
# Remove --augmented if data is not augmented
# Remove --normalization if you don't want MinMax scaling
python preprocessing.py \
    --input_dir "/Users/emmatosato/Documents/PhD/ANM_Verona/data/FCmaps_augmented/" \
    --output_dir "/Users/emmatosato/Documents/PhD/ANM_Verona/data/FCmaps_augmented_processed" \
    --mask_path "/Users/emmatosato/Documents/PhD/ANM_Verona/data_utils/GM_mask.nii" \
    --threshold 0.2 \
    --normalization \
    --augmented \

