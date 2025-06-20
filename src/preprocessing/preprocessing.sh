#!/bin/bash

# ---- Pre-preprocessing of 3D maps-----
# Before launching:
# - Set the paths below
# - Remove --threshold if data is not to be thresholded
# - Remove --augmented if data is not augmented
# - Remove --normalization if you don't want MinMax scaling

python processed3d.py \
    --input_dir "/Users/emmatosato/Documents/PhD/ANM_Verona/data/FCmaps/" \
    --output_dir "/Users/emmatosato/Documents/PhD/ANM_Verona/data/FCmaps_processed" \
    --mask_path "/Users/emmatosato/Documents/PhD/ANM_Verona/utils/masks/GM_mask.nii" \
    --normalization \
