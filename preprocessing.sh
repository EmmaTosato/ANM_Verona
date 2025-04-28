#!/bin/bash

python preprocess_fc_maps.py \
    --input_dir "/Users/emmatosato/Documents/PhD/ANM_Verona/data" \
    --output_dir "/Users/emmatosato/Documents/PhD/ANM_Verona/data" \
    --mask_path "/Users/emmatosato/Documents/PhD/ANM_Verona/data_utils/GM_mask.nii" \
    --threshold 0.2 \
    --normalization minmax \
    --augmented
