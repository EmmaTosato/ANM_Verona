#!/bin/bash

# Default arguments
LABEL_COLUMN="Group"
RATIOS="0.8 0.1 0.1"
N_AUGMENTATIONS=10
SUBSET_SIZE=17
LABELS_FILE="/Users/emmatosato/Documents/PhD/ANM_Verona/data_utils/labels.csv"
HCP_ROOT="./hcp_subjects"
OUTPUT_ROOT="./data/"
SPLITS_CSV="./data_utils/subset_info.csv"

# Run Python script
python generate_augmented_fc.py \
  --label_column "$LABEL_COLUMN" \
  --ratios "$RATIOS" \
  --n_augmentations $N_AUGMENTATIONS \
  --subset_size $SUBSET_SIZE \
  --labels_file "$LABELS_FILE" \
  --hcp_root "$HCP_ROOT" \
  --output_root "$OUTPUT_ROOT" \
  --splits_csv "$SPLITS_CSV" \
  --augment_val
