#!/bin/bash
#SBATCH --job-name=augment_4_S_5003
#SBATCH --mail-user=emmamariasole.tosato@studenti.unipd.it
#SBATCH --mail-type=ALL
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p brains
#SBATCH --output=logsEmma/augment_4_S_5003_%j.out
#SBATCH --error=logsEmma/augment_4_S_5003_%j.err

# Variables
LISTA_HCP="/data/lorenzo/ANM_Verona/lista_HCP"
DATASET_DIR="/data/lorenzo/ANM_Verona/dataset/"
OUTPUT_DIR="/data/etosato/ANM_Verona/FCmaps_augmented/"
TRACKING_CSV="/data/etosato/ANM_Verona/aug_tracking.csv"
N_AUG=10
SUBSET_SIZE=17
SUBJECT_ID="4_S_5003"

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"
mkdir -p logsEmma

# Logging
echo "----------------------------------"
echo "Processing subject: $SUBJECT_ID"
echo "Start time: $(date)"

# Run the augmentation
python3 augmentation.py \
  --subject_id "$SUBJECT_ID" \
  --dataset_dir "$DATASET_DIR" \
  --hcp_list "$LISTA_HCP" \
  --output_dir "$OUTPUT_DIR" \
  --csv_out "$TRACKING_CSV" \
  --n_augmentations "$N_AUG" \
  --subset_size "$SUBSET_SIZE"

echo "End time: $(date)"
echo "----------------------------------"
