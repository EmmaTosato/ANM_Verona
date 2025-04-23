#!/bin/bash
#SBATCH --mail-user=emmamariasole.tosato@studenti.unipd.it
#SBATCH --mail-type=ALL
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=6
#SBATCH --mem=24G
#SBATCH -p brains


LISTA_HCP="/data/lorenzo/ANM_Verona//lista_HCP"
DATASET_DIR="/data/lorenzo/ANM_Verona/dataset"
OUTPUT_DIR="/data/etosato/ANM_Verona/FCmaps"
TRACKING_CSV="/data/etosato/ANM_Verona/aug_tracking.csv"
N_AUG=10
SUBSET_SIZE=17

# Output dir if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run script for each subject
for SUBJECT in "$DATASET_DIR"/*; do
    SUBJECT_ID=$(basename "$SUBJECT")
    echo "Processing $SUBJECT_ID"
    python generate_patient_maps.py \
      --subject_id "$SUBJECT_ID" \
      --dataset_dir "$DATASET_DIR" \
      --hcp_list "$LISTA_HCP" \
      --output_dir "$OUTPUT_DIR" \
      --csv_out "$TRACKING_CSV" \
      --n_augmentations "$N_AUG" \
      --subset_size "$SUBSET_SIZE"
done

