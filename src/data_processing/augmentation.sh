#!/bin/bash
#SBATCH --mail-user=emmamariasole.tosato@studenti.unipd.it
#SBATCH --mail-type=ALL
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH -p brains
#SBATCH --output=/data/lorenzo/ANM_Verona/logsEmma/augment_%j.out
#SBATCH --error=/data/lorenzo/ANM_Verona/logsEmma/augment_%j.err

# Variabili
LISTA_HCP="/data/lorenzo/ANM_Verona/lista_HCP"
DATASET_DIR="/data/lorenzo/ANM_Verona/dataset/"
OUTPUT_DIR="/data/etosato/ANM_Verona/FCmaps_augmented/"
TRACKING_CSV="/data/etosato/ANM_Verona/aug_tracking.csv"
N_AUG=10
SUBSET_SIZE=17
COUNT=0

# Crea cartella output e logs se non esistono
mkdir -p "$OUTPUT_DIR"

echo "Job started at $(date)"
echo "----------------------------------------------"

# Esegui script per ogni soggetto nella lista
while read -r SUBJECT_ID; do
    COUNT=$((COUNT + 1))

    python3 augmentation.py \
      --subject_id "$SUBJECT_ID" \
      --dataset_dir "$DATASET_DIR" \
      --hcp_list "$LISTA_HCP" \
      --output_dir "$OUTPUT_DIR" \
      --csv_out "$TRACKING_CSV" \
      --n_augmentations "$N_AUG" \
      --subset_size "$SUBSET_SIZE"\
      --index "$COUNT"

done < "$DATASET_DIR/list"

echo "----------------------------------------------"
echo "Job finished at $(date)"
