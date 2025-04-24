#!/bin/bash
#SBATCH --mail-user=emmamariasole.tosato@studenti.unipd.it
#SBATCH --mail-type=ALL
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH -p brains
#SBATCH --output=logsEmma/augment_%j.out
#SBATCH --error=logsEmma/augment_%j.err

# Variabili
LISTA_HCP="/data/lorenzo/ANM_Verona/lista_HCP"
DATASET_DIR="/data/lorenzo/ANM_Verona/dataset/"
OUTPUT_DIR="/data/etosato/ANM_Verona/FCmaps_augmented/"
TRACKING_CSV="/data/etosato/ANM_Verona/aug_tracking.csv"
N_AUG=10
SUBSET_SIZE=17

# Crea cartella output e logs se non esistono
mkdir -p "$OUTPUT_DIR"

# Esegui script per ogni soggetto nella lista
while read -r SUBJECT_ID; do
    python3 augmentation.py \
      --subject_id "$SUBJECT_ID" \
      --dataset_dir "$DATASET_DIR" \
      --hcp_list "$LISTA_HCP" \
      --output_dir "$OUTPUT_DIR" \
      --csv_out "$TRACKING_CSV" \
      --n_augmentations "$N_AUG" \
      --subset_size "$SUBSET_SIZE"

done < "$DATASET_DIR/list"

echo "Job finished at $(date)"