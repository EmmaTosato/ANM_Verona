import os
import pandas as pd
import pytest
import json
from preprocessing.loading import load_metadata

from preprocessing.loading import load_fdc_maps, load_metadata, gmm_label_cdr

# Load config
with open("src/config/ml_paths.json", "r") as f:
    config = json.load(f)


def test_load_fdc_maps():
    files_path, subject_id, raw_df = load_fdc_maps(config)

    # Check number of files and subjects
    assert len(files_path) == 176, f"Expected 176 files, got {len(files_path)}"
    assert len(subject_id) == 176, f"Expected 176 subject IDs, got {len(subject_id)}"
    assert len(subject_id) == len(set(subject_id)), "Duplicate subject IDs found"

    # Check filename correctness
    for fp, sid in zip(files_path, subject_id):
        fname = os.path.basename(fp)
        expected = sid + '.FDC.nii.gz'
        assert fname == expected, f"Filename '{fname}' does not match expected ID '{expected}'"

    # Check final DataFrame shape
    expected_shape = (176, 902630)
    assert raw_df.shape == expected_shape, f"Expected shape {expected_shape}, got {raw_df.shape}"


def test_load_metadata():
    # Load the original Excel file directly
    df_original = pd.read_excel(config["cognitive_dataset"], sheet_name='Sheet1')
    original_len = len(df_original)

    # Apply the function
    df_meta = load_metadata(config["cognitive_dataset"])

    # Check length reduced by exactly 1
    assert len(df_meta) == original_len - 1, (
        f"Expected {original_len - 1} rows after removal, got {len(df_meta)}"
    )

    # Ensure the removed subject is not in the DataFrame
    assert "4_S_5003" not in df_meta["ID"].values, "Subject '4_S_5003' was not removed"

    # Optionally, check ID uniqueness
    assert df_meta["ID"].is_unique, "Duplicate IDs found in df_meta"


def test_split_consistency():
    split_dir = "data/dataframes/split"
    meta_path = "data/dataframes/meta/df_meta.csv"
    df_meta = pd.read_csv(meta_path)

    count = 1

    for csv_name in os.listdir(split_dir):
        if not csv_name.endswith(".csv"):
            continue
        path = os.path.join(split_dir, csv_name)
        df_split = pd.read_csv(path)

        for _, row in df_split.iterrows():
            subj_id = row["ID"]
            assert subj_id in df_meta["ID"].values, f"ID '{subj_id}' not in df_meta"

            row_meta = df_meta[df_meta["ID"] == subj_id].iloc[0]

            for col in ["Group", "Sex", "Age", "Education", "CDR_SB", "MMSE"]:
                val_csv = row[col]
                val_meta = row_meta[col]
                equal = (
                    pd.isna(val_csv) and pd.isna(val_meta)
                    or val_csv == val_meta
                )
                assert equal, f"Mismatch for subject {subj_id} column '{col}': CSV={val_csv}, META={val_meta}"

            print(f"{count}) ID {subj_id}: data correspond")
            count += 1


