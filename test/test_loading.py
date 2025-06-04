import os
import pandas as pd
import pytest
import json

from src.data_processing.loading import load_FDCmaps, load_metadata, gmm_label_CDR, load_Yeo

# Load config
with open("/Users/emmatosato/Documents/PhD/ANM_Verona/src/data_processing/config.json", "r") as f:
    config = json.load(f)


def test_load_FDCmaps():
    files_path, subject_id, raw_df = load_FDCmaps(config)

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



def test_load_Yeo():
    # Load metadata and apply GMM labels
    df_meta = load_metadata(config["cognitive_dataset"])
    df_meta = gmm_label_CDR(df_meta)

    # Load Yeo networks
    df_no_thr, df_thr01, df_thr02 = load_Yeo(config, df_meta)

    # Expected shape including ID column
    expected_shape = (176, 9)
    assert df_no_thr.shape == expected_shape, f"df_no_thr shape mismatch: {df_no_thr.shape}"
    assert df_thr01.shape == expected_shape, f"df_thr01 shape mismatch: {df_thr01.shape}"
    assert df_thr02.shape == expected_shape, f"df_thr02 shape mismatch: {df_thr02.shape}"