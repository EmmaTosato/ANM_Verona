# prcoessflat.py
import os
import json
import warnings
import pandas as pd
import numpy as np
import nibabel as nib
from sklearn.mixture import GaussianMixture
from preprocessing.loading import ConfigLoader


# Suppress all warnings to keep output clean
warnings.filterwarnings("ignore")

def apply_threshold(dataframe, threshold):
    """
    Applies a lower-bound threshold to voxel intensities.
    All values below the threshold are set to 0 (excluding the ID column).
    """
    df_thr = dataframe.copy()
    df_thr.iloc[:, 1:] = df_thr.iloc[:, 1:].mask(df_thr.iloc[:, 1:] < threshold, 0)
    return df_thr

def apply_mask(df_thr, mask_path):
    """
    Applies a binary 3D mask to the voxel data (excluding 'ID').
    The mask is flattened and used to retain only selected voxels.
    """

    mask = nib.load(mask_path).get_fdata().flatten().astype(bool)
    assert mask.shape[0] == df_thr.shape[1] - 1, "Mask and data length mismatch"

    voxel_data = df_thr.iloc[:, 1:]
    voxel_data_masked = voxel_data.loc[:, mask]

    df_masked = pd.concat([df_thr[['ID']], voxel_data_masked], axis=1)
    df_masked.columns = ['ID'] + list(range(voxel_data_masked.shape[1]))
    return df_masked

def summarize_voxel_data(df_masked, threshold=None):
    """
    Generates a summary of voxel intensity statistics.
    Includes shape, zero-map count, and global/nonzero voxel stats.
    """

    summary = {'Shape': df_masked.shape}
    values = df_masked.iloc[:, 1:]

    if threshold is not None:
        has_low = ((values > 0) & (values < threshold)).any().any()
        summary[f'Values 0 - {threshold}'] = has_low

    zero_rows = (values == 0).all(axis=1).sum()
    summary['Zero maps'] = f"{zero_rows} of {df_masked.shape[0]}"

    voxel_data = values.values
    nonzero_voxels = voxel_data[voxel_data != 0]

    summary.update({
        'All Min': round(voxel_data.min(), 3),
        'All Max': round(voxel_data.max(), 3),
        'All Mean': round(voxel_data.mean(), 3),
        'All Std': round(voxel_data.std(), 3),
        'Nonzero Min': round(nonzero_voxels.min(), 3),
        'Nonzero Max': round(nonzero_voxels.max(), 3),
        'Nonzero Mean': round(nonzero_voxels.mean(), 3),
        'Nonzero Std': round(nonzero_voxels.std(), 3),
    })
    return summary

def x_features_return(df_voxel, df_labels):
    """
    Merges voxel dataframe with metadata and returns:
    - the full merged dataframe (with metadata + features)
    - the matrix of features only (X), excluding metadata columns.
    """
    meta_columns = list(df_labels.columns)
    dataframe_merge = pd.merge(df_voxel, df_labels, on='ID', how='left', validate='one_to_one')
    ordered_cols = meta_columns + [col for col in dataframe_merge.columns if col not in meta_columns]
    dataframe_merge = dataframe_merge[ordered_cols]
    x = dataframe_merge.drop(columns=meta_columns)

    print("\n-------------------- Dataset Info --------------------")
    print(f"{'Meta columns (Labels and Covariates):':40s} {len(meta_columns):>5d}")
    print(f"{'Feature matrix shape (X):':40s} {x.shape}")
    print(f"{'Complete dataframe shape after merge:':40s} {dataframe_merge.shape}")
    print("-------------------------------------------------------\n")

    return dataframe_merge, x

def preprocessing_pipeline(params):
    """
    Runs the full preprocessing pipeline:
    - loads config
    - loads voxel and metadata
    - applies thresholding and masking
    - performs optional EDA
    - saves all outputs
    """
    print("Starting preprocessing...")

    # Load data
    df = pd.read_pickle(params['raw_df'])
    df_meta = pd.read_csv(params['df_meta'])

    # Align metadata
    df_meta = df_meta.set_index('ID').loc[df['ID']].reset_index()
    assert all(df['ID'] == df_meta['ID']), "Mismatch between ID of df and df_meta_ordered"
    print("The IDs are now perfectly aligned...")

    # Apply thresholds
    df_thr_01 = apply_threshold(df, threshold=0.1)
    df_thr_02 = apply_threshold(df, threshold=0.2)
    print("Thresholds applied...")

    # Apply GM masks
    df_gm_masked = apply_mask(df, params['gm_mask_path'])
    df_thr01_gm_masked = apply_mask(df_thr_01, params['gm_mask_path'])
    df_thr02_gm_masked = apply_mask(df_thr_02, params['gm_mask_path'])

    # Apply Harvard masks
    df_har_masked = apply_mask(df, params['harvard_oxford_mask_path'])
    df_thr01_har_masked = apply_mask(df_thr_01, params['harvard_oxford_mask_path'])
    df_thr02_har_masked = apply_mask(df_thr_02, params['harvard_oxford_mask_path'])
    print("Masks applied...")

    # Collect outputs
    outputs = {
        'df_thr01_gm': df_thr01_gm_masked,
        'df_thr02_gm': df_thr02_gm_masked,
        'df_thr01_har': df_thr01_har_masked,
        'df_thr02_har': df_thr02_har_masked,
        'df_gm': df_gm_masked,
        'df_har': df_har_masked
    }

    # EDA summary
    eda_list = []
    for name, dfm in outputs.items():
        thr_val = 0.1 if 'thr01' in name else 0.2 if 'thr02' in name else None
        summary = summarize_voxel_data(dfm, threshold=thr_val)
        summary['Dataset'] = name
        eda_list.append(summary)
    df_summary = pd.DataFrame(eda_list).set_index('Dataset')

    # Save everything
    print("Saving...")
    for key, df_out in outputs.items():
        out_path = os.path.join(params['dir_fdc_df'], f"{key}.pkl")
        df_out.to_pickle(out_path)

    df_summary.to_csv(os.path.join(params['dir_dataframe'], "meta/df_summary.csv"))
    print("Done.")

if __name__ == "__main__":
    args = ConfigLoader().args
    preprocessing_pipeline(args)
