# processing_flat.py
import pandas as pd
import numpy as np
import nibabel as nib

def remove_missing_values(raw_df, df_meta, target_col):
    # Remove subjects without target values
    subjects_nan = df_meta[df_meta[target_col].isna()]['ID'].tolist()
    df = raw_df[~raw_df['ID'].isin(subjects_nan)].reset_index(drop=True)
    return df


def apply_threshold(dataframe, threshold):
    # Apply a threshold to voxel data in a DataFrame.
    df_thr = dataframe.copy()
    df_thr.iloc[:, 1:] = df_thr.iloc[:, 1:].mask(df_thr.iloc[:, 1:] < threshold, 0)
    return df_thr


def apply_mask(df_thr, mask_path):
    # Load and flatten the 3D mask
    mask = nib.load(mask_path).get_fdata().flatten().astype(bool)
    assert mask.shape[0] == df_thr.shape[1] - 1, "Mask and data length mismatch"

    # Apply mask to voxel columns (excluding 'ID')
    voxel_data = df_thr.iloc[:, 1:]
    voxel_data_masked = voxel_data.loc[:, mask]

    # Combine with 'ID' and rename voxel columns to consecutive integers
    df_masked = pd.concat([df_thr[['ID']], voxel_data_masked], axis=1)
    df_masked.columns = ['ID'] + list(range(voxel_data_masked.shape[1]))

    return df_masked



def summarize_voxel_data(df_masked, threshold=None):
    # Compute summary statistics for voxel data
    summary = {}
    summary['Shape'] = df_masked.shape
    if threshold is not None:
        values = df_masked.iloc[:, 1:]
        has_low = ((values > 0) & (values < threshold)).any().any()
        summary[f'Values between 0 and {threshold}'] = has_low
    zero_rows = (df_masked.iloc[:, 1:] == 0).all(axis=1).sum()
    summary['Zero maps'] = f"{zero_rows} of {df_masked.shape[0]}"
    voxel_data = df_masked.iloc[:, 1:].values
    nonzero_voxels = voxel_data[voxel_data != 0]
    summary.update({
        'All Min': voxel_data.min(),
        'All Max': voxel_data.max(),
        'All Mean': voxel_data.mean(),
        'All Std': voxel_data.std(),
        'Nonzero Min': nonzero_voxels.min(),
        'Nonzero Max': nonzero_voxels.max(),
        'Nonzero Mean': nonzero_voxels.mean(),
        'Nonzero Std': nonzero_voxels.std(),
    })
    return summary


def main_processing_flat(raw_df, df_meta, target_col, gm_mask_path, harvard_mask_path, thresholds=[0.1, 0.2], do_eda=False):
    # Align metadata to raw_df
    df_meta = df_meta.set_index('ID').loc[raw_df['ID']].reset_index()
    assert all(raw_df['ID'] == df_meta['ID']), "Mismatch between ID of df and df_meta_ordered"
    print("The IDs are now perfectly aligned")

    # Remove subjects without target
    df_clean = remove_missing_values(raw_df, df_meta, target_col)

    # Apply thresholding
    df_thr_01 = apply_threshold(df_clean, threshold=0.1)
    df_thr_02 = apply_threshold(df_clean, threshold=0.2)
    print("Thresholds applied")

    # GM masking
    df_gm_masked = apply_mask(df_clean, gm_mask_path)
    df_thr01_gm_masked = apply_mask(df_thr_01, gm_mask_path)
    df_thr02_gm_masked = apply_mask(df_thr_02, gm_mask_path)

    # Harvard masking
    df_har_masked = apply_mask(df_clean, harvard_mask_path)
    df_thr01_har_masked = apply_mask(df_thr_01, harvard_mask_path)
    df_thr02_har_masked = apply_mask(df_thr_02, harvard_mask_path)
    print("Masks applied")

    # Collect all outputs
    outputs = {
        'thr_01_gm': df_thr01_gm_masked,
        'thr_02_gm': df_thr02_gm_masked,
        'thr_01_har': df_thr01_har_masked,
        'thr_02_har': df_thr02_har_masked,
        'gm_no_thr': df_gm_masked,
        'har_no_thr': df_har_masked
    }

    # Optional EDA
    if do_eda:
        eda_list = []
        for name, dfm in outputs.items():
            thr_val = None
            if name.startswith('thr_'):
                thr_val = float(name.split('_')[1].replace('_', '.'))
            summary = summarize_voxel_data(dfm, threshold=thr_val)
            summary['Dataset'] = name
            eda_list.append(summary)
        df_summary = pd.DataFrame(eda_list).set_index('Dataset')

    print("Processing completed")
    return outputs, df_summary

