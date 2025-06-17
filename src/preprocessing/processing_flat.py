# processing_flat.py
import pandas as pd
import json
import nibabel as nib
import os
import pathlib
import pickle
import warnings
warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# Apply a threshold to voxel data in a DataFrame (0.1 or 0.2).
# ------------------------------------------------------------
def apply_threshold(dataframe, threshold):
    df_thr = dataframe.copy()
    df_thr.iloc[:, 1:] = df_thr.iloc[:, 1:].mask(df_thr.iloc[:, 1:] < threshold, 0)
    return df_thr

# ------------------------------------------------------------
# Apply a flatten mask.
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# EDA summary.
# ------------------------------------------------------------
def summarize_voxel_data(df_masked, threshold=None):
    summary = {}
    summary['Shape'] = df_masked.shape

    if threshold is not None:
        values = df_masked.iloc[:, 1:]
        has_low = ((values > 0) & (values < threshold)).any().any()
        summary[f'Values 0 - {threshold}'] = has_low  # renamed

    zero_rows = (df_masked.iloc[:, 1:] == 0).all(axis=1).sum()
    summary['Zero maps'] = f"{zero_rows} of {df_masked.shape[0]}"

    voxel_data = df_masked.iloc[:, 1:].values
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


def main_processing_flat(df, df_meta, gm_mask_path, harvard_mask_path, do_eda=False):
    df_summary = None

    # Align metadata to raw_df
    df_meta = df_meta.set_index('ID').loc[df['ID']].reset_index()
    assert all(df['ID'] == df_meta['ID']), "Mismatch between ID of df and df_meta_ordered"
    print("The IDs are now perfectly aligned...")

    # Apply thresholding
    df_thr_01 = apply_threshold(df, threshold=0.1)
    df_thr_02 = apply_threshold(df, threshold=0.2)
    print("Thresholds applied...")

    # GM masking
    df_gm_masked = apply_mask(df, gm_mask_path)
    df_thr01_gm_masked = apply_mask(df_thr_01, gm_mask_path)
    df_thr02_gm_masked = apply_mask(df_thr_02, gm_mask_path)

    # Harvard masking
    df_har_masked = apply_mask(df, harvard_mask_path)
    df_thr01_har_masked = apply_mask(df_thr_01, harvard_mask_path)
    df_thr02_har_masked = apply_mask(df_thr_02, harvard_mask_path)
    print("Masks applied...")

    # Collect all outputs
    outputs = {
        'df_thr01_gm': df_thr01_gm_masked,
        'df_thr02_gm': df_thr02_gm_masked,
        'df_thr01_har': df_thr01_har_masked,
        'df_thr02_har': df_thr02_har_masked,
        'df_gm': df_gm_masked,
        'df_har': df_har_masked
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

    return outputs, df_summary


if __name__ == "__main__":
    with open("src/preprocessing/paths.json", "r") as f:
        config = json.load(f)

    # Generate outputs and summary
    print("Starting preprocessing...")
    outputs, df_summary = main_processing_flat(
        df=pd.read_pickle(config['raw_df']),
        df_meta=pd.read_csv(config['df_meta']),
        gm_mask_path=config['gm_mask_path'],
        harvard_mask_path=config['harvard_oxford_mask_path'],
        do_eda=True
    )

    # Save outputs
    print("Saving...")
    for key, df_out in outputs.items():
        out_path = os.path.join(config['dir_fdc_df'], f"{key}.pkl")
        df_out.to_pickle(out_path)


    # Save EDA summary
    df_summary.to_csv(os.path.join(config['dir_dataframe'], "meta/df_summary.csv"))
    print("Done.")
