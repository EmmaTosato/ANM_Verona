# loading.py
import pandas as pd
import os
import glob
import nibabel as nib
import numpy as np
from sklearn.mixture import GaussianMixture
import json
import pickle

def load_FDCmaps(config):
    # All nii.gz files in the directory
    dir_FCmaps = config['dir_FCmaps']
    files_path = sorted(glob.glob(os.path.join(dir_FCmaps, '*gz')))

    # Extract Subject IDs from filenames
    subject_id = [os.path.basename(f).replace('.FDC.nii.gz', '') for f in files_path]

    # Sanity checks
    assert len(files_path) == len(subject_id), (
        f"Mismatch count: {len(files_path)} files vs {len(subject_id)} IDs"
    )
    assert len(subject_id) == len(set(subject_id)), "ID duplicated"
    for fp, sid in zip(files_path, subject_id):
        fname = os.path.basename(fp)
        expected = sid + '.FDC.nii.gz'
        assert fname == expected, (
            f"Filename “{fname}” does not correspond to the extracted ID “{sid}”"
        )

    # Load image data and flatten
    maps_FDC = []
    for path in files_path:
        #print(path)
        data = nib.load(path).get_fdata().flatten()
        maps_FDC.append(data)

    # Create DataFrame
    raw_df = pd.DataFrame(maps_FDC)
    raw_df.insert(0, 'ID', subject_id)

    # Save the raw dataframe as csv
    raw_df.to_pickle(os.path.join(config['raw_df']))

    return files_path, subject_id, raw_df


def load_metadata(cognitive_dataset):
    # Load the metadata and align to FC map order
    df_meta = pd.read_excel(cognitive_dataset, sheet_name='Sheet1')
    df_meta['Age'] = df_meta['Age'].round(1)

    # Remove the subject with ID "4_S_5003"
    df_meta = df_meta[df_meta['ID'] != '4_S_5003'].reset_index(drop=True)

    return df_meta



def gmm_label_CDR(df_meta):
    # Filter valid CDR_SB values
    df_cdr = df_meta[['ID', 'CDR_SB']].dropna().copy()

    # Fit GMM and predict raw labels
    np.random.seed(42)
    x_gmm = df_cdr['CDR_SB'].values.reshape(-1, 1)
    gmm = GaussianMixture(n_components=3, random_state=42).fit(x_gmm)
    df_cdr['GMM_Label'] = gmm.predict(x_gmm)

    # Reorder labels by CDR_SB severity
    means = df_cdr.groupby('GMM_Label')['CDR_SB'].mean().sort_values()
    label_map = {old: new for new, old in enumerate(means.index)}
    df_cdr['GMM_Label'] = df_cdr['GMM_Label'].map(label_map)

    # Include in metadata
    label_map = dict(zip(df_cdr['ID'], df_cdr['GMM_Label']))
    df_meta = df_meta.drop(columns=['GMM_Label'], errors='ignore')
    df_meta['GMM_Label'] = df_meta['ID'].map(label_map).astype('Int64')

    return df_meta

def load_Yeo(config, df_meta):
    # Load the csv files for Yeo networks
    df_no_thr = pd.read_csv(config["yeo_noThr"])
    df_thr01 = pd.read_csv(config["yeo_01thr"])
    df_thr02 = pd.read_csv(config["yeo_02thr"])

    # Rename the columns
    df_no_thr = df_no_thr.rename(columns={"CODE": "ID"})
    df_thr01 = df_thr01.rename(columns={"CODE": "ID"})
    df_thr02 = df_thr02.rename(columns={"CODE": "ID"})

    # Reorder the columns to match the metadata dataframe
    df_no_thr = df_no_thr.set_index("ID").loc[df_meta['ID']].reset_index()
    df_thr01 = df_thr01.set_index("ID").loc[df_meta['ID']].reset_index()
    df_thr02 = df_thr02.set_index("ID").loc[df_meta['ID']].reset_index()

    # Save the new csv
    dir_dataframe = os.path.join(config['dir_dataframe'], "networks")
    df_no_thr.to_csv(os.path.join(dir_dataframe, "networks_noTHR.csv"), index=False)
    df_thr01.to_csv(os.path.join(dir_dataframe, "networks_thr01.csv"), index=False)
    df_thr02.to_csv(os.path.join(dir_dataframe, "networks_thr02.csv"), index=False)

    return df_no_thr, df_thr01, df_thr02


if __name__ == "__main__":
    print("Loading config and metadata...")
    with open("config.json", "r") as f:
        config = json.load(f)

    # Load the metadata
    df_meta = load_metadata(config["cognitive_dataset"])

    # Apply GMM labeling to CDR_SB
    df_meta = gmm_label_CDR(df_meta)

    # Save the metadata dataframe as csv
    df_meta.to_csv(os.path.join(config['df_meta']))

    # Load the raw dataframe
    print("Loading FC maps...")
    files_path, subject_id, raw_df = load_FDCmaps(config)

    # Load the Yeo Network
    print("Loading Yeo networks...")
    df_no_thr, df_thr01, df_thr02 =  load_Yeo(config, df_meta)

    print("Done.")


