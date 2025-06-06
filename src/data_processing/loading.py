# loading.py
import pandas as pd
import os
import glob
import nibabel as nib
import numpy as np
from sklearn.mixture import GaussianMixture
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

def load_FDCmaps(config):
    # All nii.gz files in the directory
    dir_FCmaps = config['dir_FCmaps']
    files_path = sorted(glob.glob(os.path.join(dir_FCmaps, '*gz')))

    # Extract Subject IDs from filenames
    subject_id = [os.path.basename(f).replace('.FDC.nii.gz', '') for f in files_path]

    # Load image data and flatten
    maps_FDC = []
    for path in files_path:
        #print(path)
        data = nib.load(path).get_fdata().flatten()
        maps_FDC.append(data)

    # Create DataFrame
    raw_df = pd.DataFrame(maps_FDC)
    raw_df.insert(0, 'ID', subject_id)

    # Save the raw dataframes as csv
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
    df_cdr['labels_gmm_cdr'] = gmm.predict(x_gmm)

    # Reorder labels by CDR_SB severity
    means = df_cdr.groupby('labels_gmm_cdr')['CDR_SB'].mean().sort_values()
    label_map = {old: new for new, old in enumerate(means.index)}
    df_cdr['labels_gmm_cdr'] = df_cdr['labels_gmm_cdr'].map(label_map)

    # Include in metadata
    label_map = dict(zip(df_cdr['ID'], df_cdr['labels_gmm_cdr']))
    df_meta = df_meta.drop(columns=['labels_gmm_cdr'], errors='ignore')
    df_meta['labels_gmm_cdr'] = df_meta['ID'].map(label_map).astype('Int64')

    return df_meta

def load_Yeo(config, df_meta):
    save_dir = config["dir_yeo_df"]

    # Mapping from config keys to output filenames
    mapping = {
        "yeo_noThr": "networks_noTHR.csv",
        "yeo_01thr": "networks_thr01.csv",
        "yeo_02thr": "networks_thr02.csv"
    }

    dfs = []
    for key, out_name in mapping.items():
        df = pd.read_csv(config[key]).rename(columns={"CODE": "ID"})
        df = df.set_index("ID").loc[df_meta['ID']].reset_index()
        df.to_csv(os.path.join(save_dir, out_name), index=False)
        dfs.append(df)

    return tuple(dfs)


if __name__ == "__main__":
    print("Loading config and metadata...")
    with open("src/data_processing/config.json", "r") as f:
        config = json.load(f)

    # Load the metadata
    df_meta = load_metadata(config["cognitive_dataset"])

    # Apply GMM labeling to CDR_SB
    df_meta = gmm_label_CDR(df_meta)

    # Save the metadata dataframes as csv
    df_meta.to_csv(os.path.join(config['df_meta']), index=False)

    # Load the raw dataframes
    print("Loading FC maps...")
    files_path, subject_id, raw_df = load_FDCmaps(config)

    # Load the Yeo Network
    print("Loading Yeo networks...")
    df_no_thr, df_thr01, df_thr02 =  load_Yeo(config, df_meta)

    print("Done.")


