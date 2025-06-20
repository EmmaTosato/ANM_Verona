import os
import glob
import pandas as pd
import nibabel as nib
import numpy as np
from sklearn.mixture import GaussianMixture


def load_fdc_maps(params):
    dir_fc_maps = params['dir_FCmaps']
    path_files = sorted(glob.glob(os.path.join(dir_fc_maps, '*gz')))
    patients_id = [os.path.basename(file).replace('.FDC.nii.gz', '') for file in path_files]

    maps_fdc = []
    for path in path_files:
        data = nib.load(path).get_fdata().flatten()
        maps_fdc.append(data)

    raw_df = pd.DataFrame(maps_fdc)
    raw_df.insert(0, 'ID', patients_id)
    raw_df.to_pickle(params['raw_df'])
    return path_files, patients_id, raw_df


def load_metadata(cognitive_dataset):
    df_meta = pd.read_excel(cognitive_dataset, sheet_name='Sheet1')
    df_meta['Age'] = df_meta['Age'].round(1)
    df_meta = df_meta[df_meta['ID'] != '4_S_5003'].reset_index(drop=True)
    return df_meta


def gmm_label_cdr(df_meta):
    df_cdr = df_meta[['ID', 'CDR_SB']].dropna().copy()
    np.random.seed(42)
    x_gmm = df_cdr['CDR_SB'].values.reshape(-1, 1)
    gmm = GaussianMixture(n_components=3, random_state=42).fit(x_gmm)
    df_cdr['labels_gmm_cdr'] = gmm.predict(x_gmm)

    means = df_cdr.groupby('labels_gmm_cdr')['CDR_SB'].mean().sort_values()
    label_map = {old: new for new, old in enumerate(means.index)}
    df_cdr['labels_gmm_cdr'] = df_cdr['labels_gmm_cdr'].map(label_map)

    label_map = dict(zip(df_cdr['ID'], df_cdr['labels_gmm_cdr']))
    df_meta = df_meta.drop(columns=['labels_gmm_cdr'], errors='ignore')
    df_meta['labels_gmm_cdr'] = df_meta['ID'].map(label_map).astype('Int64')
    return df_meta


def load_yeo(params, df_meta):
    save_dir = params["dir_yeo_df"]
    mapping = {
        "yeo_noThr": "networks_noTHR.csv",
        "yeo_01thr": "networks_thr01.csv",
        "yeo_02thr": "networks_thr02.csv"
    }
    dfs = []
    for key, out_name in mapping.items():
        df = pd.read_csv(params[key]).rename(columns={"CODE": "ID"})
        df = df.set_index("ID").loc[df_meta['ID']].reset_index()
        df.to_csv(os.path.join(save_dir, out_name), index=False)
        dfs.append(df)
    return tuple(dfs)
