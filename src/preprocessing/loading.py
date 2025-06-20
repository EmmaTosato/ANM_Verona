# loading.py
import pandas as pd
import os
import glob
import nibabel as nib
import numpy as np
from sklearn.mixture import GaussianMixture
import json
import warnings
warnings.filterwarnings("ignore")

def load_args_resolved(config_path="src/parameters/config.json", paths_path="src/parameters/paths.json"):
    with open(config_path) as f:
        config = json.load(f)
    with open(paths_path) as f:
        paths = json.load(f)

    args = {}
    for section in config.values():
        if isinstance(section, dict):
            args.update(section)

    args["df_path"] = resolve_data_path(
        dataset_type=args["dataset_type"],
        threshold=args["threshold"],
        paths=paths
    )

    return args

def load_all(config_path="src/parameters/config.json", paths_path="src/parameters/paths.json"):
    args = load_args_resolved(config_path, paths_path)
    df = pd.read_pickle(args["df_path"])
    meta = pd.read_csv(args["df_meta"])
    return args, df, meta

def resolve_data_path(dataset_type: str, threshold, paths: dict):
    flat_paths = {}
    for section in paths.values():
        flat_paths.update(section)

    if dataset_type == "fdc":
        if not threshold:
            return flat_paths["df_masked"]
        elif threshold == 0.2:
            return flat_paths["df_masked_02"]
        else:
            raise ValueError(f"No FDC path defined for threshold={threshold}")

    elif dataset_type == "networks":
        if not threshold:
            return flat_paths["net_noThr"]
        elif threshold == 0.2:
            return flat_paths["net_thr02"]
        elif threshold == 0.1:
            return flat_paths["net_thr01"]
        else:
            raise ValueError(f"No network path defined for threshold={threshold}")

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

def load_fdc_maps(params):
    # All nii.gz files in the directory
    dir_fc_maps = params['dir_FCmaps']
    path_files = sorted(glob.glob(os.path.join(dir_fc_maps, '*gz')))

    # Extract Subject IDs from filenames
    patients_id = [os.path.basename(file).replace('.FDC.nii.gz', '') for file in path_files]

    # Load image data and flatten
    maps_fdc = []
    for path in path_files:
        data = nib.load(path).get_fdata().flatten()
        maps_fdc.append(data)

    # Create DataFrame
    raw_df = pd.DataFrame(maps_fdc)
    raw_df.insert(0, 'ID', patients_id)

    # Save the raw dataframes as csv
    raw_df.to_pickle(params['raw_df'])

    return path_files, patients_id, raw_df


def load_metadata(cognitive_dataset):
    # Load the metadata and align to FC map order
    df_meta = pd.read_excel(cognitive_dataset, sheet_name='Sheet1')
    df_meta['Age'] = df_meta['Age'].round(1)

    # Remove the subject with ID "4_S_5003"
    df_meta = df_meta[df_meta['ID'] != '4_S_5003'].reset_index(drop=True)

    return df_meta


def gmm_label_cdr(df_meta):
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

def load_yeo(params, df_meta):
    save_dir = params["dir_yeo_df"]

    # Mapping from params keys to output filenames
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


if __name__ == "__main__":
    print("Loading config and metadata...")
    with open("src/parameters/paths.json", "r") as f:
        config = json.load(f)

    # Load the metadata
    df_metadata = load_metadata(config["cognitive_dataset"])

    # Apply GMM labeling to CDR_SB
    df_metadata = gmm_label_cdr(df_metadata)

    # Save the metadata dataframes as csv
    df_metadata.to_csv(os.path.join(config['df_meta']), index=False)

    # Load the raw dataframes
    print("Loading FC maps...")
    files_path, subject_id, raw_dataframe = load_fdc_maps(config)

    # Load the Yeo Network
    print("Loading Yeo networks...")
    df_no_thr, df_thr01, df_thr02 =  load_yeo(config, df_metadata)

    print("Done.")


