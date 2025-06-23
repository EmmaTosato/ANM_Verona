import os
import json
import glob
import warnings
import pandas as pd
import numpy as np
import nibabel as nib
from sklearn.mixture import GaussianMixture
warnings.filterwarnings("ignore")

class ConfigLoader:
    """
    A class to load and merge configuration parameters from JSON files.
    Produces a unified dictionary `args` that includes all values from config.json and paths.json,
    plus derived paths such as df_path based on dataset type and threshold.
    """
    def __init__(self, config_path="src/parameters/config.json", paths_path="src/parameters/paths.json"):
        # Load configuration and paths from JSON
        with open(config_path) as f:
            self.config = json.load(f)
        with open(paths_path) as f:
            self.paths = json.load(f)

        # Merge both configs into a single flat dictionary
        self.args = self._resolve_args()

    def _resolve_args(self):
        args = {}

        # Flatten sections from config.json
        for section_name, section in self.config.items():
            if isinstance(section, dict):
                for k, v in section.items():
                    args[k] = v
            else:
                args[section_name] = section

        # Flatten sections from paths.json
        for section_name, section in self.paths.items():
            if isinstance(section, dict):
                for k, v in section.items():
                    args[k] = v
            else:
                args[section_name] = section

        # Compute and store df_path based on dataset type and threshold
        args["df_path"] = self._resolve_data_path(
            dataset_type=args["dataset_type"],
            threshold=args["threshold"],
            flat_args=args
        )

        return args

    def _resolve_data_path(self, dataset_type, threshold, flat_args):
        """
        Selects the correct dataframe path based on dataset type and threshold.
        Values must match keys defined in paths.json.
        """
        if dataset_type == "fdc":
            if not threshold:
                return flat_args["df_masked"]
            elif threshold == 0.2:
                return flat_args["df_masked_02"]
        elif dataset_type == "networks":
            if not threshold:
                return flat_args["net_noThr"]
            elif threshold == 0.2:
                return flat_args["net_thr02"]
            elif threshold == 0.1:
                return flat_args["net_thr01"]

        raise ValueError(f"Invalid dataset_type={dataset_type} or threshold={threshold}")

    def load_all(self):
        """
        Loads the main dataframe (df) and metadata (meta) using resolved paths.
        Returns the args dictionary along with both dataframes.
        """
        df = pd.read_pickle(self.args["df_path"])
        meta = pd.read_csv(self.args["df_meta"])
        return self.args, df, meta


def load_fdc_maps(params):
    """
    Loads and flattens FDC NIfTI maps into a dataframe with ID column.
    Saves the raw matrix to disk.
    """
    path_files = sorted(glob.glob(os.path.join(params['dir_FCmaps'], '*gz')))
    ids = [os.path.basename(p).replace('.FDC.nii.gz', '') for p in path_files]
    maps = [nib.load(p).get_fdata().flatten() for p in path_files]

    df = pd.DataFrame(maps)
    df.insert(0, 'ID', ids)
    df.to_pickle(params['raw_df'])

    return path_files, ids, df


def load_metadata(cognitive_dataset):
    """
    Loads cognitive metadata from Excel and removes subject '4_S_5003'.
    Rounds age to one decimal.
    """
    df = pd.read_excel(cognitive_dataset, sheet_name='Sheet1')
    df['Age'] = df['Age'].round(1)
    return df[df['ID'] != '4_S_5003'].reset_index(drop=True)


def gmm_label_cdr(df_meta):
    """
    Applies Gaussian Mixture Model (GMM) clustering to the CDR_SB scores.
    Reorders GMM labels by ascending severity and maps them to the full metadata.
    """
    df_cdr = df_meta[['ID', 'CDR_SB']].dropna().copy()
    x = df_cdr['CDR_SB'].values.reshape(-1, 1)
    gmm = GaussianMixture(n_components=3, random_state=42).fit(x)
    df_cdr['labels_gmm_cdr'] = gmm.predict(x)

    # Reorder labels by mean CDR_SB
    means = df_cdr.groupby('labels_gmm_cdr')['CDR_SB'].mean().sort_values()
    label_map = {old: new for new, old in enumerate(means.index)}
    df_cdr['labels_gmm_cdr'] = df_cdr['labels_gmm_cdr'].map(label_map)

    # Assign labels back to full metadata
    full_map = dict(zip(df_cdr['ID'], df_cdr['labels_gmm_cdr']))
    df_meta = df_meta.drop(columns=['labels_gmm_cdr'], errors='ignore')
    df_meta['labels_gmm_cdr'] = df_meta['ID'].map(full_map).astype('Int64')
    return df_meta


def load_yeo(params, df_meta):
    """
    Loads the mean Yeo network values and aligns them with subject metadata.
    Saves the result to disk for different thresholds (noThr, 0.1, 0.2).
    """
    mapping = {
        "yeo_noThr": "networks_noTHR.csv",
        "yeo_01thr": "networks_thr01.csv",
        "yeo_02thr": "networks_thr02.csv"
    }
    dfs = []
    for key, out_file in mapping.items():
        df = pd.read_csv(params[key]).rename(columns={"CODE": "ID"})
        df = df.set_index("ID").loc[df_meta['ID']].reset_index()
        df.to_csv(os.path.join(params['dir_yeo_df'], out_file), index=False)
        dfs.append(df)
    return tuple(dfs)

def loading_pipeline(params):
    # Load and label metadata
    df_metadata = gmm_label_cdr(load_metadata(params["cognitive_dataset"]))
    df_metadata.to_csv(params["df_meta"], index=False)

    # Generate raw FDC matrix
    print("Loading FC maps...")
    load_fdc_maps(params)

    # Align and save Yeo networks
    print("Loading Yeo networks...")
    load_yeo(params, df_metadata)

    print("Done.")


if __name__ == "__main__":
    # Initialize configuration loader and retrieve flattened args
    loader = ConfigLoader()
    args = loader.args

    loading_pipeline(args)



