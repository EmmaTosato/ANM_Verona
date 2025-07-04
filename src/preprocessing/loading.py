# loading.py
import os
import glob
import warnings
import pandas as pd
import nibabel as nib
from preprocessing.config import ConfigLoader
warnings.filterwarnings("ignore")

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


if __name__ == "__main__":
    loader = ConfigLoader()
    params = loader.args

    # Load metadata
    df_metadata = load_metadata(params["cognitive_dataset"])
    df_metadata.to_csv(params["df_meta"], index=False)

    # Generate raw FDC matrix
    print("Loading FC maps...")
    load_fdc_maps(params)



