# clustering.py

import os
import re
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import hdbscan
from preprocessing.config import ConfigLoader
from preprocessing.processflat import x_features_return
from analysis.clustering_evaluation import evaluate_kmeans, evaluate_gmm, evaluate_hdbscan, evaluate_consensus
from analysis.plotting import plot_umap_embedding, plot_clusters_vs_groups
from analysis.utils import log_to_file, reset_stdout, run_umap, build_output_path

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
np.random.seed(42)

def group_value_to_str(value):
    """
    Converts a group value to a string for use in folder/file names.
    """
    if pd.isna(value): return "nan"
    if isinstance(value, float) and value.is_integer(): return str(int(value))
    return str(value)

def run_clustering(x_umap):
    """
    Performs HDBSCAN and K-Means clustering on the UMAP embedding.
    Returns a dictionary with clustering labels.
    """
    return {
        #"HDBSCAN": hdbscan.HDBSCAN(min_cluster_size=5).fit_predict(x_umap),
        "K-Means": KMeans(n_clusters=3, random_state=42).fit_predict(x_umap)
    }

def clustering_umap_pipeline(params, df_input, df_meta):
    """
    Executes the full clustering pipeline:
    - Embeds data using UMAP
    - Applies clustering algorithms
    - Evaluates clustering quality (optional)
    - Plots clustering results (optional)
    - Returns a DataFrame with standard cluster labels (no threshold suffix)
    """
    df_merged, x = x_features_return(df_input, df_meta)

    # Run umap
    x_umap = run_umap(x)

    labels_dict = run_clustering(x_umap)
    labeling_umap = pd.DataFrame({
        'ID': df_merged['ID'],
        'Group': df_merged['Group'],
        'X1': x_umap[:, 0],
        'X2': x_umap[:, 1],
        #'labels_hdb': labels_dict['HDBSCAN'],
        'labels_km': labels_dict['K-Means'],
        'labels_gmm_cdr': df_merged['labels_gmm_cdr']
    })

    # Plotting
    plot_umap_embedding(labeling_umap, title=params['prefix'], save_path=params['path_umap'],
                        plot_flag=params['plot_cluster'], save_flag=params['save_flag'],
                        title_flag=params['title_flag'])

    print(len(labeling_umap['labels_gmm_cdr']) , " e ", len(labeling_umap['Group']))

    # Plotting
    if params.get("do_evaluation"):
        clean_title = re.sub(r'[\s\-]+', '_', params['prefix'].strip().lower())
        evaluate_kmeans(x_umap, save_path=params['path_opt_cluster'], prefix=clean_title, plot_flag=params['plot_cluster'])
        evaluate_gmm(x_umap, save_path=params['path_opt_cluster'], prefix=clean_title, plot_flag=params['plot_cluster'])
        evaluate_consensus(x_umap, save_path=params['path_opt_cluster'], prefix=clean_title, plot_flag=params['plot_cluster'])
        evaluate_hdbscan(x_umap)

    plot_clusters_vs_groups(x_umap, labels_dict, labeling_umap['Group'], params['path_cluster'],params['prefix'] + " - Group label",
                            plot_flag=params['plot_cluster'], save_flag = params['save_flag'], title_flag = params['title_flag'], separated= params['separated_plots'])
    plot_clusters_vs_groups(x_umap, labels_dict, labeling_umap['labels_gmm_cdr'], params['path_cluster'],params['prefix'] + " - GMM label",
                            plot_flag=params['plot_cluster'], save_flag = params['save_flag'], title_flag = params['title_flag'], colors_gmm=True, separated= params['separated_plots'])

    return labeling_umap

def main_clustering(params, df_input, df_meta):
    """
    Handles clustering execution and metadata update:
    - Calls pipeline
    - Renames label columns if threshold is active
    - Merges with df_meta
    - Saves updated metadata
    """
    params['path_umap'] = build_output_path(params['output_dir'], "", params['dataset_type'], params['umap'])
    params['path_cluster'] = build_output_path(params['output_dir'], params['task_type'], params['dataset_type'], params['umap'])
    params['path_opt_cluster'] = build_output_path(params['output_dir'], "optimal_cluster", params['dataset_type'], params['umap'])
    params['prefix'] = f"{params['threshold']} Threshold" if params['threshold'] in [0.1, 0.2] else "No Threshold"

    # Unsupervised clustering after umap
    if params['umap']:
        labeling_umap = clustering_umap_pipeline(params, df_input, df_meta)

    # Decide suffix and rename columns if needed
    if params.get("threshold") in [0.1, 0.2]:
        suffix = f"_thr{str(params['threshold']).replace('.', '')}"
        labeling_umap = labeling_umap.rename(columns={
            "labels_km": f"labels_km{suffix}",
            "labels_hdb": f"labels_hdb{suffix}"
        })
        km_col = f"labels_km{suffix}"
        hdb_col = f"labels_hdb{suffix}"
    else:
        km_col = "labels_km"
        hdb_col = "labels_hdb"

    # Merge into df_meta
    if km_col not in df_meta.columns and hdb_col not in df_meta.columns:
        df_meta = df_meta.merge(labeling_umap[['ID', km_col, hdb_col]], on='ID', how='left')

    # Standard fix
    df_meta['labels_gmm_cdr'] = df_meta['labels_gmm_cdr'].astype('Int64')
    df_meta.to_csv(params['df_meta'], index=False)


if __name__ == "__main__":
    loader = ConfigLoader()
    args, input_dataframe, metadata_dataframe = loader.load_all()
    main_clustering(args, input_dataframe, metadata_dataframe)

