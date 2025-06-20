# clustering.py
import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import hdbscan
import numpy as np
from sklearn.cluster import KMeans
from analysis.umap_run import run_umap
from preprocessing.processflat import x_features_return
from analysis.clustering_evaluation import evaluate_kmeans, evaluate_gmm, evaluate_hdbscan, evaluate_consensus
from preprocessing.config import ConfigLoader
from analysis.utils import threshold_prefix, ensure_dir
import argparse
import json
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ---------------------------
# Run clustering algorithms
# ---------------------------
def run_clustering(x_umap):
    cluster_hdb = hdbscan.HDBSCAN(min_cluster_size=5)
    labels_hdb = cluster_hdb.fit_predict(x_umap)

    kmeans = KMeans(n_clusters=3, random_state=42)
    labels_km = kmeans.fit_predict(x_umap)

    return {
        "HDBSCAN": labels_hdb,
        "K-Means": labels_km
    }

# ---------------------------------------
# Plot clustering results vs group labels
# ---------------------------------------
def plot_clusters_vs_groups(x_umap, labels_dictionary, group_column, save_path, title_prefix, margin=2.0, plot_flag=True, colors_gmm=False):
    n = len(labels_dictionary)
    n_cols = 2
    n_rows = n

    # Get global min and max across both UMAP axes
    x_vals = x_umap[:, 0]
    y_vals = x_umap[:, 1]
    min_val = min(x_vals.min(), y_vals.min()) - margin
    max_val = max(x_vals.max(), y_vals.max()) + margin

    # Color palettes
    left_plot_col = ['#74c476', '#f44f39', '#7BD3EA', '#fd8d3c', '#37659e',
                     '#fbbabd', '#ffdb24', '#413d7b', '#9dd569', '#e84a9b',
                     '#056c39', '#6788ee']
    right_plot_col = sns.color_palette("Set2")[2:] if colors_gmm else sns.color_palette("Set2")

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))

    if n_rows == 1:
        axes = [axes]

    for i, (title, labels) in enumerate(labels_dictionary.items()):
        ax_left = axes[i][0]
        ax_right = axes[i][1]

        df_plot = pd.DataFrame({
            'X1': x_vals,
            'X2': y_vals,
            'cluster': labels,
            'label': group_column
        }).dropna(subset=['label'])

        # Left: clustering result
        sns.scatterplot(data=df_plot, x='X1', y='X2', hue='cluster', palette=left_plot_col, s=50, ax=ax_left)
        ax_left.legend(loc='best', title='cluster', fontsize='small', title_fontsize='medium')
        ax_left.set_title(f'{title}', fontweight='bold')
        ax_left.set_xlabel("X1", fontsize=11, fontweight='bold')
        ax_left.set_ylabel("X2", fontsize=11, fontweight='bold')
        ax_left.set_xlim(min_val, max_val)
        ax_left.set_ylim(min_val, max_val)

        # Right: true group label
        sns.scatterplot(data=df_plot, x='X1', y='X2', hue='label', palette=right_plot_col, s=50, ax=ax_right)
        ax_right.legend(loc='best', title='label', fontsize='small', title_fontsize='medium')
        ax_right.set_title(f'{title} - Labeling according to {group_column.name}', fontweight='bold')
        ax_right.set_xlabel("X1", fontsize=11, fontweight='bold')
        ax_right.set_ylabel("X2", fontsize=11, fontweight='bold')
        ax_right.set_xlim(min_val, max_val)
        ax_right.set_ylim(min_val, max_val)

    # Layout and title
    fig.subplots_adjust(top=0.82, hspace=0.45)
    fig.suptitle("Clustering Results", fontsize=22, fontweight='bold', y=0.95)
    fig.text(0.5, 0.88, title_prefix, fontsize=16, ha='center')

    # Save
    if save_path:
        clean_prefix = re.sub(r'[\s\-]+', '_', title_prefix.strip().lower())
        save_file = os.path.join(save_path, f"{clean_prefix}_clustering.png")
        fig.savefig(save_file, dpi=300, bbox_inches='tight')

    if plot_flag:
        plt.show()
    plt.close(fig)

# ---------------------------
# Main function
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run clustering analysis")
    parser.add_argument("--config", default="src/parameters/config.json", help="Path to config file")
    parser.add_argument("--paths", default="src/parameters/paths.json", help="Path to paths file")
    return parser.parse_args()


def main_clustering(df_masked, df_meta, params):
    # Check if threshold is set
    params['prefix_cluster'], _ = threshold_prefix(params.get("threshold"))

    # Variable definition
    title_prefix = params['prefix_cluster']
    path_cluster = params['path_cluster']
    path_opt_cluster = params['path_opt_cluster']
    plot_clustering = params['plot_cluster']
    do_evaluation = params['do_evaluation']


    # Merge voxel and metadata
    df_merged, x = x_features_return(df_masked, df_meta)

    # Reduce dimensionality with UMAP
    x_umap = run_umap(x, plot_flag=plot_clustering, save_path=params['path_umap'], title=title_prefix)

    # Evaluation of clustering
    if do_evaluation:
        print("Evaluating clustering algorithms...")
        clean_title = re.sub(r'[\s\-]+', '_', title_prefix.strip().lower())
        evaluate_kmeans(x_umap, save_path=path_opt_cluster, prefix=clean_title, plot_flag=plot_clustering)
        evaluate_gmm(x_umap, save_path=path_opt_cluster, prefix=clean_title, plot_flag=plot_clustering)
        evaluate_consensus(x_umap, save_path=path_opt_cluster, prefix=clean_title, plot_flag=plot_clustering)
        evaluate_hdbscan(x_umap)

    # Clustering and collect results
    labels_dict = run_clustering(x_umap)
    labeling_umap = pd.DataFrame({
        'ID': df_merged['ID'],
        'group': df_merged['Group'],
        'X1': x_umap[:, 0],
        'X2': x_umap[:, 1],
        'labels_hdb': labels_dict['HDBSCAN'],
        'labels_km': labels_dict['K-Means'],
        'labels_gmm_cdr': df_merged['labels_gmm_cdr']
    })

    # Plot and save clusters
    if plot_clustering or path_cluster:
        print("Running clustering...")
        # Plot according to diagnostic group
        title_cluster1 = title_prefix + " - Group label"
        plot_clusters_vs_groups(x_umap, labels_dict, labeling_umap['group'], path_cluster, title_cluster1, margin=2.0, plot_flag=plot_clustering)

        # Plot according to gmm labels cdr
        title_cluster2 = title_prefix + " - GMM label"
        plot_clusters_vs_groups(x_umap, labels_dict, labeling_umap['labels_gmm_cdr'], path_cluster, title_cluster2, margin= 2.0, plot_flag=plot_clustering, colors_gmm= True)

    # Adding clustering columns according threshold
    if params.get("threshold") in [0.1, 0.2]:
        thr_suffix = f"_thr{str(params['threshold']).replace('.', '')}"
        km_col = f"labels_km{thr_suffix}"
        hdb_col = f"labels_hdb{thr_suffix}"

        # Rinomina colonne
        labeling_umap = labeling_umap.rename(columns={
            "labels_km": km_col,
            "labels_hdb": hdb_col
        })
    else:
        km_col = "labels_km"
        hdb_col = "labels_hdb"

    if km_col not in df_meta.columns and hdb_col not in df_meta.columns:
        df_meta = df_meta.merge(labeling_umap[['ID', km_col, hdb_col]], on='ID', how='left')
    df_meta['labels_gmm_cdr'] = df_meta['labels_gmm_cdr'].astype('Int64')
    df_meta.to_csv(params['df_meta'], index=False)

    return labeling_umap


if __name__ == "__main__":
    cli_args = parse_args()
    loader = ConfigLoader(cli_args.config, cli_args.paths)
    args, _, _ = loader.load()

    df_masked_raw = pd.read_pickle(args['df_path'])
    df_metadata = pd.read_csv(args['df_meta'])

    # Run UMAP and clustering
    umap_summary = main_clustering(df_masked_raw, df_metadata, args )




