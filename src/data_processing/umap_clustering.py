# umap_clustering.py

import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import hdbscan
import numpy as np
from sklearn.cluster import KMeans
from umap_run import x_features_return, run_umap
from clustering_evaluation import evaluate_kmeans, evaluate_gmm, evaluate_hdbscan, evaluate_consensus
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
def plot_clusters_vs_groups(x_umap, labels_dictionary, group_column, save_path, title_prefix, margin = 1.8, plot_flag=True):
    n = len(labels_dictionary)
    n_cols = 2
    n_rows = n

    x_min, x_max = x_umap[:, 0].min() - margin, x_umap[:, 0].max() + margin
    y_min, y_max = x_umap[:, 1].min() - margin, x_umap[:, 1].max() + margin

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))


    for i, (title, labels) in enumerate(labels_dictionary.items()):
        ax_left = axes[i, 0]
        ax_right = axes[i, 1]

        plot_df = pd.DataFrame({
            'X1': x_umap[:, 0],
            'X2': x_umap[:, 1],
            'cluster': labels,
            'label': group_column
        })

        sns.scatterplot(data=plot_df, x='X1', y='X2', hue='cluster',palette='Set1', s=50, ax=ax_left, legend='full')
        ax_left.set_title(f'{title} - Clustering after UMAP')
        ax_left.set_xlim(x_min, x_max)
        ax_left.set_ylim(y_min, y_max)

        sns.scatterplot(data=plot_df, x='X1', y='X2', hue='label',palette='Set2', s=50, ax=ax_right, legend='full')
        ax_right.set_title(f'{title} - Labeling according to {group_column.name}')
        ax_right.set_xlim(x_min, x_max)
        ax_right.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    main_title = f"{title_prefix} - Clustering Results"
    plt.suptitle(main_title, fontsize=18)

    # Save figure if path provided
    if save_path:
        clean_prefix = re.sub(r'[\s\-]+', '_', title_prefix.strip())
        save_file = os.path.join(save_path, f"{clean_prefix}_Clustering.png")
        plt.savefig(save_file, dpi=300)

    # Show figure if requested
    if plot_flag:
        plt.show()

    plt.close()




# ---------------------------
# Main function
# ---------------------------
def main_clustering(df_masked, df_meta, title, path_umap=None, path_cluster=None, path_opt_cluster=None, plot_flag=True, do_eval=False):
    # Merge voxel and metadata
    df_merged, x = x_features_return(df_masked, df_meta)

    # Reduce dimensionality with UMAP
    x_umap = run_umap(x, plot_flag=True, save_path=path_umap, title=title)

    # Evaluation of clustering
    if do_eval:
        print("Evaluating clustering algorithms...")
        evaluate_kmeans(x_umap, save_path=path_opt_cluster, prefix=title, plot_flag=plot_flag)
        evaluate_gmm(x_umap, save_path=path_opt_cluster, prefix=title, plot_flag=plot_flag)
        evaluate_consensus(x_umap, save_path=path_opt_cluster, prefix=title, plot_flag=plot_flag)
        evaluate_hdbscan(x_umap)

    # Clustering
    labels_dict = run_clustering(x_umap)

    # Collect results
    labeling_umap = pd.DataFrame({
        'labels_hdb': labels_dict['HDBSCAN'],
        'labels_km': labels_dict['K-Means'],
        'X1': x_umap[:, 0],
        'X2': x_umap[:, 1],
        'group': df_merged['Group'],
        'labels_gmm_cdr': df_merged['labels_gmm_cdr'],
        'ID': df_merged['ID']
    })

    # Plot and save clusters
    if plot_flag or path_cluster:
        print("Running clustering...")
        # Plot according to group
        plot_clusters_vs_groups(x_umap, labels_dict, labeling_umap['group'], path_cluster, title, margin=1.5, plot_flag=plot_flag)

        # Plot according to gmm labels cdr
        title_cluster = title + " GMM Label"
        plot_clusters_vs_groups(x_umap, labels_dict, labeling_umap['gmm_label_cdr'], path_cluster, title_cluster, margin=1.5, plot_flag=plot_flag)

    # Merge and save only if clustering columns are not already present
    if 'labels_km' not in df_meta.columns and 'labels_hdb' not in df_meta.columns:
        df_meta = df_meta.merge(labeling_umap[['ID', 'labels_km', 'labels_hdb']], on='ID', how='left')
        df_meta.to_csv(config['df_meta'], index=False)

    return labeling_umap, x_umap


if __name__ == "__main__":
    # Load json
    print("\nLoading config and metadata...")
    with open("src/data_processing/config.json", "r") as f:
        config = json.load(f)

    with open("src/data_processing/run.json", "r") as f:
        run = json.load(f)

    config.update(run)

    # Load configuration and metadata
    df_masked = pd.read_pickle(config['df_masked'])
    df_meta = pd.read_csv(config['df_meta'])

    # Run UMAP and clustering
    labeling, x_umap = main_clustering(
        df_masked,
        df_meta,
        title=config['prefix_cluster'],
        path_umap=config['path_umap'],
        path_cluster=config['path_cluster'],
        path_opt_cluster=config['path_opt_cluster'],
        plot_flag=config['plot_cluster'],
        do_eval=config['plot_evaluation']
    )