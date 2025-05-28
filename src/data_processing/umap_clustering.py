# umap_clustering.py

import os
import warnings
warnings.filterwarnings("ignore", message="n_jobs value 1 overridden to 1 by setting random_state")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import hdbscan
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from umap_run import x_features_return, run_umap
from clustering_evaluation import evaluate_kmeans, evaluate_gmm, evaluate_hdbscan, evaluate_consensus

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
def plot_clusters_vs_groups(x_umap, labels_dictionary, group_column,save_path, title_prefix, margin = 1.5, plot_flag=True):

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
            'group': group_column
        })

        sns.scatterplot(data=plot_df, x='X1', y='X2', hue='cluster',
                        palette='Set1', s=50, ax=ax_left, legend='full')
        ax_left.set_title(f'{title} - Clustering')
        ax_left.set_xlim(x_min, x_max)
        ax_left.set_ylim(y_min, y_max)

        sns.scatterplot(data=plot_df, x='X1', y='X2', hue='group',
                        palette='Set2', s=50, ax=ax_right, legend='full')
        ax_right.set_title(f'{title} - Group Labeling')
        ax_right.set_xlim(x_min, x_max)
        ax_right.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle(title_prefix, fontsize=18)

    # Save figure if path provided
    if save_path:
        save_file = os.path.join(save_path, f"{title_prefix.replace(' ', '_')}_UMAP_Clustering.png")
        plt.savefig(save_file, dpi=300)
        print(f"Clustering plot saved to: {save_file}")

    # Show figure if requested
    if plot_flag:
        plt.show()

    plt.close()




# ---------------------------
# Main function
# ---------------------------
def main_clustering(df_masked, df_meta, save_path, title_umap, title_cluster, plot_flag=True, do_eval=False, eval_save_path=None):
    # Check that the directory exist
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # Merge voxel and metadata
    df_merged, x = x_features_return(df_masked, df_meta)

    # Reduce dimensionality with UMAP
    x_umap = run_umap(x, plot_flag=True, save_path = save_path,title = title_umap )  # UMAP plotting is disabled

    # Clustering
    labels_dict = run_clustering(x_umap)

    # Evaluation of clustering
    if do_eval:
        print("\nEvaluating clustering algorithms...")
        evaluate_kmeans(x_umap, save_path=eval_save_path, prefix=title_cluster, plot_flag=plot_flag)
        evaluate_gmm(x_umap, save_path=eval_save_path, prefix=title_cluster, plot_flag=plot_flag)
        evaluate_consensus(x_umap, save_path=eval_save_path, prefix=title_cluster, plot_flag=plot_flag)
        print("\n")
        evaluate_hdbscan(x_umap)

    # Collect results
    labeling_umap = pd.DataFrame({
        'labels_hdb': labels_dict['HDBSCAN'],
        'labels_km': labels_dict['K-Means'],
        'X1': x_umap[:, 0],
        'X2': x_umap[:, 1],
        'group': df_merged['Group'],
        'gmm_label':df_merged['GMM_Label'],
        'subject_id': df_merged['ID']
    })

    # Plot and save clusters
    if plot_flag or save_path:
        print("\nClustering results...")
        plot_clusters_vs_groups(x_umap, labels_dict, labeling_umap['group'],save_path, title_cluster, margin=1.5, plot_flag =plot_flag)

        # Plotting also GMM label
        title_cluster = "clustering_gm_02_mask_gmmLabel"
        save_path = None
        plot_clusters_vs_groups(x_umap, labels_dict, labeling_umap['gmm_label'],save_path, title_cluster, margin=1.5, plot_flag =plot_flag)


    return labeling_umap, x_umap
