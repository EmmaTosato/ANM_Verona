import os
import warnings
warnings.filterwarnings("ignore", message="n_jobs value 1 overridden to 1 by setting random_state")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import umap
import hdbscan
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

# ---------------------------
# Merge voxel data with metadata
# ---------------------------

def x_features_return(df_voxel, df_labels):
    # Metadata columns
    meta_columns = list(df_labels.columns)

    # Merge on subject ID
    dataframe_merge = pd.merge(df_voxel, df_labels, on='ID', how='left', validate='one_to_one')

    # Reorder columns: metadata first, then voxel features
    ordered_cols = meta_columns + [col for col in dataframe_merge.columns if col not in meta_columns]
    dataframe_merge = dataframe_merge[ordered_cols]

    # Check row alignment
    assert (dataframe_merge['ID'].values == df_voxel['ID'].values).all(), "Row order mismatch after merge"

    # Extract features only (drop metadata)
    x = dataframe_merge.drop(columns=meta_columns)

    return dataframe_merge, x

# ---------------------------
# Run UMAP and save plot
# ---------------------------

def run_umap(x_input, plot_flag=True, save_path=None, title="UMAP_Embedding"):
    reducer = umap.UMAP(n_neighbors=15, n_components=2, metric='euclidean', n_epochs=1000, learning_rate=1.0, init='spectral', min_dist=0.1, spread=1.0, low_memory=False, set_op_mix_ratio=1.0, local_connectivity=1, repulsion_strength=1.0, negative_sample_rate=5, transform_queue_size=4.0, a=None, b=None, random_state=42, metric_kwds=None, angular_rp_forest=False, target_n_neighbors=-1, transform_seed=42, verbose=False, unique=False)

    x_umap = reducer.fit_transform(x_input)

    if plot_flag:
        plt.figure(figsize=(6, 4))
        plt.scatter(x_umap[:, 0], x_umap[:, 1], s=10, alpha=0.6)
        plt.title(title)
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.grid(True)
        save_file = os.path.join(save_path, f"{title.replace(' ', '_')}.png")
        plt.savefig(save_file, dpi=300)
        print(f"Embeddings plot saved to: {save_file}")
        plt.close()
    return x_umap

# ---------------------------
# Run clustering algorithms
# ---------------------------

def run_clustering(x_umap):
    min_cluster_size = 5
    dbscan_eps = 0.5
    kmeans_n = 3
    gmm_n = 3

    cluster_hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels_hdb = cluster_hdb.fit_predict(x_umap)

    kmeans = KMeans(n_clusters=kmeans_n, random_state=42)
    labels_km = kmeans.fit_predict(x_umap)

    '''
    cluster_db = DBSCAN(eps=dbscan_eps, min_samples=5)
    labels_db = cluster_db.fit_predict(x_umap)


    gmm = GaussianMixture(n_components=gmm_n, random_state=42)
    labels_gmm = gmm.fit_predict(x_umap)
    '''

    labels_dict = {
        "HDBSCAN": labels_hdb,
        #"DBSCAN": labels_db,
        "K-Means": labels_km
        #"GMM": labels_gmm
    }

    return labels_dict

# ---------------------------
# Plot clustering results vs group labels
# ---------------------------

def plot_clusters_vs_groups(x_umap, labels_dictionary, group_column, save_path, title_prefix, plot_flag=True):
    if plot_flag:

        n = len(labels_dictionary)
        n_cols = 2
        n_rows = n

        margin = 1.5  # Ridotto per non schiacciare troppo

        x_min, x_max = x_umap[:, 0].min() - margin, x_umap[:, 0].max() + margin
        y_min, y_max = x_umap[:, 1].min() - margin, x_umap[:, 1].max() + margin

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))  # Più largo e meno alto

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
        plt.subplots_adjust(top=0.9)  # Spazio per il suptitle
        plt.suptitle(title_prefix, fontsize=18)
        save_file = os.path.join(save_path, f"{title_prefix.replace(' ', '_')}_UMAP_Clustering.png")
        plt.savefig(save_file, dpi=300)
        plt.close()
        print(f"Clustering plot saved to: {save_file}")

# ---------------------------
# Main function
# ---------------------------


def main(df_masked, df_meta, save_path, title_umap, title_cluster,  plot_umap_flag=True, plot_cluster_flag=True):
    os.makedirs(save_path, exist_ok=True)

    # Step 1 - Merge voxel and metadata
    df_merged, x = x_features_return(df_masked, df_meta)

    # Step 2 - Run UMAP
    x_umap = run_umap(x, plot_flag=plot_umap_flag, save_path=save_path, title=title_umap)

    # Step 3 - Clustering
    labels_dict = run_clustering(x_umap)

    # Step 4 - Collect results
    labeling_umap = pd.DataFrame({
        'labels_hdb': labels_dict['HDBSCAN'],
        #'labels_db': labels_dict['DBSCAN'],
        'labels_km': labels_dict['K-Means'],
        #'labels_gmm': labels_dict['GMM'],
        'X1': x_umap[:, 0],
        'X2': x_umap[:, 1],
        'group': df_merged['Group'],
        'subject_id': df_merged['ID']
    })

    # Step 5 - Plot and save clusters
    plot_clusters_vs_groups(x_umap, labels_dict, labeling_umap['group'], save_path, title_cluster, plot_flag=plot_cluster_flag)

    return labeling_umap, x_umap
